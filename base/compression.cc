#if HAVE_CONFIG_H
#include "config.h"
#endif

#include "base/compression.h"

#if HAVE_ZLIB
#include <zlib.h>
#endif

#include "base/stringref.h"
#include "base/thread-pool.h"

namespace {

uint32_t CombineAdler32(const uint32_t adler1, const uint32_t adler2,
                        const size_t length) {
  const uint32_t kBase = 65521;

  const auto remainder = length % kBase;
  auto sum1 = adler1 & 0xffff;
  auto sum2 = (remainder * sum1) % kBase;
  sum1 += (adler2 & 0xffff) + kBase - 1;
  sum2 += (adler1 >> 16) + (adler2 >> 16) + kBase - remainder;

  if (sum1 >= kBase) {
    sum1 -= kBase;
    if (sum1 >= kBase) sum1 -= kBase;
  }

  if (sum2 >= kBase * 2) sum2 -= kBase * 2;
  if (sum2 >= kBase) sum2 -= kBase;

  return sum1 | (sum2 << 16);
}

}  // namespace

namespace ev {

void CompressZLIB(std::string& output, const ev::StringRef& input,
                  ev::ThreadPool& thread_pool) {
#if HAVE_ZLIB
  static const size_t kBlockSize = 512 * 1024;
  static const auto kCompressionLevel = Z_DEFAULT_COMPRESSION;

  // First, emit 2 byte ZLIB header.
  output.push_back(0x78);  // Deflate, 32KB window
  output.push_back(0x01);  // Default compression level

  static const auto kChecksum0 = adler32(0, nullptr, 0);
  auto checksum = kChecksum0;

  const auto concurrency = std::max(std::thread::hardware_concurrency(), 1U);

  std::deque<std::future<std::tuple<std::string, uint32_t, size_t>>> queue;

  for (size_t i = 0; i < input.size(); i += kBlockSize) {
    const auto block_size = std::min(input.size() - i, kBlockSize);
    const auto finish = i + block_size == input.size();

    const auto block_data = input.substr(i, block_size);

    while (queue.size() >= concurrency) {
      const auto block = queue.front().get();
      queue.pop_front();

      output += std::get<std::string>(block);

      checksum = CombineAdler32(checksum, std::get<uint32_t>(block),
                                std::get<size_t>(block));
    }

    queue.emplace_back(thread_pool.Launch([block_data, finish] {
      z_stream zs;
      memset(&zs, 0, sizeof(zs));

      KJ_REQUIRE(Z_OK ==
                 deflateInit2(&zs, 6, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY));
      KJ_DEFER(deflateEnd(&zs));

      deflateParams(&zs, kCompressionLevel, Z_DEFAULT_STRATEGY);

      zs.next_in =
          reinterpret_cast<Bytef*>(const_cast<char*>(block_data.data()));
      zs.avail_in = block_data.size();
      zs.total_in = block_data.size();

      std::string result;
      result.resize(deflateBound(&zs, block_data.size()));

      zs.next_out = reinterpret_cast<uint8_t*>(&result[0]);
      zs.avail_out = result.size();

      if (finish) {
        const auto deflate_ret = deflate(&zs, Z_FINISH);
        KJ_REQUIRE(Z_STREAM_END == deflate_ret, deflate_ret);
      } else {
        auto deflate_ret = deflate(&zs, Z_BLOCK);
        KJ_REQUIRE(Z_OK == deflate_ret, deflate_ret);

        int bits;
        deflatePending(&zs, Z_NULL, &bits);

        if (bits & 1) {
          deflate_ret = deflate(&zs, Z_SYNC_FLUSH);
          KJ_REQUIRE(Z_OK == deflate_ret, deflate_ret);
        } else if (bits & 7) {
          do {
            bits = deflatePrime(&zs, 10, 2);
            KJ_REQUIRE(bits == Z_OK, bits);
            deflatePending(&zs, Z_NULL, &bits);
          } while (bits & 7);

          deflate_ret = deflate(&zs, Z_BLOCK);
          KJ_REQUIRE(Z_OK == deflate_ret, deflate_ret);
        }
      }

      const auto compressed_length = zs.total_out;
      KJ_REQUIRE(compressed_length <= result.size());
      result.resize(compressed_length);

      const uint32_t block_checksum =
          adler32(kChecksum0, reinterpret_cast<const Bytef*>(block_data.data()),
                  block_data.size());

      return std::make_tuple(result, block_checksum, block_data.size());
    }));
  }

  while (!queue.empty()) {
    const auto block = queue.front().get();
    queue.pop_front();

    output += std::get<std::string>(block);

    checksum = CombineAdler32(checksum, std::get<uint32_t>(block),
                              std::get<size_t>(block));
  }

  output.push_back(checksum >> 24);
  output.push_back(checksum >> 16);
  output.push_back(checksum >> 8);
  output.push_back(checksum);
#else
  KJ_FAIL_REQUIRE("No ZLIB support");
#endif
}

}  // namespace ev
