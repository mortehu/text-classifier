#include <fcntl.h>

#include <capnp/schema-parser.h>
#include <capnp/serialize.h>

#include "base/cat.h"
#include "base/columnfile-capnp.h"
#include "base/columnfile-internal.h"
#include "base/columnfile.h"
#include "base/file.h"
#include "base/string.h"
#include "third_party/gtest/gtest.h"

using namespace ev;

struct ColumnFileTest : public testing::Test {};

TEST_F(ColumnFileTest, WriteTableToFile) {
  const auto tmp_dir = TemporaryDirectory();
  DirectoryTreeRemover rm_tmp(tmp_dir);

  const auto tmp_path = ev::cat(tmp_dir, "/test00");
  ColumnFileWriter writer(tmp_path.c_str());

  writer.Put(0, "2000-01-01");
  writer.Put(1, "January");
  writer.Put(2, "First");

  writer.Put(0, "2000-01-02");
  writer.Put(1, "January");
  writer.Put(2, "Second");

  writer.Put(0, "2000-02-02");
  writer.Put(1, "February");
  writer.Put(2, "Second");
  writer.Flush();

  writer.Put(0, "2000-02-03");
  writer.Put(1, "February");
  writer.Put(2, "Third");

  writer.Put(0, "2000-02-03");
  writer.PutNull(1);
  writer.PutNull(2);

  writer.Finalize();

  ColumnFileReader reader(OpenFile(tmp_path.c_str(), O_RDONLY));

  EXPECT_FALSE(reader.End());

  auto row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-01-01", row[0].second.StringRef().str());
  EXPECT_EQ("January", row[1].second.StringRef().str());
  EXPECT_EQ("First", row[2].second.StringRef().str());

  row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-01-02", row[0].second.StringRef().str());
  EXPECT_EQ("January", row[1].second.StringRef().str());
  EXPECT_EQ("Second", row[2].second.StringRef().str());

  row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-02-02", row[0].second.StringRef().str());
  EXPECT_EQ("February", row[1].second.StringRef().str());
  EXPECT_EQ("Second", row[2].second.StringRef().str());

  row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-02-03", row[0].second.StringRef().str());
  EXPECT_EQ("February", row[1].second.StringRef().str());
  EXPECT_EQ("Third", row[2].second.StringRef().str());

  EXPECT_FALSE(reader.End());

  row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-02-03", row[0].second.StringRef().str());
  EXPECT_TRUE(row[1].second.IsNull());
  EXPECT_TRUE(row[2].second.IsNull());

  EXPECT_TRUE(reader.End());
}

TEST_F(ColumnFileTest, WriteTableToString) {
  std::string buffer;

  ColumnFileWriter writer(buffer);
  writer.Put(0, "2000-01-01");
  writer.Put(1, "January");
  writer.Put(2, "First");

  writer.Put(0, "2000-01-02");
  writer.Put(1, "January");
  writer.Put(2, "Second");
  writer.Flush();

  writer.Put(0, "2000-02-02");
  writer.Put(1, "February");
  writer.Put(2, "Second");

  std::string long_string(0xfff, 'x');
  writer.Put(0, "2000-02-03");
  writer.Put(1, "February");
  writer.Put(2, long_string);

  writer.Put(0, "2000-02-03");
  writer.PutNull(1);
  writer.PutNull(2);
  writer.Finalize();

  ColumnFileReader reader(buffer);

  EXPECT_FALSE(reader.End());

  auto row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-01-01", row[0].second.StringRef().str());
  EXPECT_EQ("January", row[1].second.StringRef().str());
  EXPECT_EQ("First", row[2].second.StringRef().str());

  row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-01-02", row[0].second.StringRef().str());
  EXPECT_EQ("January", row[1].second.StringRef().str());
  EXPECT_EQ("Second", row[2].second.StringRef().str());

  row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-02-02", row[0].second.StringRef().str());
  EXPECT_EQ("February", row[1].second.StringRef().str());
  EXPECT_EQ("Second", row[2].second.StringRef().str());

  row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-02-03", row[0].second.StringRef().str());
  EXPECT_EQ("February", row[1].second.StringRef().str());
  EXPECT_EQ(long_string, row[2].second.StringRef().str());

  EXPECT_FALSE(reader.End());

  row = reader.GetRow();
  EXPECT_EQ(3U, row.size());
  EXPECT_EQ("2000-02-03", row[0].second.StringRef().str());
  EXPECT_TRUE(row[1].second.IsNull());
  EXPECT_TRUE(row[2].second.IsNull());

  EXPECT_TRUE(reader.End());
}

TEST_F(ColumnFileTest, WriteMessageToString) {
  capnp::SchemaParser schema_parser;
  kj::ArrayPtr<const kj::StringPtr> import_path;
  auto parsed_schema = schema_parser.parseDiskFile(
      "base/testdata/addressbook.capnp", "base/testdata/addressbook.capnp",
      import_path);
  auto address_book_schema = parsed_schema.getNested("AddressBook");

  kj::Array<capnp::word> words;

  capnp::MallocMessageBuilder orig_message;
  auto orig_address_book = orig_message.initRoot<capnp::DynamicStruct>(
      address_book_schema.asStruct());

  {
    auto people = orig_address_book.init("people", 2).as<capnp::DynamicList>();

    auto alice = people[0].as<capnp::DynamicStruct>();
    alice.set("id", 123);
    alice.set("name", "Alice");
    alice.set("email", "alice@example.com");
    auto alice_phones = alice.init("phones", 1).as<capnp::DynamicList>();
    auto phone0 = alice_phones[0].as<capnp::DynamicStruct>();
    phone0.set("number", "555-1212");
    phone0.set("type", "mobile");
#if 0
    // Requires support for unions
    alice.get("employment").as<capnp::DynamicStruct>().set("school", "MIT");
#endif

    auto bob = people[1].as<capnp::DynamicStruct>();
    bob.set("id", 456);
    bob.set("name", "Bob");
    bob.set("email", "bob@example.com");

    words = capnp::messageToFlatArray(orig_message);
  }

  std::string buffer;

  {
    ColumnFileWriter writer(buffer);

    capnp::FlatArrayMessageReader message_reader(words);
    WriteMessageToColumnFile(writer,
                             message_reader.getRoot<capnp::DynamicStruct>(
                                 address_book_schema.asStruct()));
    writer.Finalize();
  }

  {
    ColumnFileReader reader(buffer);

    capnp::MallocMessageBuilder message;
    auto address_book =
        message.initRoot<capnp::DynamicStruct>(address_book_schema.asStruct());

    ReadMessageFromColumnFile(reader, address_book);

    EXPECT_TRUE(address_book.asReader().as<capnp::AnyStruct>() ==
                orig_address_book.asReader().as<capnp::AnyStruct>());
  }
}

TEST_F(ColumnFileTest, AFLTestCases) {
  std::vector<std::string> test_cases;

  ev::FindFiles("base/testdata", [&test_cases](auto path) {
    if (ev::HasSuffix(path, ".col")) test_cases.emplace_back(std::move(path));
  });

  EXPECT_LT(0U, test_cases.size());

  for (const auto& path : test_cases) {
    try {
      ev::ColumnFileReader reader(ev::OpenFile(path.c_str(), O_RDONLY));
      while (!reader.End()) reader.GetRow();
    } catch (kj::Exception e) {
    } catch (std::bad_alloc e) {
    }
  }
}

TEST_F(ColumnFileTest, IntegerCoding) {
  static const uint32_t kTestNumbers[] = {
      0,          0x10U,      0x7fU,       0x80U,      0x100U,    0x1000U,
      0x3fffU,    0x4000U,    0x10000U,    0x100000U,  0x1fffffU, 0x200000U,
      0x1000000U, 0xfffffffU, 0x10000000U, 0xffffffffU};

  for (auto i : kTestNumbers) {
    std::string buffer;
    ev::columnfile_internal::PutInt(buffer, i);

    EXPECT_TRUE((static_cast<uint8_t>(buffer[0]) & 0xc0) != 0xc0);

    ev::StringRef read_buffer(buffer);
    auto decoded_int = ev::columnfile_internal::GetInt(read_buffer);
    EXPECT_EQ(i, decoded_int);
    EXPECT_TRUE(read_buffer.empty());
  }
}
