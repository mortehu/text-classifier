#ifndef TOOLS_TEXT_CLASSIFIER_TABLE_STORAGE_H_
#define TOOLS_TEXT_CLASSIFIER_TABLE_STORAGE_H_

#include <experimental/string_view>
#include <fstream>
#include <tuple>
#include <utility>

#include <kj/debug.h>

namespace ev {

using string_view = std::experimental::string_view;

class TableWriter {
 public:
  TableWriter(std::ofstream output) : output_(std::move(output)) {
    KJ_REQUIRE(output_.is_open());
    output_.exceptions(output_.badbit | output_.failbit);
  }

  template <typename... T>
  void PutRow(const std::tuple<T...>& data) {
    Put(std::index_sequence_for<T...>{}, data);
  }

  void Flush() {
    output_.flush();
  }

 private:
  std::ofstream output_;

  template <std::size_t... I, typename... T>
  void Put(std::index_sequence<I...>, const std::tuple<T...>& value) {
    const auto l __attribute__((unused)) = {(Put(std::get<I>(value)), 0)...};
  }

  void Put(const string_view& str) {
    output_ << str.size() << ' ';
    output_.write(str.data(), str.size());
  }
};

class TableReader {
 public:
  TableReader(std::ifstream input) : input_(std::move(input)) {
    KJ_REQUIRE(input_.is_open());
    input_.exceptions(input_.badbit | input_.failbit);
  }

  template <typename... T>
  bool GetRow(std::tuple<T...>& result) {
    return Get(std::index_sequence_for<T...>{}, result);
  }

  void SeekToStart() {
    input_.seekg(0);
    input_.clear();
  }

 private:
  std::ifstream input_;

  template <std::size_t... I, typename... T>
  bool Get(std::index_sequence<I...>, std::tuple<T...>& value) {
    if (std::ifstream::traits_type::eof() == input_.peek()) return false;

    const auto l __attribute__((unused)) = {(Get(std::get<I>(value)), 0)...};

    return true;
  }

  void Get(std::string& str) {
    size_t size;
    input_ >> size;

    char space = input_.get();
    KJ_REQUIRE(space == ' ', "Data error", (int)space);

    str.resize(size);
    input_.read(const_cast<char*>(str.data()), str.size());
  }
};

}  // namespace ev

#endif  // !TOOLS_TEXT_CLASSIFIER_TABLE_STORAGE_H_
