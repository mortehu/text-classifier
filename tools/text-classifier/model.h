#ifndef TOOLS_TEXT_CLASSIFIER_MODEL_H_
#define TOOLS_TEXT_CLASSIFIER_MODEL_H_ 1

#include <cinttypes>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include <kj/debug.h>
#include <kj/io.h>

#include "tools/text-classifier/common.h"

namespace ev {

class ColumnFileReader;

class TextClassifierModel {
 public:
  virtual ~TextClassifierModel();

  virtual void Train(ev::ColumnFileReader input) = 0;

  virtual void Save(kj::AutoCloseFd output) const = 0;

  virtual void Load(kj::AutoCloseFd input) = 0;

  virtual void Print() const = 0;

  virtual double Classify(const std::vector<uint64_t>& features) const = 0;
};

class TextClassifierModelFactory {
 public:
  typedef std::function<std::unique_ptr<TextClassifierModel>(
      TextClassifierParams params)> Generator;

  static TextClassifierModelFactory* GetInstance();

  void RegisterModelType(Generator generator, std::string name) {
    generators_.emplace(std::move(name), std::move(generator));
  }

  std::vector<std::string> GetModelTypes() const {
    std::vector<std::string> result;
    for (const auto& g : generators_) result.emplace_back(g.first);
    return result;
  }

  std::unique_ptr<TextClassifierModel> CreateModel(
      const std::string& type, TextClassifierParams params) {
    auto i = generators_.find(type);
    KJ_REQUIRE(i != generators_.end(), type);
    return i->second(std::move(params));
  }

 private:
  std::unordered_map<std::string, Generator> generators_;
};

template <typename T>
class TextClassifierModelRegistrar {
 public:
  TextClassifierModelRegistrar(const char* prefix) {
    TextClassifierModelFactory::GetInstance()->RegisterModelType(
        [](TextClassifierParams params) {
          return std::make_unique<T>(std::move(params));
        },
        prefix);
  }
};

}  // namespace ev

#define REGISTER_TEXT_CLASSIFIER_MODEL(type, prefix)                     \
  namespace {                                                            \
  ::ev::TextClassifierModelRegistrar<type> counter_registrar_for_##type( \
      prefix);                                                           \
  }

#endif  // !TOOLS_TEXT_CLASSIFIER_MODEL_H_
