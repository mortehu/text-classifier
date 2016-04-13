#include "tools/text-classifier/model.h"

namespace ev {

TextClassifierModel::~TextClassifierModel() {}

TextClassifierModelFactory* TextClassifierModelFactory::GetInstance() {
  static TextClassifierModelFactory instance;
  return &instance;
}

}  // namespace ev
