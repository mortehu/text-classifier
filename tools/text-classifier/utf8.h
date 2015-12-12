#ifndef TOOLS_TEXT_CLASSIFIER_UTF8_H_
#define TOOLS_TEXT_CLASSIFIER_UTF8_H_ 1

namespace ev {

// Reads the next UTF-8 character from `input`.  The input object is modified
// so that the character
inline uint32_t GetUtf8Char(ev::StringRef* input) {
  uint32_t result = static_cast<uint8_t>(input->front());
  input->pop_front();

  if (result < 0x80) return result;

  unsigned int n;
  if ((result & 0xe0) == 0xc0) {
    result &= 0x1f;
    n = 1;
  } else if ((result & 0xf0) == 0xe0) {
    result &= 0x0f;
    n = 2;
  } else if ((result & 0xf8) == 0xf0) {
    result &= 0x07;
    n = 3;
  } else if ((result & 0xfC) == 0xf8) {
    result &= 0x03;
    n = 4;
  } else if ((result & 0xfE) == 0xfc) {
    result &= 0x01;
    n = 5;
  } else {
    return result;
  }

  while (n--) {
    if (input->empty()) return result;  // Parse error.

    const auto b = static_cast<uint8_t>(input->front());

    if ((b & 0xc0) != 0x80) return result;  // Parse error.

    result <<= 6;
    result |= (b & 0x3f);

    input->pop_front();
  }

  return result;
}

}  // namespace ev

#endif  // !TOOLS_TEXT_CLASSIFIER_UTF8_H_
