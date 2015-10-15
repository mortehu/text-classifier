#ifndef INDEX_WEB_TAGSOUP_H_
#define INDEX_WEB_TAGSOUP_H_ 1

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "base/region.h"
#include "base/stringref.h"

namespace ev {

extern const char* TAGSOUP_NODE_ROOT;
extern const char* TAGSOUP_NODE_CONTENT;
extern const char* TAGSOUP_NODE_COMMENT;

struct TagsoupAttribute {
  ev::StringRef name = nullptr;
  ev::StringRef content = nullptr;

  TagsoupAttribute* next_sibling = nullptr;
};

struct TagsoupNode {
  ev::StringRef GetAttribute(const ev::StringRef& name);

  ev::StringRef name = nullptr;
  ev::StringRef content = nullptr;

  TagsoupNode* parent = nullptr;
  TagsoupNode* next_sibling = nullptr;

  TagsoupNode* first_child = nullptr;
  TagsoupNode* last_child = nullptr;

  TagsoupAttribute* first_attribute = nullptr;
  TagsoupAttribute* last_attribute = nullptr;
};

class Tagsoup {
 public:
  Tagsoup(const ev::StringRef& input,
          ev::concurrency::RegionPool::Region&& region);

  ~Tagsoup();

  void Parse(const ev::StringRef& input);

  ev::StringRef DocType() const { return doctype_; }

  const TagsoupNode* Root() const { return root_; }

 private:
  ev::StringRef AddString(const char* begin, const char* end);

  // Adds text content to the given node.  `type' can be one of
  // TAGSOUP_NODE_CONTENT and TAGSOUP_NODE_COMMENT.
  void AddContent(TagsoupNode* current_node, const char* begin, const char* end,
                  const char* type, bool need_escape);

  template <typename T>
  T* Allocate(size_t n = 1) {
    auto result =
        reinterpret_cast<T*>(region_.EphemeralAllocate(n * sizeof(T)));
    if (!std::is_pod<T>::value) {
      for (size_t i = 0; i < n; ++i) new (result + i) T();
    }

    return result;
  }

  ev::concurrency::RegionPool::Region region_;

  ev::StringRef doctype_;

  TagsoupNode* root_ = nullptr;
};

// Writes an entire document to a file, including any DOCTYPE declaration.
void tagsoup_print_doc(FILE* output, const Tagsoup* doc);

// Writes a document subtree to a file.
void tagsoup_print(FILE* output, const TagsoupNode* node);

}  // namespace ev

#endif  // !INDEX_WEB_TAGSOUP_H_
