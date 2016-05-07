#include "index/web/tagsoup.h"

#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <experimental/string_view>
#include <unordered_map>
#include <unordered_set>

#include <kj/debug.h>

#include "base/macros.h"

namespace ev {

using string_view = std::experimental::string_view;

const char* TAGSOUP_NODE_ROOT = "#<ROOT>#";
const char* TAGSOUP_NODE_CONTENT = "#<CONTENT>#";
const char* TAGSOUP_NODE_COMMENT = "#<COMMENT>#";

namespace {

static const ev::StringRef kScriptTagName("script");
static const ev::StringRef kStyleTagName("style");
static const ev::StringRef kScriptEndTag("</script>");
static const ev::StringRef kStyleEndTag("</style>");

enum parse_state {
  s_doctype,
  s_tag_name,
  s_attrib_delim,
  s_attrib_name,
  s_attrib_value,
  s_content
};

static const std::unordered_map<string_view, char16_t> kEntities = {
    {"Mu", 924},         {"Nu", 925},         {"Pi", 928},
    {"Xi", 926},         {"ge", 8805},        {"gt", '>'},
    {"le", 8804},        {"lt", '<'},         {"mu", 956},
    {"ne", 8800},        {"ni", 8715},        {"nu", 957},
    {"or", 8744},        {"pi", 960},         {"xi", 958},

    {"Chi", 935},        {"ETH", L'\320'},    {"Eta", 919},
    {"Phi", 934},        {"Psi", 936},        {"Rho", 929},
    {"Tau", 932},        {"amp", '&'},        {"and", 8743},
    {"ang", 8736},       {"cap", 8745},       {"chi", 967},
    {"cup", 8746},       {"deg", L'\260'},    {"eta", 951},
    {"eth", L'\360'},    {"int", 8747},       {"loz", 9674},
    {"lrm", 0x8206},     {"not", L'\254'},    {"phi", 966},
    {"piv", 982},        {"psi", 968},        {"reg", L'\256'},
    {"rho", 961},        {"rlm", 0x8207},     {"shy", L'\255'},
    {"sim", 8764},       {"sub", 8834},       {"sum", 8721},
    {"sup", 8835},       {"tau", 964},        {"uml", L'\250'},
    {"yen", L'\245'},    {"zwj", 0x8205},

    {"Auml", L'\304'},   {"Beta", 914},       {"Euml", L'\313'},
    {"Iota", 921},       {"Iuml", L'\317'},   {"Ouml", L'\326'},
    {"Uuml", L'\334'},   {"Yuml", 0x376},     {"Zeta", 918},
    {"auml", L'\344'},   {"beta", 946},       {"bull", 8226},
    {"cent", L'\242'},   {"circ", 0x710},     {"cong", 8773},
    {"copy", 169},       {"dArr", 8659},      {"darr", 8595},
    {"emsp", 0x8195},    {"ensp", 0x8194},    {"euml", L'\353'},
    {"euro", 0x8364},    {"fnof", 402},       {"hArr", 8660},
    {"harr", 8596},      {"iota", 953},       {"isin", 8712},
    {"iuml", L'\357'},   {"lArr", 8656},      {"lang", 9001},
    {"larr", 8592},      {"macr", L'\257'},   {"nbsp", L'\240'},
    {"nsub", 8836},      {"ordf", L'\252'},   {"ordm", L'\272'},
    {"ouml", L'\366'},   {"para", L'\266'},   {"part", 8706},
    {"perp", 8869},      {"prod", 8719},      {"prop", 8733},
    {"quot", '"'},       {"rArr", 8658},      {"rang", 9002},
    {"rarr", 8594},      {"real", 8476},      {"sdot", 8901},
    {"sect", L'\247'},   {"sube", 8838},      {"sup1", L'\271'},
    {"sup2", L'\262'},   {"sup3", L'\263'},   {"supe", 8839},
    {"uArr", 8657},      {"uarr", 8593},      {"uuml", L'\374'},
    {"yuml", L'\377'},   {"zeta", 950},       {"zwnj", 0x8204},

    {"AElig", 198},      {"Acirc", L'\302'},  {"Alpha", 913},
    {"Aring", 197},      {"Delta", 916},      {"Ecirc", L'\312'},
    {"Gamma", 915},      {"Icirc", L'\316'},  {"Kappa", 922},
    {"OElig", 0x338},    {"Ocirc", L'\324'},  {"Omega", 937},
    {"Prime", 8243},     {"Sigma", 931},      {"THORN", L'\336'},
    {"Theta", 920},      {"Ucirc", L'\333'},  {"acirc", L'\342'},
    {"acute", L'\264'},  {"aelig", 230},      {"alpha", 945},
    {"aring", 229},      {"asymp", 8776},     {"bdquo", 0x8222},
    {"cedil", L'\270'},  {"clubs", 9827},     {"crarr", 8629},
    {"delta", 948},      {"diams", 9830},     {"ecirc", L'\352'},
    {"empty", 8709},     {"equiv", 8801},     {"exist", 8707},
    {"frasl", 8260},     {"gamma", 947},      {"icirc", L'\356'},
    {"iexcl", 161},      {"image", 8465},     {"infin", 8734},
    {"kappa", 954},      {"laquo", L'\253'},  {"lceil", 8968},
    {"ldquo", 0x8220},   {"lsquo", 0x8216},   {"mdash", 0x8212},
    {"micro", L'\265'},  {"minus", 8722},     {"nabla", 8711},
    {"ndash", 0x8211},   {"notin", 8713},     {"ocirc", L'\364'},
    {"oelig", 0x339},    {"oline", 8254},     {"omega", 969},
    {"oplus", 8853},     {"pound", L'\243'},  {"prime", 8242},
    {"radic", 8730},     {"raquo", L'\273'},  {"rceil", 8969},
    {"rdquo", 0x8221},   {"rsquo", 0x8217},   {"sbquo", 0x8218},
    {"sigma", 963},      {"szlig", L'\337'},  {"theta", 952},
    {"thorn", L'\376'},  {"tilde", 0x732},    {"times", L'\327'},
    {"trade", 8482},     {"ucirc", L'\373'},  {"upsih", 978},
    {"Lambda", 923},

    {"Aacute", L'\301'}, {"Agrave", L'\300'}, {"Atilde", L'\303'},
    {"Ccedil", L'\307'}, {"Dagger", 0x8225},  {"Eacute", L'\311'},
    {"Egrave", L'\310'}, {"Iacute", L'\315'}, {"Igrave", L'\314'},
    {"Ntilde", L'\321'}, {"Oacute", L'\323'}, {"Ograve", L'\322'},
    {"Oslash", 216},     {"Otilde", L'\325'}, {"Scaron", 0x352},
    {"Uacute", L'\332'}, {"Ugrave", L'\331'}, {"Yacute", L'\335'},
    {"aacute", L'\341'}, {"agrave", L'\340'}, {"atilde", L'\343'},
    {"brvbar", L'\246'}, {"ccedil", L'\347'}, {"curren", L'\244'},
    {"dagger", 0x8224},  {"divide", L'\367'}, {"eacute", L'\351'},
    {"egrave", L'\350'}, {"forall", 8704},    {"frac12", L'\275'},
    {"frac14", L'\274'}, {"frac34", L'\276'}, {"hearts", 9829},
    {"hellip", 8230},    {"iacute", L'\355'}, {"igrave", L'\354'},
    {"iquest", L'\277'}, {"lambda", 955},     {"lfloor", 8970},
    {"lowast", 8727},    {"lsaquo", 0x8249},  {"middot", L'\267'},
    {"ntilde", L'\361'}, {"oacute", L'\363'}, {"ograve", L'\362'},
    {"oslash", 248},     {"otilde", L'\365'}, {"otimes", 8855},
    {"permil", 0x8240},  {"plusmn", L'\261'}, {"rfloor", 8971},
    {"rsaquo", 0x8250},  {"scaron", 0x353},   {"sigmaf", 962},
    {"spades", 9824},    {"there4", 8756},    {"thinsp", 0x8201},
    {"uacute", L'\372'}, {"ugrave", L'\371'}, {"weierp", 8472},
    {"yacute", L'\375'},

    {"Epsilon", 917},    {"Omicron", 927},    {"Upsilon", 933},
    {"alefsym", 8501},   {"epsilon", 949},    {"omicron", 959},
    {"upsilon", 965},

    {"thetasym", 977}};

static const std::unordered_set<string_view> kEmptyElements{
    "br",   "hr",    "col",   "img",   "area",    "base",     "link",
    "meta", "frame", "input", "param", "isindex", "basefront"};

static const std::unordered_map<string_view, int> kAutocloseeElements{
    {"p", 0},     {"dd", 1},    {"dt", 1},       {"li", 2},
    {"td", 3},    {"th", 3},    {"tr", 4},       {"tbody", 5},
    {"tfoot", 5}, {"thead", 5}, {"colgroup", 4}, {"option", 6},
};

static const std::unordered_map<string_view, int> kAutocloserElements{
    {"colgroup", (1 << 4)}, {"dd", (1 << 0) | (1 << 1)},
    {"div", (1 << 0)},      {"dt", (1 << 0)},
    {"dt", (1 << 1)},       {"h1", (1 << 0)},
    {"h2", (1 << 0)},       {"h3", (1 << 0)},
    {"h4", (1 << 0)},       {"hr", (1 << 0)},
    {"li", (1 << 2)},       {"ol", (1 << 0)},
    {"option", (1 << 6)},   {"p", (1 << 0)},
    {"pre", (1 << 0)},      {"table", (1 << 0)},
    {"tbody", (1 << 5)},    {"td", (1 << 3)},
    {"tfoot", (1 << 5)},    {"th", (1 << 3)},
    {"thead", (1 << 5)},    {"tr", (1 << 3) | (1 << 4)},
    {"ul", (1 << 0)},
};

static int get_autoclosee_group(const char* name, size_t length) {
  const auto i = kAutocloseeElements.find(string_view{name, length});
  if (i == kAutocloseeElements.end()) return -1;
  return std::get<int>(*i);
}

static int does_autoclose(const char* name, size_t length, int group) {
  const auto i = kAutocloserElements.find(string_view{name, length});
  if (i == kAutocloserElements.end()) return 0;
  return 0 != (std::get<int>(*i) & (1 << group));
}

static int is_empty_element(const char* name, size_t length) {
  return kEmptyElements.count(string_view{name, length});
}

static char* tagsoup_utf8_put(char* output, char32_t ch) {
  if (ch < 0x80) {
    *output++ = ch;
  } else if (ch < 0x800) {
    *output++ = 0xc0 | (ch >> 6);
    *output++ = 0x80 | (ch & 0x3f);
  } else if (ch < 0x10000) {
    *output++ = 0xe0 | (ch >> 12);
    *output++ = 0x80 | ((ch >> 6) & 0x3f);
    *output++ = 0x80 | (ch & 0x3f);
  } else if (ch < 0x200000) {
    *output++ = 0xf0 | (ch >> 18);
    *output++ = 0x80 | ((ch >> 12) & 0x3f);
    *output++ = 0x80 | ((ch >> 6) & 0x3f);
    *output++ = 0x80 | (ch & 0x3f);
  } else if (ch < 0x4000000) {
    *output++ = 0xf8 | (ch >> 24);
    *output++ = 0x80 | ((ch >> 18) & 0x3f);
    *output++ = 0x80 | ((ch >> 12) & 0x3f);
    *output++ = 0x80 | ((ch >> 6) & 0x3f);
    *output++ = 0x80 | (ch & 0x3f);
  } else if (ch < 0x80000000) {
    *output++ = 0xfc | (ch >> 30);
    *output++ = 0x80 | ((ch >> 24) & 0x3f);
    *output++ = 0x80 | ((ch >> 18) & 0x3f);
    *output++ = 0x80 | ((ch >> 12) & 0x3f);
    *output++ = 0x80 | ((ch >> 6) & 0x3f);
    *output++ = 0x80 | (ch & 0x3f);
  }

  return output;
}

bool isspace_HTML(uint8_t ch) {
  static const uint64_t kSpaceBits =
      (1ULL << 0x20ULL) | (1 << 0x09) | (1 << 0x0a) | (1 << 0x0c) | (1 << 0x0d);

  return ch <= 0x20 && ((1ULL << ch) & kSpaceBits);
}

uint8_t tolower_ASCII(uint8_t ch) {
  if (ch >= 'A' && ch <= 'Z') ch |= 0x40;

  return ch;
}

}  // namespace

Tagsoup::Tagsoup(const ev::StringRef& input,
                 ev::concurrency::RegionPool::Region region)
    : region_(std::move(region)) {
  Parse(input);
}

Tagsoup::~Tagsoup() {}

ev::StringRef Tagsoup::AddString(const char* begin, const char* end) {
  const char* i;
  size_t len;
  char* result;
  char* o;

  len = end - begin + 1;
  result = Allocate<char>(len);
  o = result;

  i = begin;

  while (i != end) {
    auto next_ent = std::find(i, end, '&');

    memcpy(o, i, next_ent - i);
    o += next_ent - i;
    i = next_ent;

    if (i == end) break;

    const char* ent_begin = i + 1;
    const char* ent_end = ent_begin;

    while (ent_end != end &&
           (isalnum(*ent_end) || *ent_end == '#' || *ent_end == 'x'))
      ++ent_end;

    if (ent_begin[0] == '#') {
      unsigned int ch;
      char* end;

      // TODO(mortehu): Replace the use of strtol, so that we don't require
      // the input string to be NUL terminate.

      if (ent_begin[1] == 'x')
        ch = strtol(ent_begin + 2, &end, 16);
      else
        ch = strtol(ent_begin + 1, &end, 10);

      if (end == ent_end) {
        o = tagsoup_utf8_put(o, ch);

        i = ent_end;

        if (*ent_end == ';') ++i;

        continue;
      }
    } else {
      const auto ent = kEntities.find(
          string_view{ent_begin, static_cast<size_t>(ent_end - ent_begin)});

      if (ent != kEntities.end()) {
        o = tagsoup_utf8_put(o, std::get<char16_t>(*ent));
        i = ent_end;
        if (*ent_end == ';') ++i;

        continue;
      }
    }

    *o++ = *i++;
  }

  *o++ = 0;

  assert((size_t)(o - result) <= len);

  // TODO(mortehu): It would be possible to return `len - (o - result)' bytes to
  // the region allocator here.

  return ev::StringRef(result, o - result - 1);
}

void Tagsoup::AddContent(TagsoupNode* parent, const char* begin,
                         const char* end, const char* type, bool need_escape) {
  TagsoupNode* content_node, *prev_sibling;

  content_node = Allocate<TagsoupNode>();
  content_node->parent = parent;

  if (!parent->first_child) {
    parent->first_child = content_node;
  } else {
    prev_sibling = parent->last_child;
    prev_sibling->next_sibling = content_node;
  }

  parent->last_child = content_node;

  content_node->name = type;

  if (need_escape) {
    content_node->content = AddString(begin, end);
  } else {
    // Comments, scripts, and style sheets don't need unescaping.
    auto buffer = Allocate<char>(end - begin + 1);
    memcpy(buffer, begin, end - begin);
    buffer[end - begin] = 0;

    content_node->content = ev::StringRef(begin, end);
  }
}

void Tagsoup::Parse(const ev::StringRef& input) {
  const char* end, *i, *val_begin, *val_end, *name;

  TagsoupNode* current_node, *prev_node;
  TagsoupAttribute* current_attribute = 0;

  enum parse_state state;

  size_t val_len;
  int quote_char = 0, is_closed = 0;
  int prev_autoclosee_group;
  bool in_cdata = false;
  ev::StringRef raw_text_end;

  state = s_content;

  end = input.data() + input.size();
  val_begin = val_end = input.data();

  root_ = Allocate<TagsoupNode>();
  root_->name = TAGSOUP_NODE_ROOT;

  current_node = root_;

  i = input.data();

  while (i != end) {
    switch (state) {
      case s_doctype:

        if (*i != '>') {
          ++i;

          break;
        }

        if (i != val_begin) doctype_ = AddString(val_begin, i);

        state = s_content;
        ++i;

        break;

      case s_tag_name:

        // Unless we've encountered a space character or a '>', or a '/' in the
        // second or later position, we're still in the tag name.
        if (!((isspace_HTML(*i) && i != val_begin) || *i == '>' ||
              (i > val_begin && *i == '/'))) {
          ++i;
          ++val_end;

          break;
        }

        val_end = i;

        if (val_end != val_begin) {
          val_len = val_end - val_begin;

          if (!isalpha(*val_begin)) {
            if (*val_begin == '/') {
              ++val_begin;
              --val_len;

              /* XXX: Will this close all open tags in case of a misspelled end
               * tag? */

              while (current_node && (current_node->name.size() != val_len ||
                                      strncasecmp(current_node->name.data(),
                                                  val_begin, val_len))) {
                current_node = current_node->parent;
              }

              if (current_node)
                current_node = current_node->parent;
              else
                current_node = root_;
            } else if (*val_begin == '!' && end - val_begin > 3 &&
                       val_begin[1] == '-' && val_begin[2] == '-') {
              const char* comment_end = 0;

              i = val_begin + 6;

              if (i >= end)
                i = end;
              else {
                while (i != end && (i[-1] != '-' || i[-2] != '-')) ++i;

                comment_end = i - 2;

                while (i != end && i[-1] != '>') ++i;
              }

              if (!comment_end) comment_end = i;

              AddContent(current_node, val_begin + 3, comment_end,
                         TAGSOUP_NODE_COMMENT, false);

              val_begin = i;
              state = s_content;

              break;
            }

            while (*i != '>' && i != end) ++i;
            if (i != end) ++i;
            val_begin = i;
            state = s_content;

            break;
          }

          prev_node = current_node;

          if (is_empty_element(val_begin, val_len)) is_closed = 1;

          while (prev_node->name != TAGSOUP_NODE_CONTENT &&
                 prev_node->name != TAGSOUP_NODE_ROOT) {
            name = prev_node->name.data();
            prev_autoclosee_group = get_autoclosee_group(name, strlen(name));

            if (!does_autoclose(val_begin, val_len, prev_autoclosee_group))
              break;
            prev_node = prev_node->parent;
          }

          current_node = Allocate<TagsoupNode>();
          current_node->parent = prev_node;

          if (!prev_node->first_child)
            prev_node->first_child = current_node;
          else {
            auto prev_sibling = prev_node->last_child;
            prev_sibling->next_sibling = current_node;
          }

          prev_node->last_child = current_node;

          current_node->name = AddString(val_begin, val_end);

          for (auto& c : current_node->name)
            const_cast<char&>(c) = tolower_ASCII(c);

          if (current_node->name == kScriptTagName) {
            raw_text_end = kScriptEndTag;
          } else if (current_node->name == kStyleTagName) {
            raw_text_end = kStyleEndTag;
          }
        }

        if (*i == '>') {
          if (is_closed) current_node = current_node->parent;

          ++i;
          state = s_content;
        } else {
          state = s_attrib_delim;
        }

        val_begin = i;

        break;

      case s_attrib_delim:

        if (*i == '>') {
          if (is_closed) current_node = current_node->parent;

          state = s_content;
          ++i;
          val_begin = i;
        } else if (*i == '/') {
          is_closed = 1;

          ++i;
        } else if (!isspace_HTML(*i)) {
          val_begin = i;
          state = s_attrib_name;
          ++i;
        } else {
          ++i;
        }

        break;

      case s_attrib_name:

        if (isspace_HTML(*i) || *i == '=' || *i == '>' || *i == '/' ||
            *i == '>') {
          val_end = i;

          while (i + 1 != end && isspace_HTML(*i)) ++i;

          current_attribute = Allocate<TagsoupAttribute>();

          if (!current_node->first_attribute) {
            current_node->first_attribute = current_attribute;
          } else {
            auto prev_sibling = current_node->last_attribute;
            prev_sibling->next_sibling = current_attribute;
          }

          current_node->last_attribute = current_attribute;

          current_attribute->name = AddString(val_begin, val_end);

          for (auto& c : current_attribute->name)
            const_cast<char&>(c) = tolower_ASCII(c);

          /* Translate ``checked'' to ``checked="checked"'' */
          current_attribute->content = current_attribute->name;

          if (*i == '/') is_closed = 1;

          if (*i == '=') {
            state = s_attrib_value;
            ++i;
            while (i != end && isspace_HTML(*i)) ++i;
          } else {
            if (*i == '>')
              state = s_content;
            else
              state = s_attrib_delim;
            ++i;
          }
          val_begin = i;
        } else
          ++i;

        break;

      case s_attrib_value:

        if (i == val_begin) {
          if (*i == '"' || *i == '\'') {
            quote_char = *i;
            ++i;
            break;
          } else
            quote_char = 0;
        }

        if ((!quote_char && isspace_HTML(*i)) ||
            (quote_char && *i == quote_char) || *i == '>') {
          val_end = i;

          if (quote_char && *i != '>') ++val_begin;

          current_attribute->content = AddString(val_begin, val_end);

          if (*i == '>')
            state = s_content;
          else
            state = s_attrib_delim;
          ++i;
          val_begin = i;
        } else
          ++i;

        break;

      case s_content: {
        // Nothing interesting will happen until we hit a delimiter, so we jump
        // there now.
        i = std::find(i, end, in_cdata ? ']' : '<');
        if (i == end) break;

        bool need_escape = true;

        if (raw_text_end != nullptr) {
          if (0 ==
              raw_text_end.compare(ev::StringRef(i, raw_text_end.size()))) {
            raw_text_end = nullptr;
            need_escape = false;
          } else {
            ++i;
            break;
          }
        }

        if (in_cdata) {
          if (!strncmp(i, "]]>", 3)) {
            in_cdata = false;
            need_escape = false;
            i += 3;
          } else
            ++i;
        } else {
          KJ_ASSERT(*i == '<');
          if (!strncmp(i + 1, "![CDATA[", 8)) {
            in_cdata = true;
            i += 9;
          } else if (!strncmp(i + 1, "!DOCTYPE ", 9)) {
            state = s_doctype;
            i += 10;
            val_begin = i;
          } else {
            val_end = i;

            if (val_end != val_begin)
              AddContent(current_node, val_begin, val_end, TAGSOUP_NODE_CONTENT,
                         need_escape);

            state = s_tag_name;
            is_closed = 0;
            ++i;
            val_begin = i;
          }
        }

      } break;
    }
  }

  if (state == s_content && i != val_begin)
    AddContent(current_node, val_begin, i, TAGSOUP_NODE_CONTENT,
               !in_cdata && raw_text_end == nullptr);
}

ev::StringRef TagsoupNode::GetAttribute(const ev::StringRef& name) {
  for (auto attribute = first_attribute; attribute;
       attribute = attribute->next_sibling) {
    if (attribute->name == name) return attribute->content;
  }

  return ev::StringRef();
}

void tagsoup_print_doc(FILE* output, const Tagsoup* doc) {
  if (doc->DocType() != nullptr)
    fprintf(output, "<!DOCTYPE %s>", doc->DocType().data());

  tagsoup_print(output, doc->Root());
}

void tagsoup_print(FILE* output, const TagsoupNode* node) {
  TagsoupAttribute* attr = node->first_attribute;

  do {
    if (node->name == TAGSOUP_NODE_ROOT) {
      if (node->first_child) tagsoup_print(output, node->first_child);
    } else if (node->name == TAGSOUP_NODE_CONTENT) {
      fprintf(output, "%s", node->content.data());
    } else if (node->name == TAGSOUP_NODE_COMMENT) {
      fprintf(output, "<!--%s-->", node->content.data());
    } else {
      fprintf(output, "<%s", node->name.data());

      attr = node->first_attribute;

      while (attr) {
        fprintf(output, " %s=\"%s\"", attr->name.data(), attr->content.data());

        attr = attr->next_sibling;
      }

      if (node->first_child) {
        fprintf(output, ">");

        tagsoup_print(output, node->first_child);

        fprintf(output, "</%s>", node->name.data());
      } else
        fprintf(output, " />");
    }

    node = node->next_sibling;
  } while (node);
}

}  // namespace ev
