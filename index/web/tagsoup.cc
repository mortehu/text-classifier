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

static const struct {
  int len;
  const char* name;
  int value;
} entities[] = {
    {2, "Mu", 924},         {2, "Nu", 925},         {2, "Pi", 928},
    {2, "Xi", 926},         {2, "ge", 8805},        {2, "gt", '>'},
    {2, "le", 8804},        {2, "lt", '<'},         {2, "mu", 956},
    {2, "ne", 8800},        {2, "ni", 8715},        {2, "nu", 957},
    {2, "or", 8744},        {2, "pi", 960},         {2, "xi", 958},

    {3, "Chi", 935},        {3, "ETH", L'\320'},    {3, "Eta", 919},
    {3, "Phi", 934},        {3, "Psi", 936},        {3, "Rho", 929},
    {3, "Tau", 932},        {3, "amp", '&'},        {3, "and", 8743},
    {3, "ang", 8736},       {3, "cap", 8745},       {3, "chi", 967},
    {3, "cup", 8746},       {3, "deg", L'\260'},    {3, "eta", 951},
    {3, "eth", L'\360'},    {3, "int", 8747},       {3, "loz", 9674},
    {3, "lrm", 0x8206},     {3, "not", L'\254'},    {3, "phi", 966},
    {3, "piv", 982},        {3, "psi", 968},        {3, "reg", L'\256'},
    {3, "rho", 961},        {3, "rlm", 0x8207},     {3, "shy", L'\255'},
    {3, "sim", 8764},       {3, "sub", 8834},       {3, "sum", 8721},
    {3, "sup", 8835},       {3, "tau", 964},        {3, "uml", L'\250'},
    {3, "yen", L'\245'},    {3, "zwj", 0x8205},

    {4, "Auml", L'\304'},   {4, "Beta", 914},       {4, "Euml", L'\313'},
    {4, "Iota", 921},       {4, "Iuml", L'\317'},   {4, "Ouml", L'\326'},
    {4, "Uuml", L'\334'},   {4, "Yuml", 0x376},     {4, "Zeta", 918},
    {4, "auml", L'\344'},   {4, "beta", 946},       {4, "bull", 8226},
    {4, "cent", L'\242'},   {4, "circ", 0x710},     {4, "cong", 8773},
    {4, "copy", 169},       {4, "dArr", 8659},      {4, "darr", 8595},
    {4, "emsp", 0x8195},    {4, "ensp", 0x8194},    {4, "euml", L'\353'},
    {4, "euro", 0x8364},    {4, "fnof", 402},       {4, "hArr", 8660},
    {4, "harr", 8596},      {4, "iota", 953},       {4, "isin", 8712},
    {4, "iuml", L'\357'},   {4, "lArr", 8656},      {4, "lang", 9001},
    {4, "larr", 8592},      {4, "macr", L'\257'},   {4, "nbsp", L'\240'},
    {4, "nsub", 8836},      {4, "ordf", L'\252'},   {4, "ordm", L'\272'},
    {4, "ouml", L'\366'},   {4, "para", L'\266'},   {4, "part", 8706},
    {4, "perp", 8869},      {4, "prod", 8719},      {4, "prop", 8733},
    {4, "quot", '"'},       {4, "rArr", 8658},      {4, "rang", 9002},
    {4, "rarr", 8594},      {4, "real", 8476},      {4, "sdot", 8901},
    {4, "sect", L'\247'},   {4, "sube", 8838},      {4, "sup1", L'\271'},
    {4, "sup2", L'\262'},   {4, "sup3", L'\263'},   {4, "supe", 8839},
    {4, "uArr", 8657},      {4, "uarr", 8593},      {4, "uuml", L'\374'},
    {4, "yuml", L'\377'},   {4, "zeta", 950},       {4, "zwnj", 0x8204},

    {5, "AElig", 198},      {5, "Acirc", L'\302'},  {5, "Alpha", 913},
    {5, "Aring", 197},      {5, "Delta", 916},      {5, "Ecirc", L'\312'},
    {5, "Gamma", 915},      {5, "Icirc", L'\316'},  {5, "Kappa", 922},
    {5, "OElig", 0x338},    {5, "Ocirc", L'\324'},  {5, "Omega", 937},
    {5, "Prime", 8243},     {5, "Sigma", 931},      {5, "THORN", L'\336'},
    {5, "Theta", 920},      {5, "Ucirc", L'\333'},  {5, "acirc", L'\342'},
    {5, "acute", L'\264'},  {5, "aelig", 230},      {5, "alpha", 945},
    {5, "aring", 229},      {5, "asymp", 8776},     {5, "bdquo", 0x8222},
    {5, "cedil", L'\270'},  {5, "clubs", 9827},     {5, "crarr", 8629},
    {5, "delta", 948},      {5, "diams", 9830},     {5, "ecirc", L'\352'},
    {5, "empty", 8709},     {5, "equiv", 8801},     {5, "exist", 8707},
    {5, "frasl", 8260},     {5, "gamma", 947},      {5, "icirc", L'\356'},
    {5, "iexcl", 161},      {5, "image", 8465},     {5, "infin", 8734},
    {5, "kappa", 954},      {5, "laquo", L'\253'},  {5, "lceil", 8968},
    {5, "ldquo", 0x8220},   {5, "lsquo", 0x8216},   {5, "mdash", 0x8212},
    {5, "micro", L'\265'},  {5, "minus", 8722},     {5, "nabla", 8711},
    {5, "ndash", 0x8211},   {5, "notin", 8713},     {5, "ocirc", L'\364'},
    {5, "oelig", 0x339},    {5, "oline", 8254},     {5, "omega", 969},
    {5, "oplus", 8853},     {5, "pound", L'\243'},  {5, "prime", 8242},
    {5, "radic", 8730},     {5, "raquo", L'\273'},  {5, "rceil", 8969},
    {5, "rdquo", 0x8221},   {5, "rsquo", 0x8217},   {5, "sbquo", 0x8218},
    {5, "sigma", 963},      {5, "szlig", L'\337'},  {5, "theta", 952},
    {5, "thorn", L'\376'},  {5, "tilde", 0x732},    {5, "times", L'\327'},
    {5, "trade", 8482},     {5, "ucirc", L'\373'},  {5, "upsih", 978},
    {6, "Lambda", 923},

    {6, "Aacute", L'\301'}, {6, "Agrave", L'\300'}, {6, "Atilde", L'\303'},
    {6, "Ccedil", L'\307'}, {6, "Dagger", 0x8225},  {6, "Eacute", L'\311'},
    {6, "Egrave", L'\310'}, {6, "Iacute", L'\315'}, {6, "Igrave", L'\314'},
    {6, "Ntilde", L'\321'}, {6, "Oacute", L'\323'}, {6, "Ograve", L'\322'},
    {6, "Oslash", 216},     {6, "Otilde", L'\325'}, {6, "Scaron", 0x352},
    {6, "Uacute", L'\332'}, {6, "Ugrave", L'\331'}, {6, "Yacute", L'\335'},
    {6, "aacute", L'\341'}, {6, "agrave", L'\340'}, {6, "atilde", L'\343'},
    {6, "brvbar", L'\246'}, {6, "ccedil", L'\347'}, {6, "curren", L'\244'},
    {6, "dagger", 0x8224},  {6, "divide", L'\367'}, {6, "eacute", L'\351'},
    {6, "egrave", L'\350'}, {6, "forall", 8704},    {6, "frac12", L'\275'},
    {6, "frac14", L'\274'}, {6, "frac34", L'\276'}, {6, "hearts", 9829},
    {6, "hellip", 8230},    {6, "iacute", L'\355'}, {6, "igrave", L'\354'},
    {6, "iquest", L'\277'}, {6, "lambda", 955},     {6, "lfloor", 8970},
    {6, "lowast", 8727},    {6, "lsaquo", 0x8249},  {6, "middot", L'\267'},
    {6, "ntilde", L'\361'}, {6, "oacute", L'\363'}, {6, "ograve", L'\362'},
    {6, "oslash", 248},     {6, "otilde", L'\365'}, {6, "otimes", 8855},
    {6, "permil", 0x8240},  {6, "plusmn", L'\261'}, {6, "rfloor", 8971},
    {6, "rsaquo", 0x8250},  {6, "scaron", 0x353},   {6, "sigmaf", 962},
    {6, "spades", 9824},    {6, "there4", 8756},    {6, "thinsp", 0x8201},
    {6, "uacute", L'\372'}, {6, "ugrave", L'\371'}, {6, "weierp", 8472},
    {6, "yacute", L'\375'},

    {7, "Epsilon", 917},    {7, "Omicron", 927},    {7, "Upsilon", 933},
    {7, "alefsym", 8501},   {7, "epsilon", 949},    {7, "omicron", 959},
    {7, "upsilon", 965},

    {8, "thetasym", 977}};

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

static char* tagsoup_utf8_put(char* output, unsigned int ch) {
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
      size_t first = 0, half, middle, count;
      int cmp;

      count = sizeof(entities) / sizeof(entities[0]);

      while (count > 0) {
        half = count / 2;
        middle = first + half;

        if (entities[middle].len != ent_end - ent_begin)
          cmp = entities[middle].len - (ent_end - ent_begin);
        else
          cmp = memcmp(entities[middle].name, ent_begin, ent_end - ent_begin);

        if (cmp == 0) {
          o = tagsoup_utf8_put(o, entities[middle].value);

          break;
        }

        if (cmp < 0) {
          first = middle + 1;
          count -= half - 1;
        } else
          count = half;
      }

      if (count) {
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
