#include "characters.h"

namespace Fortran {
namespace parser {

std::optional<int> UTF8CharacterBytes(const char *p) {
  if ((*p & 0x80) == 0) {
    return {1};
  }
  if ((*p & 0xf8) == 0xf0) {
    if ((p[1] & 0xc0) == 0x80 && (p[2] & 0xc0) == 0x80 &&
        (p[3] & 0xc0) == 0x80) {
      return {4};
    }
  } else if ((*p & 0xf0) == 0xe0) {
    if ((p[1] & 0xc0) == 0x80 && (p[2] & 0xc0) == 0x80) {
      return {3};
    }
  } else if ((*p & 0xe0) == 0xc0) {
    if ((p[1] & 0xc0) == 0x80) {
      return {2};
    }
  }
  return {};
}

std::optional<int> EUC_JPCharacterBytes(const char *p) {
  int b1 = *p & 0xff;
  if (b1 <= 0x7f) {
    return {1};
  }
  if (b1 >= 0xa1 && b1 <= 0xfe) {
    int b2 = p[1] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xfe) {
      // JIS X 0208 (code set 1)
      return {2};
    }
  } else if (b1 == 0x8e) {
    int b2 = p[1] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xdf) {
      // upper half JIS 0201 (half-width kana, code set 2)
      return {2};
    }
  } else if (b1 == 0x8f) {
    int b2 = p[1] & 0xff;
    int b3 = p[2] & 0xff;
    if (b2 >= 0xa1 && b2 <= 0xfe && b3 >= 0xa1 && b3 <= 0xfe) {
      // JIS X 0212 (code set 3)
      return {3};
    }
  }
  return {};
}

std::optional<size_t> CountCharacters(
    const char *p, size_t bytes, std::optional<int> (*cbf)(const char *)) {
  size_t chars{0};
  const char *limit{p + bytes};
  while (p < limit) {
    ++chars;
    std::optional<int> cb{cbf(p)};
    if (!cb.has_value()) {
      return {};
    }
    p += *cb;
  }
  return {chars};
}
}  // namespace parser
}  // namespace Fortran
