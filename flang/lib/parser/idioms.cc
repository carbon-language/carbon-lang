#include "idioms.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace Fortran {
namespace parser {

[[noreturn]] void die(const char *msg, ...) {
  va_list ap;
  va_start(ap, msg);
  std::fputs("\nfatal internal error: ", stderr);
  std::vfprintf(stderr, msg, ap);
  va_end(ap);
  fputc('\n', stderr);
  std::abort();
}

// Convert the int index of an enumerator to a string.
// enumNames is a list of the names, separated by commas with optional spaces.
// This is intended for use from the expansion of ENUM_CLASS.
std::string EnumIndexToString(int index, const char *enumNames) {
  const char *p{enumNames};
  for (; index > 0; --index, ++p) {
    for (; *p && *p != ','; ++p) {
    }
  }
  for (; *p == ' '; ++p) {
  }
  CHECK(*p != '\0');
  const char *q = p;
  for (; *q && *q != ' ' && *q != ','; ++q) {
  }
  return std::string(p, q - p);
}

}  // namespace parser
}  // namespace Fortran
