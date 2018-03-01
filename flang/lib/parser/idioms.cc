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
}  // namespace parser
}  // namespace Fortran
