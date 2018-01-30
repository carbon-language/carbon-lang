#include "position.h"

namespace Fortran {
std::ostream &operator<<(std::ostream &o, const Position &x) {
  return o << "(at line " << x.lineNumber() << ", column " << x.column() << ')';
}
}  // namespace Fortran
