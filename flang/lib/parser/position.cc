#include "position.h"

namespace Fortran {
namespace parser {
std::ostream &operator<<(std::ostream &o, const Position &x) {
  return o << "(at line " << x.lineNumber() << ", column " << x.column() << ')';
}
}  // namespace parser
}  // namespace Fortran
