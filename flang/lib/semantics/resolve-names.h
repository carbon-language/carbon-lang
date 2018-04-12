#include <iosfwd>

namespace Fortran::parser {
class Program;
class CookedSource;
}  // namespace Fortran::parser

namespace Fortran::semantics {
void ResolveNames(const parser::Program &, const parser::CookedSource &);
}  // namespace Fortran::semantics
