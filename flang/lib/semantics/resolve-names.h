#include <iosfwd>

namespace Fortran::parser {
struct Program;
class CookedSource;
}  // namespace Fortran::parser

namespace Fortran::semantics {

void ResolveNames(parser::Program &, const parser::CookedSource &);
void DumpSymbols(std::ostream &);

}  // namespace Fortran::semantics
