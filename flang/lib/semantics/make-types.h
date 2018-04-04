#include <iosfwd>

namespace Fortran::parser {
class Program;
class CookedSource;
}  // namespace Fortran::parser

namespace Fortran::semantics {
void MakeTypes(
    const parser::Program &program, const parser::CookedSource &cookedSource);
}
