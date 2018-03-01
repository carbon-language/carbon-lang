#include <iosfwd>

namespace Fortran {

namespace parser {
class Program;
}

namespace semantics {
void MakeTypes(std::ostream &out, const parser::Program &program);
}

}  // namespace Fortran
