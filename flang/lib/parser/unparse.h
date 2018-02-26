#ifndef FORTRAN_PARSER_UNPARSE_H_
#define FORTRAN_PARSER_UNPARSE_H_

#include <iosfwd>

namespace Fortran {
namespace parser {

class Program;

/// Convert parsed program to out as Fortran.
void Unparse(std::ostream &out, const Program &program);

}  // namespace parser
}  // namespace Fortran

#endif
