#ifndef FORTRAN_PARSER_UNPARSE_H_
#define FORTRAN_PARSER_UNPARSE_H_

#include "characters.h"
#include <iosfwd>

namespace Fortran {
namespace parser {

struct Program;

/// Convert parsed program to out as Fortran.
void Unparse(std::ostream &out, const Program &program,
    Encoding encoding = Encoding::UTF8, bool capitalizeKeywords = true);

}  // namespace parser
}  // namespace Fortran

#endif
