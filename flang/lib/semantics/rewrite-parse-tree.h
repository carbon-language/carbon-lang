namespace Fortran::parser {
class Program;
}  // namespace Fortran::parser

namespace Fortran::semantics {
void RewriteParseTree(parser::Program &);
}  // namespace Fortran::semantics
