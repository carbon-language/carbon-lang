#include "../../lib/parser/parsing.h"
#include "../../lib/semantics/attr.h"
#include "../../lib/semantics/type.h"
#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <string>
#include <stddef.h>

using namespace Fortran;
using namespace parser;

extern void DoSemanticAnalysis(const CookedSource &, const Program &);

//static void visitProgramUnit(const ProgramUnit &unit);

int main(int argc, char *const argv[]) {
  if (argc != 2) {
    std::cerr << "Expected 1 source file, got " << (argc - 1) << "\n";
    return EXIT_FAILURE;
  }
  std::string path{argv[1]};
  if (std::optional<Program> parseTree{Parsing::ForTesting(path, std::cerr)}) {
    DoSemanticAnalysis(parsing.messages().cooked(), *parseTree);
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
