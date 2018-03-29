#include "../../lib/parser/parsing.h"
#include "../../lib/semantics/make-types.h"

#include <iostream>
#include <optional>
#include <sstream>
#include <string>

using namespace Fortran;
using namespace parser;

int main(int argc, char *const argv[]) {
  if (argc != 2) {
    std::cerr << "Expected 1 source file, got " << (argc - 1) << "\n";
    return EXIT_FAILURE;
  }
  std::string path{argv[1]};
  Parsing parsing;
  if (!parsing.Prescan(path, Options{}) || !parsing.Parse()) {
    std::cerr << "parse FAILED\n";
    parsing.messages().Emit(std::cerr);
    return EXIT_FAILURE;
  }
  semantics::MakeTypes(std::cout, parsing.parseTree());
  return EXIT_SUCCESS;
}
