#include "../../lib/parser/parsing.h"
#include "../../lib/semantics/make-types.h"

#include <iostream>
#include <string>

using namespace Fortran;
using namespace parser;

int main(int argc, char *const argv[]) {
  if (argc != 2) {
    std::cerr << "Expected 1 source file, got " << (argc - 1) << "\n";
    return EXIT_FAILURE;
  }
  std::string path{argv[1]};
  parser::Parsing parsing;
  if (parsing.ForTesting(path, std::cerr)) {
    semantics::MakeTypes(*parsing.parseTree(), parsing.messages().cooked());
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
