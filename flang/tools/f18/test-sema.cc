// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../../lib/parser/parsing.h"
#include "../../lib/semantics/attr.h"
#include "../../lib/semantics/type.h"
#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <sstream>
#include <stddef.h>
#include <string>

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
  Parsing parsing;
  if (parsing.ForTesting(path, std::cerr)) {
    DoSemanticAnalysis(parsing.cooked(), *parsing.parseTree());
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
