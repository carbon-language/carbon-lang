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

#ifndef FORTRAN_SEMANTICS_SEMANTICS_H_
#define FORTRAN_SEMANTICS_SEMANTICS_H_

#include "scope.h"
#include "../parser/message.h"
#include <string>
#include <vector>

namespace Fortran::parser {
  struct Program;
}

namespace Fortran::semantics {

class Semantics {
public:
  Semantics() { directories_.push_back("."s); }
  const parser::Messages &messages() const { return messages_; }
  Semantics &set_searchDirectories(const std::vector<std::string> &);
  Semantics &set_moduleDirectory(const std::string &);
  bool AnyFatalError() const { return messages_.AnyFatalError(); }
  bool Perform(parser::Program &);
  void DumpSymbols(std::ostream &);

private:
  Scope globalScope_;
  std::vector<std::string> directories_;
  std::string moduleDirectory_{"."s};
  parser::Messages messages_;
};
}  // namespace Fortran::semantics

#endif
