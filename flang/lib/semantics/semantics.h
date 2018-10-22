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

#include "default-kinds.h"
#include "expression.h"
#include "scope.h"
#include "../evaluate/intrinsics.h"
#include "../parser/message.h"
#include <iosfwd>
#include <string>
#include <vector>

namespace Fortran::parser {
struct Program;
class CookedSource;
}  // namespace Fortran::parser

namespace Fortran::semantics {

class SemanticsContext {
public:
  SemanticsContext(const IntrinsicTypeDefaultKinds &defaultKinds)
    : defaultKinds_{defaultKinds}, intrinsics_{
                                       evaluate::IntrinsicProcTable::Configure(
                                           defaultKinds)} {}

  const IntrinsicTypeDefaultKinds &defaultKinds() const {
    return defaultKinds_;
  }
  const std::vector<std::string> &searchDirectories() const {
    return searchDirectories_;
  }
  const std::string &moduleDirectory() const { return moduleDirectory_; }
  const bool warningsAreErrors() const { return warningsAreErrors_; }
  const bool debugExpressions() const { return debugExpressions_; }
  const evaluate::IntrinsicProcTable &intrinsics() const { return intrinsics_; }
  Scope &globalScope() { return globalScope_; }
  parser::Messages &messages() { return messages_; }

  SemanticsContext &set_searchDirectories(const std::vector<std::string> &x) {
    searchDirectories_ = x;
    return *this;
  }
  SemanticsContext &set_moduleDirectory(const std::string &x) {
    moduleDirectory_ = x;
    return *this;
  }
  SemanticsContext &set_warningsAreErrors(bool x) {
    warningsAreErrors_ = x;
    return *this;
  }
  SemanticsContext &set_debugExpressions(bool x) {
    debugExpressions_ = x;
    return *this;
  }

  bool AnyFatalError() const;
  template<typename... A> parser::Message &Say(A... args) {
    return messages_.Say(std::forward<A>(args)...);
  }

private:
  const IntrinsicTypeDefaultKinds &defaultKinds_;
  std::vector<std::string> searchDirectories_;
  std::string moduleDirectory_{"."s};
  bool warningsAreErrors_{false};
  bool debugExpressions_{false};
  const evaluate::IntrinsicProcTable intrinsics_;
  Scope globalScope_;
  parser::Messages messages_;
};

class Semantics {
public:
  explicit Semantics(SemanticsContext &context, parser::Program &program,
      parser::CookedSource &cooked)
    : context_{context}, program_{program}, cooked_{cooked} {}

  SemanticsContext &context() const { return context_; }
  bool Perform();
  bool AnyFatalError() const { return context_.AnyFatalError(); }
  void EmitMessages(std::ostream &) const;
  void DumpSymbols(std::ostream &);

private:
  SemanticsContext &context_;
  parser::Program &program_;
  const parser::CookedSource &cooked_;
};

}  // namespace Fortran::semantics

#endif
