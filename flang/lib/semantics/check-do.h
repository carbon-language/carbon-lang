// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_SEMANTICS_CHECK_DO_H_
#define FORTRAN_SEMANTICS_CHECK_DO_H_

#include "semantics.h"

namespace Fortran::parser {
struct DoConstruct;
struct CycleStmt;
struct ExitStmt;
}

namespace Fortran::semantics {

using NamePtr = parser::Name const *;

class DoChecker : public virtual BaseChecker {
public:
  explicit DoChecker(SemanticsContext &context) : context_{context} {}
  void Leave(const parser::DoConstruct &);
  void Enter(const parser::CycleStmt &);
  void Enter(const parser::ExitStmt &);

private:
  SemanticsContext &context_;

  void SayBadLeave(const char *stmtChecked, const char *enclosingStmt,
      const ConstructNode &) const;
  void CheckDoConcurrentExit(const char *s, const ConstructNode &) const;
  void CheckForBadLeave(const char *, const ConstructNode &) const;
  void CheckNesting(const char *, NamePtr) const;
};
}
#endif  // FORTRAN_SEMANTICS_CHECK_DO_H_
