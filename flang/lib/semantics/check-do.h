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
#include "../common/idioms.h"

namespace Fortran::parser {
struct DoConstruct;
struct CycleStmt;
struct ExitStmt;
}

namespace Fortran::semantics {

// To specify different statement types used in semantic checking.
ENUM_CLASS(StmtType, CYCLE, EXIT, ALLOCATE, DEALLOCATE)

class DoChecker : public virtual BaseChecker {
public:
  explicit DoChecker(SemanticsContext &context) : context_{context} {}
  void Leave(const parser::DoConstruct &);
  void Enter(const parser::CycleStmt &);
  void Enter(const parser::ExitStmt &);

private:
  SemanticsContext &context_;

  void SayBadLeave(
      StmtType, const char *enclosingStmt, const ConstructNode &) const;
  void CheckDoConcurrentExit(StmtType, const ConstructNode &) const;
  void CheckForBadLeave(StmtType, const ConstructNode &) const;
  void CheckNesting(StmtType, const parser::Name *) const;
};
}
#endif  // FORTRAN_SEMANTICS_CHECK_DO_H_
