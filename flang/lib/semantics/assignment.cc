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

#include "assignment.h"
#include "expression.h"
#include "semantics.h"
#include "symbol.h"
#include "../common/idioms.h"
#include "../evaluate/expression.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"

using namespace Fortran::parser::literals;

namespace Fortran::semantics {

template<typename A>
void AnalyzeExecutableStmt(SemanticsContext &, const parser::Statement<A> &) {}
template<>
void AnalyzeExecutableStmt(SemanticsContext &context,
    const parser::Statement<parser::AssignmentStmt> &stmt) {}
template<>
void AnalyzeExecutableStmt(SemanticsContext &context,
    const parser::Statement<parser::PointerAssignmentStmt> &stmt) {}
template<>
void AnalyzeExecutableStmt(SemanticsContext &context,
    const parser::Statement<parser::WhereStmt> &stmt) {}
template<>
void AnalyzeExecutableStmt(SemanticsContext &context,
    const parser::Statement<parser::ForallStmt> &stmt) {}

void AnalyzeAssignment(SemanticsContext &context,
    const parser::Statement<parser::AssignmentStmt> &stmt) {
  AnalyzeExecutableStmt(context, stmt);
}
void AnalyzeAssignment(SemanticsContext &context,
    const parser::Statement<parser::PointerAssignmentStmt> &stmt) {
  AnalyzeExecutableStmt(context, stmt);
}
void AnalyzeAssignment(SemanticsContext &context,
    const parser::Statement<parser::WhereStmt> &stmt) {
  AnalyzeExecutableStmt(context, stmt);
}
void AnalyzeAssignment(SemanticsContext &context,
    const parser::Statement<parser::ForallStmt> &stmt) {
  AnalyzeExecutableStmt(context, stmt);
}

class Mutator {
public:
  Mutator(SemanticsContext &context) : context_{context} {}

  template<typename A> bool Pre(A &) { return true /* visit children */; }
  template<typename A> void Post(A &) {}

  bool Pre(parser::Statement<parser::AssignmentStmt> &stmt) {
    AnalyzeAssignment(context_, stmt);
    return false;
  }

private:
  SemanticsContext &context_;
};

void AnalyzeAssignments(parser::Program &program, SemanticsContext &context) {
  Mutator mutator{context};
  parser::Walk(program, mutator);
}
}
