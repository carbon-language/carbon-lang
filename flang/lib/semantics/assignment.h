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

#ifndef FORTRAN_SEMANTICS_ASSIGNMENT_H_
#define FORTRAN_SEMANTICS_ASSIGNMENT_H_

namespace Fortran::parser {
template<typename> struct Statement;
struct AssignmentStmt;
struct ConcurrentHeader;
struct ForallStmt;
struct PointerAssignmentStmt;
struct Program;
struct WhereStmt;
}

namespace Fortran::semantics {
class SemanticsContext;

// Semantic analysis of an assignment statement or WHERE/FORALL construct.
void AnalyzeAssignment(
    SemanticsContext &, const parser::Statement<parser::AssignmentStmt> &);
void AnalyzeAssignment(SemanticsContext &,
    const parser::Statement<parser::PointerAssignmentStmt> &);
void AnalyzeAssignment(
    SemanticsContext &, const parser::Statement<parser::WhereStmt> &);
void AnalyzeAssignment(
    SemanticsContext &, const parser::Statement<parser::ForallStmt> &);

// R1125 concurrent-header is used in FORALL statements & constructs as
// well as in DO CONCURRENT loops.
void AnalyzeConcurrentHeader(
    SemanticsContext &, const parser::ConcurrentHeader &);

// Semantic analysis of all assignment statements and related constructs.
void AnalyzeAssignments(parser::Program &, SemanticsContext &);
}
#endif  // FORTRAN_SEMANTICS_ASSIGNMENT_H_
