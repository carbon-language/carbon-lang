//===-- lib/semantics/assignment.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_ASSIGNMENT_H_
#define FORTRAN_SEMANTICS_ASSIGNMENT_H_

#include "semantics.h"
#include "../common/indirection.h"
#include "../evaluate/expression.h"
#include <string>

namespace Fortran::parser {
template<typename> struct Statement;
struct AssignmentStmt;
struct ConcurrentHeader;
struct ForallStmt;
struct PointerAssignmentStmt;
struct Program;
struct WhereStmt;
struct WhereConstruct;
struct ForallStmt;
struct ForallConstruct;
}

namespace Fortran::evaluate::characteristics {
struct DummyDataObject;
}

namespace Fortran::evaluate {
class FoldingContext;
void CheckPointerAssignment(
    FoldingContext &, const Symbol &lhs, const Expr<SomeType> &rhs);
void CheckPointerAssignment(FoldingContext &, parser::CharBlock source,
    const std::string &description, const characteristics::DummyDataObject &,
    const Expr<SomeType> &rhs);
}

namespace Fortran::semantics {
class AssignmentContext;
}

extern template class Fortran::common::Indirection<
    Fortran::semantics::AssignmentContext>;

namespace Fortran::semantics {
// Applies checks from C1594(1-2) on definitions in pure subprograms
void CheckDefinabilityInPureScope(parser::ContextualMessages &, const Symbol &,
    const Scope &context, const Scope &pure);
// Applies checks from C1594(5-6) on copying pointers in pure subprograms
void CheckCopyabilityInPureScope(parser::ContextualMessages &,
    const evaluate::Expr<evaluate::SomeType> &, const Scope &);

class AssignmentChecker : public virtual BaseChecker {
public:
  explicit AssignmentChecker(SemanticsContext &);
  ~AssignmentChecker();
  void Enter(const parser::AssignmentStmt &);
  void Enter(const parser::PointerAssignmentStmt &);
  void Enter(const parser::WhereStmt &);
  void Enter(const parser::WhereConstruct &);
  void Enter(const parser::ForallStmt &);
  void Enter(const parser::ForallConstruct &);

private:
  common::Indirection<AssignmentContext> context_;
};

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
}
#endif  // FORTRAN_SEMANTICS_ASSIGNMENT_H_
