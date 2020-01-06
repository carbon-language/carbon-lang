//===-- lib/semantics/pointer-assignment.cc -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pointer-assignment.h"
#include "expression.h"
#include "symbol.h"
#include "tools.h"
#include "../common/idioms.h"
#include "../common/restorer.h"
#include "../evaluate/characteristics.h"
#include "../evaluate/expression.h"
#include "../evaluate/fold.h"
#include "../evaluate/tools.h"
#include "../parser/message.h"
#include "../parser/parse-tree-visitor.h"
#include "../parser/parse-tree.h"
#include <optional>
#include <set>
#include <string>
#include <type_traits>

// Semantic checks for pointer assignment.

namespace Fortran::semantics {

using namespace parser::literals;
using evaluate::characteristics::DummyDataObject;
using evaluate::characteristics::FunctionResult;
using evaluate::characteristics::Procedure;
using evaluate::characteristics::TypeAndShape;

class PointerAssignmentChecker {
public:
  PointerAssignmentChecker(parser::CharBlock source,
      const std::string &description, evaluate::FoldingContext &context)
    : source_{source}, description_{description}, context_{context} {}
  PointerAssignmentChecker &set_lhs(const Symbol &);
  PointerAssignmentChecker &set_lhsType(std::optional<TypeAndShape> &&);
  PointerAssignmentChecker &set_procedure(std::optional<Procedure> &&);
  PointerAssignmentChecker &set_isContiguous(bool);
  void Check(const SomeExpr &);

private:
  template<typename A> void Check(const A &);
  template<typename T> void Check(const evaluate::Expr<T> &);
  template<typename T> void Check(const evaluate::FunctionRef<T> &);
  template<typename T> void Check(const evaluate::Designator<T> &);
  void Check(const evaluate::NullPointer &);
  void Check(const evaluate::ProcedureDesignator &);
  void Check(const evaluate::ProcedureRef &);
  // Target is a procedure
  void Check(
      parser::CharBlock rhsName, bool isCall, const Procedure * = nullptr);

  template<typename... A> parser::Message *Say(A &&...);

  const parser::CharBlock source_;
  const std::string &description_;
  evaluate::FoldingContext &context_;
  const Symbol *lhs_{nullptr};
  std::optional<TypeAndShape> lhsType_;
  std::optional<Procedure> procedure_;
  bool isContiguous_{false};
};

PointerAssignmentChecker &PointerAssignmentChecker::set_lhs(const Symbol &lhs) {
  lhs_ = &lhs;
  return *this;
}

PointerAssignmentChecker &PointerAssignmentChecker::set_lhsType(
    std::optional<TypeAndShape> &&lhsType) {
  lhsType_ = std::move(lhsType);
  return *this;
}

PointerAssignmentChecker &PointerAssignmentChecker::set_procedure(
    std::optional<Procedure> &&procedure) {
  procedure_ = std::move(procedure);
  return *this;
}

PointerAssignmentChecker &PointerAssignmentChecker::set_isContiguous(
    bool isContiguous) {
  isContiguous_ = isContiguous;
  return *this;
}

template<typename A> void PointerAssignmentChecker::Check(const A &) {
  // Catch-all case for really bad target expression
  Say("Target associated with %s must be a designator or a call to a"
      " pointer-valued function"_err_en_US,
      description_);
}

template<typename T>
void PointerAssignmentChecker::Check(const evaluate::Expr<T> &x) {
  std::visit([&](const auto &x) { Check(x); }, x.u);
}

void PointerAssignmentChecker::Check(const SomeExpr &rhs) {
  if (HasVectorSubscript(rhs)) {  // C1025
    Say("An array section with a vector subscript may not be a pointer target"_err_en_US);
  } else if (ExtractCoarrayRef(rhs)) {  // C1026
    Say("A coindexed object may not be a pointer target"_err_en_US);
  } else {
    std::visit([&](const auto &x) { Check(x); }, rhs.u);
  }
}

void PointerAssignmentChecker::Check(const evaluate::NullPointer &) {
  // P => NULL() without MOLD=; always OK
}

template<typename T>
void PointerAssignmentChecker::Check(const evaluate::FunctionRef<T> &f) {
  std::string funcName;
  const auto *symbol{f.proc().GetSymbol()};
  if (symbol) {
    funcName = symbol->name().ToString();
  } else if (const auto *intrinsic{f.proc().GetSpecificIntrinsic()}) {
    funcName = intrinsic->name;
  }
  auto proc{Procedure::Characterize(f.proc(), context_.intrinsics())};
  if (!proc) {
    return;
  }
  std::optional<parser::MessageFixedText> msg;
  const auto &funcResult{proc->functionResult};  // C1025
  if (!funcResult) {
    msg = "%s is associated with the non-existent result of reference to"
          " procedure"_err_en_US;
  } else if (procedure_) {
    // Shouldn't be here in this function unless lhs is an object pointer.
    msg = "Procedure %s is associated with the result of a reference to"
          " function '%s' that does not return a procedure pointer"_err_en_US;
  } else if (funcResult->IsProcedurePointer()) {
    msg = "Object %s is associated with the result of a reference to"
          " function '%s' that is a procedure pointer"_err_en_US;
  } else if (!funcResult->attrs.test(FunctionResult::Attr::Pointer)) {
    msg = "%s is associated with the result of a reference to function '%s'"
          " that is a not a pointer"_err_en_US;
  } else if (isContiguous_ &&
      !funcResult->attrs.test(FunctionResult::Attr::Contiguous)) {
    msg = "CONTIGUOUS %s is associated with the result of reference to"
          " function '%s' that is not contiguous"_err_en_US;
  } else if (lhsType_) {
    const auto *frTypeAndShape{funcResult->GetTypeAndShape()};
    CHECK(frTypeAndShape);
    if (!lhsType_->IsCompatibleWith(context_.messages(), *frTypeAndShape)) {
      msg = "%s is associated with the result of a reference to function '%s'"
            " whose pointer result has an incompatible type or shape"_err_en_US;
    }
  }
  if (msg) {
    auto restorer{common::ScopedSet(lhs_, symbol)};
    Say(*msg, description_, funcName);
  }
}

template<typename T>
void PointerAssignmentChecker::Check(const evaluate::Designator<T> &d) {
  const Symbol *last{d.GetLastSymbol()};
  const Symbol *base{d.GetBaseObject().symbol()};
  if (!last || !base) {
    // P => "character literal"(1:3)
    context_.messages().Say("Pointer target is not a named entity"_err_en_US);
    return;
  }
  std::optional<parser::MessageFixedText> msg;
  if (procedure_) {
    // Shouldn't be here in this function unless lhs is an object pointer.
    msg = "In assignment to procedure %s, the target is not a procedure or"
          " procedure pointer"_err_en_US;
  } else if (!evaluate::GetLastTarget(GetSymbolVector(d))) {  // C1025
    msg = "In assignment to object %s, the target '%s' is not an object with"
          " POINTER or TARGET attributes"_err_en_US;
  } else if (auto rhsTypeAndShape{
                 TypeAndShape::Characterize(*last, context_)}) {
    if (!lhsType_ ||
        !lhsType_->IsCompatibleWith(context_.messages(), *rhsTypeAndShape)) {
      msg = "%s associated with object '%s' with incompatible type or"
            " shape"_err_en_US;
    }
  }
  if (msg) {
    auto restorer{common::ScopedSet(lhs_, last)};
    Say(*msg, description_, last->name());
  }
}

// Common handling for procedure pointer right-hand sides
void PointerAssignmentChecker::Check(
    parser::CharBlock rhsName, bool isCall, const Procedure *targetChars) {
  if (!procedure_) {
    Say("In assignment to object %s, the target '%s' is a procedure designator"_err_en_US,
        description_, rhsName);
  } else if (!targetChars) {
    Say("In assignment to procedure %s, the characteristics of the target"
        " procedure '%s' could not be determined"_err_en_US,
        description_, rhsName);
  } else if (*procedure_ == *targetChars) {
    // OK
  } else if (isCall) {
    Say("Procedure %s associated with result of reference to function '%s' that"
        " is an incompatible procedure pointer"_err_en_US,
        description_, rhsName);
  } else {
    Say("Procedure %s associated with incompatible procedure designator '%s'"_err_en_US,
        description_, rhsName);
  }
}

void PointerAssignmentChecker::Check(const evaluate::ProcedureDesignator &d) {
  if (auto chars{Procedure::Characterize(d, context_.intrinsics())}) {
    Check(d.GetName(), false, &*chars);
  } else {
    Check(d.GetName(), false);
  }
}

void PointerAssignmentChecker::Check(const evaluate::ProcedureRef &ref) {
  const Procedure *procedure{nullptr};
  auto chars{Procedure::Characterize(ref, context_.intrinsics())};
  if (chars) {
    procedure = &*chars;
    if (chars->functionResult) {
      if (const auto *proc{chars->functionResult->IsProcedurePointer()}) {
        procedure = proc;
      }
    }
  }
  Check(ref.proc().GetName(), true, procedure);
}

template<typename... A>
parser::Message *PointerAssignmentChecker::Say(A &&... x) {
  auto *msg{context_.messages().Say(std::forward<A>(x)...)};
  if (lhs_) {
    return evaluate::AttachDeclaration(msg, *lhs_);
  } else if (!source_.empty()) {
    msg->Attach(source_, "Declaration of %s"_en_US, description_);
  }
  return msg;
}

void CheckPointerAssignment(
    evaluate::FoldingContext &context, const Symbol &lhs, const SomeExpr &rhs) {
  // TODO: Acquire values of deferred type parameters &/or array bounds
  // from the RHS.
  if (!IsPointer(lhs)) {
    evaluate::SayWithDeclaration(
        context.messages(), lhs, "'%s' is not a pointer"_err_en_US, lhs.name());
  } else {
    std::string description{"pointer '"s + lhs.name().ToString() + '\''};
    PointerAssignmentChecker{lhs.name(), description, context}
        .set_lhsType(TypeAndShape::Characterize(lhs, context))
        .set_procedure(Procedure::Characterize(lhs, context.intrinsics()))
        .set_lhs(lhs)
        .set_isContiguous(lhs.attrs().test(Attr::CONTIGUOUS))
        .Check(rhs);
  }
}

void CheckPointerAssignment(evaluate::FoldingContext &context,
    parser::CharBlock source, const std::string &description,
    const DummyDataObject &lhs, const SomeExpr &rhs) {
  PointerAssignmentChecker{source, description, context}
      .set_lhsType(common::Clone(lhs.type))
      .set_isContiguous(lhs.attrs.test(DummyDataObject::Attr::Contiguous))
      .Check(rhs);
}

}
