//===-- lib/semantics/assignment.cc ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "assignment.h"
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

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

class PointerAssignmentChecker {
public:
  PointerAssignmentChecker(const Symbol *pointer, parser::CharBlock source,
      const std::string &description, const characteristics::TypeAndShape *type,
      FoldingContext &context, const characteristics::Procedure *procedure,
      bool isContiguous)
    : pointer_{pointer}, source_{source},
      description_{description}, type_{type}, context_{context},
      procedure_{procedure}, isContiguous_{isContiguous} {}

  template<typename A> void Check(const A &) {
    // Catch-all case for really bad target expression
    Say("Target associated with %s must be a designator or a call to a pointer-valued function"_err_en_US,
        description_);
  }

  template<typename T> void Check(const Expr<T> &x) {
    std::visit([&](const auto &x) { Check(x); }, x.u);
  }
  void Check(const Expr<SomeType> &);
  void Check(const NullPointer &) {}  // P => NULL() without MOLD=; always OK

  template<typename T> void Check(const FunctionRef<T> &f) {
    std::string funcName;
    const auto *symbol{f.proc().GetSymbol()};
    if (symbol) {
      funcName = symbol->name().ToString();
    } else if (const auto *intrinsic{f.proc().GetSpecificIntrinsic()}) {
      funcName = intrinsic->name;
    }
    if (auto proc{characteristics::Procedure::Characterize(
            f.proc(), context_.intrinsics())}) {
      std::optional<parser::MessageFixedText> error;
      if (const auto &funcResult{proc->functionResult}) {  // C1025
        const auto *frProc{funcResult->IsProcedurePointer()};
        if (procedure_) {
          // Shouldn't be here in this function unless lhs
          // is an object pointer.
          error =
              "Procedure %s is associated with the result of a reference to function '%s' that does not return a procedure pointer"_err_en_US;
        } else if (frProc) {
          error =
              "Object %s is associated with the result of a reference to function '%s' that is a procedure pointer"_err_en_US;
        } else if (!funcResult->attrs.test(
                       characteristics::FunctionResult::Attr::Pointer)) {
          error =
              "%s is associated with the result of a reference to function '%s' that is a not a pointer"_err_en_US;
        } else if (isContiguous_ &&
            !funcResult->attrs.test(
                characteristics::FunctionResult::Attr::Contiguous)) {
          error =
              "CONTIGUOUS %s is associated with the result of reference to function '%s' that is not contiguous"_err_en_US;
        } else if (type_) {
          const auto *frTypeAndShape{funcResult->GetTypeAndShape()};
          CHECK(frTypeAndShape);
          if (!type_->IsCompatibleWith(context_.messages(), *frTypeAndShape)) {
            error =
                "%s is associated with the result of a reference to function '%s' whose pointer result has an incompatible type or shape"_err_en_US;
          }
        }
      } else {
        error =
            "%s is associated with the non-existent result of reference to procedure"_err_en_US;
      }
      if (error) {
        auto restorer{common::ScopedSet(pointer_, symbol)};
        Say(*error, description_, funcName);
      }
    }
  }

  template<typename T> void Check(const Designator<T> &d) {
    const Symbol *last{d.GetLastSymbol()};
    const Symbol *base{d.GetBaseObject().symbol()};
    if (last && base) {
      std::optional<parser::MessageFixedText> error;
      if (procedure_) {
        // Shouldn't be here in this function unless lhs is an
        // object pointer.
        error =
            "In assignment to procedure %s, the target is not a procedure or procedure pointer"_err_en_US;
      } else if (!GetLastTarget(GetSymbolVector(d))) {  // C1025
        error =
            "In assignment to object %s, the target '%s' is not an object with POINTER or TARGET attributes"_err_en_US;
      } else if (auto rhsTypeAndShape{
                     characteristics::TypeAndShape::Characterize(
                         *last, context_)}) {
        if (!type_ ||
            !type_->IsCompatibleWith(context_.messages(), *rhsTypeAndShape)) {
          error =
              "%s associated with object '%s' with incompatible type or shape"_err_en_US;
        }
      }
      if (error) {
        auto restorer{common::ScopedSet(pointer_, last)};
        Say(*error, description_, last->name());
      }
    } else {
      // P => "character literal"(1:3)
      context_.messages().Say("Pointer target is not a named entity"_err_en_US);
    }
  }

  void Check(const ProcedureDesignator &);
  void Check(const ProcedureRef &);

private:
  // Target is a procedure
  void Check(parser::CharBlock rhsName, bool isCall,
      const characteristics::Procedure * = nullptr);

  template<typename... A> parser::Message *Say(A &&... x) {
    auto *msg{context_.messages().Say(std::forward<A>(x)...)};
    if (pointer_) {
      return AttachDeclaration(msg, *pointer_);
    } else if (!source_.empty()) {
      msg->Attach(source_, "Declaration of %s"_en_US, description_);
    }
    return msg;
  }

  const Symbol *pointer_{nullptr};
  const parser::CharBlock source_;
  const std::string &description_;
  const characteristics::TypeAndShape *type_{nullptr};
  FoldingContext &context_;
  const characteristics::Procedure *procedure_{nullptr};
  bool isContiguous_{false};
};

void PointerAssignmentChecker::Check(const Expr<SomeType> &rhs) {
  if (HasVectorSubscript(rhs)) {  // C1025
    Say("An array section with a vector subscript may not be a pointer target"_err_en_US);
  } else if (ExtractCoarrayRef(rhs)) {  // C1026
    Say("A coindexed object may not be a pointer target"_err_en_US);
  } else {
    std::visit([&](const auto &x) { Check(x); }, rhs.u);
  }
}

// Common handling for procedure pointer right-hand sides
void PointerAssignmentChecker::Check(parser::CharBlock rhsName, bool isCall,
    const characteristics::Procedure *targetChars) {
  if (procedure_) {
    if (targetChars) {
      if (*procedure_ != *targetChars) {
        if (isCall) {
          Say("Procedure %s associated with result of reference to function '%s' that is an incompatible procedure pointer"_err_en_US,
              description_, rhsName);
        } else {
          Say("Procedure %s associated with incompatible procedure designator '%s'"_err_en_US,
              description_, rhsName);
        }
      }
    } else {
      Say("In assignment to procedure %s, the characteristics of the target procedure '%s' could not be determined"_err_en_US,
          description_, rhsName);
    }
  } else {
    Say("In assignment to object %s, the target '%s' is a procedure designator"_err_en_US,
        description_, rhsName);
  }
}

void PointerAssignmentChecker::Check(const ProcedureDesignator &d) {
  if (auto chars{
          characteristics::Procedure::Characterize(d, context_.intrinsics())}) {
    Check(d.GetName(), false, &*chars);
  } else {
    Check(d.GetName(), false);
  }
}

void PointerAssignmentChecker::Check(const ProcedureRef &ref) {
  const characteristics::Procedure *procedure{nullptr};
  auto chars{
      characteristics::Procedure::Characterize(ref, context_.intrinsics())};
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

void CheckPointerAssignment(
    FoldingContext &context, const Symbol &lhs, const Expr<SomeType> &rhs) {
  // TODO: Acquire values of deferred type parameters &/or array bounds
  // from the RHS.
  if (!IsPointer(lhs)) {
    SayWithDeclaration(
        context.messages(), lhs, "'%s' is not a pointer"_err_en_US, lhs.name());
  } else {
    auto type{characteristics::TypeAndShape::Characterize(lhs, context)};
    auto proc{
        characteristics::Procedure::Characterize(lhs, context.intrinsics())};
    std::string description{"pointer '"s + lhs.name().ToString() + '\''};
    PointerAssignmentChecker{&lhs, lhs.name(), description,
        type ? &*type : nullptr, context, proc ? &*proc : nullptr,
        lhs.attrs().test(semantics::Attr::CONTIGUOUS)}
        .Check(rhs);
  }
}

void CheckPointerAssignment(FoldingContext &context, parser::CharBlock source,
    const std::string &description, const characteristics::DummyDataObject &lhs,
    const Expr<SomeType> &rhs) {
  PointerAssignmentChecker{nullptr, source, description, &lhs.type, context,
      nullptr /* proc */,
      lhs.attrs.test(characteristics::DummyDataObject::Attr::Contiguous)}
      .Check(rhs);
}

}

namespace Fortran::semantics {

using ControlExpr = evaluate::Expr<evaluate::SubscriptInteger>;
using MaskExpr = evaluate::Expr<evaluate::LogicalResult>;

// The context tracks some number of active FORALL statements/constructs
// and some number of active WHERE statements/constructs.  WHERE can nest
// in FORALL but not vice versa.  Pointer assignments are allowed in
// FORALL but not in WHERE.  These constraints are manifest in the grammar
// and don't need to be rechecked here, since errors cannot appear in the
// parse tree.
struct Control {
  Symbol *name;
  ControlExpr lower, upper, step;
};

struct ForallContext {
  explicit ForallContext(const ForallContext *that) : outer{that} {}

  std::optional<int> GetActiveIntKind(const parser::CharBlock &name) const {
    const auto iter{activeNames.find(name)};
    if (iter != activeNames.cend()) {
      return {integerKind};
    } else if (outer) {
      return outer->GetActiveIntKind(name);
    } else {
      return std::nullopt;
    }
  }

  const ForallContext *outer{nullptr};
  std::optional<parser::CharBlock> constructName;
  int integerKind;
  std::vector<Control> control;
  std::optional<MaskExpr> maskExpr;
  std::set<parser::CharBlock> activeNames;
};

struct WhereContext {
  WhereContext(MaskExpr &&x, const WhereContext *o, const ForallContext *f)
    : outer{o}, forall{f}, thisMaskExpr{std::move(x)} {}
  const WhereContext *outer{nullptr};
  const ForallContext *forall{nullptr};  // innermost enclosing FORALL
  std::optional<parser::CharBlock> constructName;
  MaskExpr thisMaskExpr;  // independent of outer WHERE, if any
  MaskExpr cumulativeMaskExpr{thisMaskExpr};
};

class AssignmentContext {
public:
  explicit AssignmentContext(SemanticsContext &c) : context_{c} {}
  AssignmentContext(const AssignmentContext &c, WhereContext &w)
    : context_{c.context_}, where_{&w} {}
  AssignmentContext(const AssignmentContext &c, ForallContext &f)
    : context_{c.context_}, forall_{&f} {}

  bool operator==(const AssignmentContext &x) const { return this == &x; }

  void Analyze(const parser::AssignmentStmt &);
  void Analyze(const parser::PointerAssignmentStmt &);
  void Analyze(const parser::WhereStmt &);
  void Analyze(const parser::WhereConstruct &);
  void Analyze(const parser::ForallStmt &);
  void Analyze(const parser::ForallConstruct &);
  void Analyze(const parser::ForallConstructStmt &);
  void Analyze(const parser::ConcurrentHeader &);

  template<typename A> void Analyze(const parser::UnlabeledStatement<A> &stmt) {
    context_.set_location(stmt.source);
    Analyze(stmt.statement);
  }
  template<typename A> void Analyze(const common::Indirection<A> &x) {
    Analyze(x.value());
  }
  template<typename A> std::enable_if_t<UnionTrait<A>> Analyze(const A &x) {
    std::visit([&](const auto &y) { Analyze(y); }, x.u);
  }
  template<typename A> void Analyze(const std::list<A> &list) {
    for (const auto &elem : list) {
      Analyze(elem);
    }
  }
  template<typename A> void Analyze(const std::optional<A> &x) {
    if (x) {
      Analyze(*x);
    }
  }

private:
  void Analyze(const parser::WhereConstruct::MaskedElsewhere &);
  void Analyze(const parser::MaskedElsewhereStmt &);
  void Analyze(const parser::WhereConstruct::Elsewhere &);

  int GetIntegerKind(const std::optional<parser::IntegerTypeSpec> &);
  void CheckForImpureCall(const SomeExpr &);
  void CheckForImpureCall(const SomeExpr *);
  void CheckForPureContext(const SomeExpr &lhs, const SomeExpr &rhs,
      parser::CharBlock rhsSource, bool isPointerAssignment);

  MaskExpr GetMask(const parser::LogicalExpr &, bool defaultValue = true);

  template<typename... A>
  parser::Message *Say(parser::CharBlock at, A &&... args) {
    return &context_.Say(at, std::forward<A>(args)...);
  }

  SemanticsContext &context_;
  WhereContext *where_{nullptr};
  ForallContext *forall_{nullptr};
};

void AssignmentContext::Analyze(const parser::AssignmentStmt &stmt) {
  // Assignment statement analysis is in expression.cc where user-defined
  // assignments can be recognized and replaced.
  if (const evaluate::Assignment * asst{GetAssignment(stmt)}) {
    if (const auto *intrinsicAsst{
            std::get_if<evaluate::Assignment::IntrinsicAssignment>(&asst->u)}) {
      CheckForImpureCall(intrinsicAsst->lhs);
      CheckForImpureCall(intrinsicAsst->rhs);
      if (forall_) {
        // TODO: Warn if some name in forall_->activeNames or its outer
        // contexts does not appear on LHS
      }
      CheckForPureContext(intrinsicAsst->lhs, intrinsicAsst->rhs,
          std::get<parser::Expr>(stmt.t).source, false /* not => */);
    }
  }
  // TODO: Fortran 2003 ALLOCATABLE assignment semantics (automatic
  // (re)allocation of LHS array when unallocated or nonconformable)
}

void AssignmentContext::Analyze(const parser::PointerAssignmentStmt &stmt) {
  CHECK(!where_);
  if (const evaluate::Assignment * asst{GetAssignment(stmt)}) {
    auto [lhs, rhs]{std::visit(
        common::visitors{
            [&](const evaluate::Assignment::IntrinsicAssignment &x) {
              return std::make_pair(&x.lhs, &x.rhs);
            },
            [&](const evaluate::ProcedureRef &x) {
              return std::make_pair(x.arguments()[0]->UnwrapExpr(),
                  x.arguments()[1]->UnwrapExpr());
            },
            [&](const evaluate::Assignment::PointerAssignment &x) {
              std::visit(
                  common::visitors{
                      [&](const evaluate::Assignment::PointerAssignment::
                              BoundsSpec &bounds) {
                        for (const auto &bound : bounds) {
                          CheckForImpureCall(SomeExpr{bound});
                        }
                      },
                      [&](const evaluate::Assignment::PointerAssignment::
                              BoundsRemapping &bounds) {
                        for (const auto &bound : bounds) {
                          CheckForImpureCall(SomeExpr{bound.first});
                          CheckForImpureCall(SomeExpr{bound.second});
                        }
                      },
                  },
                  x.bounds);
              return std::make_pair(&x.lhs, &x.rhs);
            },
        },
        asst->u)};
    CheckForImpureCall(lhs);
    CheckForImpureCall(rhs);
    if (forall_) {
      // TODO: Warn if some name in forall_->activeNames or its outer
      // contexts does not appear on LHS
    }
    if (lhs && rhs) {
      CheckForPureContext(
          *lhs, *rhs, std::get<parser::Expr>(stmt.t).source, true /* => */);
    }
    // TODO continue here, using CheckPointerAssignment()
  }
}

void AssignmentContext::Analyze(const parser::WhereStmt &stmt) {
  WhereContext where{
      GetMask(std::get<parser::LogicalExpr>(stmt.t)), where_, forall_};
  AssignmentContext nested{*this, where};
  nested.Analyze(std::get<parser::AssignmentStmt>(stmt.t));
}

// N.B. Construct name matching is checked during label resolution.
void AssignmentContext::Analyze(const parser::WhereConstruct &construct) {
  const auto &whereStmt{
      std::get<parser::Statement<parser::WhereConstructStmt>>(construct.t)};
  WhereContext where{
      GetMask(std::get<parser::LogicalExpr>(whereStmt.statement.t)), where_,
      forall_};
  if (const auto &name{
          std::get<std::optional<parser::Name>>(whereStmt.statement.t)}) {
    where.constructName = name->source;
  }
  AssignmentContext nested{*this, where};
  nested.Analyze(std::get<std::list<parser::WhereBodyConstruct>>(construct.t));
  nested.Analyze(std::get<std::list<parser::WhereConstruct::MaskedElsewhere>>(
      construct.t));
  nested.Analyze(
      std::get<std::optional<parser::WhereConstruct::Elsewhere>>(construct.t));
}

void AssignmentContext::Analyze(const parser::ForallStmt &stmt) {
  CHECK(!where_);
  ForallContext forall{forall_};
  AssignmentContext nested{*this, forall};
  nested.Analyze(
      std::get<common::Indirection<parser::ConcurrentHeader>>(stmt.t));
  nested.Analyze(
      std::get<parser::UnlabeledStatement<parser::ForallAssignmentStmt>>(
          stmt.t));
}

// N.B. Construct name matching is checked during label resolution;
// index name distinction is checked during name resolution.
void AssignmentContext::Analyze(const parser::ForallConstruct &construct) {
  CHECK(!where_);
  ForallContext forall{forall_};
  AssignmentContext nested{*this, forall};
  nested.Analyze(
      std::get<parser::Statement<parser::ForallConstructStmt>>(construct.t));
  nested.Analyze(std::get<std::list<parser::ForallBodyConstruct>>(construct.t));
}

void AssignmentContext::Analyze(const parser::ForallConstructStmt &stmt) {
  Analyze(std::get<common::Indirection<parser::ConcurrentHeader>>(stmt.t));
}

void AssignmentContext::Analyze(
    const parser::WhereConstruct::MaskedElsewhere &elsewhere) {
  CHECK(where_);
  Analyze(
      std::get<parser::Statement<parser::MaskedElsewhereStmt>>(elsewhere.t));
  Analyze(std::get<std::list<parser::WhereBodyConstruct>>(elsewhere.t));
}

void AssignmentContext::Analyze(const parser::MaskedElsewhereStmt &elsewhere) {
  MaskExpr mask{GetMask(std::get<parser::LogicalExpr>(elsewhere.t))};
  MaskExpr copyCumulative{where_->cumulativeMaskExpr};
  MaskExpr notOldMask{evaluate::LogicalNegation(std::move(copyCumulative))};
  if (!evaluate::AreConformable(notOldMask, mask)) {
    context_.Say("mask of ELSEWHERE statement is not conformable with "
                 "the prior mask(s) in its WHERE construct"_err_en_US);
  }
  MaskExpr copyMask{mask};
  where_->cumulativeMaskExpr =
      evaluate::BinaryLogicalOperation(evaluate::LogicalOperator::Or,
          std::move(where_->cumulativeMaskExpr), std::move(copyMask));
  where_->thisMaskExpr = evaluate::BinaryLogicalOperation(
      evaluate::LogicalOperator::And, std::move(notOldMask), std::move(mask));
  if (where_->outer &&
      !evaluate::AreConformable(
          where_->outer->thisMaskExpr, where_->thisMaskExpr)) {
    context_.Say("effective mask of ELSEWHERE statement is not conformable "
                 "with the mask of the surrounding WHERE construct"_err_en_US);
  }
}

void AssignmentContext::Analyze(
    const parser::WhereConstruct::Elsewhere &elsewhere) {
  MaskExpr copyCumulative{DEREF(where_).cumulativeMaskExpr};
  where_->thisMaskExpr = evaluate::LogicalNegation(std::move(copyCumulative));
  Analyze(std::get<std::list<parser::WhereBodyConstruct>>(elsewhere.t));
}

void AssignmentContext::Analyze(const parser::ConcurrentHeader &header) {
  DEREF(forall_).integerKind = GetIntegerKind(
      std::get<std::optional<parser::IntegerTypeSpec>>(header.t));
  for (const auto &control :
      std::get<std::list<parser::ConcurrentControl>>(header.t)) {
    const parser::Name &name{std::get<parser::Name>(control.t)};
    bool inserted{forall_->activeNames.insert(name.source).second};
    CHECK(inserted || context_.HasError(name));
    CheckForImpureCall(GetExpr(std::get<1>(control.t)));
    CheckForImpureCall(GetExpr(std::get<2>(control.t)));
    if (const auto &stride{std::get<3>(control.t)}) {
      CheckForImpureCall(GetExpr(*stride));
    }
  }
  if (const auto &mask{
          std::get<std::optional<parser::ScalarLogicalExpr>>(header.t)}) {
    CheckForImpureCall(GetExpr(*mask));
  }
}

int AssignmentContext::GetIntegerKind(
    const std::optional<parser::IntegerTypeSpec> &spec) {
  std::optional<parser::KindSelector> empty;
  evaluate::Expr<evaluate::SubscriptInteger> kind{AnalyzeKindSelector(
      context_, TypeCategory::Integer, spec ? spec->v : empty)};
  if (auto value{evaluate::ToInt64(kind)}) {
    return static_cast<int>(*value);
  } else {
    context_.Say("Kind of INTEGER type must be a constant value"_err_en_US);
    return context_.GetDefaultKind(TypeCategory::Integer);
  }
}

void AssignmentContext::CheckForImpureCall(const SomeExpr &expr) {
  if (forall_) {
    const auto &intrinsics{context_.foldingContext().intrinsics()};
    if (auto bad{FindImpureCall(intrinsics, expr)}) {
      context_.Say(
          "Impure procedure '%s' may not be referenced in a FORALL"_err_en_US,
          *bad);
    }
  }
}

void AssignmentContext::CheckForImpureCall(const SomeExpr *expr) {
  if (expr) {
    CheckForImpureCall(*expr);
  }
}

// C1594 checks
static bool IsPointerDummyOfPureFunction(const Symbol &x) {
  return IsPointerDummy(x) && FindPureProcedureContaining(x.owner()) &&
      x.owner().symbol() && IsFunction(*x.owner().symbol());
}

static const char *WhyBaseObjectIsSuspicious(
    const Symbol &x, const Scope &scope) {
  // See C1594, first paragraph.  These conditions enable checks on both
  // left-hand and right-hand sides in various circumstances.
  if (IsHostAssociated(x, scope)) {
    return "host-associated";
  } else if (IsUseAssociated(x, scope)) {
    return "USE-associated";
  } else if (IsPointerDummyOfPureFunction(x)) {
    return "a POINTER dummy argument of a pure function";
  } else if (IsIntentIn(x)) {
    return "an INTENT(IN) dummy argument";
  } else if (FindCommonBlockContaining(x)) {
    return "in a COMMON block";
  } else {
    return nullptr;
  }
}

// Checks C1594(1,2)
void CheckDefinabilityInPureScope(parser::ContextualMessages &messages,
    const Symbol &lhs, const Scope &context, const Scope &pure) {
  if (pure.symbol()) {
    if (const char *why{WhyBaseObjectIsSuspicious(lhs, context)}) {
      evaluate::SayWithDeclaration(messages, lhs,
          "Pure subprogram '%s' may not define '%s' because it is %s"_err_en_US,
          pure.symbol()->name(), lhs.name(), why);
    }
  }
}

static std::optional<std::string> GetPointerComponentDesignatorName(
    const SomeExpr &expr) {
  if (const auto *derived{
          evaluate::GetDerivedTypeSpec(evaluate::DynamicType::From(expr))}) {
    UltimateComponentIterator ultimates{*derived};
    if (auto pointer{
            std::find_if(ultimates.begin(), ultimates.end(), IsPointer)}) {
      return pointer.BuildResultDesignatorName();
    }
  }
  return std::nullopt;
}

// Checks C1594(5,6)
void CheckCopyabilityInPureScope(parser::ContextualMessages &messages,
    const SomeExpr &expr, const Scope &scope) {
  if (const Symbol * base{GetFirstSymbol(expr)}) {
    if (const char *why{WhyBaseObjectIsSuspicious(*base, scope)}) {
      if (auto pointer{GetPointerComponentDesignatorName(expr)}) {
        evaluate::SayWithDeclaration(messages, *base,
            "A pure subprogram may not copy the value of '%s' because it is %s and has the POINTER component '%s'"_err_en_US,
            base->name(), why, *pointer);
      }
    }
  }
}

void AssignmentContext::CheckForPureContext(const SomeExpr &lhs,
    const SomeExpr &rhs, parser::CharBlock source, bool isPointerAssignment) {
  const Scope &scope{context_.FindScope(source)};
  if (const Scope * pure{FindPureProcedureContaining(scope)}) {
    parser::ContextualMessages messages{
        context_.location().value(), &context_.messages()};
    if (evaluate::ExtractCoarrayRef(lhs)) {
      messages.Say(
          "A pure subprogram may not define a coindexed object"_err_en_US);
    } else if (const Symbol * base{GetFirstSymbol(lhs)}) {
      if (const auto *assoc{base->detailsIf<AssocEntityDetails>()}) {
        if (auto dataRef{ExtractDataRef(assoc->expr())}) {
          // ASSOCIATE(a=>x) -- check x, not a, for "a=..."
          CheckDefinabilityInPureScope(
              messages, dataRef->GetFirstSymbol(), scope, *pure);
        }
      } else {
        CheckDefinabilityInPureScope(messages, *base, scope, *pure);
      }
    }
    if (isPointerAssignment) {
      if (const Symbol * base{GetFirstSymbol(rhs)}) {
        if (const char *why{
                WhyBaseObjectIsSuspicious(*base, scope)}) {  // C1594(3)
          evaluate::SayWithDeclaration(messages, *base,
              "A pure subprogram may not use '%s' as the target of pointer assignment because it is %s"_err_en_US,
              base->name(), why);
        }
      }
    } else {
      if (auto type{evaluate::DynamicType::From(lhs)}) {
        // C1596 checks for polymorphic deallocation in a pure subprogram
        // due to automatic reallocation on assignment
        if (type->IsPolymorphic()) {
          context_.Say(
              "Deallocation of polymorphic object is not permitted in a pure subprogram"_err_en_US);
        }
        if (const DerivedTypeSpec * derived{GetDerivedTypeSpec(type)}) {
          if (auto bad{FindPolymorphicAllocatableNonCoarrayUltimateComponent(
                  *derived)}) {
            evaluate::SayWithDeclaration(messages, *bad,
                "Deallocation of polymorphic non-coarray component '%s' is not permitted in a pure subprogram"_err_en_US,
                bad.BuildResultDesignatorName());
          } else {
            CheckCopyabilityInPureScope(messages, rhs, scope);
          }
        }
      }
    }
  }
}

MaskExpr AssignmentContext::GetMask(
    const parser::LogicalExpr &logicalExpr, bool defaultValue) {
  MaskExpr mask{defaultValue};
  if (const SomeExpr * expr{GetExpr(logicalExpr)}) {
    CheckForImpureCall(*expr);
    auto *logical{std::get_if<evaluate::Expr<evaluate::SomeLogical>>(&expr->u)};
    mask = evaluate::ConvertTo(mask, common::Clone(DEREF(logical)));
  }
  return mask;
}

void AnalyzeConcurrentHeader(
    SemanticsContext &context, const parser::ConcurrentHeader &header) {
  AssignmentContext{context}.Analyze(header);
}

AssignmentChecker::~AssignmentChecker() {}

AssignmentChecker::AssignmentChecker(SemanticsContext &context)
  : context_{new AssignmentContext{context}} {}
void AssignmentChecker::Enter(const parser::AssignmentStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::PointerAssignmentStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::WhereStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::WhereConstruct &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::ForallStmt &x) {
  context_.value().Analyze(x);
}
void AssignmentChecker::Enter(const parser::ForallConstruct &x) {
  context_.value().Analyze(x);
}

}
template class Fortran::common::Indirection<
    Fortran::semantics::AssignmentContext>;
