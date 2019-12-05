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

#ifndef FORTRAN_EVALUATE_CALL_H_
#define FORTRAN_EVALUATE_CALL_H_

#include "common.h"
#include "constant.h"
#include "formatting.h"
#include "type.h"
#include "../common/indirection.h"
#include "../common/reference.h"
#include "../parser/char-block.h"
#include "../semantics/attr.h"
#include <optional>
#include <ostream>
#include <vector>

namespace Fortran::semantics {
class Symbol;
}

// Mutually referential data structures are represented here with forward
// declarations of hitherto undefined class types and a level of indirection.
namespace Fortran::evaluate {
class Component;
class IntrinsicProcTable;
}
namespace Fortran::evaluate::characteristics {
struct DummyArgument;
struct Procedure;
}

extern template class Fortran::common::Indirection<Fortran::evaluate::Component,
    true>;
extern template class Fortran::common::Indirection<
    Fortran::evaluate::characteristics::Procedure, true>;

namespace Fortran::evaluate {

using semantics::Symbol;
using SymbolRef = common::Reference<const Symbol>;

class ActualArgument {
public:
  // Dummy arguments that are TYPE(*) can be forwarded as actual arguments.
  // Since that's the only thing one may do with them in Fortran, they're
  // represented in expressions as a special case of an actual argument.
  class AssumedType {
  public:
    explicit AssumedType(const Symbol &);
    DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(AssumedType)
    const Symbol &symbol() const { return symbol_; }
    int Rank() const;
    bool operator==(const AssumedType &that) const {
      return &*symbol_ == &*that.symbol_;
    }
    std::ostream &AsFortran(std::ostream &) const;

  private:
    SymbolRef symbol_;
  };

  // A placeholder for the passed-object argument, which will be replaced
  // with the base object of the Component that constitutes the call's
  // ProcedureDesignator.
  struct PassedObject {
    bool operator==(const PassedObject &) const { return true; }
  };

  explicit ActualArgument(Expr<SomeType> &&);
  explicit ActualArgument(common::CopyableIndirection<Expr<SomeType>> &&);
  explicit ActualArgument(AssumedType);
  explicit ActualArgument(PassedObject &&) : u_{PassedObject{}} {}
  ~ActualArgument();
  ActualArgument &operator=(Expr<SomeType> &&);

  Expr<SomeType> *UnwrapExpr() {
    if (auto *p{
            std::get_if<common::CopyableIndirection<Expr<SomeType>>>(&u_)}) {
      return &p->value();
    } else {
      return nullptr;
    }
  }
  const Expr<SomeType> *UnwrapExpr() const {
    if (const auto *p{
            std::get_if<common::CopyableIndirection<Expr<SomeType>>>(&u_)}) {
      return &p->value();
    } else {
      return nullptr;
    }
  }

  const Symbol *GetAssumedTypeDummy() const {
    if (const AssumedType * aType{std::get_if<AssumedType>(&u_)}) {
      return &aType->symbol();
    } else {
      return nullptr;
    }
  }

  std::optional<DynamicType> GetType() const;
  int Rank() const;
  bool operator==(const ActualArgument &) const;
  std::ostream &AsFortran(std::ostream &) const;

  std::optional<parser::CharBlock> keyword() const { return keyword_; }
  void set_keyword(parser::CharBlock x) { keyword_ = x; }
  bool isAlternateReturn() const { return isAlternateReturn_; }
  void set_isAlternateReturn() { isAlternateReturn_ = true; }

  bool IsPassedObject() const {
    return std::holds_alternative<PassedObject>(u_);
  }
  bool Matches(const characteristics::DummyArgument &) const;

  // Wrap this argument in parentheses
  void Parenthesize();

  // TODO: Mark legacy %VAL and %REF arguments

private:
  // Subtlety: There is a distinction that must be maintained here between an
  // actual argument expression that is a variable and one that is not,
  // e.g. between X and (X).  The parser attempts to parse each argument
  // first as a variable, then as an expression, and the distinction appears
  // in the parse tree.
  std::variant<common::CopyableIndirection<Expr<SomeType>>, AssumedType,
      PassedObject>
      u_;
  std::optional<parser::CharBlock> keyword_;
  bool isAlternateReturn_{false};  // whether expr is a "*label" number
};

using ActualArguments = std::vector<std::optional<ActualArgument>>;

// Intrinsics are identified by their names and the characteristics
// of their arguments, at least for now.
using IntrinsicProcedure = std::string;

struct SpecificIntrinsic {
  SpecificIntrinsic(IntrinsicProcedure, characteristics::Procedure &&);
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(SpecificIntrinsic)
  ~SpecificIntrinsic();
  bool operator==(const SpecificIntrinsic &) const;
  std::ostream &AsFortran(std::ostream &) const;

  IntrinsicProcedure name;
  bool isRestrictedSpecific{false};  // if true, can only call it, not pass it
  common::CopyableIndirection<characteristics::Procedure> characteristics;
};

struct ProcedureDesignator {
  EVALUATE_UNION_CLASS_BOILERPLATE(ProcedureDesignator)
  explicit ProcedureDesignator(SpecificIntrinsic &&i) : u{std::move(i)} {}
  explicit ProcedureDesignator(const Symbol &n) : u{n} {}
  explicit ProcedureDesignator(Component &&);

  // Exactly one of these will return a non-null pointer.
  const SpecificIntrinsic *GetSpecificIntrinsic() const;
  const Symbol *GetSymbol() const;  // symbol or component symbol

  // Always null if the procedure is intrinsic.
  const Component *GetComponent() const;

  const Symbol *GetInterfaceSymbol() const;

  std::string GetName() const;
  std::optional<DynamicType> GetType() const;
  int Rank() const;
  bool IsElemental() const;
  std::optional<Expr<SubscriptInteger>> LEN() const;
  std::ostream &AsFortran(std::ostream &) const;

  std::variant<SpecificIntrinsic, SymbolRef,
      common::CopyableIndirection<Component>>
      u;
};

class ProcedureRef {
public:
  CLASS_BOILERPLATE(ProcedureRef)
  ProcedureRef(ProcedureDesignator &&p, ActualArguments &&a)
    : proc_{std::move(p)}, arguments_(std::move(a)) {}

  ProcedureDesignator &proc() { return proc_; }
  const ProcedureDesignator &proc() const { return proc_; }
  ActualArguments &arguments() { return arguments_; }
  const ActualArguments &arguments() const { return arguments_; }

  std::optional<Expr<SubscriptInteger>> LEN() const;
  int Rank() const { return proc_.Rank(); }
  bool IsElemental() const { return proc_.IsElemental(); }
  bool operator==(const ProcedureRef &) const;
  std::ostream &AsFortran(std::ostream &) const;

protected:
  ProcedureDesignator proc_;
  ActualArguments arguments_;
};

template<typename A> class FunctionRef : public ProcedureRef {
public:
  using Result = A;
  CLASS_BOILERPLATE(FunctionRef)
  FunctionRef(ProcedureRef &&pr) : ProcedureRef{std::move(pr)} {}
  FunctionRef(ProcedureDesignator &&p, ActualArguments &&a)
    : ProcedureRef{std::move(p), std::move(a)} {}

  std::optional<DynamicType> GetType() const { return proc_.GetType(); }
  std::optional<Constant<Result>> Fold(FoldingContext &);  // for intrinsics
};

FOR_EACH_SPECIFIC_TYPE(extern template class FunctionRef, )
}
#endif  // FORTRAN_EVALUATE_CALL_H_
