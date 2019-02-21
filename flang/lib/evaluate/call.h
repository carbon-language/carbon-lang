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
#include "type.h"
#include "../common/indirection.h"
#include "../parser/char-block.h"
#include "../semantics/attr.h"
#include <optional>
#include <ostream>
#include <vector>

namespace Fortran::semantics {
class Symbol;
}

namespace Fortran::evaluate {

struct ActualArgument {
  explicit ActualArgument(Expr<SomeType> &&x) : value{std::move(x)} {}
  explicit ActualArgument(CopyableIndirection<Expr<SomeType>> &&v)
    : value{std::move(v)} {}

  std::optional<DynamicType> GetType() const;
  int Rank() const;
  bool operator==(const ActualArgument &) const;
  std::ostream &AsFortran(std::ostream &) const;
  std::optional<int> VectorSize() const;

  std::optional<parser::CharBlock> keyword;
  bool isAlternateReturn{false};  // when true, "value" is a label number

  // TODO: Mark legacy %VAL and %REF arguments

  // Subtlety: There is a distinction that must be maintained here between an
  // actual argument expression that is a variable and one that is not,
  // e.g. between X and (X).  The parser attempts to parse each argument
  // first as a variable, then as an expression, and the distinction appears
  // in the parse tree.
  CopyableIndirection<Expr<SomeType>> value;
};

using ActualArguments = std::vector<std::optional<ActualArgument>>;

// Intrinsics are identified by their names and the characteristics
// of their arguments, at least for now.
using IntrinsicProcedure = std::string;

struct SpecificIntrinsic {
  explicit SpecificIntrinsic(IntrinsicProcedure n) : name{n} {}
  SpecificIntrinsic(IntrinsicProcedure n, std::optional<DynamicType> &&dt,
      int r, semantics::Attrs a)
    : name{n}, type{std::move(dt)}, rank{r}, attrs{a} {}
  SpecificIntrinsic(const SpecificIntrinsic &) = default;
  SpecificIntrinsic(SpecificIntrinsic &&) = default;
  SpecificIntrinsic &operator=(const SpecificIntrinsic &) = default;
  SpecificIntrinsic &operator=(SpecificIntrinsic &&) = default;
  bool operator==(const SpecificIntrinsic &) const;
  std::ostream &AsFortran(std::ostream &) const;

  IntrinsicProcedure name;
  bool isRestrictedSpecific{false};  // if true, can only call it
  std::optional<DynamicType> type;  // absent if subroutine call or NULL()
  int rank{0};
  semantics::Attrs attrs;  // ELEMENTAL, POINTER
};

struct ProcedureDesignator {
  EVALUATE_UNION_CLASS_BOILERPLATE(ProcedureDesignator)
  explicit ProcedureDesignator(SpecificIntrinsic &&i) : u{std::move(i)} {}
  explicit ProcedureDesignator(const semantics::Symbol &n) : u{&n} {}
  std::optional<DynamicType> GetType() const;
  int Rank() const;
  bool IsElemental() const;
  Expr<SubscriptInteger> LEN() const;
  const semantics::Symbol *GetSymbol() const;
  std::ostream &AsFortran(std::ostream &) const;

  // TODO: When calling X%F, pass X as PASS argument unless NOPASS
  std::variant<SpecificIntrinsic, const semantics::Symbol *> u;
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

  Expr<SubscriptInteger> LEN() const;
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

FOR_EACH_SPECIFIC_TYPE(extern template class FunctionRef)
}
#endif  // FORTRAN_EVALUATE_CALL_H_
