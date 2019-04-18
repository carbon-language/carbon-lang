// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

// Defines data structures to represent "characteristics" of Fortran
// procedures and other entities as they are specified in section 15.3
// of Fortran 2018.

#ifndef FORTRAN_EVALUATE_CHARACTERISTICS_H_
#define FORTRAN_EVALUATE_CHARACTERISTICS_H_

#include "common.h"
#include "expression.h"
#include "type.h"
#include "../common/Fortran.h"
#include "../common/enum-set.h"
#include "../common/idioms.h"
#include "../common/indirection.h"
#include "../semantics/symbol.h"
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

namespace Fortran::evaluate {
class IntrinsicProcTable;
}
namespace Fortran::evaluate::characteristics {
struct Procedure;
}
extern template class Fortran::common::Indirection<
    Fortran::evaluate::characteristics::Procedure, true>;

namespace Fortran::evaluate::characteristics {

// Absent components are deferred or assumed.
using Shape = std::vector<std::optional<Expr<SubscriptInteger>>>;

class TypeAndShape {
public:
  explicit TypeAndShape(DynamicType t) : type_{t} {}
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(TypeAndShape)

  DynamicType type() const { return type_; }
  const Shape &shape() const { return shape_; }
  bool IsAssumedRank() const { return isAssumedRank_; }

  bool operator==(const TypeAndShape &) const;

  static std::optional<TypeAndShape> Characterize(const semantics::Symbol &);
  static std::optional<TypeAndShape> Characterize(const semantics::Symbol *);
  static std::optional<TypeAndShape> Characterize(
      const semantics::ObjectEntityDetails &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::ProcEntityDetails &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::ProcInterface &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::DeclTypeSpec &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::DeclTypeSpec *);

  std::ostream &Dump(std::ostream &) const;

private:
  void AcquireShape(const semantics::ObjectEntityDetails &);

protected:
  DynamicType type_;
  Shape shape_;
  bool isAssumedRank_{false};
};

// 15.3.2.2
struct DummyDataObject : public TypeAndShape {
  ENUM_CLASS(Attr, Optional, Allocatable, Asynchronous, Contiguous, Value,
      Volatile, Pointer, Target)
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyDataObject)
  explicit DummyDataObject(const TypeAndShape &t) : TypeAndShape{t} {}
  explicit DummyDataObject(TypeAndShape &&t) : TypeAndShape{std::move(t)} {}
  explicit DummyDataObject(DynamicType t) : TypeAndShape{t} {}
  bool operator==(const DummyDataObject &) const;
  static std::optional<DummyDataObject> Characterize(const semantics::Symbol &);
  std::ostream &Dump(std::ostream &) const;
  std::vector<Expr<SubscriptInteger>> coshape;
  common::Intent intent{common::Intent::Default};
  common::EnumSet<Attr, 32> attrs;
};

// 15.3.2.3
struct DummyProcedure {
  ENUM_CLASS(Attr, Pointer, Optional)
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(DummyProcedure)
  explicit DummyProcedure(Procedure &&);
  bool operator==(const DummyProcedure &) const;
  static std::optional<DummyProcedure> Characterize(
      const semantics::Symbol &, const IntrinsicProcTable &);
  std::ostream &Dump(std::ostream &) const;
  common::CopyableIndirection<Procedure> procedure;
  common::EnumSet<Attr, 32> attrs;
};

// 15.3.2.4
struct AlternateReturn {
  bool operator==(const AlternateReturn &) const { return true; }
  std::ostream &Dump(std::ostream &) const;
};

// 15.3.2.1
using DummyArgument =
    std::variant<DummyDataObject, DummyProcedure, AlternateReturn>;
bool IsOptional(const DummyArgument &);
std::optional<DummyArgument> CharacterizeDummyArgument(
    const semantics::Symbol &, const IntrinsicProcTable &);

// 15.3.3
struct FunctionResult {
  ENUM_CLASS(Attr, Allocatable, Pointer, Contiguous)
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(FunctionResult)
  explicit FunctionResult(DynamicType t) : u{TypeAndShape{t}} {}
  explicit FunctionResult(TypeAndShape &&t) : u{std::move(t)} {}
  explicit FunctionResult(Procedure &&p) : u{std::move(p)} {}
  ~FunctionResult();
  bool operator==(const FunctionResult &) const;
  static std::optional<FunctionResult> Characterize(
      const Symbol &, const IntrinsicProcTable &);

  bool IsAssumedLengthCharacter() const;

  const Procedure *IsProcedurePointer() const {
    if (const auto *pp{
            std::get_if<common::CopyableIndirection<Procedure>>(&u)}) {
      return &pp->value();
    } else {
      return nullptr;
    }
  }
  std::ostream &Dump(std::ostream &) const;

  common::EnumSet<Attr, 32> attrs;
  std::variant<TypeAndShape, common::CopyableIndirection<Procedure>> u;
};

// 15.3.1
struct Procedure {
  ENUM_CLASS(Attr, Pure, Elemental, BindC, ImplicitInterface)
  Procedure() {}
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(Procedure)
  bool operator==(const Procedure &) const;

  static std::optional<Procedure> Characterize(
      const semantics::Symbol &, const IntrinsicProcTable &);
  bool IsFunction() const { return functionResult.has_value(); }
  bool IsSubroutine() const { return !IsFunction(); }
  bool IsPure() const { return attrs.test(Attr::Pure); }
  bool IsElemental() const { return attrs.test(Attr::Elemental); }
  bool IsBindC() const { return attrs.test(Attr::BindC); }
  bool HasExplicitInterface() const {
    return !attrs.test(Attr::ImplicitInterface);
  }
  std::ostream &Dump(std::ostream &) const;

  std::optional<FunctionResult> functionResult;
  std::vector<DummyArgument> dummyArguments;
  common::EnumSet<Attr, 32> attrs;
};
}
#endif  // FORTRAN_EVALUATE_CHARACTERISTICS_H_
