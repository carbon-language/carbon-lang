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
#include "shape.h"
#include "type.h"
#include "../common/Fortran.h"
#include "../common/enum-set.h"
#include "../common/idioms.h"
#include "../common/indirection.h"
#include "../semantics/symbol.h"
#include <optional>
#include <ostream>
#include <string>
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

using common::CopyableIndirection;

// Are these procedures distinguishable for a generic name?
bool Distinguishable(const Procedure &, const Procedure &);
// Are these procedures distinguishable for a generic operator or assignment?
bool DistinguishableOpOrAssign(const Procedure &, const Procedure &);

class TypeAndShape {
public:
  ENUM_CLASS(Attr, AssumedRank, AssumedShape, AssumedSize, Coarray)
  using Attrs = common::EnumSet<Attr, Attr_enumSize>;

  explicit TypeAndShape(DynamicType t) : type_{t} { AcquireLEN(); }
  TypeAndShape(DynamicType t, int rank) : type_{t}, shape_(rank) {
    AcquireLEN();
  }
  TypeAndShape(DynamicType t, Shape &&s) : type_{t}, shape_{std::move(s)} {
    AcquireLEN();
  }
  TypeAndShape(DynamicType t, std::optional<Shape> &&s) : type_{t} {
    if (s.has_value()) {
      shape_ = std::move(*s);
    }
    AcquireLEN();
  }
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(TypeAndShape)

  bool operator==(const TypeAndShape &) const;
  static std::optional<TypeAndShape> Characterize(const semantics::Symbol &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::ObjectEntityDetails &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::ProcEntityDetails &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::ProcInterface &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::DeclTypeSpec &);
  static std::optional<TypeAndShape> Characterize(
      const Expr<SomeType> &, FoldingContext &);
  template<typename A>
  static std::optional<TypeAndShape> Characterize(const A *p) {
    return p ? Characterize(*p) : std::nullopt;
  }

  DynamicType type() const { return type_; }
  TypeAndShape &set_type(DynamicType t) {
    type_ = t;
    return *this;
  }
  const std::optional<Expr<SomeInteger>> &LEN() const { return LEN_; }
  TypeAndShape &set_LEN(Expr<SomeInteger> &&len) {
    LEN_ = std::move(len);
    return *this;
  }
  const Shape &shape() const { return shape_; }
  const Attrs &attrs() const { return attrs_; }

  int Rank() const { return GetRank(shape_); }
  bool IsCompatibleWith(
      parser::ContextualMessages &, const TypeAndShape &) const;

  std::ostream &Dump(std::ostream &) const;

private:
  void AcquireShape(const semantics::ObjectEntityDetails &);
  void AcquireLEN();

protected:
  DynamicType type_;
  std::optional<Expr<SomeInteger>> LEN_;
  Shape shape_;
  Attrs attrs_;
};

// 15.3.2.2
struct DummyDataObject {
  ENUM_CLASS(Attr, Optional, Allocatable, Asynchronous, Contiguous, Value,
      Volatile, Pointer, Target)
  using Attrs = common::EnumSet<Attr, Attr_enumSize>;
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyDataObject)
  explicit DummyDataObject(const TypeAndShape &t) : type{t} {}
  explicit DummyDataObject(TypeAndShape &&t) : type{std::move(t)} {}
  explicit DummyDataObject(DynamicType t) : type{t} {}
  bool operator==(const DummyDataObject &) const;
  static std::optional<DummyDataObject> Characterize(const semantics::Symbol &);
  bool CanBePassedViaImplicitInterface() const;
  std::ostream &Dump(std::ostream &) const;
  TypeAndShape type;
  std::vector<Expr<SubscriptInteger>> coshape;
  common::Intent intent{common::Intent::Default};
  Attrs attrs;
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
  CopyableIndirection<Procedure> procedure;
  common::EnumSet<Attr, Attr_enumSize> attrs;
};

// 15.3.2.4
struct AlternateReturn {
  bool operator==(const AlternateReturn &) const { return true; }
  std::ostream &Dump(std::ostream &) const;
};

// 15.3.2.1
struct DummyArgument {
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(DummyArgument)
  DummyArgument(std::string &&name, DummyDataObject &&x)
    : name{std::move(name)}, u{std::move(x)} {}
  DummyArgument(std::string &&name, DummyProcedure &&x)
    : name{std::move(name)}, u{std::move(x)} {}
  explicit DummyArgument(AlternateReturn &&x) : u{std::move(x)} {}
  bool operator==(const DummyArgument &) const;
  static std::optional<DummyArgument> Characterize(
      const semantics::Symbol &, const IntrinsicProcTable &);
  static std::optional<DummyArgument> FromActual(
      std::string &&, const Expr<SomeType> &, FoldingContext &);
  bool IsOptional() const;
  void SetOptional(bool = true);
  bool CanBePassedViaImplicitInterface() const;
  std::ostream &Dump(std::ostream &) const;
  // name and pass are not characteristics and so does not participate in
  // operator== but are needed to determine if procedures are distinguishable
  std::string name;
  bool pass{false};  // is this the PASS argument of its procedure
  std::variant<DummyDataObject, DummyProcedure, AlternateReturn> u;
};

using DummyArguments = std::vector<DummyArgument>;

// 15.3.3
struct FunctionResult {
  ENUM_CLASS(Attr, Allocatable, Pointer, Contiguous)
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(FunctionResult)
  explicit FunctionResult(DynamicType);
  explicit FunctionResult(TypeAndShape &&);
  explicit FunctionResult(Procedure &&);
  ~FunctionResult();
  bool operator==(const FunctionResult &) const;
  static std::optional<FunctionResult> Characterize(
      const Symbol &, const IntrinsicProcTable &);

  bool IsAssumedLengthCharacter() const;

  const Procedure *IsProcedurePointer() const {
    if (const auto *pp{std::get_if<CopyableIndirection<Procedure>>(&u)}) {
      return &pp->value();
    } else {
      return nullptr;
    }
  }
  const TypeAndShape *GetTypeAndShape() const {
    return std::get_if<TypeAndShape>(&u);
  }
  void SetType(DynamicType t) { std::get<TypeAndShape>(u).set_type(t); }
  bool CanBeReturnedViaImplicitInterface() const;

  std::ostream &Dump(std::ostream &) const;

  common::EnumSet<Attr, Attr_enumSize> attrs;
  std::variant<TypeAndShape, CopyableIndirection<Procedure>> u;
};

// 15.3.1
struct Procedure {
  ENUM_CLASS(Attr, Pure, Elemental, BindC, ImplicitInterface, NullPointer)
  using Attrs = common::EnumSet<Attr, Attr_enumSize>;
  Procedure(FunctionResult &&, DummyArguments &&, Attrs);
  Procedure(DummyArguments &&, Attrs);  // for subroutines and NULL()
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(Procedure)
  bool operator==(const Procedure &) const;

  // Characterizes the procedure represented by a symbol, which may be an
  // "unrestricted specific intrinsic function".
  static std::optional<Procedure> Characterize(
      const semantics::Symbol &, const IntrinsicProcTable &);
  static std::optional<Procedure> Characterize(
      const ProcedureDesignator &, const IntrinsicProcTable &);
  static std::optional<Procedure> Characterize(
      const ProcedureRef &, const IntrinsicProcTable &);

  bool IsFunction() const { return functionResult.has_value(); }
  bool IsSubroutine() const { return !IsFunction(); }
  bool IsPure() const { return attrs.test(Attr::Pure); }
  bool IsElemental() const { return attrs.test(Attr::Elemental); }
  bool IsBindC() const { return attrs.test(Attr::BindC); }
  bool HasExplicitInterface() const {
    return !attrs.test(Attr::ImplicitInterface);
  }
  bool CanBeCalledViaImplicitInterface() const;
  std::ostream &Dump(std::ostream &) const;

  std::optional<FunctionResult> functionResult;
  DummyArguments dummyArguments;
  Attrs attrs;

private:
  Procedure() {}
};

}
#endif  // FORTRAN_EVALUATE_CHARACTERISTICS_H_
