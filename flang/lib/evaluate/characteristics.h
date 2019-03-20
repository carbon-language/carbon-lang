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
#include <memory>
#include <ostream>
#include <variant>
#include <vector>

namespace Fortran::evaluate::characteristics {

// Forward declare Procedure so dummy procedures can use it indirectly
struct Procedure;

// 15.3.2.2
struct DummyDataObject {
  ENUM_CLASS(Attr, AssumedRank, Optional, Allocatable, Asynchronous, Contiguous,
      Value, Volatile, Polymorphic, Pointer, Target)
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyDataObject)
  DynamicType type;
  std::vector<std::optional<Expr<SubscriptInteger>>> shape;
  std::vector<Expr<SubscriptInteger>> coshape;
  common::Intent intent{common::Intent::Default};
  common::EnumSet<Attr, 32> attrs;
  bool operator==(const DummyDataObject &) const;
  std::ostream &Dump(std::ostream &) const;
};

// 15.3.2.3
struct DummyProcedure {
  ENUM_CLASS(Attr, Pointer, Optional)
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(DummyProcedure)
  common::CopyableIndirection<Procedure> explicitProcedure;
  common::EnumSet<Attr, 32> attrs;
  bool operator==(const DummyProcedure &) const;
  std::ostream &Dump(std::ostream &) const;
};

// 15.3.2.4
struct AlternateReturn {
  bool operator==(const AlternateReturn &) const { return true; }
  std::ostream &Dump(std::ostream &) const;
};

// 15.3.2.1
using DummyArgument =
    std::variant<DummyDataObject, DummyProcedure, AlternateReturn>;

// 15.3.3
struct FunctionResult {
  ENUM_CLASS(
      Attr, Polymorphic, Allocatable, Pointer, Contiguous, ProcedurePointer)
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(FunctionResult)
  DynamicType type;
  int rank{0};
  common::EnumSet<Attr, 32> attrs;
  bool operator==(const FunctionResult &) const;
  std::ostream &Dump(std::ostream &) const;
};

// 15.3.1
struct Procedure {
  ENUM_CLASS(Attr, Pure, Elemental, Bind_C)
  Procedure() {}
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(Procedure)
  std::optional<FunctionResult> functionResult;  // absent means subroutine
  std::vector<DummyArgument> dummyArguments;
  common::EnumSet<Attr, 32> attrs;
  bool operator==(const Procedure &) const;
  std::ostream &Dump(std::ostream &) const;
};
}
#endif  // FORTRAN_EVALUATE_CHARACTERISTICS_H_
