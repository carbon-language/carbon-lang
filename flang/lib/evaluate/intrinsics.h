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

#ifndef FORTRAN_EVALUATE_INTRINSICS_H_
#define FORTRAN_EVALUATE_INTRINSICS_H_

#include "call.h"
#include "type.h"
#include "../common/idioms.h"
#include "../parser/message.h"
#include "../semantics/attr.h"
#include "../semantics/default-kinds.h"
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

namespace Fortran::evaluate {

// Intrinsics are distinguished by their names and the characteristics
// of their arguments, at least for now.
using IntrinsicProcedure = const char *;  // not an owning pointer

struct CallCharacteristics {
  parser::CharBlock name;
  bool isSubroutineCall{false};
};

struct SpecificIntrinsic {
  explicit SpecificIntrinsic(IntrinsicProcedure n) : name{n} {}
  SpecificIntrinsic(IntrinsicProcedure n, std::optional<DynamicType> &&dt,
      int r, semantics::Attrs a)
    : name{n}, type{std::move(dt)}, rank{r}, attrs{a} {}
  std::ostream &Dump(std::ostream &) const;

  IntrinsicProcedure name;
  bool isRestrictedSpecific{false};  // if true, can only call it
  std::optional<DynamicType> type;  // absent if and only if subroutine call
  int rank{0};
  semantics::Attrs attrs;  // ELEMENTAL, POINTER
};

struct SpecificCall {
  SpecificCall(SpecificIntrinsic &&si, ActualArguments &&as)
    : specificIntrinsic{std::move(si)}, arguments{std::move(as)} {}
  SpecificIntrinsic specificIntrinsic;
  ActualArguments arguments;
};

class IntrinsicProcTable {
private:
  struct Implementation;

public:
  ~IntrinsicProcTable();
  static IntrinsicProcTable Configure(
      const semantics::IntrinsicTypeDefaultKinds &);

  // Probe the intrinsics for a match against a specific call.
  // On success, the actual arguments are transferred to the result
  // in dummy argument order.
  std::optional<SpecificCall> Probe(const CallCharacteristics &,
      ActualArguments &, parser::ContextualMessages *messages = nullptr) const;
  std::ostream &Dump(std::ostream &) const;

private:
  Implementation *impl_{nullptr};  // owning pointer
};
}
#endif  // FORTRAN_EVALUATE_INTRINSICS_H_
