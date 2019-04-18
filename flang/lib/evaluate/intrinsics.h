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

#ifndef FORTRAN_EVALUATE_INTRINSICS_H_
#define FORTRAN_EVALUATE_INTRINSICS_H_

#include "call.h"
#include "characteristics.h"
#include "type.h"
#include "../common/default-kinds.h"
#include "../parser/char-block.h"
#include "../parser/message.h"
#include <optional>
#include <ostream>

namespace Fortran::evaluate {

class FoldingContext;

struct CallCharacteristics {
  parser::CharBlock name;
  bool isSubroutineCall{false};
};

struct SpecificCall {
  SpecificCall(SpecificIntrinsic &&si, ActualArguments &&as)
    : specificIntrinsic{std::move(si)}, arguments{std::move(as)} {}
  SpecificIntrinsic specificIntrinsic;
  ActualArguments arguments;
};

struct UnrestrictedSpecificIntrinsicFunctionInterface
  : public characteristics::Procedure {
  std::string genericName;
  // N.B. If there are multiple arguments, they all have the same type.
  // All argument and result types are intrinsic types with default kinds.
};

class IntrinsicProcTable {
private:
  class Implementation;

public:
  ~IntrinsicProcTable();
  static IntrinsicProcTable Configure(
      const common::IntrinsicTypeDefaultKinds &);

  // Check whether a name should be allowed to appear on an INTRINSIC
  // statement.
  bool IsIntrinsic(const std::string &) const;

  // Probe the intrinsics for a match against a specific call.
  // On success, the actual arguments are transferred to the result
  // in dummy argument order.
  std::optional<SpecificCall> Probe(
      const CallCharacteristics &, ActualArguments &, FoldingContext &) const;

  // Probe the intrinsics with the name of a potential unrestricted specific
  // intrinsic.
  std::optional<UnrestrictedSpecificIntrinsicFunctionInterface>
  IsUnrestrictedSpecificIntrinsicFunction(const std::string &) const;

  std::ostream &Dump(std::ostream &) const;

private:
  Implementation *impl_{nullptr};  // owning pointer
};
}
#endif  // FORTRAN_EVALUATE_INTRINSICS_H_
