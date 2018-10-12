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
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

namespace Fortran::evaluate {

// Intrinsics are distinguished by their names and the characteristics
// of their arguments, at least for now.
using IntrinsicProcedure = const char *;  // not an owning pointer

class Argument;

struct CallCharacteristics {
  parser::CharBlock name;
  const Arguments &argument;
  bool isSubroutineCall{false};
};

struct SpecificIntrinsic {
  explicit SpecificIntrinsic(IntrinsicProcedure n) : name{n} {}
  SpecificIntrinsic(IntrinsicProcedure n, bool isElem, DynamicType dt, int r)
    : name{n}, isElemental{isElem}, type{dt}, rank{r} {}
  std::ostream &Dump(std::ostream &) const;

  IntrinsicProcedure name;
  bool isElemental{false};  // TODO: consider using Attrs instead
  bool isPointer{false};  // NULL()
  bool isRestrictedSpecific{false};  // can only call it if true
  DynamicType type;
  int rank{0};
};

class IntrinsicProcTable {
private:
  struct Implementation;

public:
  ~IntrinsicProcTable();
  static IntrinsicProcTable Configure(const IntrinsicTypeDefaultKinds &);
  std::optional<SpecificIntrinsic> Probe(const CallCharacteristics &,
      parser::ContextualMessages *messages = nullptr) const;
  std::ostream &Dump(std::ostream &) const;

private:
  Implementation *impl_{nullptr};  // owning pointer
};
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_INTRINSICS_H_
