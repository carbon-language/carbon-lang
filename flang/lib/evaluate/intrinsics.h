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

#include "type.h"
#include "../common/idioms.h"
#include "../parser/char-block.h"
#include <memory>
#include <optional>
#include <vector>

namespace Fortran::semantics {
struct IntrinsicTypeDefaultKinds;
}

namespace Fortran::evaluate {

// Placeholder
ENUM_CLASS(IntrinsicProcedure, IAND, IEOR, IOR, LEN, MAX, MIN)

// Characterize an actual argument to an intrinsic procedure reference
struct ActualArgumentCharacteristics {
  std::optional<parser::CharBlock> keyword;
  bool isBOZ{false};
  bool isAssumedRank{false};
  DynamicType type;
  int rank;
  std::optional<int> vectorSize;
  std::optional<int> intValue;
};

struct CallCharacteristics {
  bool isSubroutineCall{false};
  parser::CharBlock name;
  std::vector<ActualArgumentCharacteristics> argument;
};

struct SpecificIntrinsic {
  //  SpecificIntrinsic(SpecificIntrinsic &&) = default;
  explicit SpecificIntrinsic(const char *n) : name{n} {}
  SpecificIntrinsic(const char *n, bool isElem, DynamicType dt, int r)
    : name{n}, isElemental{isElem}, type{dt}, rank{r} {}
  const char *name;  // not owned
  bool isElemental{false};
  DynamicType type;
  int rank{0};
};

class IntrinsicTable {
private:
  struct Implementation;

public:
  ~IntrinsicTable();
  static IntrinsicTable Configure(const semantics::IntrinsicTypeDefaultKinds &);
  std::optional<SpecificIntrinsic> Probe(const CallCharacteristics &) const;

private:
  Implementation *impl_{nullptr};  // owning pointer
};
}  // namespace Fortran::evaluate
#endif  // FORTRAN_EVALUATE_INTRINSICS_H_
