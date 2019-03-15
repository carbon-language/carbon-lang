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

#ifndef FORTRAN_FIR_VALUE_H_
#define FORTRAN_FIR_VALUE_H_

#include "common.h"
#include "mixin.h"
#include "../common/idioms.h"
#include <string>

namespace Fortran::FIR {

class Statement;
class BasicBlock;
class Procedure;
class DataObject;

class Value : public SumTypeCopyMixin<Nothing, DataObject *, Statement *,
                  BasicBlock *, Procedure *> {
public:
  SUM_TYPE_COPY_MIXIN(Value)
  template<typename A> Value(A *a) : SumTypeCopyMixin{a} {}
  Value(const Nothing &n) : SumTypeCopyMixin{n} {}
  Value() : SumTypeCopyMixin{NOTHING} {}
};

inline bool IsNothing(Value value) {
  return std::holds_alternative<Nothing>(value.u);
}
}

#endif  // FORTRAN_FIR_VALUE_H_
