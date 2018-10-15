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

#ifndef FORTRAN_DEFAULT_KINDS_H_
#define FORTRAN_DEFAULT_KINDS_H_

#include "../common/fortran.h"

// Represent the default values of the kind parameters of the
// various intrinsic types.  These can be configured by means of
// the compiler command line.
namespace Fortran::semantics {

using Fortran::common::TypeCategory;

class IntrinsicTypeDefaultKinds {
public:
  IntrinsicTypeDefaultKinds();
  int subscriptIntegerKind() const { return subscriptIntegerKind_; }
  int doublePrecisionKind() const { return doublePrecisionKind_; }
  int quadPrecisionKind() const { return quadPrecisionKind_; }
  int GetDefaultKind(TypeCategory) const;

private:
  int defaultIntegerKind_{4};
  int subscriptIntegerKind_{8};
  int defaultRealKind_{defaultIntegerKind_};
  int doublePrecisionKind_{2 * defaultRealKind_};
  int quadPrecisionKind_{2 * doublePrecisionKind_};
  int defaultCharacterKind_{1};
  int defaultLogicalKind_{defaultIntegerKind_};
};

}  // namespace Fortran::semantics
#endif  // FORTRAN_DEFAULT_KINDS_H_
