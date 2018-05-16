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

// TODO: Not complete; exists to check compilability of descriptor.h

#include "descriptor.h"

namespace Fortran::runtime {

Descriptor::Descriptor(const DerivedType &t, int rank = 0) {
  raw_.base_addr = nullptr;
  raw_.elem_len = t.SizeInBytes();
  raw_.version = CFI_VERSION;
  raw_.rank = rank;
  raw_.type = CFI_type_struct;
  raw_.attribute = ADDENDUM;
  new(GetAddendum()) DescriptorAddendum{t};
}

std::size_t Descriptor::SizeInBytes() const {
  const DescriptorAddendum *addendum{GetAddendum()};
  return sizeof *this + raw_.rank * sizeof(Dimension) +
         (addendum ? addendum->AddendumSizeInBytes() : 0);
}

std::int64_t DerivedTypeParameter::Value(const DescriptorAddendum *addendum) const {
  if (isLenTypeParameter_) {
    return addendum->GetLenParameterValue(value_);
  } else {
    return value_;
  }
}

std::int64_t DerivedTypeParameter::Value(const Descriptor *descriptor) const {
  if (isLenTypeParameter_) {
    return descriptor->GetAddendum()->GetLenTypeParameterValue(value_);
  } else {
    return value_;
  }
}
}  // namespace Fortran::runtime
