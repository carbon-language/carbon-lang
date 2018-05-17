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

Descriptor::Descriptor(const DerivedTypeSpecialization &dts, int rank) {
  raw_.base_addr = nullptr;
  raw_.elem_len = dts.SizeInBytes();
  raw_.version = CFI_VERSION;
  raw_.rank = rank;
  raw_.type = CFI_type_struct;
  raw_.attribute = ADDENDUM;
  Addendum()->set_derivedTypeSpecialization(&dts);
}

std::size_t Descriptor::SizeInBytes() const {
  const DescriptorAddendum *addendum{Addendum()};
  return sizeof *this + raw_.rank * sizeof(Dimension) +
      (addendum ? addendum->SizeOfAddendumInBytes() : 0);
}

std::int64_t TypeParameter::KindParameterValue(
    const DerivedTypeSpecialization &specialization) const {
  return specialization.KindParameterValue(which_);
}

std::int64_t TypeParameter::Value(const Descriptor &descriptor) const {
  const DescriptorAddendum &addendum{*descriptor.Addendum()};
  if (isLenTypeParameter_) {
    return addendum.LenParameterValue(which_);
  } else {
    return KindParameterValue(*addendum.derivedTypeSpecialization());
  }
}
}  // namespace Fortran::runtime
