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

#include "descriptor.h"
#include "../lib/common/idioms.h"
#include <cassert>
#include <cstdlib>

namespace Fortran::runtime {

Descriptor::~Descriptor() {
  // Descriptors created by Descriptor::Create() must be destroyed by
  // Descriptor::Destroy(), not by the default destructor, so that
  // the array variant operator delete[] is properly used.
  assert(!(Addendum() && (Addendum()->flags() & DescriptorAddendum::Created)));
}

void Descriptor::Establish(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  CHECK(ISO::CFI_establish(&raw_, p, attribute, t.raw(), elementBytes, rank,
            extent) == CFI_SUCCESS);
  raw_.f18Addendum = addendum;
  if (addendum) {
    new (Addendum()) DescriptorAddendum{};
  }
}

void Descriptor::Establish(TypeCategory c, int kind, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  std::size_t elementBytes = kind;
  if (c == TypeCategory::Complex) {
    elementBytes *= 2;
  }
  CHECK(ISO::CFI_establish(&raw_, p, attribute, TypeCode(c, kind).raw(),
            elementBytes, rank, extent) == CFI_SUCCESS);
  raw_.f18Addendum = addendum;
  if (addendum) {
    new (Addendum()) DescriptorAddendum{};
  }
}

void Descriptor::Establish(const DerivedType &dt, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  CHECK(ISO::CFI_establish(&raw_, p, attribute, CFI_type_struct,
            dt.SizeInBytes(), rank, extent) == CFI_SUCCESS);
  raw_.f18Addendum = true;
  new (Addendum()) DescriptorAddendum{&dt};
}

Descriptor *Descriptor::Create(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true)};
  Descriptor *result{reinterpret_cast<Descriptor *>(std::malloc(bytes))};
  CHECK(result != nullptr);
  result->Establish(t, elementBytes, p, rank, extent, attribute, true);
  result->Addendum()->flags() |= DescriptorAddendum::Created;
  return result;
}

Descriptor *Descriptor::Create(TypeCategory c, int kind, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true)};
  Descriptor *result{reinterpret_cast<Descriptor *>(std::malloc(bytes))};
  CHECK(result != nullptr);
  result->Establish(c, kind, p, rank, extent, attribute, true);
  result->Addendum()->flags() |= DescriptorAddendum::Created;
  return result;
}

Descriptor *Descriptor::Create(const DerivedType &dt, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true, dt.lenParameters())};
  Descriptor *result{reinterpret_cast<Descriptor *>(std::malloc(bytes))};
  CHECK(result != nullptr);
  result->Establish(dt, p, rank, extent, attribute);
  result->Addendum()->flags() |= DescriptorAddendum::Created;
  return result;
}

void Descriptor::Destroy() {
  if (const DescriptorAddendum * addendum{Addendum()}) {
    if (addendum->flags() & DescriptorAddendum::Created) {
      std::free(reinterpret_cast<void *>(this));
    }
  }
}

std::size_t Descriptor::SizeInBytes() const {
  const DescriptorAddendum *addendum{Addendum()};
  return sizeof *this - sizeof(Dimension) + raw_.rank * sizeof(Dimension) +
      (addendum ? addendum->SizeInBytes() : 0);
}

std::size_t Descriptor::Elements() const {
  int n{rank()};
  std::size_t elements{1};
  for (int j{0}; j < n; ++j) {
    elements *= GetDimension(j).Extent();
  }
  return elements;
}

void Descriptor::Check() const {
  // TODO
}

std::size_t DescriptorAddendum::SizeInBytes() const {
  return SizeInBytes(LenParameters());
}
}  // namespace Fortran::runtime
