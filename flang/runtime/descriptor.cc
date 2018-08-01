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
#include <cassert>
#include <cstdlib>

namespace Fortran::runtime {

Descriptor::~Descriptor() {
  // Descriptors created by Descriptor::Create() must be destroyed by
  // Descriptor::Destroy(), not by the default destructor, so that
  // the array variant operator delete[] is properly used.
  assert(!(Addendum() && (Addendum()->flags() & DescriptorAddendum::Created)));
}

int Descriptor::Establish(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  int result{
      CFI_establish(&raw_, p, attribute, t.raw(), elementBytes, rank, extent)};
  raw_.f18Addendum = addendum;
  return result;
}

int Descriptor::Establish(TypeCategory c, int kind, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute,
    bool addendum) {
  std::size_t elementBytes = kind;
  if (c == TypeCategory::Complex) {
    elementBytes *= 2;
  }
  int result{ISO::CFI_establish(&raw_, p, attribute, TypeCode(c, kind).raw(),
      elementBytes, rank, extent)};
  raw_.f18Addendum = addendum;
  return result;
}

int Descriptor::Establish(const DerivedType &dt, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  int result{ISO::CFI_establish(
      &raw_, p, attribute, CFI_type_struct, dt.SizeInBytes(), rank, extent)};
  raw_.f18Addendum = true;
  Addendum()->set_derivedType(dt);
  return result;
}

Descriptor *Descriptor::Create(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank)};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  result->Establish(t, elementBytes, p, rank, extent, attribute, true);
  result->Addendum()->flags() |= DescriptorAddendum::Created;
  return result;
}

Descriptor *Descriptor::Create(TypeCategory c, int kind, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank)};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  result->Establish(c, kind, p, rank, extent, attribute, true);
  result->Addendum()->flags() |= DescriptorAddendum::Created;
  return result;
}

Descriptor *Descriptor::Create(const DerivedType &dt, void *p, int rank,
    const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, dt.IsNontrivial(), dt.lenParameters())};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  result->Establish(dt, p, rank, extent, attribute);
  result->Addendum()->flags() |= DescriptorAddendum::Created;
  return result;
}

void Descriptor::Destroy() {
  if (const DescriptorAddendum * addendum{Addendum()}) {
    if (addendum->flags() & DescriptorAddendum::Created) {
      delete[] reinterpret_cast<char *>(this);
    }
  }
}

std::size_t Descriptor::SizeInBytes() const {
  const DescriptorAddendum *addendum{Addendum()};
  return sizeof *this - sizeof(Dimension) + raw_.rank * sizeof(Dimension) +
      (addendum ? addendum->SizeInBytes() : 0);
}

void Descriptor::Check() const {
  // TODO
}

std::size_t DescriptorAddendum::SizeInBytes() const {
  return SizeInBytes(derivedType_->lenParameters());
}
}  // namespace Fortran::runtime
