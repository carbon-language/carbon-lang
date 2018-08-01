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

Descriptor::~Descriptor() { assert(!(Attributes() & CREATED)); }

int Descriptor::Establish(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent) {
  return CFI_establish(
      &raw_, p, CFI_attribute_other, t.raw(), elementBytes, rank, extent);
}

int Descriptor::Establish(
    TypeCategory c, int kind, void *p, int rank, const SubscriptValue *extent) {
  std::size_t elementBytes = kind;
  if (c == TypeCategory::Complex) {
    elementBytes *= 2;
  }
  return ISO::CFI_establish(&raw_, p, CFI_attribute_other,
      TypeCode(c, kind).raw(), elementBytes, rank, extent);
}

int Descriptor::Establish(
    const DerivedType &dt, void *p, int rank, const SubscriptValue *extent) {
  int result{ISO::CFI_establish(
      &raw_, p, ADDENDUM, CFI_type_struct, dt.SizeInBytes(), rank, extent)};
  Addendum()->set_derivedType(dt);
  return result;
}

Descriptor *Descriptor::Create(TypeCode t, std::size_t elementBytes, void *p,
    int rank, const SubscriptValue *extent) {
  std::size_t bytes{SizeInBytes(rank)};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  result->Establish(t, elementBytes, p, rank, extent);
  result->Attributes() |= CREATED;
  return result;
}

Descriptor *Descriptor::Create(
    TypeCategory c, int kind, void *p, int rank, const SubscriptValue *extent) {
  std::size_t bytes{SizeInBytes(rank)};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  result->Establish(c, kind, p, rank, extent);
  result->Attributes() |= CREATED;
  return result;
}

Descriptor *Descriptor::Create(
    const DerivedType &dt, void *p, int rank, const SubscriptValue *extent) {
  std::size_t bytes{SizeInBytes(rank, dt.IsNontrivial(), dt.lenParameters())};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  result->Establish(dt, p, rank, extent);
  result->Attributes() |= CREATED;
  return result;
}

void Descriptor::Destroy() {
  if (Attributes() & CREATED) {
    delete[] reinterpret_cast<char *>(this);
  }
}

void Descriptor::SetDerivedType(const DerivedType &dt) {
  Attributes() |= ADDENDUM;
  Addendum()->set_derivedType(dt);
}

void Descriptor::SetLenParameterValue(int which, TypeParameterValue x) {
  Attributes() |= ADDENDUM;
  Addendum()->SetLenParameterValue(which, x);
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
  if (derivedType_ == nullptr) {
    return 0;
  }
  return SizeInBytes(derivedType_->lenParameters());
}
}  // namespace Fortran::runtime
