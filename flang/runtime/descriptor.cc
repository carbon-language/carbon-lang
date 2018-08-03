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
  if (raw_.attribute != CFI_attribute_pointer) {
    Deallocate();
  }
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

std::unique_ptr<Descriptor> Descriptor::Create(TypeCode t,
    std::size_t elementBytes, void *p, int rank, const SubscriptValue *extent,
    ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true)};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  CHECK(result != nullptr);
  result->Establish(t, elementBytes, p, rank, extent, attribute, true);
  return std::unique_ptr<Descriptor>{result};
}

std::unique_ptr<Descriptor> Descriptor::Create(TypeCategory c, int kind,
    void *p, int rank, const SubscriptValue *extent,
    ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true)};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  CHECK(result != nullptr);
  result->Establish(c, kind, p, rank, extent, attribute, true);
  return std::unique_ptr<Descriptor>{result};
}

std::unique_ptr<Descriptor> Descriptor::Create(const DerivedType &dt, void *p,
    int rank, const SubscriptValue *extent, ISO::CFI_attribute_t attribute) {
  std::size_t bytes{SizeInBytes(rank, true, dt.lenParameters())};
  Descriptor *result{reinterpret_cast<Descriptor *>(new char[bytes])};
  CHECK(result != nullptr);
  result->Establish(dt, p, rank, extent, attribute);
  return std::unique_ptr<Descriptor>{result};
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

int Descriptor::Allocate(
    const SubscriptValue lb[], const SubscriptValue ub[], std::size_t charLen) {
  int result{ISO::CFI_allocate(&raw_, lb, ub, charLen)};
  if (result == CFI_SUCCESS) {
    // TODO: derived type initialization
  }
  return result;
}

int Descriptor::Deallocate(bool finalize) {
  if (raw_.base_addr != nullptr) {
    Destroy(static_cast<char *>(raw_.base_addr), finalize);
  }
  return ISO::CFI_deallocate(&raw_);
}

void Descriptor::Destroy(char *data, bool finalize) const {
  if (data != nullptr) {
    if (const DescriptorAddendum * addendum{Addendum()}) {
      if (addendum->flags() & DescriptorAddendum::DoNotFinalize) {
        finalize = false;
      }
      if (const DerivedType * dt{addendum->derivedType()}) {
        std::size_t elements{Elements()};
        std::size_t elementBytes{ElementBytes()};
        for (std::size_t j{0}; j < elements; ++j) {
          dt->Destroy(data + j * elementBytes, finalize);
        }
      }
    }
  }
}

void Descriptor::Check() const {
  // TODO
}

std::ostream &Descriptor::Dump(std::ostream &o) const {
  o << "Descriptor @ 0x" << std::hex << reinterpret_cast<std::intptr_t>(this)
    << std::dec << ":\n";
  o << "  base_addr 0x" << std::hex
    << reinterpret_cast<std::intptr_t>(raw_.base_addr) << std::dec << '\n';
  o << "  elem_len  " << raw_.elem_len << '\n';
  o << "  version   " << raw_.version
    << (raw_.version == CFI_VERSION ? "(ok)" : "BAD!") << '\n';
  o << "  rank      " << static_cast<int>(raw_.rank) << '\n';
  o << "  type      " << static_cast<int>(raw_.type) << '\n';
  o << "  attribute " << static_cast<int>(raw_.attribute) << '\n';
  o << "  addendum? " << static_cast<bool>(raw_.f18Addendum) << '\n';
  for (int j{0}; j < raw_.rank; ++j) {
    o << "  dim[" << j << "] lower_bound " << raw_.dim[j].lower_bound << '\n';
    o << "         extent      " << raw_.dim[j].extent << '\n';
    o << "         sm          " << raw_.dim[j].sm << '\n';
  }
  if (const DescriptorAddendum * addendum{Addendum()}) {
    addendum->Dump(o);
  }
  return o;
}

std::size_t DescriptorAddendum::SizeInBytes() const {
  return SizeInBytes(LenParameters());
}

std::ostream &DescriptorAddendum::Dump(std::ostream &o) const {
  o << "  derivedType @ 0x" << std::hex
    << reinterpret_cast<std::intptr_t>(derivedType_) << std::dec << '\n';
  o << "  flags         " << flags_ << '\n';
  // TODO: LEN parameter values
  return o;
}
}  // namespace Fortran::runtime
