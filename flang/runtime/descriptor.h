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

#ifndef FORTRAN_RUNTIME_DESCRIPTOR_H_
#define FORTRAN_RUNTIME_DESCRIPTOR_H_

// Defines data structures used during execution of a Fortran program
// to implement nontrivial dummy arguments, pointers, allocatables,
// function results, and the special behaviors of instances of derived types.
// This header file includes and extends the published language
// interoperability header that is required by the Fortran 2018 standard
// as a subset of definitions suitable for exposure to user C/C++ code.
// User C code is welcome to depend on that ISO_Fortran_binding.h file,
// but should never reference this internal header.

#include "derived-type.h"
#include "type-code.h"
#include "../include/flang/ISO_Fortran_binding.h"
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstring>

namespace Fortran::runtime {

using SubscriptValue = ISO::CFI_index_t;

static constexpr int maxRank{CFI_MAX_RANK};

// A C++ view of the sole interoperable standard descriptor (ISO::CFI_cdesc_t)
// and its type and per-dimension information.

class Dimension {
public:
  SubscriptValue LowerBound() const { return raw_.lower_bound; }
  SubscriptValue Extent() const { return raw_.extent; }
  SubscriptValue UpperBound() const { return LowerBound() + Extent() - 1; }
  SubscriptValue ByteStride() const { return raw_.sm; }

private:
  ISO::CFI_dim_t raw_;
};

// The storage for this object follows the last used dim[] entry in a
// Descriptor (CFI_cdesc_t) generic descriptor.  Space matters here, since
// descriptors serve as POINTER and ALLOCATABLE components of derived type
// instances.  The presence of this structure is implied by the flag
// (CFI_cdesc_t.extra & ADDENDUM) != 0, and the number of elements in
// the len_[] array is determined by DerivedType::lenParameters().
class DescriptorAddendum {
public:
  enum Flags {
    StaticDescriptor = 0x001,
    ImplicitAllocatable = 0x002,  // compiler-created allocatable
    Created = 0x004,  // allocated by Descriptor::Create()
    DoNotFinalize = 0x008,  // compiler temporary
    Target = 0x010,  // TARGET attribute
    AllContiguous = 0x020,  // all array elements are contiguous
    LeadingDimensionContiguous = 0x040,  // only leading dimension contiguous
  };

  explicit DescriptorAddendum(const DerivedType &dt, std::uint64_t flags = 0)
    : derivedType_{&dt}, flags_{flags} {}

  const DerivedType *derivedType() const { return derivedType_; }
  DescriptorAddendum &set_derivedType(const DerivedType &dt) {
    derivedType_ = &dt;
    return *this;
  }
  std::uint64_t &flags() { return flags_; }
  const std::uint64_t &flags() const { return flags_; }

  TypeParameterValue LenParameterValue(int which) const { return len_[which]; }
  static constexpr std::size_t SizeInBytes(int lenParameters) {
    return sizeof(DescriptorAddendum) - sizeof(TypeParameterValue) +
        lenParameters * sizeof(TypeParameterValue);
  }
  std::size_t SizeInBytes() const;

  void SetLenParameterValue(int which, TypeParameterValue x) {
    len_[which] = x;
  }

private:
  const DerivedType *derivedType_{nullptr};
  std::uint64_t flags_{0};
  TypeParameterValue len_[1];  // must be the last component
  // The LEN type parameter values can also include captured values of
  // specification expressions that were used for bounds and for LEN type
  // parameters of components.  The values have been truncated to the LEN
  // type parameter's type, if shorter than 64 bits, then sign-extended.
};

// A C++ view of a standard descriptor object.
class Descriptor {
public:
  // Be advised: this class type is not suitable for use when allocating
  // a descriptor -- it is a dynamic view of the common descriptor format.
  // If used in a simple declaration of a local variable or dynamic allocation,
  // the size is going to be wrong, since the true size of a descriptor
  // depends on the number of its dimensions and the presence of an addendum.
  // Use the class template StaticDescriptor (below) to declare a descriptor
  // whose type and rank are fixed and known at compilation time.  Use the
  // Create() static member functions otherwise to dynamically allocate a
  // descriptor.
  Descriptor() = delete;

  ~Descriptor();

  int Establish(TypeCode t, std::size_t elementBytes, void *p = nullptr,
      int rank = maxRank, const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false);
  int Establish(TypeCategory, int kind, void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false);
  int Establish(const DerivedType &dt, void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);

  static Descriptor *Create(TypeCode t, std::size_t elementBytes,
      void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);
  static Descriptor *Create(TypeCategory, int kind, void *p = nullptr,
      int rank = maxRank, const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);
  static Descriptor *Create(const DerivedType &dt, void *p = nullptr,
      int rank = maxRank, const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);

  // Descriptor instances allocated via Create() above must be deallocated
  // by calling Destroy() so that operator delete[] is invoked.
  void Destroy();

  ISO::CFI_cdesc_t &raw() { return raw_; }
  const ISO::CFI_cdesc_t &raw() const { return raw_; }
  std::size_t ElementBytes() const { return raw_.elem_len; }
  int rank() const { return raw_.rank; }
  TypeCode type() const { return TypeCode{raw_.type}; }

  Descriptor &set_base_addr(void *p) {
    raw_.base_addr = p;
    return *this;
  }

  bool IsPointer() const { return raw_.attribute == CFI_attribute_pointer; }
  bool IsAllocatable() const {
    return raw_.attribute == CFI_attribute_allocatable;
  }

  Dimension &GetDimension(int dim) {
    return *reinterpret_cast<Dimension *>(&raw_.dim[dim]);
  }
  const Dimension &GetDimension(int dim) const {
    return *reinterpret_cast<const Dimension *>(&raw_.dim[dim]);
  }

  std::size_t SubscriptByteOffset(
      int dim, SubscriptValue subscriptValue) const {
    const Dimension &dimension{GetDimension(dim)};
    return (subscriptValue - dimension.LowerBound()) * dimension.ByteStride();
  }

  DescriptorAddendum *Addendum() {
    if (raw_.f18Addendum != 0) {
      return reinterpret_cast<DescriptorAddendum *>(&GetDimension(rank()));
    } else {
      return nullptr;
    }
  }
  const DescriptorAddendum *Addendum() const {
    if (raw_.f18Addendum != 0) {
      return reinterpret_cast<const DescriptorAddendum *>(
          &GetDimension(rank()));
    } else {
      return nullptr;
    }
  }

  static constexpr std::size_t SizeInBytes(
      int rank, bool addendum = false, int lengthTypeParameters = 0) {
    std::size_t bytes{sizeof(Descriptor) - sizeof(Dimension)};
    bytes += rank * sizeof(Dimension);
    if (addendum || lengthTypeParameters > 0) {
      bytes += DescriptorAddendum::SizeInBytes(lengthTypeParameters);
    }
    return bytes;
  }
  std::size_t SizeInBytes() const;

  void Check() const;

  // TODO: creation of array sections

  template<typename A> A &Element(std::size_t offset = 0) const {
    auto p = reinterpret_cast<char *>(raw_.base_addr);
    return *reinterpret_cast<A *>(p + offset);
  }

private:
  ISO::CFI_cdesc_t raw_;
};
static_assert(sizeof(Descriptor) == sizeof(ISO::CFI_cdesc_t));

// Properly configured instances of StaticDescriptor will occupy the
// exact amount of storage required for the descriptor based on its
// number of dimensions and whether it requires an addendum.  To build
// such a static descriptor, declare an instance of StaticDescriptor<>,
// extract a reference to the Descriptor via the descriptor() accessor,
// and then built a Descriptor therein via descriptor.Establish().
// e.g.:
//   StaticDescriptor<R,A,LP> statDesc;
//   Descriptor &descriptor{statDesc.descriptor()};
//   descriptor.Establish( ... );
template<int MAX_RANK = maxRank, bool ADDENDUM = false, int MAX_LEN_PARMS = 0>
class alignas(Descriptor) StaticDescriptor {
public:
  static constexpr int maxRank{MAX_RANK};
  static constexpr int maxLengthTypeParameters{MAX_LEN_PARMS};
  static constexpr bool hasAddendum{ADDENDUM || MAX_LEN_PARMS > 0};
  static constexpr std::size_t byteSize{
      Descriptor::SizeInBytes(maxRank, hasAddendum, maxLengthTypeParameters)};

  Descriptor &descriptor() { return *reinterpret_cast<Descriptor *>(storage_); }
  const Descriptor &descriptor() const {
    return *reinterpret_cast<const Descriptor *>(storage_);
  }

  void Check() {
    assert(descriptor().SizeInBytes() <= byteSize);
    assert(descriptor().rank() <= maxRank);
    if (DescriptorAddendum * addendum{descriptor().Addendum()}) {
      if (const DerivedType * dt{addendum->derivedType()}) {
        assert(dt->lenParameters() <= maxLengthTypeParameters);
      } else {
        assert(maxLengthTypeParameters == 0);
      }
    } else {
      assert(!hasAddendum);
      assert(maxLengthTypeParameters == 0);
    }
  }

private:
  char storage_[byteSize];
};
}  // namespace Fortran::runtime
#endif  // FORTRAN_RUNTIME_DESCRIPTOR_H_
