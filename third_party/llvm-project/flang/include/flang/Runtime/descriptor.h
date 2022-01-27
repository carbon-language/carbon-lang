//===-- include/flang/Runtime/descriptor.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

#include "flang/ISO_Fortran_binding.h"
#include "flang/Runtime/memory.h"
#include "flang/Runtime/type-code.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace Fortran::runtime::typeInfo {
using TypeParameterValue = std::int64_t;
class DerivedType;
} // namespace Fortran::runtime::typeInfo

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

  Dimension &SetBounds(SubscriptValue lower, SubscriptValue upper) {
    raw_.lower_bound = lower;
    raw_.extent = upper >= lower ? upper - lower + 1 : 0;
    return *this;
  }
  Dimension &SetLowerBound(SubscriptValue lower) {
    raw_.lower_bound = lower;
    return *this;
  }
  Dimension &SetUpperBound(SubscriptValue upper) {
    auto lower{raw_.lower_bound};
    raw_.extent = upper >= lower ? upper - lower + 1 : 0;
    return *this;
  }
  Dimension &SetExtent(SubscriptValue extent) {
    raw_.extent = extent;
    return *this;
  }
  Dimension &SetByteStride(SubscriptValue bytes) {
    raw_.sm = bytes;
    return *this;
  }

private:
  ISO::CFI_dim_t raw_;
};

// The storage for this object follows the last used dim[] entry in a
// Descriptor (CFI_cdesc_t) generic descriptor.  Space matters here, since
// descriptors serve as POINTER and ALLOCATABLE components of derived type
// instances.  The presence of this structure is implied by the flag
// CFI_cdesc_t.f18Addendum, and the number of elements in the len_[]
// array is determined by derivedType_->LenParameters().
class DescriptorAddendum {
public:
  explicit DescriptorAddendum(const typeInfo::DerivedType *dt = nullptr)
      : derivedType_{dt} {}
  DescriptorAddendum &operator=(const DescriptorAddendum &);

  const typeInfo::DerivedType *derivedType() const { return derivedType_; }
  DescriptorAddendum &set_derivedType(const typeInfo::DerivedType *dt) {
    derivedType_ = dt;
    return *this;
  }

  std::size_t LenParameters() const;

  typeInfo::TypeParameterValue LenParameterValue(int which) const {
    return len_[which];
  }
  static constexpr std::size_t SizeInBytes(int lenParameters) {
    // TODO: Don't waste that last word if lenParameters == 0
    return sizeof(DescriptorAddendum) +
        std::max(lenParameters - 1, 0) * sizeof(typeInfo::TypeParameterValue);
  }
  std::size_t SizeInBytes() const;

  void SetLenParameterValue(int which, typeInfo::TypeParameterValue x) {
    len_[which] = x;
  }

  void Dump(FILE * = stdout) const;

private:
  const typeInfo::DerivedType *derivedType_;
  typeInfo::TypeParameterValue len_[1]; // must be the last component
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
  // the size is going to be correct only by accident, since the true size of
  // a descriptor depends on the number of its dimensions and the presence and
  // size of an addendum, which depends on the type of the data.
  // Use the class template StaticDescriptor (below) to declare a descriptor
  // whose type and rank are fixed and known at compilation time.  Use the
  // Create() static member functions otherwise to dynamically allocate a
  // descriptor.

  Descriptor(const Descriptor &);
  Descriptor &operator=(const Descriptor &);

  static constexpr std::size_t BytesFor(TypeCategory category, int kind) {
    return category == TypeCategory::Complex ? kind * 2 : kind;
  }

  void Establish(TypeCode t, std::size_t elementBytes, void *p = nullptr,
      int rank = maxRank, const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false);
  void Establish(TypeCategory, int kind, void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false);
  void Establish(int characterKind, std::size_t characters, void *p = nullptr,
      int rank = maxRank, const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      bool addendum = false);
  void Establish(const typeInfo::DerivedType &dt, void *p = nullptr,
      int rank = maxRank, const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);

  static OwningPtr<Descriptor> Create(TypeCode t, std::size_t elementBytes,
      void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other,
      int derivedTypeLenParameters = 0);
  static OwningPtr<Descriptor> Create(TypeCategory, int kind, void *p = nullptr,
      int rank = maxRank, const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);
  static OwningPtr<Descriptor> Create(int characterKind,
      SubscriptValue characters, void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);
  static OwningPtr<Descriptor> Create(const typeInfo::DerivedType &dt,
      void *p = nullptr, int rank = maxRank,
      const SubscriptValue *extent = nullptr,
      ISO::CFI_attribute_t attribute = CFI_attribute_other);

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
  bool IsAllocated() const { return raw_.base_addr != nullptr; }

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

  std::size_t SubscriptsToByteOffset(const SubscriptValue subscript[]) const {
    std::size_t offset{0};
    for (int j{0}; j < raw_.rank; ++j) {
      offset += SubscriptByteOffset(j, subscript[j]);
    }
    return offset;
  }

  template <typename A = char> A *OffsetElement(std::size_t offset = 0) const {
    return reinterpret_cast<A *>(
        reinterpret_cast<char *>(raw_.base_addr) + offset);
  }

  template <typename A> A *Element(const SubscriptValue subscript[]) const {
    return OffsetElement<A>(SubscriptsToByteOffset(subscript));
  }

  template <typename A> A *ZeroBasedIndexedElement(std::size_t n) const {
    SubscriptValue at[maxRank];
    if (SubscriptsForZeroBasedElementNumber(at, n)) {
      return Element<A>(at);
    }
    return nullptr;
  }

  int GetLowerBounds(SubscriptValue subscript[]) const {
    for (int j{0}; j < raw_.rank; ++j) {
      subscript[j] = GetDimension(j).LowerBound();
    }
    return raw_.rank;
  }

  int GetShape(SubscriptValue subscript[]) const {
    for (int j{0}; j < raw_.rank; ++j) {
      subscript[j] = GetDimension(j).Extent();
    }
    return raw_.rank;
  }

  // When the passed subscript vector contains the last (or first)
  // subscripts of the array, these wrap the subscripts around to
  // their first (or last) values and return false.
  bool IncrementSubscripts(
      SubscriptValue subscript[], const int *permutation = nullptr) const {
    for (int j{0}; j < raw_.rank; ++j) {
      int k{permutation ? permutation[j] : j};
      const Dimension &dim{GetDimension(k)};
      if (subscript[k]++ < dim.UpperBound()) {
        return true;
      }
      subscript[k] = dim.LowerBound();
    }
    return false;
  }

  bool DecrementSubscripts(
      SubscriptValue[], const int *permutation = nullptr) const;

  // False when out of range.
  bool SubscriptsForZeroBasedElementNumber(SubscriptValue subscript[],
      std::size_t elementNumber, const int *permutation = nullptr) const {
    if (raw_.rank == 0) {
      return elementNumber == 0;
    }
    std::size_t dimCoefficient[maxRank];
    int k0{permutation ? permutation[0] : 0};
    dimCoefficient[0] = 1;
    auto coefficient{static_cast<std::size_t>(GetDimension(k0).Extent())};
    for (int j{1}; j < raw_.rank; ++j) {
      int k{permutation ? permutation[j] : j};
      const Dimension &dim{GetDimension(k)};
      dimCoefficient[j] = coefficient;
      coefficient *= dim.Extent();
    }
    if (elementNumber >= coefficient) {
      return false; // out of range
    }
    for (int j{raw_.rank - 1}; j > 0; --j) {
      int k{permutation ? permutation[j] : j};
      const Dimension &dim{GetDimension(k)};
      std::size_t quotient{elementNumber / dimCoefficient[j]};
      subscript[k] = quotient + dim.LowerBound();
      elementNumber -= quotient * dimCoefficient[j];
    }
    subscript[k0] = elementNumber + GetDimension(k0).LowerBound();
    return true;
  }

  std::size_t ZeroBasedElementNumber(
      const SubscriptValue *, const int *permutation = nullptr) const;

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

  // Returns size in bytes of the descriptor (not the data)
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

  std::size_t Elements() const;

  // Allocate() assumes Elements() and ElementBytes() work;
  // define the extents of the dimensions and the element length
  // before calling.  It (re)computes the byte strides after
  // allocation.  Does not allocate automatic components or
  // perform default component initialization.
  int Allocate();

  // Deallocates storage; does not call FINAL subroutines or
  // deallocate allocatable/automatic components.
  int Deallocate();

  // Deallocates storage, including allocatable and automatic
  // components.  Optionally invokes FINAL subroutines.
  int Destroy(bool finalize = false);

  bool IsContiguous(int leadingDimensions = maxRank) const {
    auto bytes{static_cast<SubscriptValue>(ElementBytes())};
    if (leadingDimensions > raw_.rank) {
      leadingDimensions = raw_.rank;
    }
    for (int j{0}; j < leadingDimensions; ++j) {
      const Dimension &dim{GetDimension(j)};
      if (bytes != dim.ByteStride()) {
        return false;
      }
      bytes *= dim.Extent();
    }
    return true;
  }

  // Establishes a pointer to a section or element.
  bool EstablishPointerSection(const Descriptor &source,
      const SubscriptValue *lower = nullptr,
      const SubscriptValue *upper = nullptr,
      const SubscriptValue *stride = nullptr);

  void Check() const;

  void Dump(FILE * = stdout) const;

private:
  ISO::CFI_cdesc_t raw_;
};
static_assert(sizeof(Descriptor) == sizeof(ISO::CFI_cdesc_t));

// Properly configured instances of StaticDescriptor will occupy the
// exact amount of storage required for the descriptor, its dimensional
// information, and possible addendum.  To build such a static descriptor,
// declare an instance of StaticDescriptor<>, extract a reference to its
// descriptor via the descriptor() accessor, and then built a Descriptor
// therein via descriptor.Establish(), e.g.:
//   StaticDescriptor<R,A,LP> statDesc;
//   Descriptor &descriptor{statDesc.descriptor()};
//   descriptor.Establish( ... );
template <int MAX_RANK = maxRank, bool ADDENDUM = false, int MAX_LEN_PARMS = 0>
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
    assert(descriptor().rank() <= maxRank);
    assert(descriptor().SizeInBytes() <= byteSize);
    if (DescriptorAddendum * addendum{descriptor().Addendum()}) {
      assert(hasAddendum);
      assert(addendum->LenParameters() <= maxLengthTypeParameters);
    } else {
      assert(!hasAddendum);
      assert(maxLengthTypeParameters == 0);
    }
    descriptor().Check();
  }

private:
  char storage_[byteSize]{};
};
} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_DESCRIPTOR_H_
