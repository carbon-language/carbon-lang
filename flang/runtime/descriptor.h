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

#include "../include/flang/ISO_Fortran_binding.h"
#include <cinttypes>
#include <cstddef>

namespace Fortran::runtime {

class DerivedTypeSpecialization;
class DescriptorAddendum;

// A C++ view of the sole interoperable standard descriptor (ISO_cdesc_t)
// and its type and per-dimension information.

class TypeCode {
public:
  enum class Form { Integer, Real, Complex, Logical, Character, Derived };

  TypeCode() {}
  explicit TypeCode(ISO::CFI_type_t t) : raw_{t} {}
  int raw() const { return raw_; }

  constexpr bool IsValid() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_type_struct;
  }
  constexpr bool IsInteger() const {
    return raw_ >= CFI_type_signed_char && raw_ <= CFI_type_ptrdiff_t;
  }
  constexpr bool IsReal() const {
    return raw_ >= CFI_type_float && raw_ <= CFI_type_long_double;
  }
  constexpr bool IsComplex() const {
    return raw_ >= CFI_type_float_Complex &&
        raw_ <= CFI_type_long_double_Complex;
  }
  constexpr bool IsLogical() const { return raw_ == CFI_type_Bool; }
  constexpr bool IsCharacter() const { return raw_ == CFI_type_cptr; }
  constexpr bool IsDerived() const { return raw_ == CFI_type_struct; }

  constexpr bool IsIntrinsic() const {
    return IsValid() && !IsDerived();
  }

  constexpr Form GetForm() const {
    if (IsInteger()) {
      return Form::Integer;
    }
    if (IsReal()) {
      return Form::Real;
    }
    if (IsComplex()) {
      return Form::Complex;
    }
    if (IsLogical()) {
      return Form::Logical;
    }
    if (IsCharacter()) {
      return Form::Character;
    }
    return Form::Derived;
  }

private:
  ISO::CFI_type_t raw_{CFI_type_other};
};

class Dimension {
public:
  std::int64_t LowerBound() const { return raw_.lower_bound; }
  std::int64_t Extent() const { return raw_.extent; }
  std::int64_t UpperBound() const { return LowerBound() + Extent() - 1; }
  std::int64_t ByteStride() const { return raw_.sm; }

private:
  ISO::CFI_dim_t raw_;
};
static_assert(sizeof(Dimension) == sizeof(ISO::CFI_dim_t));

class Descriptor {
public:
  Descriptor(TypeCode t, std::size_t elementBytes, int rank = 0) {
    raw_.base_addr = nullptr;
    raw_.elem_len = elementBytes;
    raw_.version = CFI_VERSION;
    raw_.rank = rank;
    raw_.type = t.raw();
    raw_.attribute = 0;
  }
  Descriptor(const DerivedTypeSpecialization &, int rank = 0);

  void Check() const;

  template<typename A> A &Element(std::size_t offset = 0) const {
    auto p = reinterpret_cast<char *>(raw_.base_addr);
    return *reinterpret_cast<A *>(p + offset);
  }

  std::size_t ElementBytes() const { return raw_.elem_len; }
  int rank() const { return raw_.rank; }
  TypeCode type() const { return TypeCode{raw_.type}; }

  bool IsPointer() const {
    return (raw_.attribute & CFI_attribute_pointer) != 0;
  }
  bool IsAllocatable() const {
    return (raw_.attribute & CFI_attribute_allocatable) != 0;
  }
  bool IsImplicitlyAllocated() const {
    return (raw_.attribute & IMPLICITLY_ALLOCATED) != 0;
  }
  bool IsDescriptorStatic() const {
    return (raw_.attribute & STATIC_DESCRIPTOR) != 0;
  }
  bool IsTarget() const {
    return (raw_.attribute & (CFI_attribute_pointer | TARGET)) != 0;
  }
  bool IsContiguous() const { return (raw_.attribute & CONTIGUOUS) != 0; }
  bool IsColumnContiguous() const {
    return (raw_.attribute & COLUMN_CONTIGUOUS) != 0;
  }
  bool IsTemporary() const { return (raw_.attribute & TEMPORARY) != 0; }

  Dimension &GetDimension(int dim) {
    return *reinterpret_cast<Dimension *>(&raw_.dim[dim]);
  }
  const Dimension &GetDimension(int dim) const {
    return *reinterpret_cast<const Dimension *>(&raw_.dim[dim]);
  }

  DescriptorAddendum *Addendum() {
    if ((raw_.attribute & ADDENDUM) != 0) {
      return reinterpret_cast<DescriptorAddendum *>(&GetDimension(rank()));
    } else {
      return nullptr;
    }
  }
  const DescriptorAddendum *Addendum() const {
    if ((raw_.attribute & ADDENDUM) != 0) {
      return reinterpret_cast<const DescriptorAddendum *>(
          &GetDimension(rank()));
    } else {
      return nullptr;
    }
  }

  std::size_t SizeInBytes() const;

private:
  // These values must coexist with the ISO_Fortran_binding.h definitions
  // for CFI_attribute_... values and fit in the "attribute" field of
  // CFI_cdesc_t.
  enum AdditionalAttributes {
    // non-pointer nonallocatable derived type component implemented as
    // an implicit allocatable due to dependence on LEN type parameters
    IMPLICITLY_ALLOCATED = 0x100,  // bounds depend on LEN type parameter
    ADDENDUM = 0x200,  // last dim[] entry is followed by DescriptorAddendum
    STATIC_DESCRIPTOR = 0x400,  // base_addr is null, get base address elsewhere
    TARGET = 0x800,  // TARGET attribute; also implied by CFI_attribute_pointer
    CONTIGUOUS = 0x1000,
    COLUMN_CONTIGUOUS = 0x2000,  // first dimension is contiguous
    TEMPORARY = 0x4000,  // compiler temp, do not finalize
  };

  ISO::CFI_cdesc_t raw_;
};
static_assert(sizeof(Descriptor) == sizeof(ISO::CFI_cdesc_t));

// Static type information is suitable for loading in a read-only section.
// Information about intrinsic types is inferable from raw CFI_type_t
// type codes (packaged as TypeCode above).
// Information about derived types and their KIND parameter specializations
// appears in the compiled program units that define or specialize the types.

class TypeParameter {
public:
  const char *name() const { return name_; }
  const TypeCode typeCode() const { return typeCode_; }
  bool isLenTypeParameter() const { return isLenTypeParameter_; }
  std::size_t which() const { return which_; }
  std::int64_t defaultValue() const { return defaultValue_; }

  std::int64_t KindParameterValue(const DerivedTypeSpecialization &) const;
  std::int64_t Value(const Descriptor &) const;

private:
  const char *name_;
  TypeCode typeCode_;  // INTEGER, but not necessarily default kind
  bool isLenTypeParameter_;  // whether value is in dynamic descriptor
  std::size_t which_;  // index of this parameter in kind/len array
  std::int64_t defaultValue_;
};

// Components that have any need for a descriptor will either reference
// a static descriptor that applies to all instances, or will *be* a
// descriptor.  Be advised: the base addresses in static descriptors
// are null.  Most runtime interfaces separate the data address from that
// of the descriptor, and ignore the encapsulated base address in the
// descriptor.  Some interfaces, e.g. calls to interoperable procedures,
// cannot pass a separate data address, and any static descriptor being used
// in that kind of situation must be copied and customized.
// Static descriptors are flagged in their attributes.
class Component {
public:
  const char *name() const { return name_; }
  TypeCode typeCode() const { return typeCode_; }
  const Descriptor *staticDescriptor() const { return staticDescriptor_; }
  bool IsParent() const { return (flags_ & PARENT) != 0; }
  bool IsPrivate() const { return (flags_ & PRIVATE) != 0; }
  bool IsDescriptor() const { return (flags_ & IS_DESCRIPTOR) != 0; }

private:
  enum Flag { PARENT = 1, PRIVATE = 2, IS_DESCRIPTOR = 4 };
  const char *name_{nullptr};
  std::uint32_t flags_{0};
  TypeCode typeCode_{CFI_type_other};
  const Descriptor *staticDescriptor_{nullptr};
};

struct ExecutableCode {
  ExecutableCode() {}
  ExecutableCode(const ExecutableCode &) = default;
  ExecutableCode &operator=(const ExecutableCode &) = default;
  std::intptr_t host{0};
  std::intptr_t device{0};
};

struct TypeBoundProcedure {
  const char *name;
  ExecutableCode code;
};

struct DefinedAssignment {
  int destinationRank, sourceRank;
  bool isElemental;
  ExecutableCode code;
};

// This static description of a derived type is not specialized by
// the values of kind type parameters.  All specializations share
// this information.
// Extended derived types have the EXTENDS flag set and place their base
// component first in the component descriptions, which is significant for
// the execution of FINAL subroutines.
class DerivedType {
public:
  DerivedType(const char *n, std::size_t kps, std::size_t lps,
      const TypeParameter *tp, std::size_t cs, const Component *ca,
      std::size_t tbps, const TypeBoundProcedure *tbp, std::size_t das,
      const DefinedAssignment *da)
    : name_{n}, kindParameters_{kps}, lenParameters_{lps}, components_{cs},
      typeParameter_{tp}, typeBoundProcedure_{tbp}, definedAssignments_{das},
      definedAssignment_{da} {}

  const char *name() const { return name_; }
  std::size_t kindParameters() const { return kindParameters_; }
  std::size_t lenParameters() const { return lenParameters_; }
  const TypeParameter &typeParameter(int n) const { return typeParameter_[n]; }
  std::size_t components() const { return components_; }
  std::size_t typeBoundProcedures() const { return typeBoundProcedures_; }
  const TypeBoundProcedure &typeBoundProcedure(int n) const {
    return typeBoundProcedure_[n];
  }

  DerivedType &set_sequence() {
    flags_ |= SEQUENCE;
    return *this;
  }
  DerivedType &set_bind_c() {
    flags_ |= BIND_C;
    return *this;
  }
  DerivedType &set_finalSubroutine(const ExecutableCode &c) {
    finalSubroutine_ = c;
    return *this;
  }

  bool Extends() const { return components_ > 0 && component_[0].IsParent(); }
  bool AnyPrivate() const;
  bool IsSequence() const { return (flags_ & SEQUENCE) != 0; }
  bool IsBindC() const { return (flags_ & BIND_C) != 0; }

  // TODO: assignment
  // TODO: finalization

private:
  enum Flag { SEQUENCE = 1, BIND_C = 2 };

  const char *name_;  // NUL-terminated constant text
  std::uint64_t flags_{0};  // needed for IsSameType() correct semantics
  std::size_t kindParameters_;
  std::size_t lenParameters_;
  std::size_t components_;  // *not* including type parameters
  std::size_t typeBoundProcedures_;
  const TypeParameter *typeParameter_;  // array
  const Component *component_;  // array
  const TypeBoundProcedure
      *typeBoundProcedure_;  // array of overridable TBP bindings
  ExecutableCode finalSubroutine_;  // can be null
  std::size_t definedAssignments_;
  const DefinedAssignment *definedAssignment_;  // array
};

class ComponentSpecialization {
public:
  template<typename A> A *Locate(char *instance) const {
    return reinterpret_cast<A *>(instance + offset_);
  }
  template<typename A> const A *Locate(const char *instance) const {
    return reinterpret_cast<const A *>(instance + offset_);
  }
  const Descriptor *GetDescriptor(
      const Component &c, const char *instance) const {
    if (const Descriptor * staticDescriptor{c.staticDescriptor()}) {
      return staticDescriptor;
    } else if (c.IsDescriptor()) {
      return Locate<const Descriptor>(instance);
    } else {
      return nullptr;
    }
  }

private:
  std::size_t offset_{0};  // relative to start of derived type instance
};

// This static representation of a derived type specialization includes
// the values of all its KIND type parameters, and reflects those values
// in the values of array bounds and static derived type descriptors that
// appear in the static descriptors of the components.
class DerivedTypeSpecialization {
public:
  DerivedTypeSpecialization(const DerivedType &dt, std::size_t n,
      const char *init, const std::int64_t *kp,
      const ComponentSpecialization *cs)
    : derivedType_{dt}, bytes_{n}, initializer_{init}, kindParameterValue_{kp},
      componentSpecialization_{cs} {}
  const DerivedType &derivedType() const { return derivedType_; }

  std::size_t SizeInBytes() const { return bytes_; }
  std::int64_t KindParameterValue(int n) const {
    return kindParameterValue_[n];
  }
  const ComponentSpecialization &GetComponent(int n) const {
    return componentSpecialization_[n];
  }
  bool IsSameType(const DerivedTypeSpecialization &) const;

  // TODO: initialization
  // TODO: sourced allocation initialization

private:
  const DerivedType &derivedType_;
  std::size_t bytes_;  // allocation size of one scalar instance, w/ alignment
  const char *initializer_;  // can be null; includes base components
  const std::int64_t *kindParameterValue_;  // array
  const ComponentSpecialization *componentSpecialization_;  // array
};

// The storage for this object follows the last used dim[] entry in a
// Descriptor (CFI_cdesc_t) generic descriptor; that is why this class
// cannot be defined as a derivation or encapsulation of the standard
// argument descriptor.  Space matters here, since dynamic descriptors
// can serve as components of derived type instances.  The presence of
// this structure is implied by (CFI_cdesc_t.attribute & ADDENDUM) != 0,
// and the number of elements in the len_[] array is determined by
// DerivedType::lenParameters().
class DescriptorAddendum {
public:
  explicit DescriptorAddendum(const DerivedTypeSpecialization *dts)
    : derivedTypeSpecialization_{dts} {}

  DescriptorAddendum &set_derivedTypeSpecialization(
      const DerivedTypeSpecialization *dts) {
    derivedTypeSpecialization_ = dts;
    return *this;
  }

  const DerivedTypeSpecialization *derivedTypeSpecialization() const {
    return derivedTypeSpecialization_;
  }

  std::int64_t LenParameterValue(std::size_t n) const { return len_[n]; }
  std::size_t SizeOfAddendumInBytes() const {
    return sizeof *this - sizeof len_[0] +
        derivedTypeSpecialization_->derivedType().lenParameters() *
        sizeof len_[0];
  }

  void SetLenParameterValue(std::size_t which, std::int64_t x) {
    len_[which] = x;
  }

private:
  const DerivedTypeSpecialization *derivedTypeSpecialization_{nullptr};
  std::int64_t len_[1];  // must be the last component
  // The LEN type parameter values can also include captured values of
  // specification expressions that were used for bounds and for LEN type
  // parameters of components.  The values have been truncated to the LEN
  // type parameter's type, if shorter than 64 bits, then sign-extended.
};

// Procedure pointers have static links for host association.
// TODO: define the target data structure of that static link
struct ProcedurePointer {
  ExecutableCode entryAddresses;
  void *staticLink;
};

// TODO: coarray hooks

}  // namespace Fortran::runtime
#endif  // FORTRAN_RUNTIME_DESCRIPTOR_H_
