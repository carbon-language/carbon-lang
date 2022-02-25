//===-- runtime/allocatable.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "allocatable.h"
#include "assign.h"
#include "derived.h"
#include "stat.h"
#include "terminator.h"
#include "type-info.h"

namespace Fortran::runtime {
extern "C" {

void RTNAME(AllocatableInitIntrinsic)(Descriptor &descriptor,
    TypeCategory category, int kind, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  descriptor.Establish(TypeCode{category, kind},
      Descriptor::BytesFor(category, kind), nullptr, rank, nullptr,
      CFI_attribute_allocatable);
}

void RTNAME(AllocatableInitCharacter)(Descriptor &descriptor,
    SubscriptValue length, int kind, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  descriptor.Establish(
      kind, length, nullptr, rank, nullptr, CFI_attribute_allocatable);
}

void RTNAME(AllocatableInitDerived)(Descriptor &descriptor,
    const typeInfo::DerivedType &derivedType, int rank, int corank) {
  INTERNAL_CHECK(corank == 0);
  descriptor.Establish(
      derivedType, nullptr, rank, nullptr, CFI_attribute_allocatable);
}

int RTNAME(MoveAlloc)(Descriptor &to, const Descriptor & /*from*/,
    bool /*hasStat*/, const Descriptor * /*errMsg*/,
    const char * /*sourceFile*/, int /*sourceLine*/) {
  INTERNAL_CHECK(false); // TODO: MoveAlloc is not yet implemented
  return StatOk;
}

void RTNAME(AllocatableSetBounds)(Descriptor &descriptor, int zeroBasedDim,
    SubscriptValue lower, SubscriptValue upper) {
  INTERNAL_CHECK(zeroBasedDim >= 0 && zeroBasedDim < descriptor.rank());
  descriptor.GetDimension(zeroBasedDim).SetBounds(lower, upper);
  // The byte strides are computed when the object is allocated.
}

void RTNAME(AllocatableSetDerivedLength)(
    Descriptor &descriptor, int which, SubscriptValue x) {
  DescriptorAddendum *addendum{descriptor.Addendum()};
  INTERNAL_CHECK(addendum != nullptr);
  addendum->SetLenParameterValue(which, x);
}

void RTNAME(AllocatableApplyMold)(
    Descriptor &descriptor, const Descriptor &mold) {
  descriptor = mold;
  descriptor.set_base_addr(nullptr);
  descriptor.raw().attribute = CFI_attribute_allocatable;
}

int RTNAME(AllocatableAllocate)(Descriptor &descriptor, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!descriptor.IsAllocatable()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  }
  if (descriptor.IsAllocated()) {
    return ReturnError(terminator, StatBaseNotNull, errMsg, hasStat);
  }
  int stat{ReturnError(terminator, descriptor.Allocate(), errMsg, hasStat)};
  if (stat == StatOk) {
    if (const DescriptorAddendum * addendum{descriptor.Addendum()}) {
      if (const auto *derived{addendum->derivedType()}) {
        if (!derived->noInitializationNeeded()) {
          stat = Initialize(descriptor, *derived, terminator, hasStat, errMsg);
        }
      }
    }
  }
  return stat;
}

int RTNAME(AllocatableDeallocate)(Descriptor &descriptor, bool hasStat,
    const Descriptor *errMsg, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!descriptor.IsAllocatable()) {
    return ReturnError(terminator, StatInvalidDescriptor, errMsg, hasStat);
  }
  if (!descriptor.IsAllocated()) {
    return ReturnError(terminator, StatBaseNull, errMsg, hasStat);
  }
  return ReturnError(terminator, descriptor.Destroy(true), errMsg, hasStat);
}

void RTNAME(AllocatableDeallocateNoFinal)(
    Descriptor &descriptor, const char *sourceFile, int sourceLine) {
  Terminator terminator{sourceFile, sourceLine};
  if (!descriptor.IsAllocatable()) {
    ReturnError(terminator, StatInvalidDescriptor);
  } else if (!descriptor.IsAllocated()) {
    ReturnError(terminator, StatBaseNull);
  } else {
    ReturnError(terminator, descriptor.Destroy(false));
  }
}

// TODO: AllocatableCheckLengthParameter, AllocatableAllocateSource
}
} // namespace Fortran::runtime
