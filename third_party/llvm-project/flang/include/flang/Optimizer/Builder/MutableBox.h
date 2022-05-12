//===-- MutableBox.h -- MutableBox utilities  -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_MUTABLEBOX_H
#define FORTRAN_OPTIMIZER_BUILDER_MUTABLEBOX_H

#include "llvm/ADT/StringRef.h"

namespace mlir {
class Value;
class ValueRange;
class Type;
class Location;
} // namespace mlir

namespace fir {
class FirOpBuilder;
class MutableBoxValue;
class ExtendedValue;
} // namespace fir

namespace fir::factory {

/// Create a fir.box of type \p boxType that can be used to initialize an
/// allocatable variable. Initialization of such variable has to be done at the
/// beginning of the variable lifetime by storing the created box in the memory
/// for the variable box.
/// \p nonDeferredParams must provide the non deferred length parameters so that
/// they can already be placed in the unallocated box (inquiries about these
/// parameters are legal even in unallocated state).
mlir::Value createUnallocatedBox(fir::FirOpBuilder &builder, mlir::Location loc,
                                 mlir::Type boxType,
                                 mlir::ValueRange nonDeferredParams);

/// Create a MutableBoxValue for a temporary allocatable.
/// The created MutableBoxValue wraps a fir.ref<fir.box<fir.heap<type>>> and is
/// initialized to unallocated/diassociated status. An optional name can be
/// given to the created !fir.ref<fir.box>.
fir::MutableBoxValue createTempMutableBox(fir::FirOpBuilder &builder,
                                          mlir::Location loc, mlir::Type type,
                                          llvm::StringRef name = {});

/// Update a MutableBoxValue to describe entity \p source (that must be in
/// memory). If \lbounds is not empty, it is used to defined the MutableBoxValue
/// lower bounds, otherwise, the lower bounds from \p source are used.
void associateMutableBox(fir::FirOpBuilder &builder, mlir::Location loc,
                         const fir::MutableBoxValue &box,
                         const fir::ExtendedValue &source,
                         mlir::ValueRange lbounds);

/// Update a MutableBoxValue to describe entity \p source (that must be in
/// memory) with a new array layout given by \p lbounds and \p ubounds.
/// \p source must be known to be contiguous at compile time, or it must have
/// rank 1 (constraint from Fortran 2018 standard 10.2.2.3 point 9).
void associateMutableBoxWithRemap(fir::FirOpBuilder &builder,
                                  mlir::Location loc,
                                  const fir::MutableBoxValue &box,
                                  const fir::ExtendedValue &source,
                                  mlir::ValueRange lbounds,
                                  mlir::ValueRange ubounds);

/// Set the association status of a MutableBoxValue to
/// disassociated/unallocated. Nothing is done with the entity that was
/// previously associated/allocated. The function generates code that sets the
/// address field of the MutableBoxValue to zero.
void disassociateMutableBox(fir::FirOpBuilder &builder, mlir::Location loc,
                            const fir::MutableBoxValue &box);

/// Generate code to conditionally reallocate a MutableBoxValue with a new
/// shape, lower bounds, and length parameters if it is unallocated or if its
/// current shape or deferred  length parameters do not match the provided ones.
/// Lower bounds are only used if the entity needs to be allocated, otherwise,
/// the MutableBoxValue will keep its current lower bounds.
/// If the MutableBoxValue is an array, the provided shape can be empty, in
/// which case the MutableBoxValue must already be allocated at runtime and its
/// shape and lower bounds will be kept. If \p shape is empty, only a length
/// parameter mismatch can trigger a reallocation. See Fortran 10.2.1.3 point 3
/// that this function is implementing for more details. The polymorphic
/// requirements are not yet covered by this function.
void genReallocIfNeeded(fir::FirOpBuilder &builder, mlir::Location loc,
                        const fir::MutableBoxValue &box,
                        mlir::ValueRange lbounds, mlir::ValueRange shape,
                        mlir::ValueRange lengthParams);

/// Finalize a mutable box if it is allocated or associated. This includes both
/// calling the finalizer, if any, and deallocating the storage.
void genFinalization(fir::FirOpBuilder &builder, mlir::Location loc,
                     const fir::MutableBoxValue &box);

void genInlinedAllocation(fir::FirOpBuilder &builder, mlir::Location loc,
                          const fir::MutableBoxValue &box,
                          mlir::ValueRange lbounds, mlir::ValueRange extents,
                          mlir::ValueRange lenParams,
                          llvm::StringRef allocName);

void genInlinedDeallocate(fir::FirOpBuilder &builder, mlir::Location loc,
                          const fir::MutableBoxValue &box);

/// When the MutableBoxValue was passed as a fir.ref<fir.box> to a call that may
/// have modified it, update the MutableBoxValue according to the
/// fir.ref<fir.box> value.
void syncMutableBoxFromIRBox(fir::FirOpBuilder &builder, mlir::Location loc,
                             const fir::MutableBoxValue &box);

/// Read all mutable properties into a normal symbol box.
/// It is OK to call this on unassociated/unallocated boxes but any use of the
/// resulting values will be undefined (only the base address will be guaranteed
/// to be null).
fir::ExtendedValue genMutableBoxRead(fir::FirOpBuilder &builder,
                                     mlir::Location loc,
                                     const fir::MutableBoxValue &box,
                                     bool mayBePolymorphic = true);

/// Returns the fir.ref<fir.box<T>> of a MutableBoxValue filled with the current
/// association / allocation properties. If the fir.ref<fir.box> already exists
/// and is-up to date, this is a no-op, otherwise, code will be generated to
/// fill it.
mlir::Value getMutableIRBox(fir::FirOpBuilder &builder, mlir::Location loc,
                            const fir::MutableBoxValue &box);

/// Generate allocation or association status test and returns the resulting
/// i1. This is testing this for a valid/non-null base address value.
mlir::Value genIsAllocatedOrAssociatedTest(fir::FirOpBuilder &builder,
                                           mlir::Location loc,
                                           const fir::MutableBoxValue &box);

} // namespace fir::factory

#endif // FORTRAN_OPTIMIZER_BUILDER_MUTABLEBOX_H
