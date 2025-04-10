// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_TYPE_H_
#define CARBON_TOOLCHAIN_CHECK_TYPE_H_

#include "llvm/ADT/ArrayRef.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Enforces that an integer type has a valid bit width.
auto ValidateIntType(Context& context, SemIRLoc loc, SemIR::IntType result)
    -> bool;

// Enforces that the bit width is 64 for a float.
auto ValidateFloatBitWidth(Context& context, SemIRLoc loc,
                           SemIR::InstId inst_id) -> bool;

// Enforces that a float type has a valid bit width.
auto ValidateFloatType(Context& context, SemIRLoc loc, SemIR::FloatType result)
    -> bool;

// Gets the type to use for an unbound associated entity declared in this
// interface. For example, this is the type of `I.T` after
// `interface I { let T:! type; }`. The name of the interface is used for
// diagnostics.
// TODO: Should we use a different type for each such entity, or the same type
// for all associated entities?
auto GetAssociatedEntityType(Context& context, SemIR::InterfaceId interface_id,
                             SemIR::SpecificId interface_specific_id)
    -> SemIR::TypeId;

// Gets a singleton type. The returned type will be complete. Requires that
// `singleton_id` is already validated to be a singleton.
auto GetSingletonType(Context& context, SemIR::InstId singleton_id)
    -> SemIR::TypeId;

// Gets a class type.
auto GetClassType(Context& context, SemIR::ClassId class_id,
                  SemIR::SpecificId specific_id) -> SemIR::TypeId;

// Gets a function type. The returned type will be complete.
auto GetFunctionType(Context& context, SemIR::FunctionId fn_id,
                     SemIR::SpecificId specific_id) -> SemIR::TypeId;

// Gets the type of an associated function with the `Self` parameter bound to
// a particular value. The returned type will be complete.
auto GetFunctionTypeWithSelfType(Context& context,
                                 SemIR::InstId interface_function_type_id,
                                 SemIR::InstId self_id) -> SemIR::TypeId;

// Gets a generic class type, which is the type of a name of a generic class,
// such as the type of `Vector` given `class Vector(T:! type)`. The returned
// type will be complete.
auto GetGenericClassType(Context& context, SemIR::ClassId class_id,
                         SemIR::SpecificId enclosing_specific_id)
    -> SemIR::TypeId;

// Gets a generic interface type, which is the type of a name of a generic
// interface, such as the type of `AddWith` given
// `interface AddWith(T:! type)`. The returned type will be complete.
auto GetGenericInterfaceType(Context& context, SemIR::InterfaceId interface_id,
                             SemIR::SpecificId enclosing_specific_id)
    -> SemIR::TypeId;

// Gets the facet type corresponding to a particular interface.
auto GetInterfaceType(Context& context, SemIR::InterfaceId interface_id,
                      SemIR::SpecificId specific_id) -> SemIR::TypeId;

// Returns a pointer type whose pointee type is `pointee_type_id`.
auto GetPointerType(Context& context, SemIR::InstId pointee_type_id)
    -> SemIR::TypeId;

// Returns a struct type with the given fields.
auto GetStructType(Context& context, SemIR::StructTypeFieldsId fields_id)
    -> SemIR::TypeId;

// Returns a tuple type with the given element types.
auto GetTupleType(Context& context, llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId;

// Returns an unbound element type.
auto GetUnboundElementType(Context& context, SemIR::InstId class_type_id,
                           SemIR::InstId element_type_id) -> SemIR::TypeId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_TYPE_H_
