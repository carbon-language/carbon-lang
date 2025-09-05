// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ID_KIND_H_
#define CARBON_TOOLCHAIN_SEM_IR_ID_KIND_H_

#include "common/type_enum.h"
#include "toolchain/base/int.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// An enum of all the ID types used as instruction operands.
//
// As instruction operands, the types listed here can appear as fields of typed
// instructions (`toolchain/sem_ir/typed_insts.h`) and must implement the
// `FromRaw` and `ToRaw` protocol in `Inst`. In most cases this is done by
// inheriting from `IdBase` or `IndexBase`.
//
// clang-format off: We want one per line.
using IdKind = TypeEnum<
    // From base/value_store.h.
    FloatId,
    IntId,
    RealId,
    StringLiteralValueId,
    // From sem_ir/ids.h.
    AbsoluteInstBlockId,
    AbsoluteInstId,
    AnyRawId,
    AssociatedConstantId,
    BoolValue,
    CallParamIndex,
    CharId,
    ClassId,
    CompileTimeBindIndex,
    ConstantId,
    CppOverloadSetId,
    CustomLayoutId,
    DeclInstBlockId,
    DestInstId,
    ElementIndex,
    EntityNameId,
    ExprRegionId,
    FacetTypeId,
    FloatKind,
    FunctionId,
    GenericId,
    ImplId,
    ImportIRId,
    ImportIRInstId,
    InstBlockId,
    InstId,
    InterfaceId,
    IntKind,
    LabelId,
    LibraryNameId,
    LocId,
    MetaInstId,
    NameId,
    NameScopeId,
    SpecificId,
    SpecificInterfaceId,
    StructTypeFieldsId,
    TypeInstId,
    VtableId>;
// clang-format on

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ID_KIND_H_
