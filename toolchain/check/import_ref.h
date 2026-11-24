// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Sets the IRs for `ImportIRId::ApiForImpl` from `import_ir`, and
// `ImportIRId::Local` and `ImportIRId::Cpp` from defaults. Must be called
// before `AddImportIR` in order to ensure the correct IDs are assigned.
auto SetSpecialImportIRs(Context& context, SemIR::ImportIR import_ir) -> void;

// Adds `import_ir` to `context.import_irs`, or treats `import_ir` as a
// partial update to an existing entry with the same `sem_ir` field, and
// returns its ID. In the update case, `is_export` will be true if it was
// true for either of the inputs; `decl_id` is not updated.
auto GetOrAddImportIR(Context& context, SemIR::ImportIR import_ir)
    -> SemIR::ImportIRId;

// Adds an import_ref instruction for the specified instruction in the
// specified IR. The import_ref is initially marked as unused.
auto AddImportRef(Context& context, SemIR::ImportIRInst import_ir_inst,
                  SemIR::EntityNameId entity_name_id) -> SemIR::InstId;

// Returns the canonical IR inst for an entity. Returns an `ImportIRInst` with
// a `Local` ir_id for an entity that was not imported. This may add an
// `ImportIR` if it finds an indirectly referenced `File` that hasn't previously
// been found.
auto GetCanonicalImportIRInst(Context& context, SemIR::InstId inst_id)
    -> SemIR::ImportIRInst;

// Verifies a new instruction is the same as a previous instruction.
// prev_import_ir_inst should come from GetCanonicalImportIRInst.
auto VerifySameCanonicalImportIRInst(Context& context, SemIR::NameId name_id,
                                     SemIR::InstId prev_id,
                                     SemIR::ImportIRInst prev_import_ir_inst,
                                     SemIR::ImportIRId new_ir_id,
                                     const SemIR::File* new_import_ir,
                                     SemIR::InstId new_inst_id) -> void;

// Result type of `LoadImportIRInst`.
struct LoadImportIRInstResult {
  SemIR::TypeId type_id;
  SemIR::ConstantId constant_id;
};

// Loads the given `ImportIRInst`, returning its type and constant value in
// the local file.
auto LoadImportIRInst(Context& context, SemIR::ImportIRInst import_ir_inst)
    -> LoadImportIRInstResult;

// If the passed in instruction ID is an ImportRefUnloaded, turns it into an
// ImportRefLoaded for use.
auto LoadImportRef(Context& context, SemIR::InstId inst_id) -> void;

// Load all impls declared in the api file corresponding to this impl file.
auto ImportImplsFromApiFile(Context& context) -> void;

// Imports `id` into the local file. `id` must have a value and be in the ID
// space of `import_ir_id`, and `import_ir_id` cannot be `Local`. Repeated calls
// with the same `id` will return the same local ID.
//
// Not all ID kinds are supported; only ones that can be meaningfully mapped
// from one file to another without additional context.
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::AbsoluteInstBlockId id) -> SemIR::AbsoluteInstBlockId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::AbsoluteInstId id) -> SemIR::AbsoluteInstId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::AssociatedConstantId id) -> SemIR::AssociatedConstantId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::ClangDeclId id) -> SemIR::ClangDeclId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::ClangFunctionPointerTypeId id)
    -> SemIR::ClangFunctionPointerTypeId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::ClassId id) -> SemIR::ClassId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::ConstantId id) -> SemIR::ConstantId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::CppOverloadSetId id) -> SemIR::CppOverloadSetId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::CustomLayoutId id) -> SemIR::CustomLayoutId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id, FloatId id)
    -> FloatId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::FunctionId id) -> SemIR::FunctionId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::GenericId id) -> SemIR::GenericId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::ImplId id) -> SemIR::ImplId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::ImportIRInstId id) -> SemIR::ImportIRInstId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::LibraryNameId id) -> SemIR::LibraryNameId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::MetaInstId id) -> SemIR::MetaInstId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::NamedConstraintId id) -> SemIR::NamedConstraintId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::NameScopeId id) -> SemIR::NameScopeId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::NameId id) -> SemIR::NameId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::ObserveId id) -> SemIR::ObserveId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::InterfaceId id) -> SemIR::InterfaceId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id, RealId id)
    -> RealId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::RequireImplsId id) -> SemIR::RequireImplsId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::SpecificId id) -> SemIR::SpecificId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              StringLiteralValueId id) -> StringLiteralValueId;
auto ImportId(Context& context, SemIR::ImportIRId import_ir_id,
              SemIR::VtableId id) -> SemIR::VtableId;

// These IDs are always canonical, and have the same meaning in every IR.
template <typename IdT>
  requires SameAsOneOf<IdT, SemIR::BoolValue, SemIR::CharId,
                       SemIR::CallParamIndex, SemIR::CompileTimeBindIndex,
                       SemIR::ElementIndex, SemIR::FloatKind, IntId,
                       SemIR::IntKind>
auto ImportId(Context& /*context*/, SemIR::ImportIRId /*import_ir_id*/, IdT id)
    -> IdT {
  return id;
}

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_REF_H_
