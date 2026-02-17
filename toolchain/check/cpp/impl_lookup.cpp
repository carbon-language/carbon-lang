// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/impl_lookup.h"

#include "clang/Sema/Sema.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/overload_resolution.h"
#include "toolchain/check/custom_witness.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// If the given type is a C++ class type, returns the corresponding class
// declaration. Otherwise returns nullptr.
// TODO: Handle qualified types.
static auto TypeAsClassDecl(Context& context,
                            SemIR::ConstantId query_self_const_id)
    -> clang::CXXRecordDecl* {
  auto self_inst_id = context.constant_values().GetInstId(query_self_const_id);
  auto class_type = context.insts().TryGetAs<SemIR::ClassType>(self_inst_id);
  if (!class_type) {
    // Not a class.
    return nullptr;
  }

  SemIR::NameScopeId class_scope_id =
      context.classes().Get(class_type->class_id).scope_id;
  if (!class_scope_id.has_value()) {
    return nullptr;
  }

  const auto& scope = context.name_scopes().Get(class_scope_id);
  auto decl_id = scope.clang_decl_context_id();
  if (!decl_id.has_value()) {
    return nullptr;
  }

  return dyn_cast<clang::CXXRecordDecl>(
      context.clang_decls().Get(decl_id).key.decl);
}

namespace {
// See `GetDeclForCoreInterface`.
struct DeclInfo {
  clang::NamedDecl* decl;
  SemIR::ClangDeclKey::Signature signature;
};
}  // namespace

// Retrieves a `core_interface`'s corresponding `NamedDecl`, also with the
// expected number of parameters. May return a null decl.
auto GetDeclForCoreInterface(clang::Sema& clang_sema,
                             CoreInterface core_interface,
                             clang::CXXRecordDecl* class_decl) -> DeclInfo {
  // TODO: Handle other interfaces.

  switch (core_interface) {
    case CoreInterface::Copy:
      return {.decl = clang_sema.LookupCopyingConstructor(
                  class_decl, clang::Qualifiers::Const),
              .signature = {.num_params = 1}};
    case CoreInterface::Destroy:
      return {.decl = clang_sema.LookupDestructor(class_decl),
              .signature = {.num_params = 0}};
    case CoreInterface::Unknown:
      CARBON_FATAL("shouldn't be called with `Unknown`");
  }
}

auto LookupCppImpl(Context& context, SemIR::LocId loc_id,
                   CoreInterface core_interface,
                   SemIR::ConstantId query_self_const_id,
                   SemIR::SpecificInterfaceId query_specific_interface_id,
                   const TypeStructure* best_impl_type_structure,
                   SemIR::LocId best_impl_loc_id) -> SemIR::InstId {
  // TODO: This should provide `Destroy` for enums and other trivially
  // destructible types.
  auto* class_decl = TypeAsClassDecl(context, query_self_const_id);
  if (!class_decl) {
    return SemIR::InstId::None;
  }

  auto decl_info =
      GetDeclForCoreInterface(context.clang_sema(), core_interface, class_decl);
  if (!decl_info.decl) {
    // TODO: If the impl lookup failure is an error, we should produce a
    // diagnostic explaining why the class is not copyable/destructible.
    return SemIR::InstId::None;
  }
  auto* cpp_fn = cast<clang::FunctionDecl>(decl_info.decl);

  if (context.clang_sema().DiagnoseUseOfOverloadedDecl(
          cpp_fn, GetCppLocation(context, loc_id))) {
    return SemIR::ErrorInst::InstId;
  }

  auto fn_id =
      ImportCppFunctionDecl(context, loc_id, cpp_fn, decl_info.signature);
  if (fn_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::InstId;
  }
  CheckCppOverloadAccess(
      context, loc_id, clang::DeclAccessPair::make(cpp_fn, cpp_fn->getAccess()),
      context.insts().GetAsKnownInstId<SemIR::FunctionDecl>(fn_id));

  // TODO: Infer a C++ type structure and check whether it's less strict than
  // the best Carbon type structure.
  static_cast<void>(best_impl_type_structure);
  static_cast<void>(best_impl_loc_id);

  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {fn_id});
}

}  // namespace Carbon::Check
