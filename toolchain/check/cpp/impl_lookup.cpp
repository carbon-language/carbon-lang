// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/impl_lookup.h"

#include "clang/Sema/Sema.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/operators.h"
#include "toolchain/check/cpp/overload_resolution.h"
#include "toolchain/check/custom_witness.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Given a type constant, return the corresponding class scope if there is one.
static auto GetClassScope(Context& context,
                          SemIR::ConstantId query_self_const_id)
    -> SemIR::NameScopeId {
  auto class_type = context.constant_values().TryGetInstAs<SemIR::ClassType>(
      query_self_const_id);
  if (!class_type) {
    // Not a class.
    return SemIR::NameScopeId::None;
  }

  return context.classes().Get(class_type->class_id).scope_id;
}

// If the given type is a C++ tag (class or enumeration) type, returns the
// corresponding tag declaration. Otherwise returns nullptr.
// TODO: Handle qualified types.
static auto TypeAsTagDecl(Context& context,
                          SemIR::ConstantId query_self_const_id)
    -> clang::TagDecl* {
  SemIR::NameScopeId class_scope_id =
      GetClassScope(context, query_self_const_id);
  if (!class_scope_id.has_value()) {
    return nullptr;
  }

  const auto& scope = context.name_scopes().Get(class_scope_id);
  auto decl_id = scope.clang_decl_context_id();
  if (!decl_id.has_value()) {
    return nullptr;
  }

  return dyn_cast<clang::TagDecl>(context.clang_decls().Get(decl_id).key.decl);
}

// If the given type is a C++ class type, returns the corresponding class
// declaration. Otherwise returns nullptr.
static auto TypeAsClassDecl(Context& context,
                            SemIR::ConstantId query_self_const_id)
    -> clang::CXXRecordDecl* {
  return dyn_cast_or_null<clang::CXXRecordDecl>(
      TypeAsTagDecl(context, query_self_const_id));
}

namespace {
struct DeclInfo {
  // If null, no C++ decl was found and no witness can be created.
  clang::NamedDecl* decl = nullptr;
  SemIR::ClangDeclKey::Signature signature;
};
}  // namespace

// Finds the InstId for the C++ function that is called by a specific interface.
// Returns SemIR::InstId::None if a C++ function is not found, and
// SemIR::ErrorInst::InstId if an error occurs.
static auto GetFunctionId(Context& context, SemIR::LocId loc_id,
                          DeclInfo decl_info) -> SemIR::InstId {
  if (!decl_info.decl) {
    // The C++ type is not able to implement the interface.
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

  return fn_id;
}

static auto BuildCopyWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto& clang_sema = context.clang_sema();

  auto* tag_decl = TypeAsTagDecl(context, query_self_const_id);
  if (!tag_decl) {
    return SemIR::InstId::None;
  }
  if (auto* class_decl = dyn_cast<clang::CXXRecordDecl>(tag_decl)) {
    auto class_type_id = SemIR::TypeId::ForTypeConstant(query_self_const_id);
    if (!Check::RequireCompleteType(
            context, class_type_id, SemIR::LocId::None, [&](auto& builder) {
              CARBON_DIAGNOSTIC(IncompleteTypeInCopyWitness, Context,
                                "argument to C++ call has incomplete type {0}",
                                SemIR::TypeId);
              builder.Context(loc_id, IncompleteTypeInCopyWitness,
                              class_type_id);
            })) {
      return SemIR::ErrorInst::InstId;
    }
    auto decl_info = DeclInfo{.decl = clang_sema.LookupCopyingConstructor(
                                  class_decl, clang::Qualifiers::Const),
                              .signature = {.num_params = 1}};
    auto fn_id = GetFunctionId(context, loc_id, decl_info);
    if (fn_id == SemIR::ErrorInst::InstId || fn_id == SemIR::InstId::None) {
      return fn_id;
    }
    return BuildCustomWitness(context, loc_id, query_self_const_id,
                              query_specific_interface_id, {fn_id});
  }
  // Otherwise it's an enum (or eventually a C struct type). Perform a primitive
  // copy.
  return BuildPrimitiveCopyWitness(
      context, loc_id, GetClassScope(context, query_self_const_id),
      query_self_const_id, query_specific_interface_id);
}

static auto BuildCppUnsafeDerefWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto& clang_sema = context.clang_sema();

  auto* class_decl = TypeAsClassDecl(context, query_self_const_id);
  if (!class_decl) {
    return SemIR::InstId::None;
  }
  auto candidates = class_decl->lookup(
      clang_sema.getASTContext().DeclarationNames.getCXXOperatorName(
          clang::OO_Star));
  if (candidates.empty()) {
    return SemIR::InstId::None;
  }
  if (!candidates.isSingleResult()) {
    context.TODO(loc_id, "operator* overload sets not implemented yet");
    return SemIR::ErrorInst::InstId;
  }
  auto decl_info =
      DeclInfo{.decl = *candidates.begin(), .signature = {.num_params = 0}};
  auto fn_id = GetFunctionId(context, loc_id, decl_info);
  if (fn_id == SemIR::ErrorInst::InstId || fn_id == SemIR::InstId::None) {
    return fn_id;
  }

  auto result_type_id =
      context.functions()
          .Get(context.insts().GetAs<SemIR::FunctionDecl>(fn_id).function_id)
          .return_type_inst_id;
  if (result_type_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::InstId;
  }

  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id,
                            {result_type_id, fn_id});
}

static auto BuildDefaultWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto& clang_sema = context.clang_sema();

  auto* class_decl = TypeAsClassDecl(context, query_self_const_id);
  if (!class_decl) {
    return SemIR::InstId::None;
  }
  // Clang would produce a warning for classes with uninitialized
  // [[clang::requires_init]] fields for which default initialization is
  // performed, and we don't have a good place to produce that warning.
  // That happens if class_decl->hasUninitializedExplicitInitFields() is true.
  //
  // TODO: Consider treating such types as not implementing `Default`.
  auto decl_info =
      DeclInfo{.decl = clang_sema.LookupDefaultConstructor(class_decl),
               .signature = {.num_params = 0}};
  auto fn_id = GetFunctionId(context, loc_id, decl_info);
  if (fn_id == SemIR::ErrorInst::InstId || fn_id == SemIR::InstId::None) {
    return fn_id;
  }
  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {fn_id});
}

static auto BuildDestroyWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto& clang_sema = context.clang_sema();

  // TODO: This should provide `Destroy` for enums and other trivially
  // destructible types.
  auto* class_decl = TypeAsClassDecl(context, query_self_const_id);
  if (!class_decl) {
    return SemIR::InstId::None;
  }
  auto decl_info = DeclInfo{.decl = clang_sema.LookupDestructor(class_decl),
                            .signature = {.num_params = 0}};
  auto fn_id = GetFunctionId(context, loc_id, decl_info);
  if (fn_id == SemIR::ErrorInst::InstId || fn_id == SemIR::InstId::None) {
    return fn_id;
  }
  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {fn_id});
}

// Attempts to build a witness table entry for a C++ unary operator.
static auto BuildCppUnaryOperatorWitness(
    Context& context, SemIR::LocId loc_id, SemIR::CoreInterface core_interface,
    bool has_associated_result_type, SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto self_type_id =
      context.types().GetTypeIdForTypeConstantId(query_self_const_id);
  auto fn_id = LookupCppOperator(
      context, loc_id, {.interface_name = AsCoreIdentifier(core_interface)},
      {self_type_id});
  if (fn_id == SemIR::ErrorInst::InstId || fn_id == SemIR::InstId::None) {
    return fn_id;
  }

  if (has_associated_result_type) {
    auto result_type_id =
        context.functions()
            .Get(context.insts().GetAs<SemIR::FunctionDecl>(fn_id).function_id)
            .return_type_inst_id;
    if (result_type_id == SemIR::ErrorInst::InstId) {
      return SemIR::ErrorInst::InstId;
    }

    return BuildCustomWitness(context, loc_id, query_self_const_id,
                              query_specific_interface_id,
                              {result_type_id, fn_id});
  }
  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {fn_id});
}

// Attempts to build a witness table entry for a C++ binary operator.
static auto BuildCppBinaryOperatorWitness(
    Context& context, SemIR::LocId loc_id, SemIR::CoreInterface core_interface,
    bool has_associated_result_type, SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto self_type_id =
      context.types().GetTypeIdForTypeConstantId(query_self_const_id);
  auto args =
      context.inst_blocks().Get(context.specifics()
                                    .Get(context.specific_interfaces()
                                             .Get(query_specific_interface_id)
                                             .specific_id)
                                    .args_id);
  CARBON_CHECK(args.size() == 1, "Binary operator missing an argument");
  auto arg_type_id = context.types().GetTypeIdForTypeInstId(args.front());
  auto fn_id = LookupCppOperator(
      context, loc_id, {.interface_name = AsCoreIdentifier(core_interface)},
      {self_type_id, arg_type_id});
  if (fn_id == SemIR::ErrorInst::InstId || fn_id == SemIR::InstId::None) {
    return fn_id;
  }
  if (has_associated_result_type) {
    auto result_type_id =
        context.functions()
            .Get(context.insts().GetAs<SemIR::FunctionDecl>(fn_id).function_id)
            .return_type_inst_id;
    if (result_type_id == SemIR::ErrorInst::InstId) {
      return SemIR::ErrorInst::InstId;
    }
    return BuildCustomWitness(context, loc_id, query_self_const_id,
                              query_specific_interface_id,
                              {result_type_id, fn_id});
  }
  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, {fn_id});
}

auto LookupCppImpl(Context& context, SemIR::LocId loc_id,
                   SemIR::CoreInterface core_interface,
                   SemIR::ConstantId query_self_const_id,
                   SemIR::SpecificInterfaceId query_specific_interface_id,
                   const TypeStructure* best_impl_type_structure,
                   SemIR::LocId best_impl_loc_id) -> SemIR::InstId {
  // TODO: Infer a C++ type structure and check whether it's less strict than
  // the best Carbon type structure.
  static_cast<void>(best_impl_type_structure);
  static_cast<void>(best_impl_loc_id);

  switch (core_interface) {
    case SemIR::CoreInterface::Inc:
    case SemIR::CoreInterface::Dec:
      return BuildCppUnaryOperatorWitness(context, loc_id, core_interface,
                                          /*has_associated_result_type=*/false,
                                          query_self_const_id,
                                          query_specific_interface_id);
    case SemIR::CoreInterface::Negate:
      return BuildCppUnaryOperatorWitness(
          context, loc_id, core_interface, /*has_associated_result_type=*/true,
          query_self_const_id, query_specific_interface_id);
    case SemIR::CoreInterface::AddWith:
    case SemIR::CoreInterface::SubWith:
    case SemIR::CoreInterface::MulWith:
    case SemIR::CoreInterface::DivWith:
    case SemIR::CoreInterface::ModWith:
      return BuildCppBinaryOperatorWitness(context, loc_id, core_interface,
                                           /*has_associated_result_type=*/true,
                                           query_self_const_id,
                                           query_specific_interface_id);
    case SemIR::CoreInterface::AddAssignWith:
    case SemIR::CoreInterface::SubAssignWith:
    case SemIR::CoreInterface::MulAssignWith:
    case SemIR::CoreInterface::DivAssignWith:
    case SemIR::CoreInterface::ModAssignWith:
      return BuildCppBinaryOperatorWitness(context, loc_id, core_interface,
                                           /*has_associated_result_type=*/false,
                                           query_self_const_id,
                                           query_specific_interface_id);
    case SemIR::CoreInterface::Copy:
      return BuildCopyWitness(context, loc_id, query_self_const_id,
                              query_specific_interface_id);
    case SemIR::CoreInterface::CppUnsafeDeref:
      return BuildCppUnsafeDerefWitness(context, loc_id, query_self_const_id,
                                        query_specific_interface_id);
    case SemIR::CoreInterface::Default:
      return BuildDefaultWitness(context, loc_id, query_self_const_id,
                                 query_specific_interface_id);
    case SemIR::CoreInterface::Destroy:
      return BuildDestroyWitness(context, loc_id, query_self_const_id,
                                 query_specific_interface_id);

    // IntFitsIn is for Carbon integer types only.
    case SemIR::CoreInterface::IntFitsIn:
      return SemIR::InstId::None;

    case SemIR::CoreInterface::Unknown:
      CARBON_FATAL("unexpected CoreInterface `{0}`", core_interface);
  }
}

}  // namespace Carbon::Check
