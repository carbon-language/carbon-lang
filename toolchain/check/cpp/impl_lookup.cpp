// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/impl_lookup.h"

#include <clang/AST/DeclBase.h>
#include <clang/AST/DeclCXX.h>

#include <type_traits>
#include <utility>

#include "clang/AST/DeclarationName.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "common/map.h"
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
#include "toolchain/check/operator.h"
#include "toolchain/check/type.h"
#include "toolchain/sem_ir/associated_constant.h"
#include "toolchain/sem_ir/cpp_overload_set.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
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

  // TODO: This should provide `Copy` for enums and other trivially copyable
  // types.
  auto* class_decl = TypeAsClassDecl(context, query_self_const_id);
  if (!class_decl) {
    return SemIR::InstId::None;
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

namespace {
struct CppRangeForIterateWitnessEntries {
  auto HasAllEntries() const -> bool {
    return iterator_type != SemIR::InstId::None &&
           sentinel_type != SemIR::InstId::None &&
           //  element_type == SemIR::InstId::None &&
           begin_fn != SemIR::InstId::None && end_fn != SemIR::InstId::None &&
           next_fn != SemIR::InstId::None &&
           not_equal_fn != SemIR::InstId::None;
  }

  auto HasError() const -> bool {
    return iterator_type == SemIR::ErrorInst::InstId ||
           sentinel_type == SemIR::ErrorInst::InstId ||
           //  element_type == SemIR::ErrorInst::InstId ||
           begin_fn == SemIR::ErrorInst::InstId ||
           end_fn == SemIR::ErrorInst::InstId ||
           next_fn == SemIR::ErrorInst::InstId ||
           not_equal_fn == SemIR::ErrorInst::InstId;
  }

  SemIR::InstId iterator_type = SemIR::InstId::None;
  SemIR::InstId sentinel_type = SemIR::InstId::None;
  // SemIR::InstId element_type = SemIR::InstId::None;
  SemIR::InstId begin_fn = SemIR::InstId::None;
  SemIR::InstId end_fn = SemIR::InstId::None;
  SemIR::InstId next_fn = SemIR::InstId::None;
  SemIR::InstId not_equal_fn = SemIR::InstId::None;
};
}  // namespace

static auto GetIncrementId(Context& context, SemIR::LocId loc_id,
                           const clang::CXXRecordDecl* iterator_type)
    -> SemIR::InstId {
  auto viable_candidates = llvm::SmallVector<clang::FunctionDecl*>{};
  auto candidates = iterator_type->lookup(
      context.clang_sema().getASTContext().DeclarationNames.getCXXOperatorName(
          clang::OO_PlusPlus));
  for (auto* candidate : candidates) {
    auto* function_decl = candidate->getUnderlyingDecl()->getAsFunction();
    if (function_decl->getNumParams() == 0) {
      viable_candidates.push_back(function_decl);
    }
  }

  if (viable_candidates.empty()) {
    return SemIR::InstId::None;
  }

  if (viable_candidates.size() > 1) {
    context.TODO(loc_id, "operator++ overload sets not handled yet");
    return SemIR::ErrorInst::InstId;
  }

  return GetFunctionId(
      context, loc_id,
      {.decl = viable_candidates.front(), .signature = {.num_params = 0}});
}

static auto MakeRangeWitnessEntries(Context& context, SemIR::LocId loc_id,
                                    DeclInfo begin_fn, DeclInfo end_fn)
    -> std::optional<CppRangeForIterateWitnessEntries> {
  const auto* begin_return_type =
      begin_fn.decl->getAsFunction()->getReturnType().getTypePtr();
  const auto* end_return_type =
      end_fn.decl->getAsFunction()->getReturnType().getTypePtr();

  if (begin_return_type->isPointerType()) {
    context.TODO(loc_id, "handle pointer types");
    return std::nullopt;
  }

  auto begin_fn_id = GetFunctionId(context, loc_id, begin_fn);
  auto end_fn_id = GetFunctionId(context, loc_id, end_fn);

  // CppRangeForIterate.Iterator and CppRangeForIterate.Sentinel can't be worked
  // out if `begin` or `end` don't have a Carbon ID, and we can't provide more
  // information without those types, so we should return now.
  if (begin_fn_id == SemIR::InstId::None || end_fn_id == SemIR::InstId::None ||
      begin_fn_id == SemIR::ErrorInst::InstId ||
      end_fn_id == SemIR::ErrorInst::InstId) {
    return CppRangeForIterateWitnessEntries{
        .iterator_type = SemIR::InstId::None,
        .sentinel_type = SemIR::InstId::None,
        .begin_fn = begin_fn_id,
        .end_fn = end_fn_id,
        .next_fn = SemIR::InstId::None,
        .not_equal_fn = SemIR::InstId::None,
    };
  }

  auto iterator_type = context.functions()
                           .Get(context.insts()
                                    .GetAs<SemIR::FunctionDecl>(begin_fn_id)
                                    .function_id)
                           .return_type_inst_id;
  auto sentinel_type =
      context.functions()
          .Get(
              context.insts().GetAs<SemIR::FunctionDecl>(end_fn_id).function_id)
          .return_type_inst_id;

  if (begin_return_type != end_return_type) {
    context.TODO(loc_id, "add a test case for iterator-sentinel pairs");
    return CppRangeForIterateWitnessEntries{};
  }

  return CppRangeForIterateWitnessEntries{
      .iterator_type = iterator_type,
      .sentinel_type = sentinel_type,
      .begin_fn = begin_fn_id,
      .end_fn = end_fn_id,
      .next_fn = GetIncrementId(context, loc_id,
                                begin_return_type->getAsCXXRecordDecl()),
      .not_equal_fn =
          LookupCppOperator(context, loc_id,
                            {.interface_name = CoreIdentifier::NotEqual,
                             .op_name = CoreIdentifier::NotEqual},
                            {iterator_type, sentinel_type}),
  };
}

static auto LookupCppBeginEndMethods(Context& context, SemIR::LocId loc_id,
                                     clang::CXXRecordDecl* class_decl)
    -> std::optional<CppRangeForIterateWitnessEntries> {
  auto lookup_method =
      [&](std::string_view method_name) -> std::optional<DeclInfo> {
    auto& clang_sema = context.clang_sema();
    auto lookup_info = clang::LookupResult(
        clang_sema, &clang_sema.getASTContext().Idents.get(method_name),
        clang::SourceLocation(), clang::Sema::LookupMemberName);
    clang_sema.LookupQualifiedName(lookup_info, class_decl);
    if (lookup_info.isAmbiguous()) {
      return std::nullopt;
    }

    if (!lookup_info.isSingleResult()) {
      context.TODO(loc_id, "C++ method overload sets not implemented yet");
      return std::nullopt;
    }

    if (lookup_info.empty()) {
      return DeclInfo{.decl = nullptr, .signature = {.num_params = 0}};
    }

    return DeclInfo{.decl = *lookup_info.begin(),
                    .signature = {.num_params = 0}};
  };

  auto begin_fn = lookup_method("begin");
  if (!begin_fn.has_value()) {
    return CppRangeForIterateWitnessEntries{};
  }

  auto end_fn = lookup_method("end");
  if (!end_fn.has_value()) {
    return CppRangeForIterateWitnessEntries{};
  }

  return MakeRangeWitnessEntries(context, loc_id, *begin_fn, *end_fn);
}

#if 0
static auto LookupCppUnqualifiedBeginEnd(Context& context, SemIR::LocId loc_id,
                                         clang::CXXRecordDecl* class_decl)
    -> std::optional<CppRangeForIterateWitnessEntries> {
  auto lookup_unqualified_function =
      [&](std::string_view name) -> std::optional<DeclInfo> {
    auto& clang_sema = context.clang_sema();
    auto lookup_info = clang::LookupResult(
        clang_sema, &clang_sema.getASTContext().Idents.get(name),
        clang::SourceLocation(), clang::Sema::LookupOrdinaryName);
    clang_sema.LookupQualifiedName(lookup_info, class_decl);
    if (lookup_info.isAmbiguous()) {
      return std::nullopt;
    }

    if (!lookup_info.isSingleResult()) {
      context.TODO(loc_id, "C++ method overload sets not implemented yet");
      return std::nullopt;
    }

    if (lookup_info.empty()) {
      return DeclInfo{.decl = nullptr, .signature = {.num_params = 0}};
    }

    return DeclInfo{.decl = *lookup_info.begin(),
                    .signature = {.num_params = 0}};
  };

  auto begin_fn = lookup_unqualified_function("begin");
  if (!begin_fn.has_value()) {
    return std::nullopt;
  }

  auto end_fn = lookup_unqualified_function("end");
  if (!end_fn.has_value()) {
    return std::nullopt;
  }

  return MakeRangeWitnessEntries(context, loc_id, *begin_fn, *end_fn);
}
#endif

static auto BuildCppRangeForIterateWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto* class_decl = TypeAsClassDecl(context, query_self_const_id);

  auto lookup_methods = LookupCppBeginEndMethods(context, loc_id, class_decl);
  if (!lookup_methods.has_value()) {
    return SemIR::ErrorInst::InstId;
  }

  if (lookup_methods->HasAllEntries() && !lookup_methods->HasError()) {
    auto methods = std::array{
        lookup_methods->iterator_type, lookup_methods->sentinel_type,
        lookup_methods->begin_fn,      lookup_methods->end_fn,
        lookup_methods->next_fn,       lookup_methods->not_equal_fn,
    };
    return BuildCustomWitness(context, loc_id, query_self_const_id,
                              query_specific_interface_id, methods);
  }

  // auto lookup_unqualified =
  //     LookupCppUnqualifiedBeginEnd(context, loc_id, class_decl);
  // if (!lookup_unqualified.has_value()) {
  //   return SemIR::ErrorInst::InstId;
  // }

  // if (!lookup_unqualified->HasNone() && !lookup_methods->HasError()) {
  //   return BuildCustomWitness(context, loc_id, query_self_const_id,
  //                             query_specific_interface_id,
  //                             {&lookup_unqualified->iterator_type, 6});
  // }
  llvm::errs() << '\n';
  if (lookup_methods->iterator_type == SemIR::InstId::None) {
    llvm::errs() << "CppRangeForIterate.Iterator == None\n";
  }
  if (lookup_methods->iterator_type == SemIR::ErrorInst::InstId) {
    llvm::errs() << "CppRangeForIterate.Iterator == Error\n";
  }
  if (lookup_methods->sentinel_type == SemIR::InstId::None) {
    llvm::errs() << "CppRangeForIterate.Sentinel == None\n";
  }
  if (lookup_methods->sentinel_type == SemIR::ErrorInst::InstId) {
    llvm::errs() << "CppRangeForIterate.Sentinel == Error\n";
  }
  if (lookup_methods->begin_fn == SemIR::InstId::None) {
    llvm::errs() << "CppRangeForIterate.Begin == None\n";
  }
  if (lookup_methods->begin_fn == SemIR::ErrorInst::InstId) {
    llvm::errs() << "CppRangeForIterate.Begin == Error\n";
  }
  if (lookup_methods->end_fn == SemIR::InstId::None) {
    llvm::errs() << "CppRangeForIterate.End == None\n";
  }
  if (lookup_methods->end_fn == SemIR::ErrorInst::InstId) {
    llvm::errs() << "CppRangeForIterate.End == Error\n";
  }
  if (lookup_methods->next_fn == SemIR::InstId::None) {
    llvm::errs() << "CppRangeForIterate.Next == None\n";
  }
  if (lookup_methods->next_fn == SemIR::ErrorInst::InstId) {
    llvm::errs() << "CppRangeForIterate.Next == Error\n";
  }
  if (lookup_methods->not_equal_fn == SemIR::InstId::None) {
    llvm::errs() << "CppRangeForIterate.NotEqual == None\n";
  }
  if (lookup_methods->not_equal_fn == SemIR::ErrorInst::InstId) {
    llvm::errs() << "CppRangeForIterate.NotEqual == Error\n";
  }

  context.TODO(loc_id,
               "provide helpful diagnoses for why range-for doesn't work");
  return SemIR::InstId::None;
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

auto LookupCppImpl(Context& context, SemIR::LocId loc_id,
                   CoreInterface core_interface,
                   SemIR::ConstantId query_self_const_id,
                   SemIR::SpecificInterfaceId query_specific_interface_id,
                   const TypeStructure* best_impl_type_structure,
                   SemIR::LocId best_impl_loc_id) -> SemIR::InstId {
  // TODO: Infer a C++ type structure and check whether it's less strict than
  // the best Carbon type structure.
  static_cast<void>(best_impl_type_structure);
  static_cast<void>(best_impl_loc_id);

  switch (core_interface) {
    case CoreInterface::Copy:
      return BuildCopyWitness(context, loc_id, query_self_const_id,
                              query_specific_interface_id);
    case CoreInterface::CppUnsafeDeref:
      return BuildCppUnsafeDerefWitness(context, loc_id, query_self_const_id,
                                        query_specific_interface_id);
    case CoreInterface::Default:
      return BuildDefaultWitness(context, loc_id, query_self_const_id,
                                 query_specific_interface_id);
    case CoreInterface::Destroy:
      return BuildDestroyWitness(context, loc_id, query_self_const_id,
                                 query_specific_interface_id);
    case CoreInterface::CppRangeForIterate:
      return BuildCppRangeForIterateWitness(
          context, loc_id, query_self_const_id, query_specific_interface_id);

    // IntFitsIn is for Carbon integer types only.
    case CoreInterface::IntFitsIn:
      return SemIR::InstId::None;

    // Values that should never reach this section of code.
    case CoreInterface::Unknown:
      CARBON_FATAL("unexpected CoreInterface `{0}`", core_interface);
  }
}

}  // namespace Carbon::Check
