// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/impl_lookup.h"

#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/operators.h"
#include "toolchain/check/cpp/overload_resolution.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/custom_witness.h"
#include "toolchain/check/impl.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/core_interface.h"
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
  SemIR::ClangDeclSignatureId signature_id;
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
      ImportCppFunctionDecl(context, loc_id, cpp_fn, decl_info.signature_id);
  if (fn_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::InstId;
  }
  CheckCppOverloadAccess(
      context, loc_id, clang::DeclAccessPair::make(cpp_fn, cpp_fn->getAccess()),
      context.insts().GetAsKnownInstId<SemIR::FunctionDecl>(fn_id));

  return fn_id;
}

// Creates a signature with `Normal` kind and the given parameter passing
// modes, and adds it to the value store.
static auto MakeSignature(
    Context& context,
    std::initializer_list<SemIR::ClangDeclSignature::PassingMode> modes,
    SemIR::ClangDeclSignature::PassingMode self_passing_mode =
        SemIR::ClangDeclSignature::PassingMode::ByRef)
    -> SemIR::ClangDeclSignatureId {
  return context.clang_decl_signatures().Add(SemIR::ClangDeclSignature::Make(
      modes, SemIR::ClangDeclSignature::Normal, self_passing_mode));
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

    SemIR::ClangDeclSignatureId signature_id = MakeSignature(
        context, {SemIR::ClangDeclSignature::PassingMode::ByValue});
    auto decl_info = DeclInfo{.decl = clang_sema.LookupCopyingConstructor(
                                  class_decl, clang::Qualifiers::Const),
                              .signature_id = signature_id};
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

  // TODO: Parameterize the interface by the form of the operand and compute the
  // appropriate passing mode here.
  SemIR::ClangDeclSignatureId signature_id =
      MakeSignature(context, {}, SemIR::ClangDeclSignature::PassingMode::ByRef);

  auto decl_info =
      DeclInfo{.decl = *candidates.begin(), .signature_id = signature_id};
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
  SemIR::ClangDeclSignatureId signature_id = MakeSignature(context, {});

  auto decl_info =
      DeclInfo{.decl = clang_sema.LookupDefaultConstructor(class_decl),
               .signature_id = signature_id};
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
  SemIR::ClangDeclSignatureId signature_id = MakeSignature(context, {});

  auto decl_info = DeclInfo{.decl = clang_sema.LookupDestructor(class_decl),
                            .signature_id = signature_id};
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

static auto BuildCppComparisonWitness(
    Context& context, SemIR::LocId loc_id, CoreIdentifier interface,
    llvm::ArrayRef<CoreIdentifier> operator_names,
    SemIR::ConstantId query_self_const_id,
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

  auto arg_type_id = context.types().GetTypeIdForTypeInstId(args[0]);
  auto operators = llvm::SmallVector<SemIR::InstId, 4>();
  for (auto operator_name : operator_names) {
    auto lookup_id = LookupCppOperator(
        context, loc_id,
        {.interface_name = interface, .op_name = operator_name},
        {self_type_id, arg_type_id});
    if (lookup_id == SemIR::ErrorInst::InstId ||
        lookup_id == SemIR::InstId::None) {
      // `T.(Core.EqWith(U))` is only valid if both `t == u` and `t != u` are
      // valid expressions. Looking for `t != u` is pointless if we're unable to
      // find a valid `t == u`, because `t != u` won't provide further useful
      // information at this point.
      //
      // The behavior is dependent on which expression is checked first: no
      // diagnostic will be emitted for an erroneous `operator!=` if lookup for
      // `operator==` produces `SemIR::InstId::None`.
      return lookup_id;
    }

    auto return_type_id = context.functions()
                              .Get(context.insts()
                                       .GetAs<SemIR::FunctionDecl>(lookup_id)
                                       .function_id)
                              .return_type_inst_id;
    if (!return_type_id.has_value()) {
      return SemIR::InstId::None;
    }

    operators.push_back(lookup_id);
  }

  return BuildCustomWitness(context, loc_id, query_self_const_id,
                            query_specific_interface_id, operators);
}

static auto LookupCppMethod(Context& context, clang::Sema& clang_sema,
                            SemIR::LocId loc_id,
                            const clang::DeclarationNameInfo& name_info,
                            clang::CXXRecordDecl* class_decl,
                            SemIR::ConstantId /*query_self_const_id*/)
    -> SemIR::InstId {
  constexpr auto LookupKind = clang::Sema::LookupMemberName;
  auto lookup_info = clang::LookupResult(clang_sema, name_info, LookupKind);
  clang_sema.LookupQualifiedName(lookup_info, class_decl);
  if (lookup_info.empty()) {
    return SemIR::InstId::None;
  }

  if (!lookup_info.isSingleResult()) {
    context.TODO(loc_id, "overload sets unsupported");
    return SemIR::ErrorInst::InstId;
  }

  if (lookup_info.begin()->getKind() != clang::Decl::CXXMethod) {
    return SemIR::InstId::None;
  }

  auto decl_info = DeclInfo{
      .decl = *lookup_info.begin(),
      .signature_id = MakeSignature(
          context, {}, SemIR::ClangDeclSignature::PassingMode::ByValue)};
  return GetFunctionId(context, loc_id, decl_info);
}

static auto LookupCppUnqualified(Context& context, clang::Sema& clang_sema,
                                 SemIR::LocId loc_id,
                                 const clang::DeclarationNameInfo& name_info,
                                 clang::CXXRecordDecl* /*class_decl*/,
                                 SemIR::ConstantId query_self_const_id)
    -> SemIR::InstId {
  auto lookup_result = clang_sema.CreateUnresolvedLookupExpr(
      /*NamingClass=*/nullptr, clang::NestedNameSpecifierLoc(), name_info,
      clang::UnresolvedSet<0>());
  if (lookup_result.isInvalid()) {
    return SemIR::ErrorInst::InstId;
  }

  auto* function = llvm::cast<clang::UnresolvedLookupExpr>(lookup_result.get());

  auto candidate_set = clang::OverloadCandidateSet(
      clang::SourceLocation(),
      clang::OverloadCandidateSet::CandidateSetKind::CSK_Normal);

  auto self_type_id =
      context.types().GetTypeIdForTypeConstantId(query_self_const_id);
  auto type = MapToCppType(context, self_type_id);
  if (type.isNull()) {
    return SemIR::InstId::None;
  }

  auto synthesised_expr =
      clang::OpaqueValueExpr({}, type, clang::ExprValueKind::VK_LValue);
  auto* args = static_cast<clang::Expr*>(&synthesised_expr);
  clang_sema.AddArgumentDependentLookupCandidates(name_info.getName(), {}, args,
                                                  {}, candidate_set);

  if (candidate_set.empty()) {
    return SemIR::InstId::None;
  }

  auto* best_candidate = clang::OverloadCandidateSet::iterator();
  auto overload_result =
      candidate_set.BestViableFunction(clang_sema, {}, best_candidate);
  switch (overload_result) {
    case clang::OR_No_Viable_Function:
    case clang::OR_Ambiguous:
      return SemIR::InstId::None;
    case clang::OR_Deleted: {
      auto loc = GetCppLocation(context, loc_id);
      clang_sema.DiagnoseUseOfDeletedFunction(
          loc, clang::SourceRange(loc, loc), name_info.getName(), candidate_set,
          best_candidate->Function, args);
      return SemIR::ErrorInst::InstId;
    }
    case clang::OR_Success: {
      using enum SemIR::ClangDeclSignature::PassingMode;
      auto decl_info = DeclInfo{
          .decl = best_candidate->Function,
          .signature_id = ComputeClangDeclSignatureFromBestViableFunction(
              context, best_candidate, function, args),
      };
      return GetFunctionId(context, loc_id, decl_info);
    }
  }
}

using LookupBeginEndCallees = auto(Context&, clang::Sema&, SemIR::LocId,
                                   const clang::DeclarationNameInfo&,
                                   clang::CXXRecordDecl*, SemIR::ConstantId)
    -> SemIR::InstId;

static auto BuildCppRangeForIterateWitnessImpl(
    Context& context, SemIR::LocId loc_id,
    LookupBeginEndCallees range_for_lookup, clang::CXXRecordDecl* class_decl,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto& clang_sema = context.clang_sema();
  auto begin_name_info = clang::DeclarationNameInfo(
      &clang_sema.PP.getIdentifierTable().get("begin"),
      clang::SourceLocation());
  auto begin_fn_id =
      range_for_lookup(context, clang_sema, loc_id, begin_name_info, class_decl,
                       query_self_const_id);
  if (begin_fn_id == SemIR::InstId::None ||
      begin_fn_id == SemIR::ErrorInst::InstId) {
    return begin_fn_id;
  }

  auto begin_result_type_id =
      context.functions()
          .Get(context.insts()
                   .GetAs<SemIR::FunctionDecl>(begin_fn_id)
                   .function_id)
          .return_type_inst_id;
  CARBON_CHECK(begin_result_type_id != SemIR::ErrorInst::InstId &&
               begin_result_type_id != SemIR::InstId::None);

  auto end_name_info = clang::DeclarationNameInfo(
      &clang_sema.PP.getIdentifierTable().get("end"), clang::SourceLocation());
  auto end_fn_id = range_for_lookup(context, clang_sema, loc_id, end_name_info,
                                    class_decl, query_self_const_id);
  if (end_fn_id == SemIR::InstId::None ||
      end_fn_id == SemIR::ErrorInst::InstId) {
    return end_fn_id;
  }

  auto end_result_type_id =
      context.functions()
          .Get(
              context.insts().GetAs<SemIR::FunctionDecl>(end_fn_id).function_id)
          .return_type_inst_id;
  CARBON_CHECK(end_result_type_id != SemIR::ErrorInst::InstId &&
               end_result_type_id != SemIR::InstId::None);

  return BuildCustomWitness(
      context, loc_id, query_self_const_id, query_specific_interface_id,
      {begin_result_type_id, end_result_type_id, begin_fn_id, end_fn_id});
}

static auto BuildCppRangeForIterateWitness(
    Context& context, SemIR::LocId loc_id,
    SemIR::ConstantId query_self_const_id,
    SemIR::SpecificInterfaceId query_specific_interface_id) -> SemIR::InstId {
  auto* class_decl = TypeAsClassDecl(context, query_self_const_id);
  if (auto with_members = BuildCppRangeForIterateWitnessImpl(
          context, loc_id, LookupCppMethod, class_decl, query_self_const_id,
          query_specific_interface_id);
      with_members != SemIR::InstId::None) {
    return with_members;
  }

  return BuildCppRangeForIterateWitnessImpl(
      context, loc_id, LookupCppUnqualified, class_decl, query_self_const_id,
      query_specific_interface_id);
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
    case SemIR::CoreInterface::EqWith:
      return BuildCppComparisonWitness(
          context, loc_id, CoreIdentifier::EqWith,
          {CoreIdentifier::Equal, CoreIdentifier::NotEqual},
          query_self_const_id, query_specific_interface_id);
    case SemIR::CoreInterface::OrderedWith:
      return BuildCppComparisonWitness(
          context, loc_id, CoreIdentifier::OrderedWith,
          {CoreIdentifier::Less, CoreIdentifier::LessOrEquivalent,
           CoreIdentifier::Greater, CoreIdentifier::GreaterOrEquivalent},
          query_self_const_id, query_specific_interface_id);
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

    case SemIR::CoreInterface::CppRangeForIterate:
      return BuildCppRangeForIterateWitness(
          context, loc_id, query_self_const_id, query_specific_interface_id);

    // IntFitsIn is for Carbon integer types only.
    case SemIR::CoreInterface::IntFitsIn:
      return SemIR::InstId::None;

    case SemIR::CoreInterface::Unknown:
      CARBON_FATAL("unexpected CoreInterface `{0}`", core_interface);
  }
}

}  // namespace Carbon::Check
