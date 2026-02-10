// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/call.h"

#include "clang/Sema/Sema.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/call.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/operators.h"
#include "toolchain/check/cpp/overload_resolution.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/literal.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto PerformCallToCppFunction(Context& context, SemIR::LocId loc_id,
                              SemIR::CppOverloadSetId overload_set_id,
                              SemIR::InstId self_id,
                              llvm::ArrayRef<SemIR::InstId> arg_ids,
                              bool is_operator_syntax) -> SemIR::InstId {
  SemIR::InstId callee_id = PerformCppOverloadResolution(
      context, loc_id, overload_set_id, self_id, arg_ids);
  SemIR::Callee callee = GetCallee(context.sem_ir(), callee_id);
  CARBON_KIND_SWITCH(callee) {
    case CARBON_KIND(SemIR::CalleeError _): {
      return SemIR::ErrorInst::InstId;
    }
    case CARBON_KIND(SemIR::CalleeFunction fn): {
      CARBON_CHECK(!fn.self_id.has_value());
      if (self_id.has_value()) {
        // Preserve the `self` argument from the original callee.
        fn.self_id = self_id;
      }
      return PerformCallToFunction(context, loc_id, callee_id, fn, arg_ids,
                                   is_operator_syntax);
    }
    case CARBON_KIND(SemIR::CalleeCppOverloadSet _): {
      CARBON_FATAL("overloads can't be recursive");
    }
    case CARBON_KIND(SemIR::CalleeNonFunction _): {
      CARBON_FATAL("overloads should produce functions");
    }
  }
}

// Synthesize a placeholder `void{}` template argument, that will never be a
// valid argument for any template parameter. This is used in order to get Clang
// to diagnose invalid template argument errors for us. The location of the
// Carbon expression is used as the location of the C++ expression, so
// Clang's diagnostics will point into the Carbon code.
//
// TODO: If Clang ever tries to print the type of the expression or to
// pretty-print the expression itself, it would print the wrong thing. Currently
// this doesn't appear to happen, but in principle it could. Ideally we'd add an
// extension point to Clang to represent a "foreign expression" and use it here
// instead of creating a bogus placeholder expression.
static auto MakePlaceholderTemplateArg(Context& context, SemIR::InstId arg_id)
    -> clang::TemplateArgumentLoc {
  auto arg_loc = GetCppLocation(context, SemIR::LocId(arg_id));
  auto void_type = context.ast_context().VoidTy;
  auto* arg = new (context.ast_context()) clang::CXXScalarValueInitExpr(
      void_type,
      context.ast_context().getTrivialTypeSourceInfo(void_type, arg_loc),
      arg_loc);
  return clang::TemplateArgumentLoc(
      clang::TemplateArgument(arg, /*IsCanonical=*/false), arg);
}

// Converts an argument in a call to a C++ template name into a corresponding
// clang template argument, given the template parameter it will be matched
// against.
static auto ConvertArgToTemplateArg(Context& context,
                                    const clang::NamedDecl* param_decl,
                                    SemIR::InstId arg_id)
    -> std::optional<clang::TemplateArgumentLoc> {
  if (isa<clang::TemplateTypeParmDecl>(param_decl)) {
    auto type = ExprAsType(context, SemIR::LocId(arg_id), arg_id);
    if (type.type_id == SemIR::ErrorInst::TypeId) {
      return std::nullopt;
    }
    auto clang_type = MapToCppType(context, type.type_id);
    if (clang_type.isNull()) {
      context.TODO(arg_id, "unsupported type used as template argument");
      return std::nullopt;
    }
    return clang::TemplateArgumentLoc(
        clang_type,
        context.ast_context().getTrivialTypeSourceInfo(
            clang_type, GetCppLocation(context, SemIR::LocId(arg_id))));
  }

  if (isa<clang::TemplateTemplateParmDecl>(param_decl)) {
    auto inst = context.sem_ir().insts().Get(arg_id);
    if (auto template_name_type =
            context.types().TryGetAs<SemIR::CppTemplateNameType>(
                inst.type_id())) {
      clang::TemplateName name(cast<clang::TemplateDecl>(
          context.clang_decls().Get(template_name_type->decl_id).key.decl));
      return clang::TemplateArgumentLoc(
          context.ast_context(), clang::TemplateArgument(name),
          /*TemplateKWLoc=*/clang::SourceLocation(),
          clang::NestedNameSpecifierLoc(),
          GetCppLocation(context, SemIR::LocId(arg_id)));
    }

    // TODO: Eventually we should also support passing Carbon generics as
    // template template arguments.
    return MakePlaceholderTemplateArg(context, arg_id);
  }

  if (isa<clang::NonTypeTemplateParmDecl>(param_decl)) {
    // TODO: Check the argument has a concrete constant value, and convert it to
    // a Clang constant value.
    context.TODO(arg_id, "argument for non-type template parameter");
    return std::nullopt;
  }

  CARBON_FATAL("Unknown declaration kind for template parameter");
}

// Converts a call argument list into a Clang template argument list for a given
// template. Returns true on success, or false if an error was diagnosed.
static auto ConvertArgsToTemplateArgs(Context& context,
                                      clang::TemplateDecl* template_decl,
                                      llvm::ArrayRef<SemIR::InstId> arg_ids,
                                      clang::TemplateArgumentListInfo& arg_list)
    -> bool {
  for (auto* param_decl : template_decl->getTemplateParameters()->asArray()) {
    if (arg_ids.empty()) {
      return true;
    }

    // A parameter pack consumes all remaining arguments; otherwise, it consumes
    // a single argument.
    // TODO: Handle expanded template parameter packs, which have a known, fixed
    // arity.
    llvm::ArrayRef<SemIR::InstId> args_for_param =
        param_decl->isTemplateParameterPack() ? std::exchange(arg_ids, {})
                                              : arg_ids.consume_front();
    for (auto arg_id : args_for_param) {
      if (auto arg = ConvertArgToTemplateArg(context, param_decl, arg_id)) {
        arg_list.addArgument(*arg);
      } else {
        return false;
      }
    }
  }

  // If there are any remaining arguments, that's an error; convert them to
  // placeholder template arguments so that Clang will diagnose it for us.
  for (auto arg_id : arg_ids) {
    // Synthesize a placeholder `void{}` template argument.
    arg_list.addArgument(MakePlaceholderTemplateArg(context, arg_id));
  }

  return true;
}

// Given a template and an template argument list, builds a Carbon value
// describing the corresponding C++ template-id.
static auto BuildTemplateId(Context& context, SemIR::LocId loc_id,
                            clang::SourceLocation loc,
                            clang::TemplateDecl* template_decl,
                            clang::TemplateArgumentListInfo& arg_list)
    -> SemIR::InstId {
  if (auto* var_template_decl =
          dyn_cast<clang::VarTemplateDecl>(template_decl)) {
    auto decl_result = context.clang_sema().CheckVarTemplateId(
        var_template_decl, /*TemplateLoc=*/clang::SourceLocation(), loc,
        arg_list, /*SetWrittenArgs=*/false);
    return decl_result.isInvalid()
               ? SemIR::ErrorInst::InstId
               : ImportCppDecl(context, loc_id,
                               SemIR::ClangDeclKey::ForNonFunctionDecl(
                                   decl_result.get()));
  }

  if (auto* concept_decl = dyn_cast<clang::ConceptDecl>(template_decl)) {
    auto expr_result = context.clang_sema().CheckConceptTemplateId(
        clang::CXXScopeSpec(), /*TemplateKWLoc=*/clang::SourceLocation(),
        clang::DeclarationNameInfo(concept_decl->getDeclName(), loc),
        concept_decl, concept_decl, &arg_list);
    if (expr_result.isInvalid()) {
      return SemIR::ErrorInst::InstId;
    }
    auto* expr = expr_result.getAs<clang::ConceptSpecializationExpr>();
    return MakeBoolLiteral(context, loc_id,
                           SemIR::BoolValue::From(expr->isSatisfied()));
  }

  clang::TemplateName template_name(template_decl);
  auto clang_type = context.clang_sema().CheckTemplateIdType(
      clang::ElaboratedTypeKeyword::None, template_name, loc, arg_list,
      /*Scope=*/nullptr, /*ForNestedNameSpecifier=*/false);
  if (clang_type.isNull()) {
    return SemIR::ErrorInst::InstId;
  }
  return ImportCppType(context, loc_id, clang_type).inst_id;
}

auto PerformCallToCppTemplateName(Context& context, SemIR::LocId loc_id,
                                  SemIR::ClangDeclId template_decl_id,
                                  llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::InstId {
  auto* template_decl = dyn_cast<clang::TemplateDecl>(
      context.clang_decls().Get(template_decl_id).key.decl);
  auto loc = GetCppLocation(context, loc_id);

  // Form a template argument list for this template.
  clang::TemplateArgumentListInfo arg_list(loc, loc);
  if (!ConvertArgsToTemplateArgs(context, template_decl, arg_ids, arg_list)) {
    return SemIR::ErrorInst::InstId;
  }

  return BuildTemplateId(context, loc_id, loc, template_decl, arg_list);
}

}  // namespace Carbon::Check
