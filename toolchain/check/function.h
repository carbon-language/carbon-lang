// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
#define CARBON_TOOLCHAIN_CHECK_FUNCTION_H_

#include "toolchain/check/context.h"
#include "toolchain/check/custom_witness.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Returns the ID of the self parameter pattern, or None.
// TODO: Do this during initial traversal of implicit params.
auto FindSelfPattern(Context& context,
                     SemIR::InstBlockId implicit_param_patterns_id)
    -> SemIR::InstId;

// Creates suitable return patterns for the given return form, and adds them to
// the current pattern block.
auto AddReturnPattern(Context& context, SemIR::LocId loc_id,
                      Context::FormExpr form_expr) -> SemIR::InstId;

// Returns whether `function` is a valid declaration of `builtin_kind`.
auto IsValidBuiltinDeclaration(Context& context,
                               const SemIR::Function& function,
                               SemIR::BuiltinFunctionKind builtin_kind) -> bool;

// Arguments for making a function declaration.
struct FunctionDeclArgs {
  SemIR::NameScopeId parent_scope_id;
  SemIR::NameId name_id;
  // The type of the implicit `[self: Self]` parameter, or `None` if there is
  // none.
  SemIR::TypeId self_type_id = SemIR::TypeId::None;
  // The kind of the `self` parameter.
  ParamPatternKind self_kind = ParamPatternKind::Ref;
  // The types of the explicit parameters.
  llvm::ArrayRef<SemIR::TypeId> param_type_ids = {};
  // The kind of the parameters described by `param_type_ids`.
  ParamPatternKind param_kind = ParamPatternKind::Value;
  // The return type, or `None` if the function doesn't declare a return type.
  SemIR::TypeId return_type_id = SemIR::TypeId::None;
};

// Generates and returns a function declaration. The caller should update the
// function object to add a definition. The caller is responsible for ensuring
// that the signature is non-generic.
auto MakeGeneratedFunctionDecl(Context& context, SemIR::LocId loc_id,
                               const FunctionDeclArgs& args)
    -> std::pair<SemIR::InstId, SemIR::FunctionId>;

// Checks that `new_function` has the same return type as `prev_function`, or if
// `prev_function_id` is specified, a specific version of `prev_function`.
// Prints a suitable diagnostic and returns false if not. Never checks for a
// syntactic match.
auto CheckFunctionReturnTypeMatches(Context& context,
                                    const SemIR::Function& new_function,
                                    const SemIR::Function& prev_function,
                                    SemIR::SpecificId prev_specific_id,
                                    bool diagnose = true) -> bool;

// Checks that `new_function` has the same parameter types and return type as
// `prev_function`, or if `prev_function_id` is specified, a specific version of
// `prev_function`. Prints a suitable diagnostic and returns false if not.
//
// `check_syntax` is false if the redeclaration can be called via a thunk with
// implicit conversions from the original declaration.
// `check_self` is false if the self declaration does not have to match (for
// instance in impls of virtual functions).
auto CheckFunctionTypeMatches(Context& context,
                              const SemIR::Function& new_function,
                              const SemIR::Function& prev_function,
                              SemIR::SpecificId prev_specific_id,
                              bool check_syntax, bool check_self,
                              bool diagnose = true) -> bool;

inline auto CheckFunctionTypeMatches(Context& context,
                                     const SemIR::Function& new_function,
                                     const SemIR::Function& prev_function)
    -> bool {
  return CheckFunctionTypeMatches(context, new_function, prev_function,
                                  SemIR::SpecificId::None,
                                  /*check_syntax=*/true, /*check_self=*/true);
}

// Checks that the scrutinee type of `return_pattern_id` in `specific_id` is
// concrete. If so, it returns that type; if not, it issues an error and returns
// SemIR::ErrorInst::TypeId. `return_pattern_id` must be part of a function's
// return form, or the error message will be nonsensical.
auto CheckFunctionReturnPatternType(Context& context, SemIR::LocId loc_id,
                                    SemIR::InstId return_pattern_id,
                                    SemIR::SpecificId specific_id)
    -> SemIR::TypeId;

// Checks that a function declaration's signature is suitable to support a
// function definition. This requires the parameter types to be complete and the
// return type to be concrete.
auto CheckFunctionDefinitionSignature(Context& context,
                                      SemIR::FunctionId function_id) -> void;

// Prepares for a function signature. Handles necessary stack setup. This is
// used for generated functions/thunks, not user-declared functions.
auto StartFunctionSignature(Context& context) -> void;

// Results for `FinishFunctionSignature`.
struct FinishFunctionSignatureResult {
  SemIR::InstBlockId pattern_block_id;
  SemIR::InstBlockId decl_block_id;
};

// Finishes signatures started by `StartFunctionSignature`.
auto FinishFunctionSignature(Context& context, bool check_unused = true)
    -> FinishFunctionSignatureResult;

// Creates a function object for the given function declaration. The caller must
// add the returned `decl_id` to a block (typically the current block or
// imports).
auto MakeFunctionDecl(Context& context, SemIR::LocId loc_id,
                      SemIR::InstBlockId decl_block_id, bool build_generic,
                      bool is_definition, SemIR::Function function)
    -> std::pair<SemIR::InstId, SemIR::FunctionId>;

// Starts a function definition. Handles necessary stack setup, creating the
// function scope and entry block, and definition validation. This is used for
// both generated functions/thunks and user-declared functions.
auto StartFunctionDefinition(Context& context, SemIR::InstId decl_id,
                             SemIR::FunctionId function_id) -> void;

// Finishes definitions started by `StartFunctionDefinition`.
auto FinishFunctionDefinition(Context& context, SemIR::FunctionId function_id)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
