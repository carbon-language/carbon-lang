// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
#define CARBON_TOOLCHAIN_CHECK_FUNCTION_H_

#include "toolchain/check/context.h"
#include "toolchain/check/decl_name_stack.h"
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
auto AddReturnPatterns(Context& context, SemIR::LocId loc_id,
                       Context::FormExpr form_expr) -> SemIR::InstBlockId;

// Returns whether `function` is a valid declaration of `builtin_kind`.
auto IsValidBuiltinDeclaration(Context& context,
                               const SemIR::Function& function,
                               SemIR::BuiltinFunctionKind builtin_kind) -> bool;

// The signature to declare for a builtin function.
struct BuiltinFunctionSignature {
  // The type of the implicit `[self: Self]` parameter, or `None` if there is
  // none.
  SemIR::TypeId self_type_id = SemIR::TypeId::None;
  // Whether `self` is a ref parameter.
  bool self_is_ref = true;
  // The types of the explicit parameters.
  llvm::ArrayRef<SemIR::TypeId> param_type_ids = {};
  // The return type, or `None` if the function doesn't declare a return type.
  SemIR::TypeId return_type_id = SemIR::TypeId::None;
};

// Creates and returns a new builtin function declaration.
//
// TODO: Instead of synthesizing builtin function declarations, we should
// ideally declare the builtin functions in Carbon code instead.
auto MakeBuiltinFunction(Context& context, SemIR::LocId loc_id,
                         SemIR::BuiltinFunctionKind builtin_kind,
                         SemIR::NameScopeId name_scope_id,
                         SemIR::NameId name_id,
                         BuiltinFunctionSignature signature) -> SemIR::InstId;

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

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_FUNCTION_H_
