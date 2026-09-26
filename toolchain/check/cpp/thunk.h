// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_THUNK_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_THUNK_H_

#include "clang/AST/DeclarationName.h"
#include "clang/AST/TypeBase.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Information about a C++ callee in the process of being imported. One of
// this class's key responsibilities is to track the mapping between several
// different calling conventions:
// - The native C++ function, which may or may not have an object parameter,
//   and that object parameter may or may not be present in the parameter list
//   (but is never present in the corresponding argument list).
// - The thunk that wraps it with a simple ABI, so that it can be called from
//   Carbon.
// - The Carbon function that is actually called by user code, and maps its
//   parameters to the simple ABI.
struct CalleeFunctionInfo {
  // Constructs a CalleeFunctionInfo that represents the given C++ function with
  // the given signature.
  explicit CalleeFunctionInfo(Context& context, clang::FunctionDecl* decl,
                              SemIR::ClangDeclSignatureId signature_id);

  // Returns the offset I such that callee parameter N corresponds to
  // parameter N+I of the imported Carbon function.
  auto callee_param_to_carbon_param_offset() const -> int {
    switch (self_param_kind) {
      case SelfParamKind::ImplicitObjectParam:
        return 1;
      case SelfParamKind::None:
      case SelfParamKind::ExplicitObjectParam:
        return 0;
    }
  }

  // Returns the offset I such that callee argument N corresponds to callee
  // parameter N+I.
  auto callee_arg_to_callee_param_offset() const -> int {
    switch (self_param_kind) {
      case SelfParamKind::ExplicitObjectParam:
        return 1;
      case SelfParamKind::ImplicitObjectParam:
      case SelfParamKind::None:
        return 0;
    }
  }

  // Returns the number of parameters that the imported Carbon function should
  // have.
  auto num_carbon_params() const -> int {
    return num_callee_params + callee_param_to_carbon_param_offset();
  }

  // Returns the number of parameters the simple-ABI thunk should have.
  auto num_thunk_params() const -> unsigned {
    return num_carbon_params() + !has_simple_return_type;
  }

  // Returns true if the imported Carbon function should have a self parameter.
  auto carbon_has_self_param() const -> bool {
    return self_param_kind != SelfParamKind::None;
  }

  // Returns true if the callee has an object parameter (i.e. `this`).
  auto callee_has_object_param() const -> bool {
    return self_param_kind == SelfParamKind::ImplicitObjectParam ||
           self_param_kind == SelfParamKind::ExplicitObjectParam;
  }

  // Returns the identifier for the i-th callee parameter, or null if it doesn't
  // have one.
  auto GetCalleeParamIdentifier(int i) const -> clang::IdentifierInfo*;

  // Returns the location of the i-th callee parameter declaration.
  auto GetCalleeParamLocation(int i) const -> clang::SourceLocation;

  // Information about the C++ parameter that corresponds to the `self`
  // parameter in the Carbon function.
  enum class SelfParamKind {
    // The C++ callee is an ordinary function, so it has no `self` parameter in
    // Carbon. The Nth Carbon parameter corresponds to the Nth callee argument,
    // and to the Nth callee parameter.
    None,
    // The callee is a traditional C++ method, so the Carbon `self` parameter
    // corresponds to the C++ object parameter, which is not present in
    // the callee parameter or argument list, but is passed at the callsite
    // using member access syntax. Consequently, the Nth Carbon parameter
    // corresponds to the N-1th callee argument, and the N-1th callee parameter.
    ImplicitObjectParam,
    // The callee is a C++ method with an explicit object parameter, such as
    // `F(this Foo& self)`, so the first Carbon parameter corresponds to the
    // object parameter, which is present in the callee parameter list but
    // passed at the callsite using member access syntax rather than the
    // argument list. Consequently, the Nth Carbon parameter corresponds to the
    // N-1th callee argument, and the Nth callee parameter.
    ExplicitObjectParam,
  };
  SelfParamKind self_param_kind;

  // The callee function's declaration, or null if it doesn't have one.
  clang::FunctionDecl* decl;

  // The name of the callee function.
  clang::DeclarationName decl_name;

  // The location of the callee function declaration.
  clang::SourceLocation clang_loc;

  // The SemIR representation of `clang_loc`.
  SemIR::LocId sem_ir_loc;

  // The function type of the callee. This is never null.
  const clang::FunctionProtoType* function_type;

  // The ID of `signature`.
  SemIR::ClangDeclSignatureId signature_id;

  // The signature of the function being imported.
  const SemIR::ClangDeclSignature* signature;

  // The number of explicit parameters to import. This may be less than the
  // number of parameters that the function has if default arguments are being
  // used.
  int num_callee_params;

  // The type of the callee parameter that is treated as `self` in the Carbon
  // function, or null if there isn't one.
  clang::QualType self_param_type;

  // The return type that the callee has when viewed from Carbon. This is the
  // C++ return type, except that constructors return the class type in Carbon
  // and return void in Clang's AST.
  clang::QualType effective_return_type;

  // Whether the callee has a simple return type, that we can return directly.
  // If not, we'll return through an out parameter instead.
  bool has_simple_return_type;
};

// Returns whether the given C++ imported function requires a C++ thunk to be
// used to call it. A C++ thunk is required for functions whose ABI uses any
// type except void, pointer and reference types, and signed 32-bit and 64-bit
// integers.
auto IsCppThunkRequired(Context& context, const CalleeFunctionInfo& callee_info)
    -> bool;

// Builds a C++ thunk with simple ABI (pointers, i32 and i64 types) that calls
// the specified callee. Assumes `IsCppThunkRequired()` return true for
// `callee_info`. Returns `nullptr` on failure.
auto BuildCppThunk(Context& context, const CalleeFunctionInfo& callee_info)
    -> clang::FunctionDecl*;

// Builds a call to a thunk function that forwards a call argument list built
// for `callee_function_id` to a call to `thunk_callee_id`, for use when
// building a call from a C++ thunk to its target. This is like `PerformCall`,
// except that it takes a list of call arguments for `callee_function_id`, not a
// syntactic argument list.
auto PerformCppThunkCall(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId callee_function_id,
                         llvm::ArrayRef<SemIR::InstId> callee_arg_ids,
                         SemIR::InstId thunk_callee_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_THUNK_H_
