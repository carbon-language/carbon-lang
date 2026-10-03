// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_THUNK_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_THUNK_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace clang {
class Decl;
class DeclarationName;
class FunctionDecl;
class FunctionProtoType;
class QualType;
class SourceLocation;
}  // namespace clang

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
class CalleeFunctionInfo {
 public:
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
    // The callee is the notional `__invoke` method of a C++ function pointer,
    // which is treated has having the pointer value as its implicit object
    // parameter (see the constructor comments for details). Consequently
    // the Nth thunk parameter corresponds to the N-1th callee argument and the
    // N-1th callee parameter.
    FunctionPointer,
  };

  // Constructs a CalleeFunctionInfo that represents the given C++ function with
  // the given signature.
  explicit CalleeFunctionInfo(Context* context, clang::FunctionDecl* decl,
                              SemIR::ClangDeclSignatureId signature_id);

  // Constructs a CalleeFunctionInfo that represents a C++ function pointer.
  // We treat function pointer types as having an `__invoke` method, with the
  // pointer value acting as the implicit object parameter. On the C++ side this
  // method is notional, and has no declaration: in effect, it is inlined into
  // its own simple-ABI thunk (which always exists, even if the parameter and
  // return types already have simple ABIs). The Carbon counterpart of this
  // method is real, however, and takes the pointer value as its `self`
  // parameter.
  explicit CalleeFunctionInfo(Context* context,
                              const clang::Type* function_pointer_type);

  // The declaration of the callee, or nullptr if this represents a function
  // pointer.
  auto decl() const -> clang::FunctionDecl* {
    return decl_or_pointer_type_.dyn_cast<clang::FunctionDecl*>();
  }

  // The type of the callee function. Never null.
  auto function_type() const -> const clang::FunctionProtoType* {
    return function_type_;
  }

  // Metadata about the C++ function signature.
  auto signature() const -> const SemIR::ClangDeclSignature* {
    return signature_;
  }

  // The name of the callee function. Never empty.
  auto decl_name() const -> clang::DeclarationName;

  // The location of the callee function declaration, or an invalid location
  // if there was no declaration.
  auto clang_location() const -> clang::SourceLocation;

  // The SemIR representation of `clang_location()`.
  auto sem_ir_loc() const -> SemIR::LocId { return sem_ir_loc_; }

  // The number of parameters in the C++ callee that should be imported.
  // This may be less than the number of parameters that the function has if
  // default arguments are being used.
  auto num_callee_params() const -> int { return num_callee_params_; }

  // The number of parameters that the imported Carbon function should have.
  auto num_carbon_params() const -> int {
    return num_callee_params_ + callee_param_to_carbon_param_offset();
  }

  // The number of parameters that the simple-ABI thunk should have.
  auto num_thunk_params() const -> unsigned {
    return num_carbon_params() + !has_simple_return_type_;
  }

  // The offset I such that callee parameter N corresponds to parameter N+I of
  // the imported Carbon function.
  auto callee_param_to_carbon_param_offset() const -> int {
    switch (self_param_kind_) {
      case SelfParamKind::ImplicitObjectParam:
      case SelfParamKind::FunctionPointer:
        return 1;
      case SelfParamKind::None:
      case SelfParamKind::ExplicitObjectParam:
        return 0;
    }
  }

  // The offset I such that callee argument N corresponds to callee parameter
  // N+I.
  auto callee_arg_to_callee_param_offset() const -> int {
    switch (self_param_kind_) {
      case SelfParamKind::ExplicitObjectParam:
        return 1;
      case SelfParamKind::ImplicitObjectParam:
      case SelfParamKind::FunctionPointer:
      case SelfParamKind::None:
        return 0;
    }
  }

  // Returns the declared name of the i-th callee parameter, or null if it
  // doesn't have a declaration.
  auto GetCalleeParamIdentifier(int i) const -> clang::IdentifierInfo*;

  // Returns the location of the i-th callee parameter declaration, or an
  // invalid location if it doesn't have a declaration.
  auto GetCalleeParamLocation(int i) const -> clang::SourceLocation;

  // Whether the callee has an object parameter (i.e. `this`).
  auto callee_has_object_param() const -> bool {
    return self_param_kind_ == SelfParamKind::ImplicitObjectParam ||
           self_param_kind_ == SelfParamKind::ExplicitObjectParam;
  }

  // The kind of C++ parameter that corresponds to the Carbon `self` parameter.
  auto self_param_kind() const -> SelfParamKind { return self_param_kind_; }

  // Whether the imported Carbon function should have a self parameter.
  auto carbon_has_self_param() const -> bool {
    return self_param_kind_ != SelfParamKind::None;
  }

  // The type of the callee parameter that is treated as `self` in the Carbon
  // function, or null if there isn't one.
  auto self_param_type() const -> clang::QualType;

  // The return type that the callee has when viewed from Carbon. This is the
  // C++ return type, except that constructors return the class type in Carbon
  // and return void in Clang's AST.
  auto effective_return_type() const -> clang::QualType;

  // Whether the callee's effective return type is simple, such that we can
  // return it directly rather than using an out parameter.
  auto has_simple_return_type() const -> bool {
    return has_simple_return_type_;
  }

 private:
  Context* context_;
  SelfParamKind self_param_kind_;

  // If `self_param_kind_ == FunctionPointer`, this is the type of the
  // function pointer. Otherwise it is the declaration of the callee function.
  // Never null.
  llvm::PointerUnion<clang::FunctionDecl*, const clang::Type*>
      decl_or_pointer_type_;

  SemIR::LocId sem_ir_loc_;
  const clang::FunctionProtoType* function_type_;
  SemIR::ClangDeclSignatureId signature_id_;
  const SemIR::ClangDeclSignature* signature_;
  int num_callee_params_;
  bool has_simple_return_type_;
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
