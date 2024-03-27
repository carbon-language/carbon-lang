// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_
#define CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_

#include "llvm/ADT/STLExtras.h"

// This library provides switch-like behaviors for Carbon's kind-based types.
//
// An expected use case is to mix regular switch `case` statements and
// `CARBON_KIND`. However, the `switch` must be defined using
// `CARBON_KIND_SWITCH`. For example:
//
//   CARBON_KIND_SWITCH(untyped_inst) {
//     case CARBON_KIND(SomeInstType inst): {
//       return inst.typed_field;
//     }
//     case OtherType1::Kind:
//     case OtherType2::Kind:
//       return value;
//     default:
//       return default_value;
//   }
//
// For compatibility, this requires:
//
// - The type passed to `CARBON_KIND_SWITCH` has `.kind()` to switch on, and
//   `.As<CaseT>` for `CARBON_KIND` to cast to.
// - Each type passed to `CARBON_KIND` (`CaseT` above) provides
//   `CaseT::Kind`, which is passed to the `case` keyword.
//
// Note, this is currently used primarily for Inst in toolchain. When more
// use-cases are added, it would be worth considering whether the API
// requirements should change.
namespace Carbon::Internal::Kind {

// Given `CARBON_KIND_SWITCH(value)` this handles calling `value.kind()`.
template <typename T>
auto SwitchOn(T&& switch_value) -> auto {
  return switch_value.kind();
}

// Given `CARBON_KIND(CaseT name)` this generates `CaseT::Kind`. It explicitly
// returns `KindT` because that may differ from `CaseT::Kind`, and may not be
// copyable.
template <typename FnT, typename KindT>
consteval auto ForCase() -> KindT {
  using ArgT = llvm::function_traits<FnT>::template arg_t<0>;
  return ArgT::Kind;
}

// Given `CARBON_KIND_SWITCH(value)` and `CARBON_KIND(CaseT name)` this
// generates `value.As<CaseT>()`.
template <typename FnT, typename ValueT>
auto Cast(ValueT&& kind_switch_value) -> auto {
  using CaseT = llvm::function_traits<FnT>::template arg_t<0>;
  return kind_switch_value.template As<CaseT>();
}

#define CARBON_INTERNAL_KIND_MERGE(Prefix, Line) Prefix##Line
#define CARBON_INTERNAL_KIND_LABEL(Line) \
  CARBON_INTERNAL_KIND_MERGE(carbon_internal_kind_case_, Line)

}  // namespace Carbon::Internal::Kind

// Produces a switch statement on value.kind().
#define CARBON_KIND_SWITCH(value)                                \
  switch (const auto& carbon_internal_kind_switch_value = value; \
          auto carbon_internal_kind_switch_kind =                \
              ::Carbon::Internal::Kind::SwitchOn(                \
                  carbon_internal_kind_switch_value))

// Produces a case-compatible block of code that also instantiates a local typed
// variable. typed_variable_declaration looks like `int i`, with a space.
//
// This uses `if` to scope the variable, and provides a dangling `else` in order
// to prevent accidental `else` use. The label allows `:` to follow the macro
// name, making it look more like a typical `case`.
//
// NOLINTBEGIN(bugprone-macro-parentheses)
#define CARBON_KIND(typed_variable_declaration)                                \
  ::Carbon::Internal::Kind::ForCase<                                           \
      decltype([]([[maybe_unused]] typed_variable_declaration) {}),            \
      decltype(carbon_internal_kind_switch_kind)>()                            \
      : if (typed_variable_declaration = ::Carbon::Internal::Kind::Cast<       \
                decltype([]([[maybe_unused]] typed_variable_declaration) {})>( \
                carbon_internal_kind_switch_value);                            \
            false) {}                                                          \
  else [[maybe_unused]] CARBON_INTERNAL_KIND_LABEL(__LINE__)
// NOLINTEND(bugprone-macro-parentheses)

#endif  // CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_
