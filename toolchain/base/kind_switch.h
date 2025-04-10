// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_
#define CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_

#include <type_traits>

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
//   `CaseT::Kind::RawEnumType` is the type returned by `.kind()`.
//
// Note, this is currently used primarily for Inst in toolchain. When more
// use-cases are added, it would be worth considering whether the API
// requirements should change.
namespace Carbon::Internal::Kind {

// Given `CARBON_KIND_SWITCH(value)` this returns `value.kind()` to switch on.
template <typename SwitchT>
auto SwitchOn(SwitchT&& switch_value) -> auto {
  return switch_value.kind();
}

// Given `CARBON_KIND(CaseT name)` this generates `CaseT::Kind`. It explicitly
// returns `KindT` because that may differ from `CaseT::Kind`, and may not be
// copyable.
template <typename SwitchT, typename CaseFnT>
consteval auto ForCase() -> auto {
  using KindT = llvm::function_traits<
      decltype(&std::remove_cvref_t<SwitchT>::kind)>::result_t;
  using CaseT = llvm::function_traits<CaseFnT>::template arg_t<0>;
  return static_cast<KindT::RawEnumType>(KindT::template For<CaseT>);
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
#define CARBON_KIND_SWITCH(value)                            \
  switch (                                                   \
      const auto& carbon_internal_kind_switch_value = value; \
      ::Carbon::Internal::Kind::SwitchOn(carbon_internal_kind_switch_value))

// Produces a case-compatible block of code that also instantiates a local typed
// variable. typed_variable_decl looks like `int i`, with a space.
//
// This uses `if` to scope the variable, and provides a dangling `else` in order
// to prevent accidental `else` use. The label allows `:` to follow the macro
// name, making it look more like a typical `case`.
#define CARBON_KIND(typed_variable_decl)                                \
  ::Carbon::Internal::Kind::ForCase<                                    \
      decltype(carbon_internal_kind_switch_value),                      \
      decltype([]([[maybe_unused]] typed_variable_decl) {})>()          \
      : if (typed_variable_decl = ::Carbon::Internal::Kind::Cast<       \
                decltype([]([[maybe_unused]] typed_variable_decl) {})>( \
                carbon_internal_kind_switch_value);                     \
            false) {}                                                   \
  else [[maybe_unused]] CARBON_INTERNAL_KIND_LABEL(__LINE__)

#endif  // CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_
