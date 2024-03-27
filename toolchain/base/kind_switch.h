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
namespace Carbon::Internal {

// Given `CARBON_KIND_SWITCH(value)` this handles calling `value.kind()`.
template <typename T>
auto KindSwitch(T&& switch_value) -> auto {
  return switch_value.kind();
};

// Given `CARBON_KIND(CaseT name)` this generates `CaseT::Kind`.
template <typename FnT, typename KindT>
constexpr auto KindCaseValue() -> KindT {
  using ArgT = llvm::function_traits<FnT>::template arg_t<0>;
  return ArgT::Kind;
}

// Given `CARBON_KIND_SWITCH(value)` and `CARBON_KIND(CaseT name)` this
// generates `value.As<CaseT>()`.
template <typename FnT, typename ValueT>
auto KindCaseAs(ValueT&& kind_switch_value) -> auto {
  using CaseT = llvm::function_traits<FnT>::template arg_t<0>;
  return kind_switch_value.template As<CaseT>();
}

#define CARBON_KIND_MERGE_(Prefix, Line) Prefix##Line
#define CARBON_KIND_LABEL_(Line) CARBON_KIND_MERGE_(__carbon_kind_case_, Line)

}  // namespace Carbon::Internal

// Produces a switch statement on value.kind().
#define CARBON_KIND_SWITCH(value)                    \
  switch (auto&& __carbon_kind_switch_value = value; \
          auto __carbon_kind_switch_kind =           \
              ::Carbon::Internal::KindSwitch(__carbon_kind_switch_value))

// Produces a case-compatible block of code that also instantiates a local typed
// variable.
//
// This uses `if` to scope the variable, and provides a dangling `else` in order
// to prevent accidental `else` use. The label allows `:` to follow the macro
// name, making it look more like a typical `case`.
//
// NOLINTBEGIN(bugprone-macro-parentheses)
#define CARBON_KIND(type_and_name)                                \
  ::Carbon::Internal::KindCaseValue<                              \
      decltype([]([[maybe_unused]] type_and_name) {}),            \
      decltype(__carbon_kind_switch_kind)>()                      \
      : if (type_and_name = ::Carbon::Internal::KindCaseAs<       \
                decltype([]([[maybe_unused]] type_and_name) {})>( \
                __carbon_kind_switch_value);                      \
            false) {}                                             \
  else [[maybe_unused]] CARBON_KIND_LABEL_(__LINE__)
// NOLINTEND(bugprone-macro-parentheses)

#endif  // CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_
