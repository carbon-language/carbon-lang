// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_
#define CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_

#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "toolchain/base/for_each_macro.h"

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

template <typename T>
static constexpr bool IsStdVariantValue = false;

template <typename... Ts>
static constexpr bool IsStdVariantValue<std::variant<Ts...>> = true;

template <typename T>
concept IsStdVariant = IsStdVariantValue<std::decay_t<T>>;

template <typename... Ts>
struct TypePack {};

template <typename T>
struct StdVariantTypePackValue;

template <typename... Ts>
struct StdVariantTypePackValue<std::variant<Ts...>> {
  using Types = TypePack<Ts...>;
};

template <typename T>
using StdVariantTypePack = StdVariantTypePackValue<std::decay_t<T>>::Types;

template <typename T>
struct StdVariantEnumValue;

template <typename... Ts>
struct StdVariantEnumValue<std::variant<Ts...>> {
  static_assert(sizeof...(Ts) <= 12,
                "CARBON_KIND_SWITCH supports std::variant with up to 12 types. "
                "Add more if needed.");
};

#define CARBON_INTERNAL_MAKE_ENUM_VALUE(n) VariantType##n##NotHandledInSwitch
#define CARBON_INTERNAL_MAKE_TYPENAME(name) typename name

#define CARBON_INTERNAL_MAKE_ENUM(...)                                        \
  template <CARBON_FOR_EACH(CARBON_INTERNAL_MAKE_TYPENAME,                    \
                            CARBON_FOR_EACH_COMMA, __VA_ARGS__)>              \
  struct StdVariantEnumValue<std::variant<__VA_ARGS__>> {                     \
    enum EnumType {                                                           \
      CARBON_FOR_EACH(CARBON_INTERNAL_MAKE_ENUM_VALUE, CARBON_FOR_EACH_COMMA, \
                      __VA_ARGS__)                                            \
    };                                                                        \
    using Type = EnumType;                                                    \
  }

CARBON_INTERNAL_MAKE_ENUM(T0);
CARBON_INTERNAL_MAKE_ENUM(T0, T1);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3, T4);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3, T4, T5);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3, T4, T5, T6);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3, T4, T5, T6, T7);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3, T4, T5, T6, T7, T8);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
CARBON_INTERNAL_MAKE_ENUM(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

#undef CARBON_INTERNAL_MAKE_ENUM_VALUE
#undef CARBON_INTERNAL_MAKE_TYPENAME
#undef CARBON_INTERNAL_MAKE_ENUM

template <typename T>
using StdVariantEnum = StdVariantEnumValue<std::decay_t<T>>::Type;

// Given `CARBON_KIND_SWITCH(value)` this returns `value.kind()` to switch on.
template <typename SwitchT>
constexpr auto SwitchOn(SwitchT&& switch_value) -> auto {
  if constexpr (IsStdVariant<SwitchT>) {
    return static_cast<StdVariantEnum<SwitchT>>(switch_value.index());
  } else {
    return switch_value.kind();
  }
}

template <class T>
concept TypeFoundInVariant = false;

template <class T>
  requires TypeFoundInVariant<T>
struct ValidCaseType;

template <size_t I, typename T, typename TypePack>
struct IndexOfTypeValue {
  // Error case when `T` is not found in the std::variant<...> types.
  ValidCaseType<T> Error;
};

template <size_t I, typename T, typename... Ts>
struct IndexOfTypeValue<I, T, TypePack<T, Ts...>> {
  static constexpr size_t Index = I;
};

template <size_t I, typename T, typename U, typename... Ts>
struct IndexOfTypeValue<I, T, TypePack<U, Ts...>> {
  static constexpr size_t Index =
      IndexOfTypeValue<I + 1, T, TypePack<Ts...>>::Index;
};

template <typename T, typename TypePack>
static constexpr size_t IndexOfType = IndexOfTypeValue<0, T, TypePack>::Index;

// Given `CARBON_KIND(CaseT name)` this generates `CaseT::Kind`. It explicitly
// returns `KindT` because that may differ from `CaseT::Kind`, and may not be
// copyable.
template <typename SwitchT, typename CaseFnT>
consteval auto ForCase() -> auto {
  using CaseT = llvm::function_traits<CaseFnT>::template arg_t<0>;
  if constexpr (IsStdVariant<SwitchT>) {
    return IndexOfType<CaseT, StdVariantTypePack<SwitchT>>;
  } else {
    using KindT = llvm::function_traits<
        decltype(&std::remove_cvref_t<SwitchT>::kind)>::result_t;
    return static_cast<KindT::RawEnumType>(KindT::template For<CaseT>);
  }
}

// Given `CARBON_KIND_SWITCH(value)` and `CARBON_KIND(CaseT name)` this
// generates `value.As<CaseT>()`.
template <typename CaseFnT, typename SwitchT>
auto Cast(SwitchT&& kind_switch_value) -> decltype(auto) {
  using CaseT = llvm::function_traits<CaseFnT>::template arg_t<0>;
  if constexpr (IsStdVariant<SwitchT>) {
    return std::get<CaseT>(kind_switch_value);
  } else {
    return kind_switch_value.template As<CaseT>();
  }
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

// Like `CARBON_KIND` but works with a type without a name, and does not make
// the switch's value available in the case body as a result.
#define CARBON_KIND_(type_without_name)            \
  ::Carbon::Internal::Kind::ForCase<               \
      decltype(carbon_internal_kind_switch_value), \
      decltype([]([[maybe_unused]] type_without_name) {})>()

#endif  // CARBON_TOOLCHAIN_BASE_KIND_SWITCH_H_
