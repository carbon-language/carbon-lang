// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_
#define CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_

#include <cstdint>

#include "common/ostream.h"

// Provides an enum type with names for values.
//
// Uses should look like:
//
// namespace Internal {
// enum class MyEnum : uint8_t {
// #define CARBON_MY_ENUM(Name) CARBON_ENUM_BASE_LITERAL(Name)
// #include "my_enum.def"
// };
// }  // namespace Internal
//
// class MyEnum : public EnumBase<MyEnum, Internal::MyEnum> {
//  public:
// #define CARBON_MY_ENUM(Name) CARBON_ENUM_BASE_FACTORY(MyEnum, Name)
// #include "my_enum.def"
//
//   // Gets a friendly name for the token for logging or debugging.
//   [[nodiscard]] inline auto name() const -> llvm::StringRef {
//     static constexpr llvm::StringLiteral Names[] = {
// #define CARBON_MY_ENUM(Name) CARBON_ENUM_BASE_STRING(Name)
// #include "my_enum.def"
//     };
//     return Names[static_cast<int>(val_)];
//   }
//
//  private:
//   using EnumBase::EnumBase;
// };
//
// They will be usable in a switch statement, e.g. `case MyEnum::Name():`.
//
// Uses CRTP to provide the Print function.
template <typename DerivedT, typename EnumT>
class EnumBase {
 protected:
  using InternalEnum = EnumT;

 public:
  // The default constructor is deleted because objects of this type should
  // always be constructed using the above factory functions for each unique
  // kind.
  EnumBase() = delete;

  // Enable conversion to our private enum, including in a `constexpr`
  // context, to enable usage in `switch` and `case`. The enum remains
  // private and nothing else should be using this function.
  //
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator InternalEnum() const { return val_; }

  void Print(llvm::raw_ostream& out) const {
    out << reinterpret_cast<const DerivedT*>(this)->name();
  }

 protected:
  constexpr explicit EnumBase(InternalEnum val) : val_(val) {}

  InternalEnum val_;
};

// In CARBON_ENUM_BASE, combines with X_NAMES to generate `enum class` values.
#define CARBON_ENUM_BASE_LITERAL(Name) Name,

// In CARBON_ENUM_BASE, combines with X_NAMES to generate `MyEnum::Name()`
// factory functions.
//
// `clang-format` has a bug with spacing around `->` returns in macros.
// See https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_ENUM_BASE_FACTORY(ClassName, Name) \
  static constexpr auto Name()->ClassName {       \
    return ClassName(InternalEnum::Name);         \
  }

// In CARBON_ENUM_BASE, combines with X_NAMES to generate strings for `name()`.
#define CARBON_ENUM_BASE_STRING(Name) #Name,

#endif  // CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_
