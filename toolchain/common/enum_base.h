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
//   #define CARBON_MY_ENUM(X) \
//     X(FirstValue) \
//     X(SecondValue) \
//     ...
//
//   CARBON_ENUM_BASE(MyEnumBase, CARBON_MY_ENUM)
//
//   class MyEnum : public MyEnumBase<MyEnum> {
//     using MyEnumBase::MyEnumBase;
//   };
//
// Specific macro values are available as:
//   MyEnum::Name()
//
// They will be usable in a switch statement, e.g. `case MyEnum::Name():`.
#define CARBON_ENUM_BASE(EnumBaseName, X_NAMES)                                \
  /* Uses CRTP to provide factory functions which create the derived enum. */  \
  template <typename DerivedEnumT>                                             \
  class EnumBaseName {                                                         \
   protected:                                                                  \
    /* The enum must be declared earlier in the class so that its type can be  \
     * used, for example in the conversion operator.                           \
     */                                                                        \
    enum class InternalEnum : uint8_t {                                        \
      X_NAMES(CARBON_ENUM_BASE_INTERNAL_ENUM_ENTRY)                            \
    };                                                                         \
                                                                               \
   public:                                                                     \
    /* Defines factory functions for each enum name.                           \
     *`clang-format` has a bug with spacing around `->` returns in macros. See \
     * https://bugs.llvm.org/show_bug.cgi?id=48320 for details.                \
     */                                                                        \
    X_NAMES(CARBON_ENUM_BASE_INTERNAL_FACTORY)                                 \
                                                                               \
    /* The default constructor is deleted because objects of this type should  \
     * always be constructed using the above factory functions for each unique \
     * kind.                                                                   \
     */                                                                        \
    EnumBaseName() = delete;                                                   \
                                                                               \
    /* Gets a friendly name for the token for logging or debugging. */         \
    [[nodiscard]] inline auto name() const -> llvm::StringRef {                \
      static constexpr llvm::StringLiteral Names[] = {                         \
          X_NAMES(CARBON_ENUM_BASE_INTERNAL_NAMES)};                           \
      return Names[static_cast<int>(val_)];                                    \
    }                                                                          \
                                                                               \
    /* Enable conversion to our private enum, including in a `constexpr`       \
     * context, to enable usage in `switch` and `case`. The enum remains       \
     * private and nothing else should be using this function.                 \
     */                                                                        \
    /* NOLINTNEXTLINE(google-explicit-constructor) */                          \
    constexpr operator InternalEnum() const { return val_; }                   \
                                                                               \
    void Print(llvm::raw_ostream& out) const { out << name(); }                \
                                                                               \
   protected:                                                                  \
    constexpr explicit EnumBaseName(InternalEnum val) : val_(val) {}           \
                                                                               \
    InternalEnum val_;                                                         \
  };

// In CARBON_ENUM_BASE, combines with X_NAMES to generate `enum class` values.
#define CARBON_ENUM_BASE_INTERNAL_ENUM_ENTRY(Name) Name,

// In CARBON_ENUM_BASE, combines with X_NAMES to generate `MyEnum::Name()`
// factory functions.
#define CARBON_ENUM_BASE_INTERNAL_FACTORY(Name) \
  static constexpr auto Name()->DerivedEnumT {  \
    return DerivedEnumT(InternalEnum::Name);    \
  }

// In CARBON_ENUM_BASE, combines with X_NAMES to generate strings for `name()`.
#define CARBON_ENUM_BASE_INTERNAL_NAMES(Name) #Name,

#endif  // CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_
