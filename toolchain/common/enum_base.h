// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Provides an enum type with names for values.
//
// Uses should look like:
//   #define CARBON_ENUM_BASE_NAME MyEnumBase
//   #define CARBON_ENUM_DEF_PATH "my_xmacros.def"
//   #include "toolchain/common/enum_base.def"
//
//   class MyEnum : MyEnumBase<MyEnum> {
//    public:
//     (any custom APIs)
//    protected:
//     using MyEnumBase::MyEnumBase;
//   };
//
// my_xmacros.def will provide:
//   #ifdef CARBON_ENUM_BASE_NAME
//   #define MY_XMACRO(Name) CARBON_ENUM_ENTRY(Name)
//   #endif
//
// Specific macro values are available as:
//   MyEnum::Name()
//
// They will be usable in a switch statement, e.g. `case MyEnum::Name():`.

#ifndef CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_
#define CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_

#include <cstdint>

#include "common/ostream.h"

#define CARBON_ENUM_BASE_INTERNAL_ENUM_ENTRY(Name) Name,

#define CARBON_ENUM_BASE_INTERNAL_FACTORY(Name) \
  static constexpr auto Name()->DerivedEnumT {  \
    return DerivedEnumT(InternalEnum::Name);    \
  }

#define CARBON_ENUM_BASE_INTERNAL_NAMES(Name) #Name,

// This uses CRTP to provide factory functions which create the derived enum.
#define CARBON_ENUM_BASE(EnumBaseName, X_NAMES)                                \
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

#endif  // CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_
