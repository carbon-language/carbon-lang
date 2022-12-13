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

// clang-format doesn't work well on this enum due to the separation of blocks
// across defines.
// clang-format off

// Start the base class definition.
#define CARBON_ENUM_BASE_1_OF_7(EnumBaseName) \
  /* Uses CRTP to provide factory functions which create the derived enum. */  \
  template <typename DerivedEnumT>                                             \
  class EnumBaseName {                                                         \
   protected:                                                                  \
    /* The enum must be declared earlier in the class so that its type can be  \
     * used, for example in the conversion operator.                           \
     */                                                                        \
    enum class InternalEnum : uint8_t {

// Generate entries for the `enum class`.
#define CARBON_ENUM_BASE_2_OF_7_ITER(Name) Name,

// Resume the base class definition.
#define CARBON_ENUM_BASE_3_OF_7(EnumBaseName)                                  \
    };                                                                         \
                                                                               \
   public:                                                                     \
    /* Defines factory functions for each enum name.                           \
     */

// Generate `MyEnum::Name()` factory functions.
#define CARBON_ENUM_BASE_4_OF_7_ITER(Name)                                     \
  static constexpr auto Name() -> DerivedEnumT {                               \
    return DerivedEnumT(InternalEnum::Name);    \
  }

#define CARBON_ENUM_BASE_5_OF_7(EnumBaseName)                                  \
    /* The default constructor is deleted because objects of this type should  \
     * always be constructed using the above factory functions for each unique \
     * kind.                                                                   \
     */                                                                        \
    EnumBaseName() = delete;                                                   \
                                                                               \
    /* Gets a friendly name for the token for logging or debugging. */         \
    [[nodiscard]] inline auto name() const -> llvm::StringRef {                \
      static constexpr llvm::StringLiteral Names[] = {

#define CARBON_ENUM_BASE_6_OF_7_ITER(Name) #Name,

#define CARBON_ENUM_BASE_7_OF_7(EnumBaseName)                                  \
      };                                                                       \
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

// clang-format on

#endif  // CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_
