// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_
#define CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_

#include <cstdint>
#include <type_traits>

#include "common/ostream.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Internal {

// CRTP-style base class used to define the common pattern of Carbon enum-like
// classes. The result is a class with named constants similar to enumerators,
// but that are normal classes, can contain other methods, and support a `name`
// method and printing the enums. These even work in switch statements and
// support `case MyEnum::Name:`.
//
// It is specifically designed to compose with X-MACRO style `.def` files that
// stamp out all the enumerators.
//
// It also supports some opt-in APIs that classes can enable by `using` the
// names to make them public: `AsInt` and `FromInt` allow converting to and from
// the underlying type of the enumerator.
//
// Users must be in the `Carbon` namespace and should look like the following.
//
// In `my_kind.h`:
//   ```
//   CARBON_DEFINE_RAW_ENUM_CLASS(MyKind, uint8_t) {
//   #define CARBON_MY_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
//   #include "toolchain/.../my_kind.def"
//   };
//
//   class MyKind : public CARBON_ENUM_BASE(MyKind) {
//    public:
//   #define CARBON_MY_KIND(Name) CARBON_ENUM_CONSTANT_DECLARATION(Name)
//   #include "toolchain/.../my_kind.def"
//   };
//
//   #define CARBON_MY_KIND(Name) CARBON_ENUM_CONSTANT_DEFINITION(MyKind, Name)
//   #include "toolchain/.../my_kind.def"
//   ```
//
// In `my_kind.cpp`:
//   ```
//   CARBON_DEFINE_ENUM_CLASS_NAMES(MyKind) = {
//   #define CARBON_MY_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
//   #include "toolchain/.../my_kind.def"
//   };
//   ```
template <typename DerivedT, typename EnumT>
class EnumBase {
 protected:
  // An alias for the raw enum type. This is an implementation detail and
  // shouldn't be used, but we need it for a signature so it is declared early.
  using RawEnumType = EnumT;

 public:
  using EnumType = DerivedT;
  using UnderlyingType = std::underlying_type_t<RawEnumType>;

  // Enable conversion to the raw enum type, including in a `constexpr` context,
  // to enable comparisons and usage in `switch` and `case`. The enum type
  // remains an implementation detail and nothing else should be using this
  // function.
  //
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator RawEnumType() const { return value_; }

  // Conversion to bool is deleted to prevent direct use in an `if` condition
  // instead of comparing with another value.
  explicit operator bool() const = delete;

  // Returns the name of this value.
  //
  // This method will be automatically defined using the static `names` string
  // table in the base class, which is in turn will be populated for each
  // derived type using the macro helpers in this file.
  [[nodiscard]] auto name() const -> llvm::StringRef;

  // Prints this value using its name.
  void Print(llvm::raw_ostream& out) const {
    out << reinterpret_cast<const EnumType*>(this)->name();
  }

 protected:
  // The default constructor is explicitly defaulted (and constexpr) as a
  // protected constructor to allow derived classes to be constructed but not
  // the base itself. This should only be used in the `Create` function below.
  constexpr EnumBase() = default;

  // Create an instance from the raw enumerator, for internal use.
  static constexpr auto Create(RawEnumType value) -> EnumType {
    EnumType result;
    result.value_ = value;
    return result;
  }

  // Convert to the underlying integer type. Derived types can choose to expose
  // this as part of their API.
  constexpr auto AsInt() const -> UnderlyingType {
    return static_cast<UnderlyingType>(value_);
  }

  // Convert from the underlying integer type. Derived types can choose to
  // expose this as part of their API.
  static constexpr auto FromInt(UnderlyingType value) -> EnumType {
    return Create(static_cast<RawEnumType>(value));
  }

 private:
  static llvm::StringLiteral names[];

  RawEnumType value_;
};

}  // namespace Carbon::Internal

// Use this before defining a class that derives from `EnumBase` to begin the
// definition of the raw `enum class`. It should be followed by the body of that
// raw enum class.
#define CARBON_DEFINE_RAW_ENUM_CLASS(EnumClassName, UnderlyingType) \
  namespace Internal {                                              \
  /* NOLINTNEXTLINE(bugprone-macro-parentheses) */                  \
  enum class EnumClassName##RawEnum : UnderlyingType;               \
  }                                                                 \
  enum class ::Carbon::Internal::EnumClassName##RawEnum : UnderlyingType

// In CARBON_DEFINE_RAW_ENUM_CLASS block, use this to generate each enumerator.
#define CARBON_RAW_ENUM_ENUMERATOR(Name) Name,

// Use this to compute the `Internal::EnumBase` specialization for a Carbon enum
// class. It both computes the name of the raw enum and ensures all the
// namespaces are correct.
#define CARBON_ENUM_BASE(EnumClassName)       \
  ::Carbon::Internal::EnumBase<EnumClassName, \
                               ::Carbon::Internal::EnumClassName##RawEnum>

// Use this within the Carbon enum class body to generate named constant
// declarations for each value.
#define CARBON_ENUM_CONSTANT_DECLARATION(Name) static const EnumType Name;

// Use this immediately after the Carbon enum class body to define each named
// constant.
#define CARBON_ENUM_CONSTANT_DEFINITION(EnumClassName, Name) \
  constexpr EnumClassName EnumClassName::Name =              \
      EnumClassName::Create(RawEnumType::Name);

// Use this in the `.cpp` file for an enum class to start the definition of the
// constant names array for each enumerator. It is followed by the desired
// constant initializer.
//
// `clang-format` has a bug with spacing around `->` returns in macros. See
// https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_DEFINE_ENUM_CLASS_NAMES(EnumClassName)                         \
  /* First declare an explicit specialization of the names array so we can    \
   * reference it from an explicit function specialization. */                \
  template <>                                                                 \
  llvm::StringLiteral Internal::EnumBase<                                     \
      EnumClassName, Internal::EnumClassName##RawEnum>::names[];              \
                                                                              \
  /* Now define an explicit function specialization for the `name` method, as \
   * it can now reference our specialized array. */                           \
  template <>                                                                 \
  auto                                                                        \
  Internal::EnumBase<EnumClassName, Internal::EnumClassName##RawEnum>::name() \
      const->llvm::StringRef {                                                \
    return names[static_cast<int>(value_)];                                   \
  }                                                                           \
                                                                              \
  /* Finally, open up the definition of our specialized array for the user to \
   * populate using the x-macro include. */                                   \
  template <>                                                                 \
  llvm::StringLiteral Internal::EnumBase<                                     \
      EnumClassName, Internal::EnumClassName##RawEnum>::names[]

// Use this within the names array initializer to generate a string for each
// name.
#define CARBON_ENUM_CLASS_NAME_STRING(Name) #Name,

#endif  // CARBON_TOOLCHAIN_COMMON_ENUM_BASE_H_
