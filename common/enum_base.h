// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ENUM_BASE_H_
#define CARBON_COMMON_ENUM_BASE_H_

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
// Users must be in the `Carbon` namespace and should look like the following.
//
// In `my_kind.h`:
//   ```
//   CARBON_DEFINE_RAW_ENUM_CLASS(MyKind, uint8_t) {
//   #define CARBON_MY_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
//   #include ".../my_kind.def"
//   };
//
//   class MyKind : public CARBON_ENUM_BASE(MyKind) {
//    public:
//   #define CARBON_MY_KIND(Name) CARBON_ENUM_CONSTANT_DECL(Name)
//   #include ".../my_kind.def"
//
//     // OPTIONAL: To support converting to and from the underlying type of
//     // the enumerator, add these lines:
//     using EnumBase::AsInt;
//     using EnumBase::FromInt;
//
//     // OPTIONAL: To expose the ability to create an instance from the raw
//     // enumerator (for unusual use cases), add this:
//     using EnumBase::Make;
//
//     // Plus, anything else you wish to include.
//   };
//
//   #define CARBON_MY_KIND(Name) CARBON_ENUM_CONSTANT_DEFINITION(MyKind, Name)
//   #include ".../my_kind.def"
//   ```
//
// In `my_kind.cpp`:
//   ```
//   CARBON_DEFINE_ENUM_CLASS_NAMES(MyKind) = {
//   #define CARBON_MY_KIND(Name) CARBON_ENUM_CLASS_NAME_STRING(Name)
//   #include ".../my_kind.def"
//   };
//   ```
//
// The result of the above:
// - An enum class (`RawEnumType`) defined in an `Internal` namespace with one
//   enumerator per call to CARBON_MY_KIND(Name) in `.../my_kind.def`, with name
//   `Name`. This won't generally be used directly, but may be needed for niche
//   use cases such as a template argument.
// - A type `MyKind` that extends `Carbon::Internal::EnumBase`.
//   - `MyKind` includes all the public members of `EnumBase`, like `name` and
//     `Print`. For example, you might call `name()` to construct an error
//     message:
//     ```
//     auto ErrorMessage(MyKind k) -> std::string {
//       return k.name() + " not found";
//     }
//     ```
//   - `MyKind` includes all protected members of `EnumBase`, like `AsInt`,
//     `FromInt`. They will be part of the public API of `EnumBase` if they
//     were included in a `using` declaration.
//   - `MyKind` includes a member `static const MyKind Name;` per call to
//     `CARBON_MY_KIND(Name)` in `.../my_kind.def`. It will have the
//     corresponding value from `RawEnumType`. This is the primary way to create
//     an instance of `MyKind`. For example, it might be used like:
//     ```
//     ErrorMessage(MyKind::Name1);
//     ```
//   - `MyKind` includes an implicit conversion to the `RawEnumType`, returning
//     the value of a private field in `EnumBase`. This is used when writing a
//     `switch` statement, as in this example:
//     ```
//     auto MyFunction(MyKind k) -> void {
//       // Implicitly converts `k` and every `case` expression to
//       // `RawEnumType`:
//       switch (k) {
//         case MyKind::Name1:
//           // ...
//           break;
//
//         case MyKind::Name2:
//           // ...
//           break;
//
//         // No `default` case needed if the above cases are exhaustive.
//         // Prefer no `default` case when possible, to get an error if
//         // a case is skipped.
//       }
//     }
//     ```
//
template <typename DerivedT, typename EnumT, const llvm::StringLiteral Names[]>
class EnumBase : public Printable<DerivedT> {
 public:
  // An alias for the raw enum type. This is an implementation detail and
  // should rarely be used directly, only when an actual enum type is needed.
  using RawEnumType = EnumT;

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
  auto name() const -> llvm::StringRef { return Names[AsInt()]; }

  // Prints this value using its name.
  auto Print(llvm::raw_ostream& out) const -> void { out << name(); }

 protected:
  // The default constructor is explicitly defaulted (and constexpr) as a
  // protected constructor to allow derived classes to be constructed but not
  // the base itself. This should only be used in the `Make` function below.
  constexpr EnumBase() = default;

  // Create an instance from the raw enumerator. Mainly used internally, but may
  // be exposed for unusual use cases.
  static constexpr auto Make(RawEnumType value) -> EnumType {
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
    return Make(static_cast<RawEnumType>(value));
  }

 private:
  RawEnumType value_;
};

}  // namespace Carbon::Internal

// For use when multiple enums use the same list of names.
#define CARBON_DEFINE_RAW_ENUM_CLASS_NO_NAMES(EnumClassName, UnderlyingType) \
  namespace Internal {                                                       \
  enum class EnumClassName##RawEnum : UnderlyingType;                        \
  }                                                                          \
  enum class Internal::EnumClassName##RawEnum : UnderlyingType

// Use this before defining a class that derives from `EnumBase` to begin the
// definition of the raw `enum class`. It should be followed by the body of that
// raw enum class.
#define CARBON_DEFINE_RAW_ENUM_CLASS(EnumClassName, UnderlyingType) \
  namespace Internal {                                              \
  extern const llvm::StringLiteral EnumClassName##Names[];          \
  }                                                                 \
  CARBON_DEFINE_RAW_ENUM_CLASS_NO_NAMES(EnumClassName, UnderlyingType)

// In CARBON_DEFINE_RAW_ENUM_CLASS block, use this to generate each enumerator.
#define CARBON_RAW_ENUM_ENUMERATOR(Name) Name,

// Use this to compute the `Internal::EnumBase` specialization for a Carbon enum
// class. It both computes the name of the raw enum and ensures all the
// namespaces are correct.
#define CARBON_ENUM_BASE(EnumClassName) \
  CARBON_ENUM_BASE_CRTP(EnumClassName, EnumClassName, EnumClassName)
// This variant handles the case where the external name for the Carbon enum is
// not the same as the name by which we refer to it from this context.
#define CARBON_ENUM_BASE_CRTP(EnumClassName, LocalTypeNameForEnumClass, \
                              EnumClassNameForNames)                    \
  ::Carbon::Internal::EnumBase<LocalTypeNameForEnumClass,               \
                               Internal::EnumClassName##RawEnum,        \
                               Internal::EnumClassNameForNames##Names>

// Use this within the Carbon enum class body to generate named constant
// declarations for each value.
#define CARBON_ENUM_CONSTANT_DECL(Name) static const EnumType Name;

// Use this immediately after the Carbon enum class body to define each named
// constant.
#define CARBON_ENUM_CONSTANT_DEFINITION(EnumClassName, Name) \
  constexpr EnumClassName EnumClassName::Name =              \
      EnumClassName::Make(RawEnumType::Name);

// Alternatively, use this within the Carbon enum class body to declare and
// define each named constant. Due to type completeness constraints, this will
// only work if the enum-like class is templated.
//
// This requires the template to have a member named `Base` that names the
// `EnumBase` base class.
#define CARBON_INLINE_ENUM_CONSTANT_DEFINITION(Name)     \
  static constexpr const typename Base::EnumType& Name = \
      Base::Make(Base::RawEnumType::Name);

// Use this in the `.cpp` file for an enum class to start the definition of the
// constant names array for each enumerator. It is followed by the desired
// constant initializer.
//
// `clang-format` has a bug with spacing around `->` returns in macros. See
// https://bugs.llvm.org/show_bug.cgi?id=48320 for details.
#define CARBON_DEFINE_ENUM_CLASS_NAMES(EnumClassName) \
  constexpr llvm::StringLiteral Internal::EnumClassName##Names[]

// Use this within the names array initializer to generate a string for each
// name.
#define CARBON_ENUM_CLASS_NAME_STRING(Name) #Name,

#endif  // CARBON_COMMON_ENUM_BASE_H_
