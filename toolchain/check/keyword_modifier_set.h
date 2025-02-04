// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_KEYWORD_MODIFIER_SET_H_
#define CARBON_TOOLCHAIN_CHECK_KEYWORD_MODIFIER_SET_H_

#include <optional>

#include "llvm/ADT/BitmaskEnum.h"
#include "toolchain/sem_ir/name_scope.h"

namespace Carbon::Check {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// The order of modifiers. Each of these corresponds to a group on
// KeywordModifierSet, and can be used as an array index.
enum class ModifierOrder : int8_t { Access, Extern, Decl, Last = Decl };

// Represents a set of keyword modifiers, using a separate bit per modifier.
class KeywordModifierSet {
 public:
  // Provide values as an enum. This doesn't expose these as KeywordModifierSet
  // instances just due to the duplication of declarations that would cause.
  //
  // We expect this to grow, so are using a bigger size than needed.
  // NOLINTNEXTLINE(performance-enum-size)
  enum RawEnumType : uint32_t {
    // At most one of these access modifiers allowed for a given declaration,
    // and if present it must be first:
    Private = 1 << 0,
    Protected = 1 << 1,

    // Extern is standalone.
    Extern = 1 << 2,

    // At most one of these declaration modifiers allowed for a given
    // declaration:
    Abstract = 1 << 3,
    Base = 1 << 4,
    Default = 1 << 5,
    Export = 1 << 6,
    Extend = 1 << 7,
    Final = 1 << 8,
    Impl = 1 << 9,
    Virtual = 1 << 10,
    Returned = 1 << 11,

    // Sets of modifiers:
    Access = Private | Protected,
    Class = Abstract | Base,
    Method = Abstract | Impl | Virtual,
    ImplDecl = Extend | Final,
    Interface = Default | Final,
    Decl = Class | Method | ImplDecl | Interface | Export | Returned,
    None = 0,

    LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/Returned)
  };

  // Default construct to empty.
  explicit KeywordModifierSet() : set_(None) {}

  // Support implicit conversion so that the difference with the member enum is
  // opaque.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr KeywordModifierSet(RawEnumType set) : set_(set) {}

  // Adds entries to the set.
  auto Add(KeywordModifierSet set) -> void { set_ |= set.set_; }
  // Removes entries from the set.
  auto Remove(KeywordModifierSet set) -> void { set_ &= ~set.set_; }

  // Returns true if there's a non-empty set intersection.
  constexpr auto HasAnyOf(KeywordModifierSet other) const -> bool {
    return set_ & other.set_;
  }

  // Return a builder that returns the new enumeration type once a series of
  // mapping `Case`s and a final `Default` are provided. For example:
  //   ```
  //   auto e = set.ToEnum<SomeEnum>()
  //                .Case(KeywordModifierSet::A, SomeEnum::A)
  //                .Case(KeywordModifierSet::B, SomeEnum::B)
  //                .Default(SomeEnum::DefaultValue);
  //   ```
  template <typename T>
  auto ToEnum() const -> auto {
    class Converter {
     public:
      explicit Converter(const KeywordModifierSet& set) : set_(set) {}

      auto Case(RawEnumType raw_enumerator, T result) -> Converter& {
        if (set_.HasAnyOf(raw_enumerator)) {
          result_ = result;
        }
        return *this;
      }

      auto Default(T default_value) -> T {
        if (result_) {
          return *result_;
        }
        return default_value;
      }

     private:
      const KeywordModifierSet& set_;
      std::optional<T> result_;
    };
    return Converter(*this);
  }

  // Returns the access kind from modifiers.
  auto GetAccessKind() const -> SemIR::AccessKind {
    if (HasAnyOf(KeywordModifierSet::Protected)) {
      return SemIR::AccessKind::Protected;
    }
    if (HasAnyOf(KeywordModifierSet::Private)) {
      return SemIR::AccessKind::Private;
    }
    return SemIR::AccessKind::Public;
  }

  // Returns true if empty.
  constexpr auto empty() const -> bool { return !set_; }

  // Returns the set intersection.
  constexpr auto operator&(KeywordModifierSet other) const
      -> KeywordModifierSet {
    return set_ & other.set_;
  }

  // Returns the set inverse.
  auto operator~() const -> KeywordModifierSet { return ~set_; }

 private:
  RawEnumType set_;
};

static_assert(!KeywordModifierSet(KeywordModifierSet::Access)
                      .HasAnyOf(KeywordModifierSet::Extern) &&
                  !KeywordModifierSet(KeywordModifierSet::Access |
                                      KeywordModifierSet::Extern)
                       .HasAnyOf(KeywordModifierSet::Decl),
              "Order-related sets must not overlap");
static_assert(~KeywordModifierSet::None ==
                  (KeywordModifierSet::Access | KeywordModifierSet::Extern |
                   KeywordModifierSet::Decl),
              "Modifier missing from all modifier sets");

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_KEYWORD_MODIFIER_SET_H_
