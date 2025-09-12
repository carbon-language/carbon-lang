// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_KEYWORD_MODIFIER_SET_H_
#define CARBON_TOOLCHAIN_CHECK_KEYWORD_MODIFIER_SET_H_

#include <optional>

#include "common/enum_mask_base.h"
#include "toolchain/sem_ir/name_scope.h"

namespace Carbon::Check {

// The order of modifiers. Each of these corresponds to a group on
// KeywordModifierSet, and can be used as an array index.
enum class ModifierOrder : int8_t { Access, Extern, Extend, Decl, Last = Decl };

// A single X-macro to cover modifier groups. These are split out to make groups
// clearer.
#define CARBON_KEYWORD_MODIFIER_SET(X)                                       \
  /* At most one of these access modifiers allowed for a given declaration,  \
   * and if present it must be first. */                                     \
  X(Private)                                                                 \
  X(Protected)                                                               \
                                                                             \
  /* Extern is standalone. */                                                \
  X(Extern)                                                                  \
                                                                             \
  /* Extend can be combined with Final, but no others in the group below. */ \
  X(Extend)                                                                  \
                                                                             \
  /* At most one of these declaration modifiers allowed for a given          \
   * declaration. */                                                         \
  X(Abstract)                                                                \
  X(Base)                                                                    \
  X(Default)                                                                 \
  X(Export)                                                                  \
  X(Final)                                                                   \
  X(Impl)                                                                    \
  X(Returned)                                                                \
  X(Virtual)

// We expect this to grow, so are using a bigger size than needed.
CARBON_DEFINE_RAW_ENUM_MASK(KeywordModifierSet, uint32_t) {
  CARBON_KEYWORD_MODIFIER_SET(CARBON_RAW_ENUM_MASK_ENUMERATOR)
};

// Represents a set of keyword modifiers, using a separate bit per modifier.
class KeywordModifierSet : public CARBON_ENUM_MASK_BASE(KeywordModifierSet) {
 public:
  CARBON_KEYWORD_MODIFIER_SET(CARBON_ENUM_MASK_CONSTANT_DECL)

  // Sets of modifiers.
  static const KeywordModifierSet Access;
  static const KeywordModifierSet Class;
  static const KeywordModifierSet Method;
  static const KeywordModifierSet ImplDecl;
  static const KeywordModifierSet Interface;
  static const KeywordModifierSet Decl;

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

      auto Case(KeywordModifierSet other, T result) -> Converter& {
        if (set_.HasAnyOf(other)) {
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
};

#define CARBON_KEYWORD_MODIFIER_SET_WITH_TYPE(X) \
  CARBON_ENUM_MASK_CONSTANT_DEFINITION(KeywordModifierSet, X)
CARBON_KEYWORD_MODIFIER_SET(CARBON_KEYWORD_MODIFIER_SET_WITH_TYPE)
#undef CARBON_KEYWORD_MODIFIER_SET_WITH_TYPE

constexpr KeywordModifierSet KeywordModifierSet::Access(Private | Protected);
constexpr KeywordModifierSet KeywordModifierSet::Class(Abstract | Base);
constexpr KeywordModifierSet KeywordModifierSet::Method(Abstract | Impl |
                                                        Virtual);
constexpr KeywordModifierSet KeywordModifierSet::ImplDecl(Extend | Final);
constexpr KeywordModifierSet KeywordModifierSet::Interface(Default | Final);
constexpr KeywordModifierSet KeywordModifierSet::Decl(Class | Method |
                                                      Interface | Export |
                                                      Returned);

static_assert(
    !KeywordModifierSet::Access.HasAnyOf(KeywordModifierSet::Extern) &&
        !(KeywordModifierSet::Access | KeywordModifierSet::Extern |
          KeywordModifierSet::Extend)
             .HasAnyOf(KeywordModifierSet::Decl),
    "Order-related sets must not overlap");

#define CARBON_KEYWORD_MODIFIER_SET_IN_GROUP(Modifier)                     \
  static_assert((KeywordModifierSet::Access | KeywordModifierSet::Extern | \
                 KeywordModifierSet::Extend | KeywordModifierSet::Decl)    \
                    .HasAnyOf(KeywordModifierSet::Modifier),               \
                "Modifier missing from all modifier sets: " #Modifier);
CARBON_KEYWORD_MODIFIER_SET(CARBON_KEYWORD_MODIFIER_SET_IN_GROUP)
#undef CARBON_KEYWORD_MODIFIER_SET_IN_GROUP

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_KEYWORD_MODIFIER_SET_H_
