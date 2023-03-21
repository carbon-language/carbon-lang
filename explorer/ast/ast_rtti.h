// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_AST_RTTI_H_
#define CARBON_EXPLORER_AST_AST_RTTI_H_

#include "common/enum_base.h"
#include "explorer/ast/ast_kinds.h"

namespace Carbon {

#define IGNORE(C)

// Assign numbers to all AST types.
CARBON_DEFINE_RAW_ENUM_CLASS(AstRttiNodeKind, int) {
#define ENUMERATOR(E) E,
  CARBON_AST_RTTI_KINDS(IGNORE, ENUMERATOR)
#undef ENUMERATOR
};

// An enumerated type defining the kinds of AST nodes.
class AstRttiNodeKind : public CARBON_ENUM_BASE(AstRttiNodeKind) {
 public:
  using IsCarbonAstRttiNodeKind = void;

  AstRttiNodeKind() = default;

  // All our other RTTI node kinds implicitly convert to this one.
  // TODO: We could support conversion to the base class's kind for all node
  // kinds if we find a reason to do so.
  template <typename Kind, typename = typename Kind::IsCarbonAstRttiNodeKind>
  /*implicit*/ AstRttiNodeKind(Kind kind)
      : AstRttiNodeKind(FromInt(kind.AsInt())) {}

  // Expose the integer value of this node. This is used to set the values of
  // other enumerations to match, and to implement range checks.
  using EnumBase::AsInt;

  CARBON_AST_RTTI_KINDS(IGNORE, CARBON_ENUM_CONSTANT_DECLARATION)
};
#define CONSTANT_DEFINITION(E) \
  CARBON_ENUM_CONSTANT_DEFINITION(AstRttiNodeKind, E)
CARBON_AST_RTTI_KINDS(IGNORE, CONSTANT_DEFINITION)
#undef CONSTANT_DEFINITION

// Define Kind enumerations for all base classes.
#define DEFINE_ENUM(C)                                                      \
  CARBON_DEFINE_RAW_ENUM_CLASS(C##Kind, int) {                              \
    CARBON_##C##_KINDS(IGNORE, ENUMERATOR)                                  \
  };                                                                        \
  template <typename Derived>                                               \
  class C##KindTemplate : public CARBON_ENUM_BASE_CRTP(C##Kind, Derived) {  \
   private:                                                                 \
    using Base = CARBON_ENUM_BASE_CRTP(C##Kind, Derived);                   \
    friend class AstRttiNodeKind;                                           \
                                                                            \
   public:                                                                  \
    using IsCarbonAstRttiNodeKind = void;                                   \
                                                                            \
    C##KindTemplate() = default;                                            \
                                                                            \
    /* This type can be explicitly converted from the generic node kind. */ \
    explicit C##KindTemplate(AstRttiNodeKind base_kind)                     \
        : Base(Base::FromInt(base_kind.AsInt())) {}                         \
                                                                            \
    CARBON_##C##_KINDS(IGNORE, CARBON_INLINE_ENUM_CONSTANT_DEFINITION)      \
  };                                                                        \
  class C##Kind : public C##KindTemplate<C##Kind> {                         \
    using C##KindTemplate<C##Kind>::C##KindTemplate;                        \
  };
#define ENUMERATOR(E) E = AstRttiNodeKind::E.AsInt(),
CARBON_AST_RTTI_KINDS(DEFINE_ENUM, IGNORE)
#undef DEFINE_ENUM
#undef ENUMERATOR

// Helper to compute the number of derived classes for a class by repeatedly
// adding one.
#define ADD_ONE(C) +1
#define NUMBER_OF_DERIVED_CLASSES(C) CARBON_##C##_KINDS(IGNORE, ADD_ONE)

// Helper to find the first derived class of a class.
#define EXPAND(A) A
#define LPAREN (
#define RPAREN )
#define FIRST(A, ...) A
#define COMMA_AFTER(A) A,
#define FIRST_DERIVED_CLASS_IMPL(C) \
  FIRST LPAREN CARBON_##C##_KINDS(IGNORE, COMMA_AFTER) RPAREN
#define FIRST_DERIVED_CLASS(C) EXPAND(FIRST_DERIVED_CLASS_IMPL(C))

// Helper to find the last derived class of a class. This works by converting
// the list of derived classes A B C into
//   APPLY(, DISCARD) APPLY(A, DISCARD) APPLY(B, DISCARD) APPLY(C, RETURN)
// which then expands to
//   DISCARD() DISCARD(A) DISCARD(B) RETURN(C)
// then to
//   C
#define DISCARD(...)
#define RETURN(A) A
#define APPLY(E, M) M(E)
#define LAST_HELPER(E) , DISCARD RPAREN APPLY LPAREN E
#define LAST_DERIVED_CLASS_IMPL(C) \
  APPLY LPAREN CARBON_##C##_KINDS(IGNORE, LAST_HELPER), RETURN RPAREN
#define LAST_DERIVED_CLASS(C) EXPAND(LAST_DERIVED_CLASS_IMPL(C))

// Define InheritsFrom functions for all kinds.
#define DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT(C)                            \
  inline bool InheritsFrom##C(Carbon::AstRttiNodeKind kind) {                \
    constexpr auto kFirst = Carbon::AstRttiNodeKind::FIRST_DERIVED_CLASS(C); \
    constexpr auto kLast = Carbon::AstRttiNodeKind::LAST_DERIVED_CLASS(C);   \
    static_assert(NUMBER_OF_DERIVED_CLASSES(C) ==                            \
                  kLast.AsInt() - kFirst.AsInt() + 1);                       \
    return kind >= kFirst && kind <= kLast;                                  \
  }
#define DEFINE_INHERITS_FROM_FUNCTION_FINAL(C)                \
  inline bool InheritsFrom##C(Carbon::AstRttiNodeKind kind) { \
    return kind == Carbon::AstRttiNodeKind::C;                \
  }
CARBON_AST_RTTI_KINDS(DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT,
                      DEFINE_INHERITS_FROM_FUNCTION_FINAL)
#undef DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT
#undef DEFINE_INHERITS_FROM_FUNCTION_FINAL

// Undefine all helpers.
#undef IGNORE
#undef ADD_ONE
#undef NUMBER_OF_DERIVED_CLASSES
#undef EXPAND
#undef LPAREN
#undef RPAREN
#undef FIRST
#undef COMMA_AFTER
#undef FIRST_DERIVED_CLASS
#undef DISCARD
#undef RETURN
#undef APPLY
#undef LAST_HELPER
#undef LAST_DERIVED_CLASS

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_AST_RTTI_H_
