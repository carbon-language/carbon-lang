// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_AST_RTTI_H_
#define CARBON_EXPLORER_AST_AST_RTTI_H_

#include "common/enum_base.h"
#include "explorer/ast/ast_kinds.h"

namespace Carbon {

#define CARBON_IGNORE(C)

// Assign numbers to all AST types.
CARBON_DEFINE_RAW_ENUM_CLASS(AstRttiNodeKind, int) {
#define DEFINE_ENUMERATOR(E) E,
  CARBON_AST_RTTI_KINDS(CARBON_IGNORE, DEFINE_ENUMERATOR)
#undef DEFINE_ENUMERATOR
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

  CARBON_AST_RTTI_KINDS(CARBON_IGNORE, CARBON_ENUM_CONSTANT_DECLARATION)
};

// Define the constant members for AstRttiNodeKind.
#define CONSTANT_DEFINITION(E) \
  CARBON_ENUM_CONSTANT_DEFINITION(AstRttiNodeKind, E)
CARBON_AST_RTTI_KINDS(CARBON_IGNORE, CONSTANT_DEFINITION)
#undef CONSTANT_DEFINITION

// Define Kind enumerations for all base classes.
#define DEFINE_KIND_ENUM(C)                                                   \
  CARBON_DEFINE_RAW_ENUM_CLASS(C##Kind, int) {                                \
    CARBON_##C##_KINDS(CARBON_IGNORE, DEFINE_ENUMERATOR)                      \
  };                                                                          \
  template <typename Derived>                                                 \
  class C##KindTemplate : public CARBON_ENUM_BASE_CRTP(C##Kind, Derived) {    \
   private:                                                                   \
    using Base = CARBON_ENUM_BASE_CRTP(C##Kind, Derived);                     \
    friend class AstRttiNodeKind;                                             \
                                                                              \
   public:                                                                    \
    using IsCarbonAstRttiNodeKind = void;                                     \
                                                                              \
    C##KindTemplate() = default;                                              \
                                                                              \
    /* This type can be explicitly converted from the generic node kind. */   \
    explicit C##KindTemplate(AstRttiNodeKind base_kind)                       \
        : Base(Base::FromInt(base_kind.AsInt())) {}                           \
                                                                              \
    CARBON_##C##_KINDS(CARBON_IGNORE, CARBON_INLINE_ENUM_CONSTANT_DEFINITION) \
  };                                                                          \
  class C##Kind : public C##KindTemplate<C##Kind> {                           \
    using C##KindTemplate<C##Kind>::C##KindTemplate;                          \
  };
#define DEFINE_ENUMERATOR(E) E = AstRttiNodeKind::E.AsInt(),
CARBON_AST_RTTI_KINDS(DEFINE_KIND_ENUM, CARBON_IGNORE)
#undef DEFINE_KIND_ENUM
#undef DEFINE_ENUMERATOR

// Define InheritsFrom functions for all kinds.
#define DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT(C)                             \
  inline bool InheritsFrom##C(Carbon::AstRttiNodeKind kind) {                 \
    return CARBON_##C##_KINDS(CARBON_IGNORE, INHERITS_FROM_CLASS_TEST) false; \
  }
#define INHERITS_FROM_CLASS_TEST(C) kind == Carbon::AstRttiNodeKind::C ||
#define DEFINE_INHERITS_FROM_FUNCTION_FINAL(C)                \
  inline bool InheritsFrom##C(Carbon::AstRttiNodeKind kind) { \
    return kind == Carbon::AstRttiNodeKind::C;                \
  }
CARBON_AST_RTTI_KINDS(DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT,
                      DEFINE_INHERITS_FROM_FUNCTION_FINAL)
#undef DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT
#undef INHERITS_FROM_CLASS_TEST
#undef DEFINE_INHERITS_FROM_FUNCTION_FINAL

#undef CARBON_IGNORE

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_AST_RTTI_H_
