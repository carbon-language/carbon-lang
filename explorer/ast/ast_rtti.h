// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_AST_RTTI_H_
#define CARBON_EXPLORER_AST_AST_RTTI_H_

#include "common/enum_base.h"
#include "explorer/ast/ast_kinds.h"

namespace Carbon {

// Assign numbers to all AST types.
CARBON_DEFINE_RAW_ENUM_CLASS(AstRttiNodeKind, int) {
#define DEFINE_ENUMERATOR(E) E,
  CARBON_AST_FOR_EACH_FINAL_CLASS(DEFINE_ENUMERATOR)
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

  CARBON_AST_FOR_EACH_FINAL_CLASS(CARBON_ENUM_CONSTANT_DECL)
};

// Define the constant members for AstRttiNodeKind.
#define CONSTANT_DEFINITION(E) \
  CARBON_ENUM_CONSTANT_DEFINITION(AstRttiNodeKind, E)
CARBON_AST_FOR_EACH_FINAL_CLASS(CONSTANT_DEFINITION)
#undef CONSTANT_DEFINITION

// Define Kind enumerations for all base classes.
#define DEFINE_KIND_ENUM(C)                                                 \
  CARBON_DEFINE_RAW_ENUM_CLASS_NO_NAMES(C##Kind, int) {                     \
    CARBON_AST_FOR_EACH_FINAL_CLASS_BELOW(C, DEFINE_ENUMERATOR)             \
  };                                                                        \
  template <typename Derived>                                               \
  class C##KindTemplate                                                     \
      : public CARBON_ENUM_BASE_CRTP(C##Kind, Derived, AstRttiNodeKind) {   \
   private:                                                                 \
    using Base = CARBON_ENUM_BASE_CRTP(C##Kind, Derived, AstRttiNodeKind);  \
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
    CARBON_AST_FOR_EACH_FINAL_CLASS_BELOW(                                  \
        C, CARBON_INLINE_ENUM_CONSTANT_DEFINITION)                          \
  };                                                                        \
  class C##Kind : public C##KindTemplate<C##Kind> {                         \
    using C##KindTemplate<C##Kind>::C##KindTemplate;                        \
  };
#define DEFINE_ENUMERATOR(E) E = AstRttiNodeKind::E.AsInt(),
CARBON_AST_FOR_EACH_ABSTRACT_CLASS(DEFINE_KIND_ENUM)
#undef DEFINE_KIND_ENUM
#undef DEFINE_ENUMERATOR

// Define InheritsFrom functions for each abstract class.
#define DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT(C)             \
  inline bool InheritsFrom##C(Carbon::AstRttiNodeKind kind) { \
    return CARBON_AST_FOR_EACH_FINAL_CLASS_BELOW(             \
        C, INHERITS_FROM_CLASS_TEST) false;                   \
  }
#define INHERITS_FROM_CLASS_TEST(C) kind == Carbon::AstRttiNodeKind::C ||
CARBON_AST_FOR_EACH_ABSTRACT_CLASS(DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT)
#undef DEFINE_INHERITS_FROM_FUNCTION_ABSTRACT
#undef INHERITS_FROM_CLASS_TEST

// Define trivial InheritsFrom functions for each final class.
#define DEFINE_INHERITS_FROM_FUNCTION_FINAL(C)                \
  inline bool InheritsFrom##C(Carbon::AstRttiNodeKind kind) { \
    return kind == Carbon::AstRttiNodeKind::C;                \
  }
CARBON_AST_FOR_EACH_FINAL_CLASS(DEFINE_INHERITS_FROM_FUNCTION_FINAL)
#undef DEFINE_INHERITS_FROM_FUNCTION_FINAL

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_AST_RTTI_H_
