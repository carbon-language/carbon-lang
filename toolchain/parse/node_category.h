// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_NODE_CATEGORY_H_
#define CARBON_TOOLCHAIN_PARSE_NODE_CATEGORY_H_

#include "common/enum_mask_base.h"
#include "common/ostream.h"

namespace Carbon::Parse {

// An X-macro for node categories. Uses should look like:
//
//   #define CARBON_NODE_CATEGORY_FOR_XYZ(Name) ...
//   CARBON_NODE_CATEGORY(CARBON_NODE_CATEGORY_FOR_XYZ)
//   #undef CARBON_NODE_CATEGORY_FOR_XYZ
#define CARBON_NODE_CATEGORY(X)                        \
  X(Decl)                                              \
  X(Expr)                                              \
  /* `impl <type> as` or just `impl as` */             \
  X(ImplAs)                                            \
  X(IntConst)                                          \
  X(MemberExpr)                                        \
  X(MemberName)                                        \
  X(Modifier)                                          \
  X(NonExprName)                                       \
  X(PackageName)                                       \
  X(Pattern)                                           \
  /* `require <type> impls` or just `require impls` */ \
  X(RequireImpls)                                      \
  X(Requirement)                                       \
  X(Statement)

// We expect this to grow, so are using a bigger size than needed.
CARBON_DEFINE_RAW_ENUM_MASK(NodeCategory, uint32_t) {
  CARBON_NODE_CATEGORY(CARBON_RAW_ENUM_MASK_ENUMERATOR)
};

// Represents a set of keyword modifiers, using a separate bit per modifier.
class NodeCategory : public CARBON_ENUM_MASK_BASE(NodeCategory) {
 public:
  CARBON_NODE_CATEGORY(CARBON_ENUM_MASK_CONSTANT_DECL)

  using EnumMaskBase::EnumMaskBase;
  // Provide implicit conversion because the raw enum is used in templates.
  explicit(false) constexpr NodeCategory(RawEnumType value) {
    *this = EnumBase::Make(value);
  }
};

#define CARBON_NODE_CATEGORY_WITH_TYPE(X) \
  CARBON_ENUM_MASK_CONSTANT_DEFINITION(NodeCategory, X)
CARBON_NODE_CATEGORY(CARBON_NODE_CATEGORY_WITH_TYPE)
#undef CARBON_NODE_CATEGORY_WITH_TYPE

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_NODE_CATEGORY_H_
