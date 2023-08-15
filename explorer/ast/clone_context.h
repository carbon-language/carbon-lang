// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_CLONE_CONTEXT_H_
#define CARBON_EXPLORER_AST_CLONE_CONTEXT_H_

#include <optional>
#include <type_traits>
#include <vector>

#include "common/check.h"
#include "explorer/ast/ast_rtti.h"
#include "explorer/base/arena.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

namespace Carbon {

class AstNode;
class Element;
class Value;

// A context for performing a deep copy of some fragment of the AST.
//
// This class carries the state necessary to make the copy, including ensuring
// that each node is cloned only once and mapping from old nodes to new ones.
class CloneContext {
 public:
  explicit CloneContext(Nonnull<Arena*> arena) : arena_(arena) {}

  CloneContext(const CloneContext&) = delete;
  auto operator=(const CloneContext&) -> const CloneContext& = delete;

  // Clone an AST element.
  template <typename T>
  auto Clone(Nonnull<const T*> node) -> Nonnull<T*> {
    if constexpr (std::is_convertible_v<const T*, const AstNode*>) {
      const AstNode* base_node = node;
      // Note, we can't use `llvm::cast<T>` here because we might not have
      // finished cloning `base_node` yet and its kind might not be set. This
      // happens when there is a pointer cycle in the AST.
      return static_cast<T*>(CloneBase(base_node));
    } else if constexpr (std::is_convertible_v<const T*, const Value*>) {
      const Value* base_value = node;
      return static_cast<T*>(CloneBase(base_value));
    } else {
      static_assert(std::is_convertible_v<const T*, const Element*>,
                    "unknown pointer type to clone");
      const Element* base_elem = node;
      return static_cast<T*>(CloneBase(base_elem));
    }
  }

  // Clone anything with a clone constructor, that is, a constructor of the
  // form:
  //
  //   explicit MyType(CloneContext&, const MyType&)
  //
  // Clone constructors should call Clone on their owned elements to form a new
  // value. Pointers returned by Clone should not be inspected by the clone
  // constructor, as the pointee is not necessarily fully initialized until the
  // overall cloning process completes.
  //
  // Clone constructors should call Remap on values that they do not own, such
  // as for the declaration named by an IdentifierExpression.
  template <typename T>
  auto Clone(const T& other)
      -> std::enable_if_t<std::is_constructible_v<T, CloneContext&, const T&>,
                          T> {
    return T(*this, other);
  }

  template <typename T>
  auto Clone(std::optional<T> node) -> std::optional<T> {
    if (node) {
      return Clone(*node);
    }
    return std::nullopt;
  }

  template <typename T>
  auto Clone(const std::vector<T>& nodes) -> std::vector<T> {
    std::vector<T> result;
    result.reserve(nodes.size());
    for (const auto& node : nodes) {
      result.push_back(Clone(node));
    }
    return result;
  }

  // Find the new or existing node corresponding to the given node. This should
  // be used when a cloned node has a non-owning reference to another node,
  // that might refer to something being cloned or might refer to the original
  // object. The returned node might not be fully constructed and should not be
  // inspected.
  template <typename T>
  auto Remap(Nonnull<T*> node) -> Nonnull<T*> {
    // Note, we can't use `llvm::cast<T>` here because we might not have
    // finished cloning `base_node` yet and its kind might not be set. This
    // happens when there is a pointer cycle in the AST.
    T* cloned = static_cast<T*>(nodes_[node]);
    return cloned ? cloned : node;
  }

  // It's safe to remap a `const` object by remapping the non-const version and
  // adding back the `const`.
  template <typename T>
  auto Remap(Nonnull<const T*> node) -> Nonnull<const T*> {
    return Remap(const_cast<T*>(node));
  }

  template <typename T>
  auto Remap(std::optional<T> node) -> std::optional<T> {
    if (node) {
      return Remap(*node);
    }
    return std::nullopt;
  }

  template <typename T>
  auto Remap(const std::vector<T>& nodes) -> std::vector<T> {
    std::vector<T> result;
    result.reserve(nodes.size());
    for (const auto& node : nodes) {
      result.push_back(Remap(node));
    }
    return result;
  }

  template <typename T>
  auto GetExistingClone(Nonnull<const T*> node) -> Nonnull<T*> {
    AstNode* cloned = nodes_.lookup(node);
    CARBON_CHECK(cloned) << "expected node to be cloned";
    return llvm::cast<T>(cloned);
  }

 private:
  // A value transform that remaps or clones AST elements referred to by the
  // value being transformed.
  class CloneValueTransform;

  // Clone the given node, and remember the mapping from the original to the
  // new node for remapping.
  auto CloneBase(Nonnull<const AstNode*> node) -> Nonnull<AstNode*>;

  // Clone the given value, replacing references to cloned local declarations
  // with references to the copies.
  auto CloneBase(Nonnull<const Value*> value) -> Nonnull<Value*>;

  // Clone the given element reference.
  auto CloneBase(Nonnull<const Element*> elem) -> Nonnull<Element*>;

  // Clone the given node if it's not already been cloned. This should be used
  // very sparingly, in cases where ownership is unclear.
  void MaybeCloneBase(Nonnull<const AstNode*> node);

  // Arena to allocate new nodes within.
  Nonnull<Arena*> arena_;

  // Mapping from old nodes to new nodes.
  llvm::DenseMap<const AstNode*, AstNode*> nodes_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_CLONE_CONTEXT_H_
