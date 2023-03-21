// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/clone_context.h"

#include "explorer/ast/ast_node.h"
#include "explorer/ast/value_transform.h"

namespace Carbon {

auto CloneContext::CloneBase(Nonnull<const AstNode*> node)
    -> Nonnull<AstNode*> {
  auto [it, added] = nodes_.insert({node, nullptr});
  CARBON_CHECK(added) << (it->second
                              ? "node was cloned multiple times: "
                              : "node was remapped before it was cloned: ")
                      << *node;

  // The implementation is generated in ast_rtti.cpp.
  CloneImpl(*arena_, *this, *node, &it->second);

  // Cloning may have invalidated our iterator; redo lookup.
  auto* result = nodes_[node];
  CARBON_CHECK(result) << "CloneImpl didn't set the result pointer";
  return result;
}

class CloneContext::CloneValueTransform
    : public ValueTransform<CloneValueTransform, NoOpUnwrapper> {
 public:
  CloneValueTransform(Nonnull<CloneContext*> context, Nonnull<Arena*> arena)
      : ValueTransform(arena), context_(context) {}

  using ValueTransform::operator();

  // Transforming a pointer to an AstNode should remap the node. Values do not
  // own the nodes they point to, apart from the exceptions handled below.
  template <typename NodeT>
  auto operator()(Nonnull<const NodeT*> node, int /*unused*/ = 0)
      -> std::enable_if_t<std::is_base_of_v<AstNode, NodeT>,
                          Nonnull<const NodeT*>> {
    return context_->Remap(node);
  }

  // Transforming a value node view should clone it. The value node view does
  // not itself own the node it points to, so this is a shallow clone.
  auto operator()(ValueNodeView value_node) -> ValueNodeView {
    return context_->Clone(value_node);
  }

  // A FunctionType may or may not own its bindings.
  auto operator()(Nonnull<const FunctionType*> fn_type)
      -> Nonnull<const FunctionType*> {
    for (auto* binding : fn_type->deduced_bindings()) {
      context_->MaybeCloneBase(binding);
    }
    for (auto [index, binding] : fn_type->generic_parameters()) {
      context_->MaybeCloneBase(binding);
    }
    return ValueTransform::operator()(fn_type);
  }

  // A ConstraintType owns its self binding, so we need to clone it.
  auto operator()(Nonnull<const ConstraintType*> constraint)
      -> Nonnull<const Value*> {
    context_->Clone(constraint->self_binding());
    return ValueTransform::operator()(constraint);
  }

 private:
  Nonnull<CloneContext*> context_;
};

auto CloneContext::CloneBase(Nonnull<const Value*> value) -> Nonnull<Value*> {
  return const_cast<Value*>(CloneValueTransform(this, arena_).Transform(value));
}

auto CloneContext::CloneBase(Nonnull<const Element*> elem)
    -> Nonnull<Element*> {
  return const_cast<Element*>(
      CloneValueTransform(this, arena_).Transform(elem));
}

void CloneContext::MaybeCloneBase(Nonnull<const AstNode*> node) {
  auto it = nodes_.find(node);
  if (it == nodes_.end()) {
    Clone(node);
  }
}

}  // namespace Carbon
