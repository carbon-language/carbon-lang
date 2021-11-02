// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_UNIMPLEMENTED_H_
#define EXECUTABLE_SEMANTICS_AST_UNIMPLEMENTED_H_

#include <string>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/source_location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon {

// AST node type representing an unimplemented syntax. NodeBase must be
// one of the base AST node types (Expression, Statement, Pattern, or
// Declaration).
template <typename NodeBase>
class Unimplemented : public NodeBase {
 public:
  // Constructs an Unimplemented node standing in for a future node type named
  // `kind`, with the given children. Each child must be a pointer to an
  // object that supports streaming to llvm::raw_ostream, or a movable value
  // that supports such streaming.
  template <typename... Children>
  explicit Unimplemented(std::string_view kind, SourceLocation source_loc,
                         const Children&... children);

  static auto classof(const NodeBase* other) -> bool {
    return other->kind() == NodeBase::Kind::Unimplemented;
  }

  // Returns a printable representation of the subtree rooted at this node.
  auto printed_form() const -> std::string_view { return printed_form_; }

 private:
  std::string printed_form_;
};

// Implementation details only below here.

namespace UnimplementedInternal {

// PrintChild(out, child) prints `child` to `out`. If `child` is a pointer
// to a printable type, the pointee will be printed.
template <typename Child, typename = std::enable_if_t<IsPrintable<Child>>>
void PrintChild(llvm::raw_ostream& out, Nonnull<const Child*> child) {
  out << *child;
}

template <typename Child, typename = std::enable_if_t<IsPrintable<Child>>>
void PrintChild(llvm::raw_ostream& out, const Child& child) {
  out << child;
}

// PrintChildren(out, children...) prints each of `children` to `out`,
// separated by commas.
inline void PrintChildren(llvm::raw_ostream&) {}

template <typename Child>
void PrintChildren(llvm::raw_ostream& out, const Child& child) {
  PrintChild(out, child);
}

template <typename Child, typename... Children>
void PrintChildren(llvm::raw_ostream& out, const Child& child,
                   const Children&... children) {
  PrintChild(out, child);
  out << ", ";
  PrintChildren(out, children...);
}

}  // namespace UnimplementedInternal

template <typename NodeBase>
template <typename... Children>
Unimplemented<NodeBase>::Unimplemented(std::string_view kind,
                                       SourceLocation source_loc,
                                       const Children&... children)
    : NodeBase(NodeBase::Kind::Unimplemented, source_loc) {
  printed_form_.append(llvm::join_items("", "Unimplemented", kind, "("));
  llvm::raw_string_ostream out(printed_form_);
  UnimplementedInternal::PrintChildren(out, children...);
  printed_form_.append(")");
}

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_UNIMPLEMENTED_H_
