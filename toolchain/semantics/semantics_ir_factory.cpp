// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir_factory.h"

#include <stack>

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"

namespace Carbon {

auto SemanticsIRFactory::Build(const ParseTree& parse_tree) -> SemanticsIR {
  SemanticsIRFactory builder(parse_tree);
  builder.Build();
  return builder.semantics_;
}

void SemanticsIRFactory::Build() {
  std::stack<int> depth_ends;
  for (; cursor_ != range_.end();) {
    const auto node_kind = parse_tree().node_kind(*cursor_);
    switch (node_kind) {
      case ParseNodeKind::FileEnd():
        // Discard.
        CHECK(parse_tree().node_subtree_size(*cursor_) == 1);
        ++cursor_;
        break;
      case ParseNodeKind::FunctionDeclaration():
        TransformFunctionDeclaration(semantics_.root_block_);
        break;
      default:
        FATAL() << NodeUnexpectedError();
    }
  }
}

void SemanticsIRFactory::MovePastChildlessNode() {
  CHECK(parse_tree().node_subtree_size(*cursor_) == 1);
  ++cursor_;
}

auto SemanticsIRFactory::NodeKindWrongError(ParseNodeKind expected)
    -> std::string {
  auto cursor_kind = parse_tree().node_kind(*cursor_);
  CHECK(expected == cursor_kind)
      << "At index " << (*cursor_).index() << ", expected " << expected.name()
      << ", found " << cursor_kind.name();
}

auto SemanticsIRFactory::TryHandleCursor(llvm::ArrayRef<NodeHandler> handlers,
                                         size_t& handlers_index) -> bool {
  auto kind = parse_tree().node_kind(*cursor_);
  while (handlers_index < handlers.size()) {
    const auto& candidate = handlers[handlers_index];
    if (kind == candidate.kind) {
      candidate.handler();
      if (candidate.ordering != Ordering::Repeated) {
        ++handlers_index;
      }
      return true;
    } else if (candidate.ordering != Ordering::Required) {
      ++handlers_index;
    } else {
      FATAL() << NodeKindWrongError(candidate.kind);
    }
  }
  return false;
}

void SemanticsIRFactory::TransformCursorAsOrderedNodes(
    llvm::ArrayRef<NodeHandler> handlers) {
  auto subtree_end = GetSubtreeEnd();
  ++cursor_;

  size_t handlers_index = 0;
  while (cursor_ != subtree_end) {
    ParseTree::Node n = *cursor_;
    CHECK(TryHandleCursor(handlers, handlers_index))
        << "At index " << n.index() << ", unexpected node of kind "
        << parse_tree().node_kind(n).name();
    CHECK(n != *cursor_) << "Handler didn't move past "
                         << parse_tree().node_kind(*cursor_).name();
  }
  // See if any non-optional handlers remain.
  while (handlers_index < handlers.size()) {
    const auto& candidate = handlers[handlers_index];
    CHECK(candidate.ordering != Ordering::Required)
        << "Didn't use required handler for " << candidate.kind.name();
    ++handlers_index;
  }
}

auto SemanticsIRFactory::TransformDeclaredName() -> Semantics::DeclaredName {
  CHECK(parse_tree().node_kind(*cursor_) == ParseNodeKind::DeclaredName());

  Semantics::DeclaredName name(parse_tree().GetNodeText(*cursor_), *cursor_);
  MovePastChildlessNode();
  return name;
}

void SemanticsIRFactory::TransformFunctionDeclaration(
    SemanticsIR::Block& block) {
  CHECK(parse_tree().node_kind(*cursor_) ==
        ParseNodeKind::FunctionDeclaration());

  ParseTree::Node node = *cursor_;
  llvm::Optional<Semantics::DeclaredName> name;
  TransformCursorAsOrderedNodes({
      {.kind = ParseNodeKind::CodeBlock(),
       .handler = [&]() { cursor_ = GetSubtreeEnd(); },
       .ordering = Ordering::Optional},
      {.kind = ParseNodeKind::ReturnType(),
       .handler = [&]() { cursor_ = GetSubtreeEnd(); },
       .ordering = Ordering::Optional},
      {.kind = ParseNodeKind::ParameterList(),
       .handler = [&]() { TransformParameterList(); }},
      {.kind = ParseNodeKind::DeclaredName(),
       .handler = [&]() { name = TransformDeclaredName(); }},
  });
  semantics_.AddFunction(block, Semantics::Function(node, *name));
}

void SemanticsIRFactory::TransformPatternBinding() {
  CHECK(parse_tree().node_kind(*cursor_) == ParseNodeKind::PatternBinding());

  ParseTree::Node node = *cursor_;
  llvm::Optional<Semantics::DeclaredName> name;
  TransformCursorAsOrderedNodes({
      {.kind = ParseNodeKind::Literal(),
       .handler = [&]() { MovePastChildlessNode(); }},
      {.kind = ParseNodeKind::DeclaredName(),
       .handler = [&]() { name = TransformDeclaredName(); }},
  });
  CHECK(*cursor_ != node);
}

void SemanticsIRFactory::TransformCursorAsList(
    ParseNodeKind item_kind, std::function<void()> item_handler,
    ParseNodeKind separator, ParseNodeKind list_end) {
  auto subtree_end = GetSubtreeEnd();
  ++cursor_;

  CHECK(cursor_ != subtree_end) << "No list end";
  CHECK(parse_tree().node_kind(*cursor_) == list_end)
      << NodeKindWrongError(list_end);
  MovePastChildlessNode();

  // Handly empty lists.
  if (cursor_ == subtree_end) {
    return;
  }

  while (true) {
    ParseTree::Node n = *cursor_;
    CHECK(parse_tree().node_kind(n) == item_kind)
        << NodeKindWrongError(item_kind);
    item_handler();
    CHECK(n != *cursor_) << "Handler didn't move past "
                         << parse_tree().node_kind(*cursor_).name();

    if (cursor_ == subtree_end) {
      break;
    }

    CHECK(parse_tree().node_kind(*cursor_) == separator)
        << NodeKindWrongError(separator);
    MovePastChildlessNode();
    CHECK(cursor_ != subtree_end) << "No list end";
  }
}

void SemanticsIRFactory::TransformParameterList() {
  CHECK(parse_tree().node_kind(*cursor_) == ParseNodeKind::ParameterList());

  ParseTree::Node node = *cursor_;
  TransformCursorAsOrderedNodes({
      {.kind = ParseNodeKind::ParameterListEnd(),
       .handler = [&]() { MovePastChildlessNode(); }},
      {.kind = ParseNodeKind::PatternBinding(),
       .handler = [&]() { TransformPatternBinding(); },
       .ordering = Ordering::Repeated},
  });
  CHECK(*cursor_ != node);
}

}  // namespace Carbon
