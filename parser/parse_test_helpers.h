// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PARSER_PARSE_TEST_HELPERS_H_
#define PARSER_PARSE_TEST_HELPERS_H_

#include <ostream>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "lexer/tokenized_buffer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "parser/parse_node_kind.h"
#include "parser/parse_tree.h"

namespace Carbon {

// Enable printing a parse tree from Google Mock.
inline void PrintTo(const ParseTree& tree, std::ostream* output) {
  std::string text;
  llvm::raw_string_ostream text_stream(text);
  tree.Print(text_stream);
  *output << "\n" << text_stream.str() << "\n";
}

namespace Testing {

// An aggregate used to describe an expected parse tree.
//
// This type is designed to be used via aggregate initialization with designated
// initializers. The latter make it easy to default everything and then override
// the desired aspects when writing an expectation in a test.
struct ExpectedNode {
  ParseNodeKind kind = ParseNodeKind::EmptyDeclaration();
  std::string text;
  bool has_error = false;
  bool skip_subtree = false;
  std::vector<ExpectedNode> children;
};

// Implementation of a matcher for a parse tree based on a tree of expected
// nodes.
//
// Don't create this directly, instead use `MatchParseTreeNodes` to construct a
// matcher based on this.
class ExpectedNodesMatcher
    : public ::testing::MatcherInterface<const ParseTree&> {
 public:
  explicit ExpectedNodesMatcher(
      llvm::SmallVector<ExpectedNode, 0> expected_nodess)
      : expected_nodes(std::move(expected_nodess)) {}

  auto MatchAndExplain(const ParseTree& tree,
                       ::testing::MatchResultListener* output_ptr) const
      -> bool override;
  auto DescribeTo(std::ostream* output_ptr) const -> void override;

 private:
  auto MatchExpectedNode(const ParseTree& tree, ParseTree::Node n,
                         int postorder_index, const ExpectedNode& expected_node,
                         ::testing::MatchResultListener& output) const -> bool;

  llvm::SmallVector<ExpectedNode, 0> expected_nodes;
};

// Implementation of the Google Mock interface for matching (and explaining any
// failure).
inline auto ExpectedNodesMatcher::MatchAndExplain(
    const ParseTree& tree, ::testing::MatchResultListener* output_ptr) const
    -> bool {
  auto& output = *output_ptr;
  bool matches = true;
  const auto rpo = llvm::reverse(tree.Postorder());
  const auto nodes_begin = rpo.begin();
  const auto nodes_end = rpo.end();
  auto nodes_it = nodes_begin;
  llvm::SmallVector<const ExpectedNode*, 16> expected_node_stack;
  for (const ExpectedNode& en : expected_nodes) {
    expected_node_stack.push_back(&en);
  }
  while (!expected_node_stack.empty()) {
    if (nodes_it == nodes_end) {
      // We'll check the size outside the loop.
      break;
    }

    ParseTree::Node n = *nodes_it++;
    int postorder_index = n.GetIndex();

    const ExpectedNode& expected_node = *expected_node_stack.pop_back_val();

    if (!MatchExpectedNode(tree, n, postorder_index, expected_node, output)) {
      matches = false;
    }

    if (expected_node.skip_subtree) {
      assert(expected_node.children.empty() &&
             "Must not skip an expected subtree while specifying expected "
             "children!");
      nodes_it = llvm::reverse(tree.Postorder(n)).end();
      continue;
    }

    // We want to make sure we don't end up with unsynchronized walks, so skip
    // ahead in the tree to ensure that the number of children of this node and
    // the expected number of children match.
    int num_children =
        std::distance(tree.Children(n).begin(), tree.Children(n).end());
    if (num_children != static_cast<int>(expected_node.children.size())) {
      output
          << "\nParse node (postorder index #" << postorder_index << ") has "
          << num_children << " children, expected "
          << expected_node.children.size()
          << ". Skipping this subtree to avoid any unsynchronized tree walk.";
      matches = false;
      nodes_it = llvm::reverse(tree.Postorder(n)).end();
      continue;
    }

    // Push the children onto the stack to continue matching. The expectation
    // is in preorder, but we visit the parse tree in reverse postorder. This
    // causes the siblings to be visited in reverse order from the expected
    // list. However, we use a stack which inherently does this reverse for us
    // so we simply append to the stack here.
    for (const ExpectedNode& child_expected_node : expected_node.children) {
      expected_node_stack.push_back(&child_expected_node);
    }
  }

  // We don't directly check the size because we allow expectations to skip
  // subtrees. Instead, we need to check that we successfully processed all of
  // the actual tree and consumed all of the expected tree.
  if (nodes_it != nodes_end) {
    assert(expected_node_stack.empty() &&
           "If we have unmatched nodes in the input tree, should only finish "
           "having fully processed expected tree.");
    output << "\nFinished processing expected nodes and there are still "
           << (nodes_end - nodes_it) << " unexpected nodes.";
    matches = false;
  } else if (!expected_node_stack.empty()) {
    output << "\nProcessed all " << (nodes_end - nodes_begin)
           << " nodes and still have " << expected_node_stack.size()
           << " expected nodes that were unmatched.";
    matches = false;
  }

  return matches;
}

// Implementation of the Google Mock interface for describing the expected node
// tree.
//
// This is designed to describe the expected tree node structure in as similar
// of a format to the parse tree's print format as is reasonable. There is both
// more and less information, so it won't be exact, but should be close enough
// to make it easy to visually compare the two.
inline auto ExpectedNodesMatcher::DescribeTo(std::ostream* output_ptr) const
    -> void {
  auto& output = *output_ptr;
  output << "Matches expected node pattern:\n[\n";

  // We want to walk these in RPO instead of in preorder to match the printing
  // of the actual parse tree.
  llvm::SmallVector<std::pair<const ExpectedNode*, int>, 16>
      expected_node_stack;
  for (const ExpectedNode& expected_node : llvm::reverse(expected_nodes)) {
    expected_node_stack.push_back({&expected_node, 0});
  }

  while (!expected_node_stack.empty()) {
    const ExpectedNode& expected_node = *expected_node_stack.back().first;
    int depth = expected_node_stack.back().second;
    expected_node_stack.pop_back();
    for (int indent_count = 0; indent_count < depth; ++indent_count) {
      output << "  ";
    }
    output << "{kind: '" << expected_node.kind.GetName().str() << "'";
    if (!expected_node.text.empty()) {
      output << ", text: '" << expected_node.text << "'";
    }
    if (expected_node.has_error) {
      output << ", has_error: yes";
    }
    if (expected_node.skip_subtree) {
      output << ", skip_subtree: yes";
    }

    if (!expected_node.children.empty()) {
      assert(!expected_node.skip_subtree &&
             "Must not have children and skip a subtree!");
      output << ", children: [\n";
      for (const ExpectedNode& child_expected_node :
           llvm::reverse(expected_node.children)) {
        expected_node_stack.push_back({&child_expected_node, depth + 1});
      }
      // If we have children, we know we're not popping off.
      continue;
    }

    // If this is some form of leaf we'll at least need to close it. It may also
    // be the last sibling of its parent, and we'll need to close any parents as
    // we pop up.
    output << "}";
    if (!expected_node_stack.empty()) {
      assert(depth >= expected_node_stack.back().second &&
             "Cannot have an increase in depth on a leaf node!");
      // The distance we need to pop is the difference in depth.
      int pop_depth = depth - expected_node_stack.back().second;
      for (int pop_count = 0; pop_count < pop_depth; ++pop_count) {
        // Close both the children array and the node mapping.
        output << "]}";
      }
    }
    output << "\n";
  }
  output << "]\n";
}

inline auto ExpectedNodesMatcher::MatchExpectedNode(
    const ParseTree& tree, ParseTree::Node n, int postorder_index,
    const ExpectedNode& expected_node,
    ::testing::MatchResultListener& output) const -> bool {
  bool matches = true;

  ParseNodeKind kind = tree.GetNodeKind(n);
  if (kind != expected_node.kind) {
    output << "\nParse node (postorder index #" << postorder_index << ") is a "
           << kind.GetName().str() << ", expected a "
           << expected_node.kind.GetName().str() << ".";
    matches = false;
  }

  if (tree.HasErrorInNode(n) != expected_node.has_error) {
    output << "\nParse node (postorder index #" << postorder_index << ") "
           << (tree.HasErrorInNode(n) ? "has an error"
                                      : "does not have an error")
           << ", expected that it "
           << (expected_node.has_error ? "has an error"
                                       : "does not have an error")
           << ".";
    matches = false;
  }

  llvm::StringRef node_text = tree.GetNodeText(n);
  if (!expected_node.text.empty() && node_text != expected_node.text) {
    output << "\nParse node (postorder index #" << postorder_index
           << ") is spelled '" << node_text.str() << "', expected '"
           << expected_node.text << "'.";
    matches = false;
  }

  return matches;
}

// Creates a matcher for a parse tree using a tree of expected nodes.
//
// This is intended to be used with an braced initializer list style aggregate
// initializer for an argument, allowing it to describe a tree structure via
// nested `ExpectedNode` objects.
inline auto MatchParseTreeNodes(
    llvm::SmallVector<ExpectedNode, 0> expected_nodes)
    -> ::testing::Matcher<const ParseTree&> {
  return ::testing::MakeMatcher(
      new ExpectedNodesMatcher(std::move(expected_nodes)));
}

}  // namespace Testing
}  // namespace Carbon

#endif  // PARSER_PARSE_TEST_HELPERS_H_
