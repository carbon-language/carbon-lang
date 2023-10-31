// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/typed_nodes.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "toolchain/sem_ir/node.h"

namespace Carbon::SemIR {

// A friend of `SemIR::Node` that is used to pierce the abstraction.
class NodeTestHelper {
 public:
  static auto MakeNode(NodeKind node_kind, Parse::Node parse_node,
                       TypeId type_id, int32_t arg0, int32_t arg1) -> Node {
    return Node(node_kind, parse_node, type_id, arg0, arg1);
  }
};

}  // namespace Carbon::SemIR

namespace Carbon::SemIR {
namespace {

// Check that each node kind defines a Kind member using the correct NodeKind
// enumerator.
#define CARBON_SEM_IR_NODE_KIND(Name) \
  static_assert(Name::Kind == NodeKind::Name);
#include "toolchain/sem_ir/node_kind.def"

template <typename Ignored, typename... Types>
using TypesExceptFirst = ::testing::Types<Types...>;

// Form a list of all typed node types. Use `TypesExceptFirst` and a leading
// `void` to handle the problem that we only want N-1 commas in this list.
using TypedNodeTypes = TypesExceptFirst<void
#define CARBON_SEM_IR_NODE_KIND(Name) , Name
#include "toolchain/sem_ir/node_kind.def"
                                        >;

// Set up the test fixture.
template <typename TypedNode>
class TypedNodeTest : public testing::Test {};

TYPED_TEST_SUITE(TypedNodeTest, TypedNodeTypes);

TYPED_TEST(TypedNodeTest, CommonFieldOrder) {
  using TypedNode = TypeParam;

  Node node = NodeTestHelper::MakeNode(TypeParam::Kind, Parse::Node(1),
                                       TypeId(2), 3, 4);
  EXPECT_EQ(node.kind(), TypeParam::Kind);
  EXPECT_EQ(node.parse_node(), Parse::Node(1));
  EXPECT_EQ(node.type_id(), TypeId(2));

  TypedNode typed = node.As<TypedNode>();
  if constexpr (HasParseNode<TypedNode>) {
    EXPECT_EQ(typed.parse_node, Parse::Node(1));
  }
  if constexpr (HasTypeId<TypedNode>) {
    EXPECT_EQ(typed.type_id, TypeId(2));
  }
}

TYPED_TEST(TypedNodeTest, RoundTrip) {
  using TypedNode = TypeParam;

  Node node1 = NodeTestHelper::MakeNode(TypeParam::Kind, Parse::Node(1),
                                        TypeId(2), 3, 4);
  EXPECT_EQ(node1.kind(), TypeParam::Kind);
  EXPECT_EQ(node1.parse_node(), Parse::Node(1));
  EXPECT_EQ(node1.type_id(), TypeId(2));

  TypedNode typed1 = node1.As<TypedNode>();
  Node node2 = typed1;

  EXPECT_EQ(node1.kind(), node2.kind());
  if constexpr (HasParseNode<TypedNode>) {
    EXPECT_EQ(node1.parse_node(), node2.parse_node());
  }
  if constexpr (HasTypeId<TypedNode>) {
    EXPECT_EQ(node1.type_id(), node2.type_id());
  }

  // If the typed node has no padding, we should get exactly the same thing
  // if we convert back from a node.
  TypedNode typed2 = node2.As<TypedNode>();
  if constexpr (std::has_unique_object_representations_v<TypedNode>) {
    EXPECT_EQ(std::memcmp(&typed1, &typed2, sizeof(TypedNode)), 0);
  }

  // The original node might not be identical after one round trip, because the
  // fields not carried by the typed node are lost. But they should be stable
  // if we round-trip again.
  Node node3 = typed2;
  if constexpr (std::has_unique_object_representations_v<Node>) {
    EXPECT_EQ(std::memcmp(&node2, &node3, sizeof(Node)), 0);
  }
}

TYPED_TEST(TypedNodeTest, StructLayout) {
  using TypedNode = TypeParam;

  TypedNode typed =
      NodeTestHelper::MakeNode(TypeParam::Kind, Parse::Node(1), TypeId(2), 3, 4)
          .template As<TypedNode>();

  // Check that the memory representation of the typed node is what we expect.
  // TODO: Struct layout is not guaranteed, and this test could fail in some
  // build environment. If so, we should disable it.
  int32_t fields[4] = {};
  int field = 0;
  if constexpr (HasParseNode<TypedNode>) {
    fields[field++] = 1;
  }
  if constexpr (HasTypeId<TypedNode>) {
    fields[field++] = 2;
  }
  fields[field++] = 3;
  fields[field++] = 4;

  ASSERT_LE(sizeof(TypedNode), sizeof(fields));
  // We can only do this check if the typed node has no padding.
  if constexpr (std::has_unique_object_representations_v<TypedNode>) {
    EXPECT_EQ(std::memcmp(&fields, &typed, sizeof(TypedNode)), 0);
  }
}

TYPED_TEST(TypedNodeTest, NodeKindMatches) {
  using TypedNode = TypeParam;

  // TypedNode::Kind is a NodeKind::Definition that extends NodeKind, but
  // has different definitions of the `ir_name()` and `terminator_kind()`
  // methods. Here we test that values returned by the two different versions
  // of those functions match.
  NodeKind as_kind = TypedNode::Kind;
  EXPECT_EQ(TypedNode::Kind.ir_name(), as_kind.ir_name());
  EXPECT_EQ(TypedNode::Kind.terminator_kind(), as_kind.terminator_kind());
}

}  // namespace
}  // namespace Carbon::SemIR
