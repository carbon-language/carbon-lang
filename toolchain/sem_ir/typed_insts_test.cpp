// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/typed_insts.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

// A friend of `SemIR::Inst` that is used to pierce the abstraction.
class InstTestHelper {
 public:
  static auto MakeInst(InstKind inst_kind, Parse::Node parse_node,
                       TypeId type_id, int32_t arg0, int32_t arg1) -> Inst {
    return Inst(inst_kind, parse_node, type_id, arg0, arg1);
  }
};

}  // namespace Carbon::SemIR

namespace Carbon::SemIR {
namespace {

// Check that each instruction kind defines a Kind member using the correct
// InstKind enumerator.
#define CARBON_SEM_IR_INST_KIND(Name) \
  static_assert(Name::Kind == InstKind::Name);
#include "toolchain/sem_ir/inst_kind.def"

template <typename Ignored, typename... Types>
using TypesExceptFirst = ::testing::Types<Types...>;

// Form a list of all typed instruction types. Use `TypesExceptFirst` and a
// leading `void` to handle the problem that we only want N-1 commas in this
// list.
using TypedInstTypes = TypesExceptFirst<void
#define CARBON_SEM_IR_INST_KIND(Name) , Name
#include "toolchain/sem_ir/inst_kind.def"
                                        >;

// Set up the test fixture.
template <typename TypedInst>
class TypedInstTest : public testing::Test {};

TYPED_TEST_SUITE(TypedInstTest, TypedInstTypes);

TYPED_TEST(TypedInstTest, CommonFieldOrder) {
  using TypedInst = TypeParam;

  Inst inst = InstTestHelper::MakeInst(TypeParam::Kind, Parse::Node(1),
                                       TypeId(2), 3, 4);
  EXPECT_EQ(inst.kind(), TypeParam::Kind);
  EXPECT_EQ(inst.parse_node(), Parse::Node(1));
  EXPECT_EQ(inst.type_id(), TypeId(2));

  TypedInst typed = inst.As<TypedInst>();
  if constexpr (HasParseNode<TypedInst>) {
    EXPECT_EQ(typed.parse_node, Parse::Node(1));
  }
  if constexpr (HasTypeId<TypedInst>) {
    EXPECT_EQ(typed.type_id, TypeId(2));
  }
}

TYPED_TEST(TypedInstTest, RoundTrip) {
  using TypedInst = TypeParam;

  Inst inst1 = InstTestHelper::MakeInst(TypeParam::Kind, Parse::Node(1),
                                        TypeId(2), 3, 4);
  EXPECT_EQ(inst1.kind(), TypeParam::Kind);
  EXPECT_EQ(inst1.parse_node(), Parse::Node(1));
  EXPECT_EQ(inst1.type_id(), TypeId(2));

  TypedInst typed1 = inst1.As<TypedInst>();
  Inst inst2 = typed1;

  EXPECT_EQ(inst1.kind(), inst2.kind());
  if constexpr (HasParseNode<TypedInst>) {
    EXPECT_EQ(inst1.parse_node(), inst2.parse_node());
  }
  if constexpr (HasTypeId<TypedInst>) {
    EXPECT_EQ(inst1.type_id(), inst2.type_id());
  }

  // If the typed instruction has no padding, we should get exactly the same
  // thing if we convert back from an instruction.
  TypedInst typed2 = inst2.As<TypedInst>();
  if constexpr (std::has_unique_object_representations_v<TypedInst>) {
    EXPECT_EQ(std::memcmp(&typed1, &typed2, sizeof(TypedInst)), 0);
  }

  // The original instruction might not be identical after one round trip,
  // because the fields not carried by the typed instruction are lost. But they
  // should be stable if we round-trip again.
  Inst inst3 = typed2;
  if constexpr (std::has_unique_object_representations_v<Inst>) {
    EXPECT_EQ(std::memcmp(&inst2, &inst3, sizeof(Inst)), 0);
  }
}

TYPED_TEST(TypedInstTest, StructLayout) {
  using TypedInst = TypeParam;

  TypedInst typed =
      InstTestHelper::MakeInst(TypeParam::Kind, Parse::Node(1), TypeId(2), 3, 4)
          .template As<TypedInst>();

  // Check that the memory representation of the typed instruction is what we
  // expect.
  // TODO: Struct layout is not guaranteed, and this test could fail in some
  // build environment. If so, we should disable it.
  int32_t fields[4] = {};
  int field = 0;
  if constexpr (HasParseNode<TypedInst>) {
    fields[field++] = 1;
  }
  if constexpr (HasTypeId<TypedInst>) {
    fields[field++] = 2;
  }
  fields[field++] = 3;
  fields[field++] = 4;

  ASSERT_LE(sizeof(TypedInst), sizeof(fields));
  // We can only do this check if the typed instruction has no padding.
  if constexpr (std::has_unique_object_representations_v<TypedInst>) {
    EXPECT_EQ(std::memcmp(&fields, &typed, sizeof(TypedInst)), 0);
  }
}

TYPED_TEST(TypedInstTest, InstKindMatches) {
  using TypedInst = TypeParam;

  // TypedInst::Kind is an InstKind::Definition that extends InstKind, but
  // has different definitions of the `ir_name()` and `terminator_kind()`
  // methods. Here we test that values returned by the two different versions
  // of those functions match.
  InstKind as_kind = TypedInst::Kind;
  EXPECT_EQ(TypedInst::Kind.ir_name(), as_kind.ir_name());
  EXPECT_EQ(TypedInst::Kind.terminator_kind(), as_kind.terminator_kind());
}

}  // namespace
}  // namespace Carbon::SemIR
