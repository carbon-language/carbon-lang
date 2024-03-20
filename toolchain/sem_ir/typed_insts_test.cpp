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
  static auto MakeInst(InstKind inst_kind, TypeId type_id, int32_t arg0,
                       int32_t arg1) -> Inst {
    return Inst(inst_kind, type_id, arg0, arg1);
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

auto MakeInstWithNumberedFields(InstKind kind) -> Inst {
  Inst inst = InstTestHelper::MakeInst(kind, TypeId(1), 2, 3);
  EXPECT_EQ(inst.kind(), kind);
  EXPECT_EQ(inst.type_id(), TypeId(1));
  return inst;
}

template <typename TypedInst>
auto CommonFieldOrder() -> void {
  Inst inst = MakeInstWithNumberedFields(TypedInst::Kind);
  auto typed = inst.As<TypedInst>();
  if constexpr (Internal::HasTypeIdMember<TypedInst>) {
    EXPECT_EQ(typed.type_id, TypeId(1));
  }
}

TEST(TypedInstTest, CommonFieldOrder) {
#define CARBON_SEM_IR_INST_KIND(Name) \
  {                                   \
    SCOPED_TRACE(#Name);              \
    CommonFieldOrder<Name>();         \
  }
#include "toolchain/sem_ir/inst_kind.def"
}

auto ExpectEqInsts(const Inst& inst1, const Inst& inst2, bool compare_type_id)
    -> void {
  EXPECT_EQ(inst1.kind(), inst2.kind());
  if (compare_type_id) {
    EXPECT_EQ(inst1.type_id(), inst2.type_id());
  }
}

template <typename TypedInst>
auto RoundTrip() -> void {
  Inst inst1 = MakeInstWithNumberedFields(TypedInst::Kind);
  auto typed1 = inst1.As<TypedInst>();
  Inst inst2 = typed1;

  ExpectEqInsts(inst1, inst2, Internal::HasTypeIdMember<TypedInst>);

  // If the typed instruction has no padding, we should get exactly the same
  // thing if we convert back from an instruction.
  auto typed2 = inst2.As<TypedInst>();
  if constexpr (std::has_unique_object_representations_v<TypedInst>) {
    EXPECT_EQ(std::memcmp(&typed1, &typed2, sizeof(TypedInst)), 0);
  }

  // The original instruction might not be identical after one round trip,
  // because the fields not carried by the typed instruction are lost. But they
  // should be stable if we round-trip again.
  Inst inst3 = typed2;
  ExpectEqInsts(inst2, inst3, true);
}

TEST(TypedInstTest, RoundTrip) {
#define CARBON_SEM_IR_INST_KIND(Name) \
  {                                   \
    SCOPED_TRACE(#Name);              \
    RoundTrip<Name>();                \
  }
#include "toolchain/sem_ir/inst_kind.def"
}

auto StructLayoutHelper(void* typed_inst, std::size_t typed_inst_size,
                        bool has_type_id) -> void {
  // Check that the memory representation of the typed instruction is what we
  // expect.
  // TODO: Struct layout is not guaranteed, and this test could fail in some
  // build environment. If so, we should disable it.
  int32_t fields[3] = {};
  int field = 0;
  if (has_type_id) {
    fields[field++] = 1;
  }
  fields[field++] = 2;
  fields[field++] = 3;

  ASSERT_LE(typed_inst_size, sizeof(int32_t) * field);
  EXPECT_EQ(std::memcmp(&fields, typed_inst, typed_inst_size), 0);
}

template <typename TypedInst>
auto StructLayout() -> void {
  // We can only do this check if the typed instruction has no padding.
  if constexpr (std::has_unique_object_representations_v<TypedInst>) {
    auto typed =
        MakeInstWithNumberedFields(TypedInst::Kind).template As<TypedInst>();
    StructLayoutHelper(&typed, sizeof(typed),
                       Internal::HasTypeIdMember<TypedInst>);
  }
}

TEST(TypedInstTest, StructLayout) {
#define CARBON_SEM_IR_INST_KIND(Name) \
  {                                   \
    SCOPED_TRACE(#Name);              \
    StructLayout<Name>();             \
  }
#include "toolchain/sem_ir/inst_kind.def"
}

template <typename TypedNodeId>
auto InstKindMatches(const InstKind::Definition<TypedNodeId>& def,
                     InstKind kind) {
  EXPECT_EQ(def.ir_name(), kind.ir_name());
  EXPECT_EQ(def.terminator_kind(), kind.terminator_kind());
}

TEST(TypedInstTest, InstKindMatches) {
  // TypedInst::Kind is an InstKind::Definition that extends InstKind, but has
  // different definitions of the `ir_name()` and `terminator_kind()` methods.
  // Here we test that values returned by the two different versions of those
  // functions match.
#define CARBON_SEM_IR_INST_KIND(Name)        \
  {                                          \
    SCOPED_TRACE(#Name);                     \
    InstKindMatches(Name::Kind, Name::Kind); \
  }
#include "toolchain/sem_ir/inst_kind.def"
}

}  // namespace
}  // namespace Carbon::SemIR
