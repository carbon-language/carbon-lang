// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NODE_KIND_H_
#define CARBON_TOOLCHAIN_SEM_IR_NODE_KIND_H_

#include <cstdint>

#include "common/enum_base.h"
#include "llvm/ADT/FoldingSet.h"

namespace Carbon::SemIR {

// Whether a node produces or represents a value, and if so, what kind of value.
enum class NodeValueKind : int8_t {
  // This node doesn't produce a value, and shouldn't be referenced by other
  // nodes.
  None,
  // This node represents an expression or expression-like construct that
  // produces a value of the type indicated by its `type_id` field.
  Typed,
};

// Whether a node is a terminator or part of the terminator sequence. The nodes
// in a block appear in the order NotTerminator, then TerminatorSequence, then
// Terminator, which is also the numerical order of these values.
enum class TerminatorKind : int8_t {
  // This node is not a terminator.
  NotTerminator,
  // This node is not itself a terminator, but forms part of a terminator
  // sequence.
  TerminatorSequence,
  // This node is a terminator.
  Terminator,
};

CARBON_DEFINE_RAW_ENUM_CLASS(NodeKind, uint8_t) {
#define CARBON_SEM_IR_NODE_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/sem_ir/node_kind.def"
};

class NodeKind : public CARBON_ENUM_BASE(NodeKind) {
 public:
#define CARBON_SEM_IR_NODE_KIND(Name) CARBON_ENUM_CONSTANT_DECLARATION(Name)
#include "toolchain/sem_ir/node_kind.def"

  using EnumBase::Create;

  // Returns the name to use for this node kind in Semantics IR.
  [[nodiscard]] auto ir_name() const -> llvm::StringRef;

  // Returns whether this kind of node is expected to produce a value.
  [[nodiscard]] auto value_kind() const -> NodeValueKind;

  // Returns whether this node kind is a code block terminator, such as an
  // unconditional branch instruction, or part of the termination sequence,
  // such as a conditional branch instruction. The termination sequence of a
  // code block appears after all other instructions, and ends with a
  // terminator instruction.
  [[nodiscard]] auto terminator_kind() const -> TerminatorKind;

  // Compute a fingerprint for this node kind, allowing its use as part of the
  // key in a `FoldingSet`.
  void Profile(llvm::FoldingSetNodeID& id) { id.AddInteger(AsInt()); }
};

#define CARBON_SEM_IR_NODE_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(NodeKind, Name)
#include "toolchain/sem_ir/node_kind.def"

// We expect the node kind to fit compactly into 8 bits.
static_assert(sizeof(NodeKind) == 1, "Kind objects include padding!");

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_NODE_KIND_H_
