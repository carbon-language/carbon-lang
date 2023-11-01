// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_KIND_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_KIND_H_

#include <cstdint>

#include "common/enum_base.h"
#include "llvm/ADT/FoldingSet.h"

namespace Carbon::SemIR {

// Whether an inst produces or represents a value, and if so, what kind of
// value.
enum class InstValueKind : int8_t {
  // This inst doesn't produce a value, and shouldn't be referenced by other
  // insts.
  None,
  // This inst represents an expression or expression-like construct that
  // produces a value of the type indicated by its `type_id` field.
  Typed,
};

// Whether an inst is a terminator or part of the terminator sequence. The insts
// in a block appear in the order NotTerminator, then TerminatorSequence, then
// Terminator, which is also the numerical order of these values.
enum class TerminatorKind : int8_t {
  // This inst is not a terminator.
  NotTerminator,
  // This inst is not itself a terminator, but forms part of a terminator
  // sequence.
  TerminatorSequence,
  // This inst is a terminator.
  Terminator,
};

CARBON_DEFINE_RAW_ENUM_CLASS(InstKind, uint8_t) {
#define CARBON_SEM_IR_INST_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/sem_ir/inst_kind.def"
};

class InstKind : public CARBON_ENUM_BASE(InstKind) {
 public:
#define CARBON_SEM_IR_INST_KIND(Name) CARBON_ENUM_CONSTANT_DECLARATION(Name)
#include "toolchain/sem_ir/inst_kind.def"

  using EnumBase::Create;

  // Returns the name to use for this inst kind in Semantics IR.
  [[nodiscard]] auto ir_name() const -> llvm::StringLiteral;

  // Returns whether this kind of inst is expected to produce a value.
  [[nodiscard]] auto value_kind() const -> InstValueKind;

  // Returns whether this inst kind is a code block terminator, such as an
  // unconditional branch instruction, or part of the termination sequence,
  // such as a conditional branch instruction. The termination sequence of a
  // code block appears after all other instructions, and ends with a
  // terminator instruction.
  [[nodiscard]] auto terminator_kind() const -> TerminatorKind;

  // Compute a fingerprint for this inst kind, allowing its use as part of the
  // key in a `FoldingSet`.
  void Profile(llvm::FoldingSetNodeID& id) { id.AddInteger(AsInt()); }

  class Definition;

  // Provides a definition for this inst kind. Should only be called once, to
  // construct the kind as part of defining it in `typed_insts.h`.
  constexpr auto Define(llvm::StringLiteral ir_name,
                        TerminatorKind terminator_kind =
                            TerminatorKind::NotTerminator) const -> Definition;

 private:
  // Looks up the definition for this inst kind.
  [[nodiscard]] auto definition() const -> const Definition&;
};

#define CARBON_SEM_IR_INST_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(InstKind, Name)
#include "toolchain/sem_ir/inst_kind.def"

// We expect the inst kind to fit compactly into 8 bits.
static_assert(sizeof(InstKind) == 1, "Kind objects include padding!");

// A definition of an inst kind. This is an InstKind value, plus ancillary data
// such as the name to use for the node kind in LLVM IR. These are not
// copyable, and only one instance of this type is expected to exist per inst
// kind, specifically `TypedInst::Kind`. Use `InstKind` instead as a thin
// wrapper around an inst kind index.
class InstKind::Definition : public InstKind {
 public:
  // Returns the name to use for this inst kind in Semantics IR.
  [[nodiscard]] constexpr auto ir_name() const -> llvm::StringLiteral {
    return ir_name_;
  }

  // Returns whether this inst kind is a code block terminator. See
  // InstKind::terminator_kind().
  [[nodiscard]] constexpr auto terminator_kind() const -> TerminatorKind {
    return terminator_kind_;
  }

 private:
  friend class InstKind;

  constexpr Definition(InstKind kind, llvm::StringLiteral ir_name,
                       TerminatorKind terminator_kind)
      : InstKind(kind), ir_name_(ir_name), terminator_kind_(terminator_kind) {}

  // Not copyable.
  Definition(const Definition&) = delete;
  Definition& operator=(const Definition&) = delete;

  llvm::StringLiteral ir_name_;
  TerminatorKind terminator_kind_;
};

constexpr auto InstKind::Define(llvm::StringLiteral ir_name,
                                TerminatorKind terminator_kind) const
    -> Definition {
  return Definition(*this, ir_name, terminator_kind);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_KIND_H_
