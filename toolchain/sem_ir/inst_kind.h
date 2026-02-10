// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_KIND_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_KIND_H_

#include <cstdint>
#include <optional>

#include "common/enum_base.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::SemIR {

// Forward-declared to avoid a cycle.
struct TypeId;

// The expression category of an instruction. See /docs/design/values.md for
// background.
//
// Several categories are concerned with object initialization. At the SemIR
// level, initialization consists of several phase transitions:
// 1. A _repr-initializing_ expression forms an initializing representation of
//    the object's eventual contents.
// 2. An _in-place initializing_ expression commits to writing an object
//    representation to memory.
// 3. An _ephemeral entire reference_ expression commits to a particular memory
//    location for the object.
// 4. Finally, the owner/lifetime for that object is specified. This need not be
//    an expression, so it doesn't correspond to a particular expression
//    category. Instead, the inst kinds that perform this role are marked with
//    has_cleanup = true.`
//
// If an inst combines more than one of those transitions, its category is
// determined by the last one it performs (which means a non-expression inst may
// perform some of the first three steps). Note that the language-level category
// "initializing expression" is the union of the repr-initializing and in-place
// initializing categories, which exist only in the implementation.
//
// An _initializer_ is an inst in any of those three categories. An inst that
// directly depends on it is said to _consume_ it, and typically an initializer
// must be consumed by exactly one inst.
//
// Thus, the key distinction between an initializing expression and a reference
// expression is that the storage location of a reference expression is fixed as
// soon as it is evaluated, but the storage location of an initializing
// expression is notionally set by the inst that consumes it. "Notionally",
// because that distinction is obscured by two optimizations:
// - The storage location inst is always a direct or indirect argument of the
//   in-place initializing inst. The ID of the storage argument inst is fixed
//   when the initializing inst is created, and can be found with
//   `FindStorageArgForInitializer`, but the inst stored at that ID may be
//   overwritten when the consumer is created. This makes the final SemIR appear
//   as though the location was set by the initializing inst.
// - When the initializing inst and its consumer are created together, the
//   initializing inst is typically created with its storage argument already
//   set, rather than creating and then immediately overwriting a placeholder.
//
// TODO: Add an enumerator for ephemeral entire references, when needed.
enum class ExprCategory : int8_t {
  // This instruction does not correspond to an expression, and as such has no
  // category.
  NotExpr,
  // The category of this instruction is not known due to an error.
  Error,
  // This instruction represents a pattern, not an expression.
  Pattern,
  // This instruction represents a value expression.
  Value,
  // This instruction represents a repr-initializing expression (see above),
  // which initializes an object using the type's initializing representation.
  // It must be consumed exactly once unless the type's initializing
  // representation is known not to be in-place.
  ReprInitializing,
  // This instruction represents an in-place initializing expression (see
  // above), which initializes an object in-place, regardless of the type's
  // initializing representation. It must be consumed exactly once.
  InPlaceInitializing,
  // This instruction represents a ephemeral non-entire reference, which denotes
  // an object that does not outlive the current full expression context.
  EphemeralRef,
  // This instruction represents a durable reference expression, which denotes
  // an object that outlives the current full expression context.
  DurableRef,
  // This instruction represents a syntactic combination of expressions that are
  // permitted to have different expression categories. This is used for tuple
  // and struct literals, where the subexpressions for different elements can
  // have different categories.
  Mixed,
  // This instruction is a `RefTagExpr`, and so its semantics (including its
  // expression category) depends on the usage context.
  RefTagged,
  Last = RefTagged
};

// The computation used to determine the expression category for an instruction,
// given its instruction kind. In the case where the instruction kind always has
// the same category, a value from the `ExprCategory` enumeration is used
// directly instead, so these values should not overlap with the `ExprCategory`
// values.
enum ComputedExprCategory : int8_t {
  // The expression category is `Value` if the instruction has a `type_id`
  // field, and `NotExpr` otherwise. This is the default, and is used for
  // convenience because it does the right thing for most instructions.
  ValueIfHasType = -1,
  // The expression category is the same as that of the first operand, which
  // is an `InstId`.
  SameAsFirstOperand = -2,
  // The expression category is the same as that of the first operand, which
  // is an `InstId`.
  SameAsSecondOperand = -3,
  // The expression category depends on the operands in some way not covered
  // by the above options. The category is determined by custom logic in
  // `GetExprCategory`.
  DependsOnOperands = -4,
};

// What kind of expression category an instruction kind produces. The expression
// category in general may depend on the operands of the instruction, but we can
// handle most cases based on the instruction kind alone.
class InstExprCategory {
 public:
  constexpr explicit(false) InstExprCategory(ExprCategory cat)
      : kind_(static_cast<int8_t>(cat)) {}
  constexpr explicit(false) InstExprCategory(ComputedExprCategory kind)
      : kind_(static_cast<int8_t>(kind)) {}

  // If this instruction always has the same category, returns that category.
  // Otherwise returns nullopt.
  constexpr auto TryAsFixedCategory() const -> std::optional<ExprCategory> {
    return kind_ >= 0 ? std::optional(static_cast<ExprCategory>(kind_))
                      : std::nullopt;
  }

  // If the category of this instruction depends on its operands, returns the
  // kind of computation to use to determine the category. Otherwise returns
  // nullopt.
  constexpr auto TryAsComputedCategory() const
      -> std::optional<ComputedExprCategory> {
    return kind_ < 0 ? std::optional(static_cast<ComputedExprCategory>(kind_))
                     : std::nullopt;
  }

 private:
  // A value from either the `ExprCategory` or `ComputedExprCategory`
  // enumerations.
  int8_t kind_;
};

// Whether an instruction defines a type.
enum class InstIsType : int8_t {
  // Always of type `type`, and might define a type constant.
  Always,
  // Sometimes of type `type`, and might define a type constant.
  Maybe,
  // Never defines a type constant. Note that such instructions can still have
  // type `type`, but are not the canonical definition of any type.
  Never,
};

// Whether an instruction can have a constant value, and whether it can be a
// constant inst (i.e. an inst whose canonical ID defines a constant value; see
// constant.h).
//
// This specifies whether an instruction of this kind can have a corresponding
// constant value in the `constant_values()` list, and whether an instruction of
// this kind can be added to the `constants()` list.
enum class InstConstantKind : int8_t {
  // This instruction never has a constant value, and is never a constant inst.
  // This is also used for instructions that don't produce a value at all and
  // aren't used as constants.
  Never,
  // This instruction is never a constant inst, but can reduce to a
  // constant value of a different kind. For example, `UnaryOperatorNot` is
  // never a constant inst; if its operand is a concrete constant, its
  // constant value will instead be a `BoolLiteral`, and if its operand is not a
  // concrete constant, it is non-constant. This is the default.
  Indirect,
  // This instruction can be a symbolic constant inst, depending on its
  // operands, but never a concrete constant inst. For example, a `Call`
  // instruction can be a symbolic constant inst but never a concrete constant
  // inst. The instruction may have a concrete constant value of a different
  // kind.
  SymbolicOnly,
  // This instruction may be a symbolic constant inst if it has symbolic
  // operands, and may be a concrete constant inst if it is a reference
  // expression, but it is never a concrete constant if it is a value or
  // initializing expression. For example, a `TupleAccess` instruction can be a
  // symbolic constant inst when applied to a symbolic constant, and can be a
  // concrete reference constant inst when applied to a reference constant.
  SymbolicOrReference,
  // This instruction is a metaprogramming or template instantiation action that
  // generates an instruction. Like `SymbolicOnly`, it may be a symbolic
  // constant inst depending on its operands, but never a concrete constant
  // inst. The instruction may have a concrete constant value that is a
  // generated instruction. Constant evaluation support for types with this
  // constant kind is provided automatically, by calling `PerformDelayedAction`.
  InstAction,
  // This instruction's operands determine whether it has a constant value,
  // whether it is a constant inst, and/or whether it results in a compile-time
  // error, in ways not expressed by the other InstConstantKinds. For example,
  // `ArrayType` is a compile-time constant if its operands are constant and its
  // array bound is within a valid range, and `ConstType` is a constant inst if
  // its operand is the canonical ID of a constant inst that isn't a
  // `ConstType`.
  Conditional,
  // This instruction is a constant inst if and only if its operands are all the
  // canonical IDs of constant insts, it has a constant value if and only if its
  // operands all have constant values, and that constant value is the result of
  // substituting the operands with their canonical IDs. For example, a
  // `TupleValue` has all these properties. Constant evaluation support for
  // types with this constant kind is provided automatically.
  WheneverPossible,
  // The same as `WheneverPossible`, except that the operands are known in
  // advance to always have a constant value. For example, `IntValue`.
  Always,
  // The instruction may be a unique constant, as described below for
  // `AlwaysUnique`. Otherwise the instruction is not constant. This is used for
  // `VarStorage`, where global variables are `AlwaysUnique` and other variables
  // are non-constant.
  ConditionalUnique,
  // This instruction is itself a unique constant, and its ID is always
  // canonical. This is used for declarations whose constant identity is simply
  // themselves. The `ConstantId` for this instruction will always be a concrete
  // constant whose `InstId` refers directly back to the instruction, rather
  // than to a separate instruction in the constants block.
  // TODO: Decide if this is the model we want for these cases.
  AlwaysUnique,
};

// Whether constant evaluation of an instruction needs the instruction to have
// been created and allocated an InstId, or only needs the instruction operands.
enum class InstConstantNeedsInstIdKind : int8_t {
  // This instruction kind doesn't need an InstId to be evaluated.
  No,
  // This instruction needs an InstId during evaluation, but doesn't need the
  // instruction to persist after evaluation.
  DuringEvaluation,
  // This instruction needs a permanent instruction ID, for example because that
  // instruction ID can appear in the constant result of evaluation.
  Permanent,
};

// Whether an instruction is a terminator or part of the terminator sequence.
// The instructions in a block appear in the order NotTerminator, then
// TerminatorSequence, then Terminator, which is also the numerical order of
// these values.
enum class TerminatorKind : int8_t {
  // This instruction is not a terminator.
  NotTerminator,
  // This instruction is not itself a terminator, but forms part of a terminator
  // sequence.
  TerminatorSequence,
  // This instruction is a terminator.
  Terminator,
};

CARBON_DEFINE_RAW_ENUM_CLASS(InstKind, uint8_t) {
#define CARBON_SEM_IR_INST_KIND(Name) CARBON_RAW_ENUM_ENUMERATOR(Name)
#include "toolchain/sem_ir/inst_kind.def"
};

class InstKind : public CARBON_ENUM_BASE(InstKind) {
 public:
#define CARBON_SEM_IR_INST_KIND(Name) CARBON_ENUM_CONSTANT_DECL(Name)
#include "toolchain/sem_ir/inst_kind.def"

  // Returns the `InstKind` for an instruction, for `CARBON_KIND_SWITCH`.
  template <typename InstT>
  static constexpr auto& For = InstT::Kind;

  template <typename TypedNodeId>
  class Definition;

  // Information about a definition. See associated accessors below for
  // comments.
  struct DefinitionInfo {
    llvm::StringLiteral ir_name;
    InstExprCategory expr_category = ComputedExprCategory::ValueIfHasType;
    InstIsType is_type = InstIsType::Never;
    InstConstantKind constant_kind = InstConstantKind::Indirect;
    InstConstantNeedsInstIdKind constant_needs_inst_id =
        constant_kind == InstConstantKind::AlwaysUnique
            ? InstConstantNeedsInstIdKind::Permanent
            : InstConstantNeedsInstIdKind::No;
    TerminatorKind terminator_kind = TerminatorKind::NotTerminator;
    bool is_lowered = true;
    bool deduce_through = false;
    bool has_cleanup = false;
  };

  // Provides a definition for this instruction kind. Should only be called
  // once, to construct the kind as part of defining it in `typed_insts.h`.
  template <typename TypedNodeId>
  constexpr auto Define(DefinitionInfo info) const -> Definition<TypedNodeId>;

  using EnumBase::AsInt;
  using EnumBase::FromInt;
  using EnumBase::Make;

  // Returns true if the kind matches any of the provided instructions' kinds.
  template <typename... InstT>
  constexpr auto IsAnyOf() const -> bool {
    return ((*this == InstT::Kind) || ...);
  }

  // Returns the name to use for this instruction kind in Semantics IR.
  auto ir_name() const -> llvm::StringLiteral {
    return definition_info(*this).ir_name;
  }

  // Returns the category of expression represented by this instruction kind.
  auto expr_category() const -> InstExprCategory {
    return definition_info(*this).expr_category;
  }

  // Returns whether this instruction kind defines a type.
  auto is_type() const -> InstIsType { return definition_info(*this).is_type; }

  // Returns whether this instruction kind is expected to produce a typed value.
  auto has_type() const -> bool;

  // Returns this instruction kind's category of allowed constants.
  auto constant_kind() const -> InstConstantKind {
    return definition_info(*this).constant_kind;
  }

  // Returns whether we need an `InstId` referring to the instruction to
  // constant evaluate this instruction. If this is set to `true`, then:
  //
  //  - `Check::TryEvalInst` will not allow this instruction to be directly
  //    evaluated without an `InstId`.
  //  - `Check::EvalConstantInst` will be passed an `InstId` for the original
  //    instruction being evaluated.
  //
  // This is set to true for instructions whose evaluation either might need a
  // location, for example for diagnostics or for newly-created instructions,
  // and for instructions whose evaluation needs to inspect the original form of
  // its operands.
  auto constant_needs_inst_id() const -> InstConstantNeedsInstIdKind {
    return definition_info(*this).constant_needs_inst_id;
  }

  // Returns whether this instruction kind is a code block terminator, such as
  // an unconditional branch instruction, or part of the termination sequence,
  // such as a conditional branch instruction. The termination sequence of a
  // code block appears after all other instructions, and ends with a
  // terminator instruction.
  auto terminator_kind() const -> TerminatorKind {
    return definition_info(*this).terminator_kind;
  }

  // Returns true if `Instruction(A)` == `Instruction(B)` allows deduction to
  // conclude `A` == `B`.
  auto deduce_through() const -> bool {
    return definition_info(*this).deduce_through;
  }

  // Returns true if this instruction has scoped cleanup associated, typically a
  // destructor.
  constexpr auto has_cleanup() const -> bool {
    return definition_info(*this).has_cleanup;
  }

 private:
  // Returns the DefinitionInfo for the kind.
  static auto definition_info(InstKind kind) -> const DefinitionInfo&;
};

#define CARBON_SEM_IR_INST_KIND(Name) \
  CARBON_ENUM_CONSTANT_DEFINITION(InstKind, Name)
#include "toolchain/sem_ir/inst_kind.def"

// We expect the instruction kind to fit compactly into 8 bits.
static_assert(sizeof(InstKind) == 1, "Kind objects include padding!");

// A definition of an instruction kind. This is an InstKind value, plus
// ancillary data such as the name to use for the node kind in LLVM IR. These
// are not copyable, and only one instance of this type is expected to exist
// per instruction kind, specifically `TypedInst::Kind`. Use `InstKind`
// instead as a thin wrapper around an instruction kind index.
template <typename TypedNodeIdArg>
class InstKind::Definition : public InstKind {
 public:
  using TypedNodeId = TypedNodeIdArg;

  // Not copyable.
  Definition(const Definition&) = delete;
  auto operator=(const Definition&) -> Definition& = delete;

  // Returns the name to use for this instruction kind in Semantics IR.
  constexpr auto ir_name() const -> llvm::StringLiteral {
    return info_.ir_name;
  }

  // Returns the category of expression represented by this instruction kind.
  constexpr auto expr_category() const -> InstExprCategory {
    return info_.expr_category;
  }

  // Returns whether this instruction kind defines a type.
  constexpr auto is_type() const -> InstIsType { return info_.is_type; }

  // Returns whether instructions of this kind are always symbolic whenever they
  // are types. For convenience, also returns false if the instruction cannot be
  // a type, because this is typically used in requires expressions where that
  // case is handled by a separate overload.
  constexpr auto is_symbolic_when_type() const -> bool {
    // Types are values (not references) of type `type`, so if the instruction
    // kind is always symbolic when it's a value, then it's always symbolic when
    // it's a type.
    return is_type() != InstIsType::Never &&
           (constant_kind() == InstConstantKind::SymbolicOnly ||
            constant_kind() == InstConstantKind::SymbolicOrReference);
  }

  // Returns this instruction kind's category of allowed constants.
  constexpr auto constant_kind() const -> InstConstantKind {
    return info_.constant_kind;
  }

  // Returns whether constant evaluation of this instruction needs an InstId.
  constexpr auto constant_needs_inst_id() const -> InstConstantNeedsInstIdKind {
    return info_.constant_needs_inst_id;
  }

  // Returns whether this instruction kind is a code block terminator. See
  // InstKind::terminator_kind().
  constexpr auto terminator_kind() const -> TerminatorKind {
    return info_.terminator_kind;
  }

  // Returns true if the instruction is lowered.
  constexpr auto is_lowered() const -> bool { return info_.is_lowered; }

  // Returns true if `Instruction(A)` == `Instruction(B)` allows deduction to
  // conclude `A` == `B`.
  constexpr auto deduce_through() const -> bool { return info_.deduce_through; }

  // Returns true if this instruction has scoped cleanup associated, typically a
  // destructor.
  constexpr auto has_cleanup() const -> bool { return info_.has_cleanup; }

 private:
  friend class InstKind;

  constexpr Definition(InstKind kind, InstKind::DefinitionInfo info)
      : InstKind(kind), info_(info) {}

  InstKind::DefinitionInfo info_;
};

template <typename TypedNodeId>
constexpr auto InstKind::Define(DefinitionInfo info) const
    -> Definition<TypedNodeId> {
  return Definition<TypedNodeId>(*this, info);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_KIND_H_
