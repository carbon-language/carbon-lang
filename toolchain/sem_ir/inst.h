// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_H_

#include <concepts>
#include <cstdint>

#include "common/check.h"
#include "common/hashing.h"
#include "common/ostream.h"
#include "common/struct_reflection.h"
#include "toolchain/base/index_base.h"
#include "toolchain/sem_ir/block_value_store.h"
#include "toolchain/sem_ir/builtin_inst_kind.h"
#include "toolchain/sem_ir/id_kind.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// InstLikeTypeInfo is an implementation detail, and not public API.
namespace Internal {

// Information about an instruction-like type, which is a type that an Inst can
// be converted to and from. The `Enabled` parameter is used to check
// requirements on the type in the specializations of this template.
template <typename InstLikeType>
struct InstLikeTypeInfo;

// A helper base class for instruction-like types that are structs.
template <typename InstLikeType>
struct InstLikeTypeInfoBase {
  // A corresponding std::tuple<...> type.
  using Tuple =
      decltype(StructReflection::AsTuple(std::declval<InstLikeType>()));

  static constexpr int FirstArgField =
      HasKindMemberAsField<InstLikeType> + HasTypeIdMember<InstLikeType>;

  static constexpr int NumArgs = std::tuple_size_v<Tuple> - FirstArgField;
  static_assert(NumArgs <= 2,
                "Unsupported: typed inst has more than two data fields");

  template <int N>
  using ArgType = std::tuple_element_t<FirstArgField + N, Tuple>;

  template <int N>
  static auto Get(InstLikeType inst) -> ArgType<N> {
    return std::get<FirstArgField + N>(StructReflection::AsTuple(inst));
  }
};

// A particular type of instruction is instruction-like.
template <typename TypedInst>
  requires std::same_as<const InstKind::Definition<
                            typename decltype(TypedInst::Kind)::TypedNodeId>,
                        decltype(TypedInst::Kind)>
struct InstLikeTypeInfo<TypedInst> : InstLikeTypeInfoBase<TypedInst> {
  static_assert(!HasKindMemberAsField<TypedInst>,
                "Instruction type should not have a kind field");
  static auto GetKind(TypedInst /*inst*/) -> InstKind {
    return TypedInst::Kind;
  }
  static auto IsKind(InstKind kind) -> bool { return kind == TypedInst::Kind; }
  // A name that can be streamed to an llvm::raw_ostream.
  static auto DebugName() -> InstKind { return TypedInst::Kind; }
};

// An instruction category is instruction-like.
template <typename InstCat>
  requires std::same_as<const InstKind&, decltype(InstCat::Kinds[0])>
struct InstLikeTypeInfo<InstCat> : InstLikeTypeInfoBase<InstCat> {
  static_assert(HasKindMemberAsField<InstCat>,
                "Instruction category should have a kind field");
  static auto GetKind(InstCat cat) -> InstKind { return cat.kind; }
  static auto IsKind(InstKind kind) -> bool {
    for (InstKind k : InstCat::Kinds) {
      if (k == kind) {
        return true;
      }
    }
    return false;
  }
  // A name that can be streamed to an llvm::raw_ostream.
  static auto DebugName() -> std::string {
    std::string str;
    llvm::raw_string_ostream out(str);
    out << "{";
    llvm::ListSeparator sep;
    for (auto kind : InstCat::Kinds) {
      out << sep << kind;
    }
    out << "}";
    return out.str();
  }
};

// A type is InstLike if InstLikeTypeInfo is defined for it.
template <typename T>
concept InstLikeType = requires { sizeof(InstLikeTypeInfo<T>); };

}  // namespace Internal

// A type-erased representation of a SemIR instruction, that may be constructed
// from the specific kinds of instruction defined in `typed_insts.h`. This
// provides access to common fields present on most or all kinds of
// instructions:
//
// - `kind` for run-time logic when the input Kind is unknown.
// - `type_id` for quick type checking.
//
// In addition, kind-specific data can be accessed by casting to the specific
// kind of instruction:
//
// - Use `inst.kind()` or `Is<InstLikeType>` to determine what kind of
//   instruction it is.
// - Cast to a specific type using `inst.As<InstLikeType>()`
//   - Using the wrong kind in `inst.As<InstLikeType>()` is a programming error,
//     and will CHECK-fail in debug modes (opt may too, but it's not an API
//     guarantee).
// - Use `inst.TryAs<InstLikeType>()` to safely access type-specific instruction
//   data where the instruction's kind is not known.
class Inst : public Printable<Inst> {
 public:
  template <typename TypedInst>
    requires Internal::InstLikeType<TypedInst>
  // NOLINTNEXTLINE(google-explicit-constructor)
  Inst(TypedInst typed_inst)
      // kind_ is always overwritten below.
      : kind_(),
        type_id_(TypeId::Invalid),
        arg0_(InstId::InvalidIndex),
        arg1_(InstId::InvalidIndex) {
    if constexpr (Internal::HasKindMemberAsField<TypedInst>) {
      kind_ = typed_inst.kind.AsInt();
    } else {
      kind_ = TypedInst::Kind.AsInt();
    }
    if constexpr (Internal::HasTypeIdMember<TypedInst>) {
      type_id_ = typed_inst.type_id;
    }
    using Info = Internal::InstLikeTypeInfo<TypedInst>;
    if constexpr (Info::NumArgs > 0) {
      arg0_ = ToRaw(Info::template Get<0>(typed_inst));
    }
    if constexpr (Info::NumArgs > 1) {
      arg1_ = ToRaw(Info::template Get<1>(typed_inst));
    }
  }

  // Returns whether this instruction has the specified type.
  template <typename TypedInst>
    requires Internal::InstLikeType<TypedInst>
  auto Is() const -> bool {
    return Internal::InstLikeTypeInfo<TypedInst>::IsKind(kind());
  }

  // Casts this instruction to the given typed instruction, which must match the
  // instruction's kind, and returns the typed instruction.
  template <typename TypedInst>
    requires Internal::InstLikeType<TypedInst>
  auto As() const -> TypedInst {
    using Info = Internal::InstLikeTypeInfo<TypedInst>;
    CARBON_CHECK(Is<TypedInst>()) << "Casting inst of kind " << kind()
                                  << " to wrong kind " << Info::DebugName();
    auto build_with_type_id_onwards = [&](auto... type_id_onwards) {
      if constexpr (Internal::HasKindMemberAsField<TypedInst>) {
        return TypedInst{kind(), type_id_onwards...};
      } else {
        return TypedInst{type_id_onwards...};
      }
    };

    auto build_with_args = [&](auto... args) {
      if constexpr (Internal::HasTypeIdMember<TypedInst>) {
        return build_with_type_id_onwards(type_id(), args...);
      } else {
        return build_with_type_id_onwards(args...);
      }
    };

    if constexpr (Info::NumArgs == 0) {
      return build_with_args();
    } else if constexpr (Info::NumArgs == 1) {
      return build_with_args(
          FromRaw<typename Info::template ArgType<0>>(arg0_));
    } else if constexpr (Info::NumArgs == 2) {
      return build_with_args(
          FromRaw<typename Info::template ArgType<0>>(arg0_),
          FromRaw<typename Info::template ArgType<1>>(arg1_));
    }
  }

  // If this instruction is the given kind, returns a typed instruction,
  // otherwise returns nullopt.
  template <typename TypedInst>
    requires Internal::InstLikeType<TypedInst>
  auto TryAs() const -> std::optional<TypedInst> {
    if (Is<TypedInst>()) {
      return As<TypedInst>();
    } else {
      return std::nullopt;
    }
  }

  auto kind() const -> InstKind {
    return InstKind::Make(static_cast<InstKind::RawEnumType>(kind_));
  }

  // Gets the type of the value produced by evaluating this instruction.
  auto type_id() const -> TypeId { return type_id_; }

  // Gets the kinds of IDs used for arg0 and arg1 of the specified kind of
  // instruction.
  //
  // TODO: This would ideally live on InstKind, but can't be there for layering
  // reasons.
  static auto ArgKinds(InstKind kind) -> std::pair<IdKind, IdKind> {
    return ArgKindTable[kind.AsInt()];
  }

  // Gets the kinds of IDs used for arg0 and arg1 of this instruction.
  auto ArgKinds() const -> std::pair<IdKind, IdKind> {
    return ArgKinds(kind());
  }

  // Gets the first argument of the instruction. InvalidIndex if there is no
  // such argument.
  auto arg0() const -> int32_t { return arg0_; }

  // Gets the second argument of the instruction. InvalidIndex if there is no
  // such argument.
  auto arg1() const -> int32_t { return arg1_; }

  // Sets the type of this instruction.
  auto SetType(TypeId type_id) -> void { type_id_ = type_id; }

  // Sets the arguments of this instruction.
  auto SetArgs(int32_t arg0, int32_t arg1) {
    arg0_ = arg0;
    arg1_ = arg1;
  }

  auto Print(llvm::raw_ostream& out) const -> void;

  friend auto operator==(Inst lhs, Inst rhs) -> bool {
    return std::memcmp(&lhs, &rhs, sizeof(Inst)) == 0;
  }

 private:
  friend class InstTestHelper;

  // Table mapping instruction kinds to their argument kinds.
  static const std::pair<IdKind, IdKind> ArgKindTable[];

  // Raw constructor, used for testing.
  explicit Inst(InstKind kind, TypeId type_id, int32_t arg0, int32_t arg1)
      : Inst(kind.AsInt(), type_id, arg0, arg1) {}
  explicit Inst(int32_t kind, TypeId type_id, int32_t arg0, int32_t arg1)
      : kind_(kind), type_id_(type_id), arg0_(arg0), arg1_(arg1) {}

  // Convert a field to its raw representation, used as `arg0_` / `arg1_`.
  static constexpr auto ToRaw(IdBase base) -> int32_t { return base.index; }
  static constexpr auto ToRaw(BuiltinInstKind kind) -> int32_t {
    return kind.AsInt();
  }

  // Convert a field from its raw representation.
  template <typename T>
  static constexpr auto FromRaw(int32_t raw) -> T {
    return T(raw);
  }
  template <>
  constexpr auto FromRaw<BuiltinInstKind>(int32_t raw) -> BuiltinInstKind {
    return BuiltinInstKind::FromInt(raw);
  }

  int32_t kind_;
  TypeId type_id_;

  // Use `As` to access arg0 and arg1.
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 16 bytes because we sometimes have 2 arguments for a
// pair of Insts. However, InstKind is 1 byte; if args were 3.5 bytes, we could
// potentially shrink Inst by 4 bytes. This may be worth investigating further.
// Note though that 16 bytes is an ideal size for registers, we may want more
// flags, and 12 bytes would be a more marginal improvement.
static_assert(sizeof(Inst) == 16, "Unexpected Inst size");

// Instruction-like types can be printed by converting them to instructions.
template <typename TypedInst>
  requires Internal::InstLikeType<TypedInst>
inline auto operator<<(llvm::raw_ostream& out, TypedInst inst)
    -> llvm::raw_ostream& {
  Inst(inst).Print(out);
  return out;
}

// Associates a LocId and Inst in order to provide type-checking that the
// TypedNodeId corresponds to the InstT.
struct LocIdAndInst {
  // Constructs a LocIdAndInst with no associated location. This should be used
  // very sparingly: only when it doesn't make sense to store a location even
  // when the instruction kind usually has one, such as for instructions in the
  // constants block.
  template <typename InstT>
  static auto NoLoc(InstT inst) -> LocIdAndInst {
    return LocIdAndInst(LocId::Invalid, inst, /*is_unchecked=*/true);
  }

  // Unsafely form a pair of a location and an instruction. Used in the cases
  // where we can't statically enforce the type matches.
  static auto UncheckedLoc(LocId loc_id, Inst inst) -> LocIdAndInst {
    return LocIdAndInst(loc_id, inst, /*is_unchecked=*/true);
  }

  // Construction for the common case with a typed node.
  template <typename InstT>
    requires(Internal::HasNodeId<InstT>)
  LocIdAndInst(decltype(InstT::Kind)::TypedNodeId node_id, InstT inst)
      : loc_id(node_id), inst(inst) {}

  // Construction for the case where the instruction can have any associated
  // node.
  template <typename InstT>
    requires(Internal::HasUntypedNodeId<InstT>)
  LocIdAndInst(SemIR::LocId loc_id, InstT inst) : loc_id(loc_id), inst(inst) {}

  LocId loc_id;
  Inst inst;

 private:
  // Note `is_unchecked` serves to disambiguate from public constructors.
  explicit LocIdAndInst(LocId loc_id, Inst inst, bool /*is_unchecked*/)
      : loc_id(loc_id), inst(inst) {}
};

// Provides a ValueStore wrapper for an API specific to instructions.
class InstStore {
 public:
  // Adds an instruction to the instruction list, returning an ID to reference
  // the instruction. Note that this doesn't add the instruction to any
  // instruction block. Check::Context::AddInst or InstBlockStack::AddInst
  // should usually be used instead, to add the instruction to the current
  // block.
  auto AddInNoBlock(LocIdAndInst loc_id_and_inst) -> InstId {
    loc_ids_.push_back(loc_id_and_inst.loc_id);
    return values_.Add(loc_id_and_inst.inst);
  }

  // Returns the requested instruction.
  auto Get(InstId inst_id) const -> Inst { return values_.Get(inst_id); }

  // Returns the requested instruction and its location ID.
  auto GetWithLocId(InstId inst_id) const -> LocIdAndInst {
    return LocIdAndInst::UncheckedLoc(GetLocId(inst_id), Get(inst_id));
  }

  // Returns whether the requested instruction is the specified type.
  template <typename InstT>
  auto Is(InstId inst_id) const -> bool {
    return Get(inst_id).Is<InstT>();
  }

  // Returns the requested instruction, which is known to have the specified
  // type.
  template <typename InstT>
  auto GetAs(InstId inst_id) const -> InstT {
    return Get(inst_id).As<InstT>();
  }

  // Returns the requested instruction as the specified type, if it is of that
  // type.
  template <typename InstT>
  auto TryGetAs(InstId inst_id) const -> std::optional<InstT> {
    return Get(inst_id).TryAs<InstT>();
  }

  // Returns the requested instruction as the specified type, if it is valid and
  // of that type. Otherwise returns nullopt.
  template <typename InstT>
  auto TryGetAsIfValid(InstId inst_id) const -> std::optional<InstT> {
    if (!inst_id.is_valid()) {
      return std::nullopt;
    }
    return TryGetAs<InstT>(inst_id);
  }

  auto GetLocId(InstId inst_id) const -> LocId {
    CARBON_CHECK(inst_id.index >= 0) << inst_id.index;
    CARBON_CHECK(inst_id.index < (int)loc_ids_.size())
        << inst_id.index << " " << loc_ids_.size();
    return loc_ids_[inst_id.index];
  }

  // Overwrites a given instruction with a new value.
  auto Set(InstId inst_id, Inst inst) -> void { values_.Get(inst_id) = inst; }

  // Overwrites a given instruction's location with a new value.
  auto SetLocId(InstId inst_id, LocId loc_id) -> void {
    loc_ids_[inst_id.index] = loc_id;
  }

  // Overwrites a given instruction and location ID with a new value.
  auto SetLocIdAndInst(InstId inst_id, LocIdAndInst loc_id_and_inst) -> void {
    Set(inst_id, loc_id_and_inst.inst);
    SetLocId(inst_id, loc_id_and_inst.loc_id);
  }

  // Reserves space.
  auto Reserve(size_t size) -> void {
    loc_ids_.reserve(size);
    values_.Reserve(size);
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Add(MemUsage::ConcatLabel(label, "loc_ids_"), loc_ids_);
    mem_usage.Collect(MemUsage::ConcatLabel(label, "values_"), values_);
  }

  auto array_ref() const -> llvm::ArrayRef<Inst> { return values_.array_ref(); }
  auto size() const -> int { return values_.size(); }

 private:
  llvm::SmallVector<LocId> loc_ids_;
  ValueStore<InstId> values_;
};

// Adapts BlockValueStore for instruction blocks.
class InstBlockStore : public BlockValueStore<InstBlockId> {
 public:
  using BaseType = BlockValueStore<InstBlockId>;

  using BaseType::AddDefaultValue;
  using BaseType::AddUninitialized;

  explicit InstBlockStore(llvm::BumpPtrAllocator& allocator)
      : BaseType(allocator) {
    auto empty_id = AddCanonical({});
    CARBON_CHECK(empty_id == InstBlockId::Empty);
    auto exports_id = AddDefaultValue();
    CARBON_CHECK(exports_id == InstBlockId::Exports);
    auto import_refs_id = AddDefaultValue();
    CARBON_CHECK(import_refs_id == InstBlockId::ImportRefs);
    auto global_init_id = AddDefaultValue();
    CARBON_CHECK(global_init_id == InstBlockId::GlobalInit);
  }

  // Adds a block with the given content, returning an ID to reference it.
  // Returns Empty rather than creating a unique ID if the block is empty.
  auto AddOrEmpty(llvm::ArrayRef<ElementType> content) -> InstBlockId {
    return content.empty() ? InstBlockId::Empty : Add(content);
  }

  auto Set(InstBlockId block_id, llvm::ArrayRef<InstId> content) -> void {
    CARBON_CHECK(block_id != InstBlockId::Unreachable);
    BlockValueStore<InstBlockId>::SetContent(block_id, content);
  }

  // Returns the contents of the specified block, or an empty array if the block
  // is invalid.
  auto GetOrEmpty(InstBlockId block_id) const -> llvm::ArrayRef<InstId> {
    return block_id.is_valid() ? Get(block_id) : llvm::ArrayRef<InstId>();
  }
};

// See common/hashing.h.
inline auto CarbonHashValue(const Inst& value, uint64_t seed) -> HashCode {
  Hasher hasher(seed);
  hasher.HashRaw(value);
  return static_cast<HashCode>(hasher);
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_H_
