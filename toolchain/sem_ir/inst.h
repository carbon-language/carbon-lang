// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_INST_H_
#define CARBON_TOOLCHAIN_SEM_IR_INST_H_

#include <cstdint>
#include <type_traits>

#include "common/check.h"
#include "common/ostream.h"
#include "common/struct_reflection.h"
#include "toolchain/base/index_base.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Information about an instruction-like type, which is a type that an Inst can
// be converted to and from. The `Enabled` parameter is used to check
// requirements on the type in the specializations of this template.
template <typename InstLikeType, bool Enabled = true>
struct InstLikeTypeInfo;

// A helper base class for instruction-like types that are structs.
template <typename InstLikeType>
struct InstLikeTypeInfoBase {
  // The derived class. Useful to allow SFINAE on whether a type is
  // instruction-like: `typename InstLikeTypeInfo<T>::Self` is valid only if `T`
  // is instruction-like.
  using Self = InstLikeTypeInfo<InstLikeType>;

  // A corresponding std::tuple<...> type.
  using Tuple =
      decltype(StructReflection::AsTuple(std::declval<InstLikeType>()));

  static constexpr int FirstArgField = HasKindMemberAsField<InstLikeType> +
                                       HasParseNodeMember<InstLikeType> +
                                       HasTypeIdMember<InstLikeType>;

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
struct InstLikeTypeInfo<
    TypedInst,
    (bool)std::is_same_v<const InstKind::Definition, decltype(TypedInst::Kind)>>
    : InstLikeTypeInfoBase<TypedInst> {
  static_assert(!HasKindMemberAsField<TypedInst>,
                "Instruction type should not have a kind field");
  static auto GetKind(TypedInst) -> InstKind { return TypedInst::Kind; }
  static auto IsKind(InstKind kind) -> bool { return kind == TypedInst::Kind; }
};

// An instruction category is instruction-like.
template <typename InstCat>
struct InstLikeTypeInfo<
    InstCat, (bool)std::is_same_v<const InstKind&, decltype(InstCat::Kinds[0])>>
    : InstLikeTypeInfoBase<InstCat> {
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
};

// A type-erased representation of a SemIR instruction, that may be constructed
// from the specific kinds of instruction defined in `typed_insts.h`. This
// provides access to common fields present on most or all kinds of
// instructions:
//
// - `parse_node` for error placement.
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
  template <typename TypedInst,
            typename Info = typename InstLikeTypeInfo<TypedInst>::Self>
  // NOLINTNEXTLINE(google-explicit-constructor)
  Inst(TypedInst typed_inst)
      : parse_node_(Parse::NodeId::Invalid),
        // Always overwritten below.
        kind_(InstKind::Create({})),
        type_id_(TypeId::Invalid),
        arg0_(InstId::InvalidIndex),
        arg1_(InstId::InvalidIndex) {
    if constexpr (HasParseNodeMember<TypedInst>) {
      parse_node_ = typed_inst.parse_node;
    }
    if constexpr (HasKindMemberAsField<TypedInst>) {
      kind_ = typed_inst.kind;
    } else {
      kind_ = TypedInst::Kind;
    }
    if constexpr (HasTypeIdMember<TypedInst>) {
      type_id_ = typed_inst.type_id;
    }
    if constexpr (Info::NumArgs > 0) {
      arg0_ = ToRaw(Info::template Get<0>(typed_inst));
    }
    if constexpr (Info::NumArgs > 1) {
      arg1_ = ToRaw(Info::template Get<1>(typed_inst));
    }
  }

  // Returns whether this instruction has the specified type.
  template <typename TypedInst, typename Info = InstLikeTypeInfo<TypedInst>>
  auto Is() const -> bool {
    return Info::IsKind(kind());
  }

  // Casts this instruction to the given typed instruction, which must match the
  // instruction's kind, and returns the typed instruction.
  template <typename TypedInst, typename Info = InstLikeTypeInfo<TypedInst>>
  auto As() const -> TypedInst {
    CARBON_CHECK(Is<TypedInst>())
        << "Casting inst of kind " << kind() << " to wrong kind "
        << typeid(TypedInst).name();

    auto build_with_parse_node_onwards = [&](auto... parse_node_onwards) {
      if constexpr (HasKindMemberAsField<TypedInst>) {
        return TypedInst{kind(), parse_node_onwards...};
      } else {
        return TypedInst{parse_node_onwards...};
      }
    };

    auto build_with_type_id_onwards = [&](auto... type_id_onwards) {
      if constexpr (HasParseNodeMember<TypedInst>) {
        return build_with_parse_node_onwards(
            decltype(TypedInst::parse_node)(parse_node()), type_id_onwards...);
      } else {
        return build_with_parse_node_onwards(type_id_onwards...);
      }
    };

    auto build_with_args = [&](auto... args) {
      if constexpr (HasTypeIdMember<TypedInst>) {
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
  auto TryAs() const -> std::optional<TypedInst> {
    if (Is<TypedInst>()) {
      return As<TypedInst>();
    } else {
      return std::nullopt;
    }
  }

  auto parse_node() const -> Parse::NodeId { return parse_node_; }
  auto kind() const -> InstKind { return kind_; }

  // Gets the type of the value produced by evaluating this instruction.
  auto type_id() const -> TypeId { return type_id_; }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  friend class InstTestHelper;

  // Raw constructor, used for testing.
  explicit Inst(InstKind kind, Parse::NodeId parse_node, TypeId type_id,
                int32_t arg0, int32_t arg1)
      : parse_node_(parse_node),
        kind_(kind),
        type_id_(type_id),
        arg0_(arg0),
        arg1_(arg1) {}

  // Convert a field to its raw representation, used as `arg0_` / `arg1_`.
  static constexpr auto ToRaw(IdBase base) -> int32_t { return base.index; }
  static constexpr auto ToRaw(BuiltinKind kind) -> int32_t {
    return kind.AsInt();
  }

  // Convert a field from its raw representation.
  template <typename T>
  static constexpr auto FromRaw(int32_t raw) -> T {
    return T(raw);
  }
  template <>
  constexpr auto FromRaw<BuiltinKind>(int32_t raw) -> BuiltinKind {
    return BuiltinKind::FromInt(raw);
  }

  Parse::NodeId parse_node_;
  InstKind kind_;
  TypeId type_id_;

  // Use `As` to access arg0 and arg1.
  int32_t arg0_;
  int32_t arg1_;
};

// TODO: This is currently 20 bytes because we sometimes have 2 arguments for a
// pair of Insts. However, InstKind is 1 byte; if args were 3.5 bytes, we could
// potentially shrink Inst by 4 bytes. This may be worth investigating further.
static_assert(sizeof(Inst) == 20, "Unexpected Inst size");

// Instruction-like types can be printed by converting them to instructions.
template <typename TypedInst,
          typename = typename InstLikeTypeInfo<TypedInst>::Self>
inline auto operator<<(llvm::raw_ostream& out, TypedInst inst)
    -> llvm::raw_ostream& {
  Inst(inst).Print(out);
  return out;
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_INST_H_
