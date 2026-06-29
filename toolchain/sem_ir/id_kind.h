// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ID_KIND_H_
#define CARBON_TOOLCHAIN_SEM_IR_ID_KIND_H_

#include "common/type_enum.h"
#include "toolchain/base/int.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// An enum of all the ID types used as instruction operands.
//
// As instruction operands, the types listed here can appear as fields of typed
// instructions (`toolchain/sem_ir/typed_insts.h`) and must implement the
// `FromRaw` and `ToRaw` protocol. In most cases this is done by inheriting from
// `IdBase` or `IndexBase`.
//
// clang-format off: We want one per line.
using IdKind = TypeEnum<
    // From base/value_store.h.
    FloatId,
    IntId,
    RealId,
    StringLiteralValueId,
    // From sem_ir/ids.h.
    AbsoluteInstBlockId,
    AbsoluteInstId,
    AnyRawId,
    AssociatedConstantId,
    BoolValue,
    BundleId<CalleePatternMatchAction::Args>,
    BundleId<CallerPatternMatchAction::Args>,
    CallParamIndex,
    CharId,
    ClangDeclId,
    ClassId,
    CompileTimeBindIndex,
    ConstantId,
    CppOverloadSetId,
    CustomLayoutId,
    DeclInstBlockId,
    DestInstId,
    ElementIndex,
    EntityNameId,
    ExprRegionId,
    FacetTypeId,
    FieldId,
    FloatKind,
    FunctionId,
    GenericId,
    ImplId,
    ImportIRId,
    ImportIRInstId,
    InstBlockId,
    InstId,
    InterfaceId,
    IntKind,
    LabelId,
    LibraryNameId,
    LocId,
    MetaInstId,
    NameId,
    NameScopeId,
    NamedConstraintId,
    RawBundleId,
    RequireImplsId,
    SpecificId,
    SpecificInterfaceId,
    StructTypeFieldsId,
    TypeInstId,
    VtableId>;
// clang-format on

// Convert a field to its raw representation.
static constexpr auto ToRaw(AnyIdBase base) -> int32_t { return base.index; }

// Convert a field from its raw representation.
template <typename T>
  requires IdKind::Contains<T>
static constexpr auto FromRaw(int32_t raw) -> T {
  return T(raw);
}

// Specialization for IntId.
static constexpr auto ToRaw(IntId id) -> int32_t { return id.AsRaw(); }
template <>
constexpr auto FromRaw<IntId>(int32_t raw) -> IntId {
  return IntId::MakeRaw(raw);
}

// A type-safe wrapper around any of the ID types in IdKind.
class IdAndKind {
 public:
  explicit IdAndKind(IdKind kind, int32_t value) : kind_(kind), value_(value) {}

  // Converts to `IdT`, validating the `kind` matches.
  template <typename IdT>
  auto As() const -> IdT {
    CARBON_DCHECK(kind_ == IdKind::For<IdT>);
    return IdT(value_);
  }

  // Converts to `IdT`, returning nullopt if the kind is incorrect.
  template <typename IdT>
  auto TryAs() const -> std::optional<IdT> {
    if (kind_ != IdKind::For<IdT>) {
      return std::nullopt;
    }
    return IdT(value_);
  }

  auto kind() const -> IdKind { return kind_; }
  auto value() const -> int32_t { return value_; }

  // Sentinel type that represents TypeEnum::Invalid in a Dispatch overload set.
  // TODO: Consider moving these to TypeEnum.
  struct InvalidType : public Printable<InvalidType> {
    static constexpr llvm::StringLiteral Label = "invalid";
    void Print(llvm::raw_ostream& out) const { out << Label; }
  };

  // Sentinel type that represents TypeEnum::None in a Dispatch overload set.
  struct NoneType : public Printable<NoneType> {
    static constexpr llvm::StringLiteral Label = "none";
    void Print(llvm::raw_ostream& out) const { out << Label; }
  };

  // Converts `*this` to the type corresponding to `kind()`, passes it to `f`,
  // and returns the result. If `kind()` is `Invalid` or `None`, `f` is called
  // with an `InvalidType` or `NoneType` argument. `f`'s return value must
  // be convertible to `R`.
  template <typename R, typename F>
  auto Dispatch(F&& f) const -> R {
    return GetDispatchFn<R, F>(kind_)(std::forward<F>(f), value_);
  }

 private:
  template <typename R, typename F>
  using DispatchFnT = auto(F&& f, int32_t id) -> R;

  template <typename R, typename F, typename... Ids>
  static auto GetDispatchFn(TypeEnum<Ids...> id_kind) -> DispatchFnT<R, F>* {
    static constexpr std::array<DispatchFnT<R, F>*, TypeEnum<Ids...>::NumValues>
        Table = {
            [](F&& f, int32_t id) -> R {
              return std::forward<F>(f)(SemIR::FromRaw<Ids>(id));
            }...,
            [](F&& f, int32_t /*id*/) -> R {
              return std::forward<F>(f)(InvalidType{});
            },
            [](F&& f, int32_t /*id*/) -> R {
              return std::forward<F>(f)(NoneType{});
            },
        };
    return Table[id_kind.ToIndex()];
  }

  IdKind kind_;
  int32_t value_;
};

namespace Internal {
template <typename T>
concept IsIdKindType =
    IdKind::Contains<std::remove_cvref_t<T>> ||
    SameAsOneOf<T, IdAndKind::NoneType, IdAndKind::InvalidType>;
}

// Specializations for Invalid and None.
inline auto ToRaw(IdAndKind::InvalidType /*invalid*/) -> int32_t {
  CARBON_FATAL("Invalid ID kind");
}
template <typename T>
  requires std::is_same_v<T, IdAndKind::InvalidType>
auto FromRaw(int32_t /*raw*/) -> IdAndKind::InvalidType {
  CARBON_FATAL("Invalid ID kind");
}

static constexpr auto ToRaw(IdAndKind::NoneType /*none*/) -> int32_t {
  return AnyIdBase::NoneIndex;
}
template <typename T>
  requires std::is_same_v<T, IdAndKind::NoneType>
constexpr auto FromRaw(int32_t raw) -> IdAndKind::NoneType {
  CARBON_CHECK(raw == AnyIdBase::NoneIndex);
  return {};
}

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ID_KIND_H_
