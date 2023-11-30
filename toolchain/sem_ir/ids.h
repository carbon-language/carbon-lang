// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IDS_H_
#define CARBON_TOOLCHAIN_SEM_IR_IDS_H_

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/value_store.h"
#include "toolchain/sem_ir/builtin_kind.h"

namespace Carbon::SemIR {

// Forward declare indexed types, for integration with ValueStore.
class File;
class Inst;
struct Class;
struct Function;
struct TypeInfo;

// The ID of an instruction.
struct InstId : public IdBase, public Printable<InstId> {
  using ValueType = Inst;

  // An explicitly invalid instruction ID.
  static const InstId Invalid;

// Builtin instruction IDs.
#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) static const InstId Builtin##Name;
#include "toolchain/sem_ir/builtin_kind.def"

  // Returns the cross-reference instruction ID for a builtin. This relies on
  // File guarantees for builtin cross-reference placement.
  static constexpr auto ForBuiltin(BuiltinKind kind) -> InstId {
    return InstId(kind.AsInt());
  }

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "inst";
    if (!is_valid()) {
      IdBase::Print(out);
    } else if (index < BuiltinKind::ValidCount) {
      out << BuiltinKind::FromInt(index);
    } else {
      // Use the `+` as a small reminder that this is a delta, rather than an
      // absolute index.
      out << "+" << index - BuiltinKind::ValidCount;
    }
  }
};

constexpr InstId InstId::Invalid = InstId(InstId::InvalidIndex);

#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) \
  constexpr InstId InstId::Builtin##Name =    \
      InstId::ForBuiltin(BuiltinKind::Name);
#include "toolchain/sem_ir/builtin_kind.def"

// The ID of a function.
struct FunctionId : public IdBase, public Printable<FunctionId> {
  using ValueType = Function;

  // An explicitly invalid function ID.
  static const FunctionId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "function";
    IdBase::Print(out);
  }
};

constexpr FunctionId FunctionId::Invalid = FunctionId(FunctionId::InvalidIndex);

// The ID of a class.
struct ClassId : public IdBase, public Printable<ClassId> {
  using ValueType = Class;

  // An explicitly invalid class ID.
  static const ClassId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "class";
    IdBase::Print(out);
  }
};

constexpr ClassId ClassId::Invalid = ClassId(ClassId::InvalidIndex);

// The ID of a cross-referenced IR.
struct CrossRefIRId : public IdBase, public Printable<CrossRefIRId> {
  using ValueType = const File*;

  static const CrossRefIRId Builtins;
  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "ir";
    IdBase::Print(out);
  }
};

constexpr CrossRefIRId CrossRefIRId::Builtins = CrossRefIRId(0);

// A boolean value.
struct BoolValue : public IdBase, public Printable<BoolValue> {
  static const BoolValue False;
  static const BoolValue True;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    switch (index) {
      case 0:
        out << "false";
        break;
      case 1:
        out << "true";
        break;
      default:
        CARBON_FATAL() << "Invalid bool value " << index;
    }
  }
};

constexpr BoolValue BoolValue::False = BoolValue(0);
constexpr BoolValue BoolValue::True = BoolValue(1);

// The ID of a name. A name is either a string or a special name such as
// `self`, `Self`, or `base`.
struct NameId : public IdBase, public Printable<NameId> {
  // An explicitly invalid ID.
  static const NameId Invalid;
  // The name of `self`.
  static const NameId SelfValue;
  // The name of `Self`.
  static const NameId SelfType;
  // The name of the return slot in a function.
  static const NameId ReturnSlot;
  // The name of `base`.
  static const NameId Base;

  // Returns the NameId corresponding to a particular IdentifierId.
  static auto ForIdentifier(IdentifierId id) -> NameId {
    // NOLINTNEXTLINE(misc-redundant-expression): Asserting to be sure.
    static_assert(NameId::InvalidIndex == IdentifierId::InvalidIndex);
    CARBON_CHECK(id.index >= 0 || id.index == InvalidIndex)
        << "Unexpected identifier ID";
    return NameId(id.index);
  }

  using IdBase::IdBase;

  // Returns the IdentifierId corresponding to this NameId, or an invalid
  // IdentifierId if this is a special name.
  auto AsIdentifierId() -> IdentifierId {
    return index >= 0 ? IdentifierId(index) : IdentifierId::Invalid;
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "name";
    if (*this == SelfValue) {
      out << "SelfValue";
    } else if (*this == SelfType) {
      out << "SelfType";
    } else if (*this == ReturnSlot) {
      out << "ReturnSlot";
    } else if (*this == Base) {
      out << "Base";
    } else {
      CARBON_CHECK(index >= 0) << "Unknown index";
      IdBase::Print(out);
    }
  }
};

constexpr NameId NameId::Invalid = NameId(NameId::InvalidIndex);
constexpr NameId NameId::SelfValue = NameId(NameId::InvalidIndex - 1);
constexpr NameId NameId::SelfType = NameId(NameId::InvalidIndex - 2);
constexpr NameId NameId::ReturnSlot = NameId(NameId::InvalidIndex - 3);
constexpr NameId NameId::Base = NameId(NameId::InvalidIndex - 4);

// The ID of a name scope.
struct NameScopeId : public IdBase, public Printable<NameScopeId> {
  using ValueType = llvm::DenseMap<NameId, InstId>;

  // An explicitly invalid ID.
  static const NameScopeId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "name_scope";
    IdBase::Print(out);
  }
};

constexpr NameScopeId NameScopeId::Invalid =
    NameScopeId(NameScopeId::InvalidIndex);

// The ID of an instruction block.
struct InstBlockId : public IdBase, public Printable<InstBlockId> {
  using ElementType = InstId;
  using ValueType = llvm::MutableArrayRef<ElementType>;

  // All File instances must provide the 0th instruction block as empty.
  static const InstBlockId Empty;

  // An explicitly invalid ID.
  static const InstBlockId Invalid;

  // An ID for unreachable code.
  static const InstBlockId Unreachable;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    if (index == Unreachable.index) {
      out << "unreachable";
    } else {
      out << "block";
      IdBase::Print(out);
    }
  }
};

constexpr InstBlockId InstBlockId::Empty = InstBlockId(0);
constexpr InstBlockId InstBlockId::Invalid =
    InstBlockId(InstBlockId::InvalidIndex);
constexpr InstBlockId InstBlockId::Unreachable =
    InstBlockId(InstBlockId::InvalidIndex - 1);

// The ID of a type.
struct TypeId : public IdBase, public Printable<TypeId> {
  using ValueType = TypeInfo;

  // The builtin TypeType.
  static const TypeId TypeType;

  // The builtin Error.
  static const TypeId Error;

  // An explicitly invalid ID.
  static const TypeId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "type";
    if (index == TypeType.index) {
      out << "TypeType";
    } else if (index == Error.index) {
      out << "Error";
    } else {
      IdBase::Print(out);
    }
  }
};

constexpr TypeId TypeId::TypeType = TypeId(TypeId::InvalidIndex - 2);
constexpr TypeId TypeId::Error = TypeId(TypeId::InvalidIndex - 1);
constexpr TypeId TypeId::Invalid = TypeId(TypeId::InvalidIndex);

// The ID of a type block.
struct TypeBlockId : public IdBase, public Printable<TypeBlockId> {
  using ElementType = TypeId;
  using ValueType = llvm::MutableArrayRef<ElementType>;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "typeBlock";
    IdBase::Print(out);
  }
};

// An index for element access, for structs, tuples, and classes.
struct ElementIndex : public IndexBase, public Printable<ElementIndex> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "element";
    IndexBase::Print(out);
  }
};

}  // namespace Carbon::SemIR

// Support use of Id types as DenseMap/DenseSet keys.
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::InstBlockId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::InstBlockId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::InstId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::InstId> {};

#endif  // CARBON_TOOLCHAIN_SEM_IR_IDS_H_
