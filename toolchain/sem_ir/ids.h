// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IDS_H_
#define CARBON_TOOLCHAIN_SEM_IR_IDS_H_

#include <cstdint>

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/sem_ir/builtin_kind.h"

namespace Carbon::SemIR {

// The ID of an instruction.
struct InstId : public IndexBase, public Printable<InstId> {
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

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "inst";
    if (!is_valid()) {
      IndexBase::Print(out);
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
struct FunctionId : public IndexBase, public Printable<FunctionId> {
  // An explicitly invalid function ID.
  static const FunctionId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "function";
    IndexBase::Print(out);
  }
};

constexpr FunctionId FunctionId::Invalid = FunctionId(FunctionId::InvalidIndex);

// The ID of a class.
struct ClassId : public IndexBase, public Printable<ClassId> {
  // An explicitly invalid class ID.
  static const ClassId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "class";
    IndexBase::Print(out);
  }
};

constexpr ClassId ClassId::Invalid = ClassId(ClassId::InvalidIndex);

// The ID of a cross-referenced IR.
struct CrossReferenceIRId : public IndexBase,
                            public Printable<CrossReferenceIRId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "ir";
    IndexBase::Print(out);
  }
};

// A boolean value.
struct BoolValue : public IndexBase, public Printable<BoolValue> {
  static const BoolValue False;
  static const BoolValue True;

  using IndexBase::IndexBase;
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

// The ID of a name scope.
struct NameScopeId : public IndexBase, public Printable<NameScopeId> {
  // An explicitly invalid ID.
  static const NameScopeId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "name_scope";
    IndexBase::Print(out);
  }
};

constexpr NameScopeId NameScopeId::Invalid =
    NameScopeId(NameScopeId::InvalidIndex);

// The ID of an instruction block.
struct InstBlockId : public IndexBase, public Printable<InstBlockId> {
  // All File instances must provide the 0th instruction block as empty.
  static const InstBlockId Empty;

  // An explicitly invalid ID.
  static const InstBlockId Invalid;

  // An ID for unreachable code.
  static const InstBlockId Unreachable;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    if (index == Unreachable.index) {
      out << "unreachable";
    } else {
      out << "block";
      IndexBase::Print(out);
    }
  }
};

constexpr InstBlockId InstBlockId::Empty = InstBlockId(0);
constexpr InstBlockId InstBlockId::Invalid =
    InstBlockId(InstBlockId::InvalidIndex);
constexpr InstBlockId InstBlockId::Unreachable =
    InstBlockId(InstBlockId::InvalidIndex - 1);

// The ID of a type.
struct TypeId : public IndexBase, public Printable<TypeId> {
  // The builtin TypeType.
  static const TypeId TypeType;

  // The builtin Error.
  static const TypeId Error;

  // An explicitly invalid ID.
  static const TypeId Invalid;

  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "type";
    if (index == TypeType.index) {
      out << "TypeType";
    } else if (index == Error.index) {
      out << "Error";
    } else {
      IndexBase::Print(out);
    }
  }
};

constexpr TypeId TypeId::TypeType = TypeId(TypeId::InvalidIndex - 2);
constexpr TypeId TypeId::Error = TypeId(TypeId::InvalidIndex - 1);
constexpr TypeId TypeId::Invalid = TypeId(TypeId::InvalidIndex);

// The ID of a type block.
struct TypeBlockId : public IndexBase, public Printable<TypeBlockId> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "typeBlock";
    IndexBase::Print(out);
  }
};

// An index for member access, for structs and tuples.
struct MemberIndex : public IndexBase, public Printable<MemberIndex> {
  using IndexBase::IndexBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "member";
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
