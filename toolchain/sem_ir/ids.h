// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IDS_H_
#define CARBON_TOOLCHAIN_SEM_IR_IDS_H_

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/value_store.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/builtin_kind.h"

namespace Carbon::SemIR {

// Forward declare indexed types, for integration with ValueStore.
class File;
class Inst;
struct BindNameInfo;
struct Class;
struct Function;
struct ImportIR;
struct ImportIRInst;
struct Interface;
struct Impl;
struct NameScope;
struct TypeInfo;

// The ID of an instruction.
struct InstId : public IdBase, public Printable<InstId> {
  using ValueType = Inst;

  // An explicitly invalid ID.
  static const InstId Invalid;

// Builtin instruction IDs.
#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) static const InstId Builtin##Name;
#include "toolchain/sem_ir/builtin_kind.def"

  // The namespace for a `package` expression.
  static const InstId PackageNamespace;

  // Returns the instruction ID for a builtin. This relies on File guarantees
  // for builtin ImportRefUsed placement.
  static constexpr auto ForBuiltin(BuiltinKind kind) -> InstId {
    return InstId(kind.AsInt());
  }

  using IdBase::IdBase;

  // Returns true if the instruction is a builtin. Requires is_valid.
  auto is_builtin() const -> bool {
    CARBON_CHECK(is_valid());
    return index < BuiltinKind::ValidCount;
  }

  // Returns the BuiltinKind. Requires is_builtin.
  auto builtin_kind() const -> BuiltinKind {
    CARBON_CHECK(is_builtin());
    return BuiltinKind::FromInt(index);
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "inst";
    if (!is_valid()) {
      IdBase::Print(out);
    } else if (is_builtin()) {
      out << builtin_kind();
    } else {
      // Use the `+` as a small reminder that this is a delta, rather than an
      // absolute index.
      out << "+" << index - BuiltinKind::ValidCount;
    }
  }
};

constexpr InstId InstId::Invalid = InstId(InvalidIndex);

#define CARBON_SEM_IR_BUILTIN_KIND_NAME(Name) \
  constexpr InstId InstId::Builtin##Name =    \
      InstId::ForBuiltin(BuiltinKind::Name);
#include "toolchain/sem_ir/builtin_kind.def"

// The package namespace will be the instruction after builtins.
constexpr InstId InstId::PackageNamespace = InstId(BuiltinKind::ValidCount);

// The ID of a constant value of an expression. An expression is either:
//
// - a template constant, with an immediate value, such as `42` or `i32*` or
//   `("hello", "world")`, or
// - a symbolic constant, whose value includes a symbolic parameter, such as
//   `Vector(T*)`, or
// - a runtime expression, such as `Print("hello")`.
struct ConstantId : public IdBase, public Printable<ConstantId> {
  // An ID for an expression that is not constant.
  static const ConstantId NotConstant;
  // An ID for an expression whose phase cannot be determined because it
  // contains an error. This is always modeled as a template constant.
  static const ConstantId Error;
  // An explicitly invalid ID.
  static const ConstantId Invalid;

  // Returns the constant ID corresponding to a template constant, which should
  // either be in the `constants` block in the file or should be known to be
  // unique.
  static constexpr auto ForTemplateConstant(InstId const_id) -> ConstantId {
    return ConstantId(const_id.index + IndexOffset);
  }

  // Returns the constant ID corresponding to a symbolic constant, which should
  // either be in the `constants` block in the file or should be known to be
  // unique.
  static constexpr auto ForSymbolicConstant(InstId const_id) -> ConstantId {
    return ConstantId(-const_id.index - IndexOffset);
  }

  using IdBase::IdBase;

  // Returns whether this represents a constant. Requires is_valid.
  auto is_constant() const -> bool {
    CARBON_CHECK(is_valid());
    return *this != ConstantId::NotConstant;
  }
  // Returns whether this represents a symbolic constant. Requires is_valid.
  auto is_symbolic() const -> bool {
    CARBON_CHECK(is_valid());
    return index <= -IndexOffset;
  }
  // Returns whether this represents a template constant. Requires is_valid.
  auto is_template() const -> bool {
    CARBON_CHECK(is_valid());
    return index >= IndexOffset;
  }

  // Returns the instruction that describes this constant value, or
  // InstId::Invalid for a runtime value. Requires is_valid.
  constexpr auto inst_id() const -> InstId {
    CARBON_CHECK(is_valid());
    return InstId(Abs(index) - IndexOffset);
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    if (!is_valid()) {
      IdBase::Print(out);
    } else if (is_template()) {
      out << "template " << inst_id();
    } else if (is_symbolic()) {
      out << "symbolic " << inst_id();
    } else {
      out << "runtime";
    }
  }

 private:
  // TODO: C++23 makes std::abs constexpr, but until then we mirror std::abs
  // logic here. LLVM should still optimize this.
  static constexpr auto Abs(int32_t i) -> int32_t { return i > 0 ? i : -i; }

  static constexpr int32_t NotConstantIndex = InvalidIndex - 1;
  // The offset of InstId indices to ConstantId indices.
  static constexpr int32_t IndexOffset = -NotConstantIndex + 1;
};

constexpr ConstantId ConstantId::NotConstant = ConstantId(NotConstantIndex);
static_assert(ConstantId::NotConstant.inst_id() == InstId::Invalid);
constexpr ConstantId ConstantId::Error =
    ConstantId::ForTemplateConstant(InstId::BuiltinError);
constexpr ConstantId ConstantId::Invalid = ConstantId(InvalidIndex);

// The ID of a bind name.
struct BindNameId : public IdBase, public Printable<BindNameId> {
  using ValueType = BindNameInfo;

  // An explicitly invalid ID.
  static const BindNameId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "bindName";
    IdBase::Print(out);
  }
};

constexpr BindNameId BindNameId::Invalid = BindNameId(InvalidIndex);

// The ID of a function.
struct FunctionId : public IdBase, public Printable<FunctionId> {
  using ValueType = Function;

  // An explicitly invalid ID.
  static const FunctionId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "function";
    IdBase::Print(out);
  }
};

constexpr FunctionId FunctionId::Invalid = FunctionId(InvalidIndex);

// The ID of a class.
struct ClassId : public IdBase, public Printable<ClassId> {
  using ValueType = Class;

  // An explicitly invalid ID.
  static const ClassId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "class";
    IdBase::Print(out);
  }
};

constexpr ClassId ClassId::Invalid = ClassId(InvalidIndex);

// The ID of an interface.
struct InterfaceId : public IdBase, public Printable<InterfaceId> {
  using ValueType = Interface;

  // An explicitly invalid ID.
  static const InterfaceId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "interface";
    IdBase::Print(out);
  }
};

constexpr InterfaceId InterfaceId::Invalid = InterfaceId(InvalidIndex);

// The ID of an impl.
struct ImplId : public IdBase, public Printable<ImplId> {
  using ValueType = Impl;

  // An explicitly invalid ID.
  static const ImplId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "impl";
    IdBase::Print(out);
  }
};

constexpr ImplId ImplId::Invalid = ImplId(InvalidIndex);

// The ID of an imported IR.
struct ImportIRId : public IdBase, public Printable<ImportIRId> {
  using ValueType = ImportIR;

  // An explicitly invalid ID.
  static const ImportIRId Invalid;

  // The builtin IR's import location.
  static const ImportIRId Builtins;

  // The implicit `api` import, for an `impl` file. A null entry is added if
  // there is none, as in an `api`, in which case this ID should not show up in
  // instructions.
  static const ImportIRId ApiForImpl;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "ir";
    IdBase::Print(out);
  }
};

constexpr ImportIRId ImportIRId::Invalid = ImportIRId(InvalidIndex);
constexpr ImportIRId ImportIRId::Builtins = ImportIRId(0);
constexpr ImportIRId ImportIRId::ApiForImpl = ImportIRId(1);

// A boolean value.
struct BoolValue : public IdBase, public Printable<BoolValue> {
  static const BoolValue False;
  static const BoolValue True;

  // Returns the `BoolValue` corresponding to `b`.
  static constexpr auto From(bool b) -> BoolValue { return b ? True : False; }

  // Returns the `bool` corresponding to this `BoolValue`.
  constexpr auto ToBool() -> bool {
    CARBON_CHECK(*this == False || *this == True)
        << "Invalid bool value " << index;
    return *this != False;
  }

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    if (*this == False) {
      out << "false";
    } else if (*this == True) {
      out << "true";
    } else {
      CARBON_FATAL() << "Invalid bool value " << index;
    }
  }
};

constexpr BoolValue BoolValue::False = BoolValue(0);
constexpr BoolValue BoolValue::True = BoolValue(1);

// An integer kind value -- either "signed" or "unsigned".
//
// This might eventually capture any other properties of an integer type that
// affect its semantics, such as overflow behavior.
struct IntKind : public IdBase, public Printable<IntKind> {
  static const IntKind Unsigned;
  static const IntKind Signed;

  using IdBase::IdBase;

  // Returns whether this type is signed.
  constexpr auto is_signed() -> bool { return *this == Signed; }

  auto Print(llvm::raw_ostream& out) const -> void {
    if (*this == Unsigned) {
      out << "unsigned";
    } else if (*this == Signed) {
      out << "signed";
    } else {
      CARBON_FATAL() << "Invalid int kind value " << index;
    }
  }
};

constexpr IntKind IntKind::Unsigned = IntKind(0);
constexpr IntKind IntKind::Signed = IntKind(1);

// The ID of a name. A name is either a string or a special name such as
// `self`, `Self`, or `base`.
struct NameId : public IdBase, public Printable<NameId> {
  // names().GetFormatted() is used for diagnostics.
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // An explicitly invalid ID.
  static const NameId Invalid;
  // The name of `self`.
  static const NameId SelfValue;
  // The name of `Self`.
  static const NameId SelfType;
  // The name of the return slot in a function.
  static const NameId ReturnSlot;
  // The name of `package`.
  static const NameId PackageNamespace;
  // The name of `base`.
  static const NameId Base;

  // The number of non-index (<0) that exist, and will need storage in name
  // lookup.
  static const int NonIndexValueCount;

  // Returns the NameId corresponding to a particular IdentifierId.
  static auto ForIdentifier(IdentifierId id) -> NameId {
    if (id.index >= 0) {
      return NameId(id.index);
    } else if (!id.is_valid()) {
      return NameId::Invalid;
    } else {
      CARBON_FATAL() << "Unexpected identifier ID " << id;
    }
  }

  using IdBase::IdBase;

  // Returns the IdentifierId corresponding to this NameId, or an invalid
  // IdentifierId if this is a special name.
  auto AsIdentifierId() const -> IdentifierId {
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
    } else if (*this == PackageNamespace) {
      out << "PackageNamespace";
    } else if (*this == Base) {
      out << "Base";
    } else {
      CARBON_CHECK(!is_valid() || index >= 0) << "Unknown index " << index;
      IdBase::Print(out);
    }
  }
};

constexpr NameId NameId::Invalid = NameId(InvalidIndex);
constexpr NameId NameId::SelfValue = NameId(InvalidIndex - 1);
constexpr NameId NameId::SelfType = NameId(InvalidIndex - 2);
constexpr NameId NameId::ReturnSlot = NameId(InvalidIndex - 3);
constexpr NameId NameId::PackageNamespace = NameId(InvalidIndex - 4);
constexpr NameId NameId::Base = NameId(InvalidIndex - 5);
constexpr int NameId::NonIndexValueCount = 6;
// Enforce the link between SpecialValueCount and the last special value.
static_assert(NameId::NonIndexValueCount == -NameId::Base.index);

// The ID of a name scope.
struct NameScopeId : public IdBase, public Printable<NameScopeId> {
  using ValueType = NameScope;

  // An explicitly invalid ID.
  static const NameScopeId Invalid;
  // The package (or file) name scope, guaranteed to be the first added.
  static const NameScopeId Package;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "name_scope";
    IdBase::Print(out);
  }
};

constexpr NameScopeId NameScopeId::Invalid = NameScopeId(InvalidIndex);
constexpr NameScopeId NameScopeId::Package = NameScopeId(0);

// The ID of an instruction block.
struct InstBlockId : public IdBase, public Printable<InstBlockId> {
  using ElementType = InstId;
  using ValueType = llvm::MutableArrayRef<ElementType>;

  // An empty block, reused to avoid allocating empty vectors. Always the
  // 0-index block.
  static const InstBlockId Empty;

  // Exported instructions. Always the 1-index block. Empty until the File is
  // fully checked; intermediate state is in the Check::Context.
  static const InstBlockId Exports;

  // Global declaration initialization instructions. Empty if none are present.
  // Otherwise, __global_init function will be generated and this block will
  // be inserted into it.
  static const InstBlockId GlobalInit;

  // An explicitly invalid ID.
  static const InstBlockId Invalid;

  // An ID for unreachable code.
  static const InstBlockId Unreachable;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    if (*this == Unreachable) {
      out << "unreachable";
    } else if (*this == Empty) {
      out << "empty";
    } else if (*this == Exports) {
      out << "exports";
    } else if (*this == GlobalInit) {
      out << "global_init";
    } else {
      out << "block";
      IdBase::Print(out);
    }
  }
};

constexpr InstBlockId InstBlockId::Empty = InstBlockId(0);
constexpr InstBlockId InstBlockId::Exports = InstBlockId(1);
constexpr InstBlockId InstBlockId::Invalid = InstBlockId(InvalidIndex);
constexpr InstBlockId InstBlockId::Unreachable = InstBlockId(InvalidIndex - 1);
constexpr InstBlockId InstBlockId::GlobalInit = InstBlockId(2);

// The ID of a type.
struct TypeId : public IdBase, public Printable<TypeId> {
  using ValueType = TypeInfo;
  // StringifyType() is used for diagnostics.
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // The builtin TypeType.
  static const TypeId TypeType;

  // The builtin Error.
  static const TypeId Error;

  // An explicitly invalid ID.
  static const TypeId Invalid;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "type";
    if (*this == TypeType) {
      out << "TypeType";
    } else if (*this == Error) {
      out << "Error";
    } else {
      IdBase::Print(out);
    }
  }
};

constexpr TypeId TypeId::TypeType = TypeId(InvalidIndex - 2);
constexpr TypeId TypeId::Error = TypeId(InvalidIndex - 1);
constexpr TypeId TypeId::Invalid = TypeId(InvalidIndex);

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

// The ID of an ImportIRInst.
struct ImportIRInstId : public IdBase, public Printable<InstId> {
  using ValueType = ImportIRInst;

  // An explicitly invalid ID.
  static const ImportIRInstId Invalid;

  using IdBase::IdBase;
};

constexpr ImportIRInstId ImportIRInstId::Invalid = ImportIRInstId(InvalidIndex);

// A SemIR location used exclusively for diagnostic locations.
//
// Contents:
// - index > Invalid: A Parse::NodeId in the current IR.
// - index < Invalid: An ImportIRInstId.
// - index == Invalid: Can be used for either.
struct LocId : public IdBase, public Printable<LocId> {
  // An explicitly invalid ID.
  static const LocId Invalid;

  using IdBase::IdBase;

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(Parse::InvalidNodeId /*invalid*/) : IdBase(InvalidIndex) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(Parse::NodeId node_id) : IdBase(node_id.index) {
    CARBON_CHECK(node_id.is_valid() == is_valid());
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(ImportIRInstId inst_id)
      : IdBase(InvalidIndex + ImportIRInstId::InvalidIndex - inst_id.index) {
    CARBON_CHECK(inst_id.is_valid() == is_valid());
  }

  auto is_node_id() const -> bool { return index > InvalidIndex; }
  auto is_import_ir_inst_id() const -> bool { return index < InvalidIndex; }

  // This is allowed to return an invalid NodeId, but should never be used for a
  // valid InstId.
  auto node_id() const -> Parse::NodeId {
    CARBON_CHECK(is_node_id() || !is_valid());
    return Parse::NodeId(index);
  }

  // This is allowed to return an invalid InstId, but should never be used for a
  // valid NodeId.
  auto import_ir_inst_id() const -> ImportIRInstId {
    CARBON_CHECK(is_import_ir_inst_id() || !is_valid());
    return ImportIRInstId(InvalidIndex + ImportIRInstId::InvalidIndex - index);
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "loc_";
    if (is_node_id() || !is_valid()) {
      out << node_id();
    } else {
      out << import_ir_inst_id();
    }
  }
};

constexpr LocId LocId::Invalid = LocId(Parse::NodeId::Invalid);

}  // namespace Carbon::SemIR

// Support use of Id types as DenseMap/DenseSet keys.
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::ConstantId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::ConstantId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::InstBlockId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::InstBlockId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::InstId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::InstId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::NameId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::NameId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::NameScopeId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::NameScopeId> {};
template <>
struct llvm::DenseMapInfo<Carbon::SemIR::TypeId>
    : public Carbon::IndexMapInfo<Carbon::SemIR::TypeId> {};

#endif  // CARBON_TOOLCHAIN_SEM_IR_IDS_H_
