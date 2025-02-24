// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IDS_H_
#define CARBON_TOOLCHAIN_SEM_IR_IDS_H_

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::SemIR {

// Forward declare indexed types, for integration with ValueStore.
class File;
class Inst;
class NameScope;
struct AssociatedConstant;
struct Class;
struct EntityName;
struct ExprRegion;
struct FacetTypeInfo;
struct Function;
struct Generic;
struct CompleteFacetType;
struct Specific;
struct ImportCpp;
struct ImportIR;
struct ImportIRInst;
struct Impl;
struct Interface;
struct StructTypeField;
struct TypeInfo;

// The ID of an instruction.
struct InstId : public IdBase<InstId> {
  static constexpr llvm::StringLiteral Label = "inst";
  using ValueType = Inst;

  // An ID with no value.
  static const InstId None;

  // Represents the result of a name lookup that is temporarily disallowed
  // because the name is currently being initialized.
  static const InstId InitTombstone;

  using IdBase::IdBase;

  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr InstId InstId::None = InstId(NoneIndex);
constexpr InstId InstId::InitTombstone = InstId(NoneIndex - 2);

// An ID of an instruction that is referenced absolutely by another instruction.
// This should only be used as the type of a field within a typed instruction
// class.
//
// When a typed instruction has a field of this type, that field represents an
// absolute reference to another instruction that typically resides in a
// different entity. This behaves in most respects like an InstId field, but
// substitution into the typed instruction leaves the field unchanged rather
// than substituting into it.
class AbsoluteInstId : public InstId {
 public:
  // Support implicit conversion from InstId so that InstId and AbsoluteInstId
  // have the same interface.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr AbsoluteInstId(InstId inst_id) : InstId(inst_id) {}

  using InstId::InstId;
};

// The ID of a constant value of an expression. An expression is either:
//
// - a concrete constant, whose value does not depend on any generic parameters,
//   such as `42` or `i32*` or `("hello", "world")`, or
// - a symbolic constant, whose value includes a generic parameter, such as
//   `Vector(T*)`, or
// - a runtime expression, such as `Print("hello")`.
//
// Concrete constants are a thin wrapper around the instruction ID of the
// constant instruction that defines the constant. Symbolic constants are an
// index into a separate table of `SymbolicConstant`s maintained by the constant
// value store.
struct ConstantId : public IdBase<ConstantId> {
  static constexpr llvm::StringLiteral Label = "constant";

  // An ID for an expression that is not constant.
  static const ConstantId NotConstant;
  // An ID with no value.
  static const ConstantId None;

  // Returns the constant ID corresponding to a concrete constant, which should
  // either be in the `constants` block in the file or should be known to be
  // unique.
  static constexpr auto ForConcreteConstant(InstId const_id) -> ConstantId {
    return ConstantId(const_id.index);
  }

  // Returns the constant ID corresponding to a symbolic constant index.
  static constexpr auto ForSymbolicConstantIndex(int32_t symbolic_index)
      -> ConstantId {
    return ConstantId(FirstSymbolicIndex - symbolic_index);
  }

  using IdBase::IdBase;

  // Returns whether this represents a constant. Requires has_value.
  auto is_constant() const -> bool {
    CARBON_DCHECK(has_value());
    return *this != ConstantId::NotConstant;
  }
  // Returns whether this represents a symbolic constant. Requires has_value.
  auto is_symbolic() const -> bool {
    CARBON_DCHECK(has_value());
    return index <= FirstSymbolicIndex;
  }
  // Returns whether this represents a concrete constant. Requires has_value.
  auto is_concrete() const -> bool {
    CARBON_DCHECK(has_value());
    return index >= 0;
  }

  // Prints this ID to the given output stream. `disambiguate` indicates whether
  // concrete constants should be wrapped with "concrete_constant(...)" so that
  // they aren't printed the same as an InstId. This can be set to false if
  // there is no risk of ambiguity.
  auto Print(llvm::raw_ostream& out, bool disambiguate = true) const -> void;

 private:
  friend class ConstantValueStore;

  // TODO: C++23 makes std::abs constexpr, but until then we mirror std::abs
  // logic here. LLVM should still optimize this.
  static constexpr auto Abs(int32_t i) -> int32_t { return i > 0 ? i : -i; }

  // Returns the instruction that describes this concrete constant value.
  // Requires `is_concrete()`. Use `ConstantValueStore::GetInstId` to get the
  // instruction ID of a `ConstantId`.
  constexpr auto concrete_inst_id() const -> InstId {
    CARBON_DCHECK(is_concrete());
    return InstId(index);
  }

  // Returns the symbolic constant index that describes this symbolic constant
  // value. Requires `is_symbolic()`.
  constexpr auto symbolic_index() const -> int32_t {
    CARBON_DCHECK(is_symbolic());
    return FirstSymbolicIndex - index;
  }

  static constexpr int32_t NotConstantIndex = NoneIndex - 1;
  static constexpr int32_t FirstSymbolicIndex = NoneIndex - 2;
};

constexpr ConstantId ConstantId::NotConstant = ConstantId(NotConstantIndex);
constexpr ConstantId ConstantId::None = ConstantId(NoneIndex);

// The ID of a EntityName.
struct EntityNameId : public IdBase<EntityNameId> {
  static constexpr llvm::StringLiteral Label = "entity_name";
  using ValueType = EntityName;

  // An ID with no value.
  static const EntityNameId None;

  using IdBase::IdBase;
};

constexpr EntityNameId EntityNameId::None = EntityNameId(NoneIndex);

// The index of a compile-time binding. This is the de Bruijn level for the
// binding -- that is, this is the number of other compile time bindings whose
// scope encloses this binding.
struct CompileTimeBindIndex : public IndexBase<CompileTimeBindIndex> {
  static constexpr llvm::StringLiteral Label = "comp_time_bind";

  // An index with no value.
  static const CompileTimeBindIndex None;

  using IndexBase::IndexBase;
};

constexpr CompileTimeBindIndex CompileTimeBindIndex::None =
    CompileTimeBindIndex(NoneIndex);

// The index of a runtime parameter in a function. These are allocated
// sequentially, left-to-right, to the function parameters that will have
// arguments passed to them at runtime. In a `call` instruction, a runtime
// argument will have the position in the argument list corresponding to its
// runtime parameter index.
// TODO: Rename this to CallParamIndex, for consistency with the "`Call`
// parameters" terminology in EntityWithParamsBase.
struct RuntimeParamIndex : public IndexBase<RuntimeParamIndex> {
  static constexpr llvm::StringLiteral Label = "runtime_param";

  // An index with no value.
  static const RuntimeParamIndex None;

  // An placeholder for index whose value is not yet known.
  static const RuntimeParamIndex Unknown;

  using IndexBase::IndexBase;

  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr RuntimeParamIndex RuntimeParamIndex::None =
    RuntimeParamIndex(NoneIndex);

constexpr RuntimeParamIndex RuntimeParamIndex::Unknown =
    RuntimeParamIndex(NoneIndex - 1);

// The ID of a function.
struct FunctionId : public IdBase<FunctionId> {
  static constexpr llvm::StringLiteral Label = "function";
  using ValueType = Function;

  // An ID with no value.
  static const FunctionId None;

  using IdBase::IdBase;
};

constexpr FunctionId FunctionId::None = FunctionId(NoneIndex);

// The ID of an IR within the set of all IRs being evaluated in the current
// check execution.
struct CheckIRId : public IdBase<CheckIRId> {
  static constexpr llvm::StringLiteral Label = "check_ir";
  using IdBase::IdBase;
};

// The ID of a class.
struct ClassId : public IdBase<ClassId> {
  static constexpr llvm::StringLiteral Label = "class";
  using ValueType = Class;

  // An ID with no value.
  static const ClassId None;

  using IdBase::IdBase;
};

constexpr ClassId ClassId::None = ClassId(NoneIndex);

// The ID of an interface.
struct InterfaceId : public IdBase<InterfaceId> {
  static constexpr llvm::StringLiteral Label = "interface";
  using ValueType = Interface;

  // An ID with no value.
  static const InterfaceId None;

  using IdBase::IdBase;
};

constexpr InterfaceId InterfaceId::None = InterfaceId(NoneIndex);

// The ID of an associated constant.
struct AssociatedConstantId : public IdBase<AssociatedConstantId> {
  static constexpr llvm::StringLiteral Label = "assoc_const";
  using ValueType = AssociatedConstant;

  // An ID with no value.
  static const AssociatedConstantId None;

  using IdBase::IdBase;
};

constexpr AssociatedConstantId AssociatedConstantId::None =
    AssociatedConstantId(NoneIndex);

// The ID of an facet type value.
struct FacetTypeId : public IdBase<FacetTypeId> {
  static constexpr llvm::StringLiteral Label = "facet_type";
  using ValueType = FacetTypeInfo;

  // An ID with no value.
  static const FacetTypeId None;

  using IdBase::IdBase;
};

constexpr FacetTypeId FacetTypeId::None = FacetTypeId(NoneIndex);

// The ID of an resolved facet type value.
struct CompleteFacetTypeId : public IdBase<CompleteFacetTypeId> {
  static constexpr llvm::StringLiteral Label = "complete_facet_type";
  using ValueType = CompleteFacetType;

  // An ID with no value.
  static const CompleteFacetTypeId None;

  using IdBase::IdBase;
};

constexpr CompleteFacetTypeId CompleteFacetTypeId::None =
    CompleteFacetTypeId(NoneIndex);

// The ID of an impl.
struct ImplId : public IdBase<ImplId> {
  static constexpr llvm::StringLiteral Label = "impl";
  using ValueType = Impl;

  // An ID with no value.
  static const ImplId None;

  using IdBase::IdBase;
};

constexpr ImplId ImplId::None = ImplId(NoneIndex);

// The ID of a generic.
struct GenericId : public IdBase<GenericId> {
  static constexpr llvm::StringLiteral Label = "generic";
  using ValueType = Generic;

  // An ID with no value.
  static const GenericId None;

  using IdBase::IdBase;
};

constexpr GenericId GenericId::None = GenericId(NoneIndex);

// The ID of a specific, which is the result of specifying the generic arguments
// for a generic.
struct SpecificId : public IdBase<SpecificId> {
  static constexpr llvm::StringLiteral Label = "specific";
  using ValueType = Specific;

  // An ID with no value. This is typically used to represent a non-generic
  // entity.
  static const SpecificId None;

  using IdBase::IdBase;
};

constexpr SpecificId SpecificId::None = SpecificId(NoneIndex);

// The index of an instruction that depends on generic parameters within a
// region of a generic. A corresponding specific version of the instruction can
// be found in each specific corresponding to that generic. This is a pair of a
// region and an index, stored in 32 bits.
struct GenericInstIndex : public IndexBase<GenericInstIndex> {
  // Where the value is first used within the generic.
  enum Region : uint8_t {
    // In the declaration.
    Declaration,
    // In the definition.
    Definition,
  };

  // An index with no value.
  static const GenericInstIndex None;

  explicit constexpr GenericInstIndex(Region region, int32_t index)
      : IndexBase(region == Declaration ? index
                                        : FirstDefinitionIndex - index) {
    CARBON_CHECK(index >= 0);
  }

  // Returns the index of the instruction within the region.
  auto index() const -> int32_t {
    CARBON_CHECK(has_value());
    return IndexBase::index >= 0 ? IndexBase::index
                                 : FirstDefinitionIndex - IndexBase::index;
  }

  // Returns the region within which this instruction was first used.
  auto region() const -> Region {
    CARBON_CHECK(has_value());
    return IndexBase::index >= 0 ? Declaration : Definition;
  }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  static constexpr auto MakeNone() -> GenericInstIndex {
    GenericInstIndex result(Declaration, 0);
    result.IndexBase::index = NoneIndex;
    return result;
  }

  static constexpr int32_t FirstDefinitionIndex = NoneIndex - 1;
};

constexpr GenericInstIndex GenericInstIndex::None =
    GenericInstIndex::MakeNone();

struct ImportCppId : public IdBase<ImportCppId> {
  static constexpr llvm::StringLiteral Label = "import_cpp";
  using ValueType = ImportCpp;

  // An ID with no value.
  static const ImportCppId None;

  using IdBase::IdBase;
};

constexpr ImportCppId ImportCppId::None = ImportCppId(NoneIndex);

// The ID of an IR within the set of imported IRs, both direct and indirect.
struct ImportIRId : public IdBase<ImportIRId> {
  static constexpr llvm::StringLiteral Label = "ir";
  using ValueType = ImportIR;

  // An ID with no value.
  static const ImportIRId None;

  // The implicit `api` import, for an `impl` file. A null entry is added if
  // there is none, as in an `api`, in which case this ID should not show up in
  // instructions.
  static const ImportIRId ApiForImpl;

  using IdBase::IdBase;
};

constexpr ImportIRId ImportIRId::None = ImportIRId(NoneIndex);
constexpr ImportIRId ImportIRId::ApiForImpl = ImportIRId(0);

// A boolean value.
struct BoolValue : public IdBase<BoolValue> {
  // Not used by `Print`, but for `IdKind`.
  static constexpr llvm::StringLiteral Label = "bool";

  static const BoolValue False;
  static const BoolValue True;

  // Returns the `BoolValue` corresponding to `b`.
  static constexpr auto From(bool b) -> BoolValue { return b ? True : False; }

  // Returns the `bool` corresponding to this `BoolValue`.
  constexpr auto ToBool() -> bool {
    CARBON_CHECK(*this == False || *this == True, "Invalid bool value {0}",
                 index);
    return *this != False;
  }

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr BoolValue BoolValue::False = BoolValue(0);
constexpr BoolValue BoolValue::True = BoolValue(1);

// An integer kind value -- either "signed" or "unsigned".
//
// This might eventually capture any other properties of an integer type that
// affect its semantics, such as overflow behavior.
struct IntKind : public IdBase<IntKind> {
  // Not used by `Print`, but for `IdKind`.
  static constexpr llvm::StringLiteral Label = "int_kind";

  static const IntKind Unsigned;
  static const IntKind Signed;

  using IdBase::IdBase;

  // Returns whether this type is signed.
  constexpr auto is_signed() -> bool { return *this == Signed; }

  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr IntKind IntKind::Unsigned = IntKind(0);
constexpr IntKind IntKind::Signed = IntKind(1);

// A float kind value.
struct FloatKind : public IdBase<FloatKind> {
  // Not used by `Print`, but for `IdKind`.
  static constexpr llvm::StringLiteral Label = "float_kind";

  using IdBase::IdBase;

  auto Print(llvm::raw_ostream& out) const -> void { out << "float"; }
};

// The ID of a name. A name is either a string or a special name such as
// `self`, `Self`, or `base`.
struct NameId : public IdBase<NameId> {
  static constexpr llvm::StringLiteral Label = "name";

  // names().GetFormatted() is used for diagnostics.
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // An ID with no value.
  static const NameId None;
  // The name of `base`.
  static const NameId Base;
  // The name of the package `Core`.
  static const NameId Core;
  // The name of `package`.
  static const NameId PackageNamespace;
  // The name of `.Self`.
  static const NameId PeriodSelf;
  // The name of the return slot in a function.
  static const NameId ReturnSlot;
  // The name of `Self`.
  static const NameId SelfType;
  // The name of `self`.
  static const NameId SelfValue;
  // The name of `vptr`.
  static const NameId Vptr;
  // The name of the discriminant field (if any) in a choice.
  static const NameId ChoiceDiscriminant;

  // The number of non-index (<0) that exist, and will need storage in name
  // lookup.
  static const int NonIndexValueCount;

  // Returns the NameId corresponding to a particular IdentifierId.
  static auto ForIdentifier(IdentifierId id) -> NameId;

  // Returns the NameId corresponding to a particular PackageNameId. This is the
  // name that is declared when the package is imported.
  static auto ForPackageName(PackageNameId id) -> NameId;

  using IdBase::IdBase;

  // Returns the IdentifierId corresponding to this NameId, or `None` if this is
  // a special name.
  auto AsIdentifierId() const -> IdentifierId {
    return index >= 0 ? IdentifierId(index) : IdentifierId::None;
  }

  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr NameId NameId::None = NameId(NoneIndex);
constexpr NameId NameId::Base = NameId(NoneIndex - 1);
constexpr NameId NameId::Core = NameId(NoneIndex - 2);
constexpr NameId NameId::PackageNamespace = NameId(NoneIndex - 3);
constexpr NameId NameId::PeriodSelf = NameId(NoneIndex - 4);
constexpr NameId NameId::ReturnSlot = NameId(NoneIndex - 5);
constexpr NameId NameId::SelfType = NameId(NoneIndex - 6);
constexpr NameId NameId::SelfValue = NameId(NoneIndex - 7);
constexpr NameId NameId::Vptr = NameId(NoneIndex - 8);
constexpr NameId NameId::ChoiceDiscriminant = NameId(NoneIndex - 9);
constexpr int NameId::NonIndexValueCount = 10;
// Enforce the link between SpecialValueCount and the last special value.
static_assert(NameId::NonIndexValueCount == -NameId::ChoiceDiscriminant.index);

// The ID of a name scope.
struct NameScopeId : public IdBase<NameScopeId> {
  static constexpr llvm::StringLiteral Label = "name_scope";
  using ValueType = NameScope;

  // An ID with no value.
  static const NameScopeId None;
  // The package (or file) name scope, guaranteed to be the first added.
  static const NameScopeId Package;

  using IdBase::IdBase;
};

constexpr NameScopeId NameScopeId::None = NameScopeId(NoneIndex);
constexpr NameScopeId NameScopeId::Package = NameScopeId(0);

// The ID of an instruction block.
struct InstBlockId : public IdBase<InstBlockId> {
  static constexpr llvm::StringLiteral Label = "inst_block";
  // Types for BlockValueStore<InstBlockId>.
  using ElementType = InstId;
  using ValueType = llvm::MutableArrayRef<ElementType>;

  // The canonical empty block, reused to avoid allocating empty vectors. Always
  // the 0-index block.
  static const InstBlockId Empty;

  // Exported instructions. Empty until the File is fully checked; intermediate
  // state is in the Check::Context.
  static const InstBlockId Exports;

  // ImportRef instructions. Empty until the File is fully checked; intermediate
  // state is in the Check::Context.
  static const InstBlockId ImportRefs;

  // Global declaration initialization instructions. Empty if none are present.
  // Otherwise, __global_init function will be generated and this block will
  // be inserted into it.
  static const InstBlockId GlobalInit;

  // An ID with no value.
  static const InstBlockId None;

  // An ID for unreachable code.
  static const InstBlockId Unreachable;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr InstBlockId InstBlockId::Empty = InstBlockId(0);
constexpr InstBlockId InstBlockId::Exports = InstBlockId(1);
constexpr InstBlockId InstBlockId::ImportRefs = InstBlockId(2);
constexpr InstBlockId InstBlockId::GlobalInit = InstBlockId(3);
constexpr InstBlockId InstBlockId::None = InstBlockId(NoneIndex);
constexpr InstBlockId InstBlockId::Unreachable = InstBlockId(NoneIndex - 1);

// An ID of an instruction block that is referenced absolutely by an
// instruction. This should only be used as the type of a field within a typed
// instruction class. See AbsoluteInstId.
class AbsoluteInstBlockId : public InstBlockId {
 public:
  // Support implicit conversion from InstBlockId so that InstBlockId and
  // AbsoluteInstBlockId have the same interface.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr AbsoluteInstBlockId(InstBlockId inst_block_id)
      : InstBlockId(inst_block_id) {}

  using InstBlockId::InstBlockId;
};

// An ID of an instruction block that is used as the declaration block within a
// declaration instruction. This is a block that is nested within the
// instruction, but doesn't contribute to its value. Such blocks are not
// included in the fingerprint of the declaration. This should only be used as
// the type of a field within a typed instruction class.
class DeclInstBlockId : public InstBlockId {
 public:
  // Support implicit conversion from InstBlockId so that InstBlockId and
  // DeclInstBlockId have the same interface.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr DeclInstBlockId(InstBlockId inst_block_id)
      : InstBlockId(inst_block_id) {}

  using InstBlockId::InstBlockId;
};

// An ID of an instruction block that is used as a label in a branch instruction
// or similar. This is a block that is not nested within the instruction, but
// instead exists elsewhere in the enclosing executable region. This should
// only be used as the type of a field within a typed instruction class.
class LabelId : public InstBlockId {
 public:
  // Support implicit conversion from InstBlockId so that InstBlockId and
  // LabelId have the same interface.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LabelId(InstBlockId inst_block_id) : InstBlockId(inst_block_id) {}

  using InstBlockId::InstBlockId;
};

// TODO: Move this out of sem_ir and into check, if we don't wind up using it
// in the SemIR for expression patterns.
struct ExprRegionId : public IdBase<ExprRegionId> {
  static constexpr llvm::StringLiteral Label = "region";
  using ValueType = ExprRegion;

  // An ID with no value.
  static const ExprRegionId None;

  using IdBase::IdBase;
};

constexpr ExprRegionId ExprRegionId::None = ExprRegionId(NoneIndex);

// The ID of a struct type field block.
struct StructTypeFieldsId : public IdBase<StructTypeFieldsId> {
  static constexpr llvm::StringLiteral Label = "struct_type_fields";
  // Types for BlockValueStore<StructTypeFieldsId>.
  using ElementType = StructTypeField;
  using ValueType = llvm::MutableArrayRef<StructTypeField>;

  // An ID with no value.
  static const StructTypeFieldsId None;

  // The canonical empty block, reused to avoid allocating empty vectors. Always
  // the 0-index block.
  static const StructTypeFieldsId Empty;

  using IdBase::IdBase;
};

constexpr StructTypeFieldsId StructTypeFieldsId::None =
    StructTypeFieldsId(NoneIndex);
constexpr StructTypeFieldsId StructTypeFieldsId::Empty = StructTypeFieldsId(0);

// The ID of a type.
struct TypeId : public IdBase<TypeId> {
  static constexpr llvm::StringLiteral Label = "type";

  // `StringifyTypeExpr` is used for diagnostics. However, where possible, an
  // `InstId` describing how the type was written should be preferred, using
  // `InstIdAsType` or `TypeOfInstId` as the diagnostic argument type.
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // An ID with no value.
  static const TypeId None;

  using IdBase::IdBase;

  // Returns the ID of the type corresponding to the constant `const_id`, which
  // must be of type `type`. As an exception, the type `Error` is of type
  // `Error`.
  static constexpr auto ForTypeConstant(ConstantId const_id) -> TypeId {
    return TypeId(const_id.index);
  }

  // Returns the constant ID that defines the type.
  auto AsConstantId() const -> ConstantId { return ConstantId(index); }

  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr TypeId TypeId::None = TypeId(NoneIndex);

// The ID of a type block.
struct TypeBlockId : public IdBase<TypeBlockId> {
  static constexpr llvm::StringLiteral Label = "type_block";
  // Types for BlockValueStore<TypeBlockId>.
  using ElementType = TypeId;
  using ValueType = llvm::MutableArrayRef<ElementType>;

  // An ID with no value.
  static const TypeBlockId None;

  // The canonical empty block, reused to avoid allocating empty vectors. Always
  // the 0-index block.
  static const TypeBlockId Empty;

  using IdBase::IdBase;
};

constexpr TypeBlockId TypeBlockId::None = TypeBlockId(NoneIndex);
constexpr TypeBlockId TypeBlockId::Empty = TypeBlockId(0);

// An index for element access, for structs, tuples, and classes.
struct ElementIndex : public IndexBase<ElementIndex> {
  static constexpr llvm::StringLiteral Label = "element";
  using IndexBase::IndexBase;

  // An ID with no value.
  static const ElementIndex None;
};

constexpr ElementIndex ElementIndex::None = ElementIndex(NoneIndex);

// The ID of a library name. This is either a string literal or `default`.
struct LibraryNameId : public IdBase<LibraryNameId> {
  static constexpr llvm::StringLiteral Label = "library_name";
  using DiagnosticType = DiagnosticTypeInfo<std::string>;

  // An ID with no value.
  static const LibraryNameId None;
  // The name of `default`.
  static const LibraryNameId Default;
  // Track cases where the library name was set, but has been diagnosed and
  // shouldn't be used anymore.
  static const LibraryNameId Error;

  // Returns the LibraryNameId for a library name as a string literal.
  static auto ForStringLiteralValueId(StringLiteralValueId id) -> LibraryNameId;

  using IdBase::IdBase;

  // Converts a LibraryNameId back to a string literal.
  auto AsStringLiteralValueId() const -> StringLiteralValueId {
    CARBON_CHECK(index >= NoneIndex, "{0} must be handled directly", *this);
    return StringLiteralValueId(index);
  }

  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr LibraryNameId LibraryNameId::None = LibraryNameId(NoneIndex);
constexpr LibraryNameId LibraryNameId::Default = LibraryNameId(NoneIndex - 1);
constexpr LibraryNameId LibraryNameId::Error = LibraryNameId(NoneIndex - 2);

// The ID of an ImportIRInst.
struct ImportIRInstId : public IdBase<ImportIRInstId> {
  static constexpr llvm::StringLiteral Label = "import_ir_inst";
  using ValueType = ImportIRInst;

  // An ID with no value.
  static const ImportIRInstId None;

  using IdBase::IdBase;
};

constexpr ImportIRInstId ImportIRInstId::None = ImportIRInstId(NoneIndex);

// A SemIR location used as the location of instructions.
//
// Contents:
// - index > None: A Parse::NodeId in the current IR.
// - index < None: An ImportIRInstId.
// - index == None: Can be used for either.
struct LocId : public IdBase<LocId> {
  static constexpr llvm::StringLiteral Label = "loc";

  // This bit, if set for a node ID location, indicates a location for
  // operations performed implicitly.
  static const int32_t ImplicitBit = 1 << 30;

  // An ID with no value.
  static const LocId None;

  using IdBase::IdBase;

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(Parse::NoneNodeId /*none*/) : IdBase(NoneIndex) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(Parse::NodeId node_id) : IdBase(node_id.index) {
    CARBON_CHECK(node_id.has_value() == has_value());
    CARBON_CHECK(!is_implicit());
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(ImportIRInstId inst_id)
      : IdBase(NoneIndex + ImportIRInstId::NoneIndex - inst_id.index) {
    CARBON_CHECK(inst_id.has_value() == has_value());
    CARBON_CHECK(index & ImplicitBit);
  }

  // Forms an equivalent LocId for an implicit location.
  auto ToImplicit() const -> LocId {
    // For import IR locations and the `None` location, the implicit bit is
    // always set, so this is a no-op.
    return LocId(index | ImplicitBit);
  }

  auto is_node_id() const -> bool { return index > NoneIndex; }
  auto is_import_ir_inst_id() const -> bool { return index < NoneIndex; }
  auto is_implicit() const -> bool {
    return is_node_id() && (index & ImplicitBit) != 0;
  }

  // This is allowed to return `NodeId::None`, but should never be used for
  // `InstId` other than `InstId::None`.
  auto node_id() const -> Parse::NodeId {
    if (!has_value()) {
      return Parse::NodeId::None;
    }
    CARBON_CHECK(is_node_id());
    return Parse::NodeId(index & ~ImplicitBit);
  }

  // This is allowed to return `InstId::None`, but should never be used for
  // `NodeId` other than `NodeId::None`.
  auto import_ir_inst_id() const -> ImportIRInstId {
    CARBON_CHECK(is_import_ir_inst_id() || !has_value());
    return ImportIRInstId(NoneIndex + ImportIRInstId::NoneIndex - index);
  }

  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr LocId LocId::None = LocId(Parse::NodeId::None);

// Polymorphic id for fields in `Any[...]` typed instruction category. Used for
// fields where the specific instruction structs have different field types in
// that position or do not have a field in that position at all. Allows
// conversion with `Inst::As<>` from the specific typed instruction to the
// `Any[...]` instruction category.
//
// This type participates in `Inst::FromRaw` in order to convert from specific
// instructions to an `Any[...]` instruction category:
// - In the case the specific instruction has a field of some `IdKind` in the
//   same position, the `Any[...]` type will hold its raw value in the
//   `AnyRawId` field.
// - In the case the specific instruction has no field in the same position, the
//   `Any[...]` type will hold a default constructed `AnyRawId` with a `None`
//   value.
struct AnyRawId : public AnyIdBase {
  // For IdKind.
  static constexpr llvm::StringLiteral Label = "any_raw";

  constexpr explicit AnyRawId() : AnyIdBase(AnyIdBase::NoneIndex) {}
  constexpr explicit AnyRawId(int32_t id) : AnyIdBase(id) {}
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_IDS_H_
