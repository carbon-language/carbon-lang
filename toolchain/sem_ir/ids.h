// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_IDS_H_
#define CARBON_TOOLCHAIN_SEM_IR_IDS_H_

#include <limits>

#include "common/check.h"
#include "common/ostream.h"
#include "toolchain/base/index_base.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parse/node_ids.h"

namespace Carbon::SemIR {

// TODO: This is in use, but not here.
class File;

// The ID of an `Inst`.
struct InstId : public IdBase<InstId> {
  static constexpr llvm::StringLiteral Label = "inst";

  // The maximum ID, inclusive.
  static constexpr int Max = std::numeric_limits<int32_t>::max();

  // Represents the result of a name lookup that is temporarily disallowed
  // because the name is currently being initialized.
  static const InstId InitTombstone;

  using IdBase::IdBase;

  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr InstId InstId::InitTombstone = InstId(NoneIndex - 1);

// An InstId whose value is a type. The fact it's a type must be validated
// before construction, and this allows that validation to be represented in the
// type system.
struct TypeInstId : public InstId {
  static const TypeInstId None;

  using InstId::InstId;

  static constexpr auto UnsafeMake(InstId id) -> TypeInstId {
    return TypeInstId(UnsafeCtor(), id);
  }

 private:
  struct UnsafeCtor {};
  explicit constexpr TypeInstId(UnsafeCtor /*unsafe*/, InstId id)
      : InstId(id) {}
};

constexpr TypeInstId TypeInstId::None = TypeInstId::UnsafeMake(InstId::None);

// An InstId whose type is known to be T. The fact it's a type must be validated
// before construction, and this allows that validation to be represented in the
// type system.
//
// Unlike TypeInstId, this type can *not* be an operand in instructions, since
// being a template prevents it from being used in non-generic contexts such as
// switches.
template <class T>
struct KnownInstId : public InstId {
  static const KnownInstId None;

  using InstId::InstId;

  static constexpr auto UnsafeMake(InstId id) -> KnownInstId {
    return KnownInstId(UnsafeCtor(), id);
  }

 private:
  struct UnsafeCtor {};
  explicit constexpr KnownInstId(UnsafeCtor /*unsafe*/, InstId id)
      : InstId(id) {}
};

template <class T>
constexpr KnownInstId<T> KnownInstId<T>::None =
    KnownInstId<T>::UnsafeMake(InstId::None);

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
  static constexpr llvm::StringLiteral Label = "absolute_inst";

  // Support implicit conversion from InstId so that InstId and AbsoluteInstId
  // have the same interface.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr AbsoluteInstId(InstId inst_id) : InstId(inst_id) {}

  using InstId::InstId;
};

// An ID of an instruction that is used as the destination of an initializing
// expression. This should only be used as the type of a field within a typed
// instruction class.
//
// This behaves in most respects like an InstId field, but constant evaluation
// of an instruction with a destination field will not evaluate this field, and
// substitution will not substitute into it.
//
// TODO: Decide on how substitution should handle this. Multiple instructions
// can refer to the same destination, so these don't have the tree structure
// that substitution expects, but we might need to substitute into the result of
// an instruction.
class DestInstId : public InstId {
 public:
  static constexpr llvm::StringLiteral Label = "dest_inst";

  // Support implicit conversion from InstId so that InstId and DestInstId
  // have the same interface.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr DestInstId(InstId inst_id) : InstId(inst_id) {}

  using InstId::InstId;
};

// An ID of an instruction that is referenced as a meta-operand of an action.
// This should only be used as the type of a field within a typed instruction
// class.
//
// This is used to model cases where an action's operand is not the value
// produced by another instruction, but is the other instruction itself. This is
// common for actions representing template instantiation.
//
// This behaves in most respects like an InstId field, but evaluation of the
// instruction that has this field will not fail if the instruction does not
// have a constant value. If the instruction has a constant value, it will still
// be replaced by its constant value during evaluation like normal, but if it
// has a non-constant value, the field is left unchanged by evaluation.
class MetaInstId : public InstId {
 public:
  static constexpr llvm::StringLiteral Label = "meta_inst";

  // Support implicit conversion from InstId so that InstId and MetaInstId
  // have the same interface.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr MetaInstId(InstId inst_id) : InstId(inst_id) {}

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

  // Returns the constant ID corresponding to a concrete constant, which should
  // either be in the `constants` block in the file or should be known to be
  // unique.
  static constexpr auto ForConcreteConstant(InstId const_id) -> ConstantId {
    return ConstantId(const_id.index);
  }

  using IdBase::IdBase;

  // Returns whether this represents a constant. Requires has_value.
  constexpr auto is_constant() const -> bool {
    CARBON_DCHECK(has_value());
    return *this != ConstantId::NotConstant;
  }
  // Returns whether this represents a symbolic constant. Requires has_value.
  constexpr auto is_symbolic() const -> bool {
    CARBON_DCHECK(has_value());
    return index <= FirstSymbolicId;
  }
  // Returns whether this represents a concrete constant. Requires has_value.
  constexpr auto is_concrete() const -> bool {
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

  // For Dump.
  friend auto MakeSymbolicConstantId(int id) -> ConstantId;

  // A symbolic constant.
  struct SymbolicId : public IdBase<SymbolicId> {
    static constexpr llvm::StringLiteral Label = "symbolic_constant";
    using IdBase::IdBase;
  };

  // Returns the constant ID corresponding to a symbolic constant index.
  static constexpr auto ForSymbolicConstantId(SymbolicId symbolic_id)
      -> ConstantId {
    return ConstantId(FirstSymbolicId - symbolic_id.index);
  }

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
  constexpr auto symbolic_id() const -> SymbolicId {
    CARBON_DCHECK(is_symbolic());
    return SymbolicId(FirstSymbolicId - index);
  }

  static constexpr int32_t NotConstantIndex = NoneIndex - 1;
  static constexpr int32_t FirstSymbolicId = NoneIndex - 2;
};

constexpr ConstantId ConstantId::NotConstant = ConstantId(NotConstantIndex);

// The ID of a `EntityName`.
struct EntityNameId : public IdBase<EntityNameId> {
  static constexpr llvm::StringLiteral Label = "entity_name";

  using IdBase::IdBase;
};

// The index of a compile-time binding. This is the de Bruijn level for the
// binding -- that is, this is the number of other compile time bindings whose
// scope encloses this binding.
struct CompileTimeBindIndex : public IndexBase<CompileTimeBindIndex> {
  static constexpr llvm::StringLiteral Label = "comp_time_bind";

  using IndexBase::IndexBase;
};

// The index of a `Call` parameter in a function. These are allocated
// sequentially, left-to-right, to the function parameters that will have
// arguments passed to them at runtime. In a `Call` instruction, a runtime
// argument will have the position in the argument list corresponding to its
// `Call` parameter index.
struct CallParamIndex : public IndexBase<CallParamIndex> {
  static constexpr llvm::StringLiteral Label = "call_param";

  using IndexBase::IndexBase;
};

// The ID of a `Function`.
struct FunctionId : public IdBase<FunctionId> {
  static constexpr llvm::StringLiteral Label = "function";

  using IdBase::IdBase;
};

// The ID of an IR within the set of all IRs being evaluated in the current
// check execution.
struct CheckIRId : public IdBase<CheckIRId> {
  static constexpr llvm::StringLiteral Label = "check_ir";

  // Used when referring to the imported C++.
  static const CheckIRId Cpp;

  using IdBase::IdBase;
};

constexpr CheckIRId CheckIRId::Cpp = CheckIRId(NoneIndex - 1);

// The ID of a `Class`.
struct ClassId : public IdBase<ClassId> {
  static constexpr llvm::StringLiteral Label = "class";

  using IdBase::IdBase;
};

// The ID of a `Vtable`.
struct VtableId : public IdBase<VtableId> {
  static constexpr llvm::StringLiteral Label = "vtable";

  using IdBase::IdBase;
};

// The ID of an `Interface`.
struct InterfaceId : public IdBase<InterfaceId> {
  static constexpr llvm::StringLiteral Label = "interface";

  using IdBase::IdBase;
};

// The ID of an `AssociatedConstant`.
struct AssociatedConstantId : public IdBase<AssociatedConstantId> {
  static constexpr llvm::StringLiteral Label = "assoc_const";

  using IdBase::IdBase;
};

// The ID of a `FacetTypeInfo`.
struct FacetTypeId : public IdBase<FacetTypeId> {
  static constexpr llvm::StringLiteral Label = "facet_type";

  using IdBase::IdBase;
};

// The ID of an resolved facet type value.
struct IdentifiedFacetTypeId : public IdBase<IdentifiedFacetTypeId> {
  static constexpr llvm::StringLiteral Label = "identified_facet_type";

  using IdBase::IdBase;
};

// The ID of an `Impl`.
struct ImplId : public IdBase<ImplId> {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  static constexpr llvm::StringLiteral Label = "impl";

  using IdBase::IdBase;
};

// The ID of a `Generic`.
struct GenericId : public IdBase<GenericId> {
  static constexpr llvm::StringLiteral Label = "generic";

  using IdBase::IdBase;
};

// The ID of a `Specific`, which is the result of specifying the generic
// arguments for a generic.
struct SpecificId : public IdBase<SpecificId> {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  static constexpr llvm::StringLiteral Label = "specific";

  using IdBase::IdBase;
};

// The ID of a `SpecificInterface`, which is an interface and a specific pair.
struct SpecificInterfaceId : public IdBase<SpecificInterfaceId> {
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  static constexpr llvm::StringLiteral Label = "specific_interface";

  using IdBase::IdBase;
};

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

// The ID of an `ImportCpp`.
struct ImportCppId : public IdBase<ImportCppId> {
  static constexpr llvm::StringLiteral Label = "import_cpp";

  using IdBase::IdBase;
};

// The ID of an `ImportIR` within the set of imported IRs, both direct and
// indirect.
struct ImportIRId : public IdBase<ImportIRId> {
  static constexpr llvm::StringLiteral Label = "ir";

  // The implicit `api` import, for an `impl` file. A null entry is added if
  // there is none, as in an `api`, in which case this ID should not show up in
  // instructions.
  static const ImportIRId ApiForImpl;

  // The `Cpp` import. A null entry is added if there is none, in which case
  // this ID should not show up in instructions.
  static const ImportIRId Cpp;

  using IdBase::IdBase;
};

constexpr ImportIRId ImportIRId::ApiForImpl = ImportIRId(0);
constexpr ImportIRId ImportIRId::Cpp = ImportIRId(ApiForImpl.index + 1);

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

// A character literal value as a unicode codepoint.
struct CharId : public IdBase<CharId> {
  // Not used by `Print`, but for `IdKind`.
  static constexpr llvm::StringLiteral Label = "";

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void;
};

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

// An X-macro for special names. Uses should look like:
//
//   #define CARBON_SPECIAL_NAME_ID_FOR_XYZ(Name) ...
//   CARBON_SPECIAL_NAME_ID(CARBON_SPECIAL_NAME_ID_FOR_XYZ)
//   #undef CARBON_SPECIAL_NAME_ID_FOR_XYZ
#define CARBON_SPECIAL_NAME_ID(X)                                \
  /* The name of `base`. */                                      \
  X(Base)                                                        \
  /* The name of the discriminant field (if any) in a choice. */ \
  X(ChoiceDiscriminant)                                          \
  /* The name of the package `Core`. */                          \
  X(Core)                                                        \
  /* The name of `destroy`. */                                   \
  X(Destroy)                                                     \
  /* The name of `package`. */                                   \
  X(PackageNamespace)                                            \
  /* The name of `.Self`. */                                     \
  X(PeriodSelf)                                                  \
  /* The name of the return slot in a function. */               \
  X(ReturnSlot)                                                  \
  /* The name of `Self`. */                                      \
  X(SelfType)                                                    \
  /* The name of `self`. */                                      \
  X(SelfValue)                                                   \
  /* The name of `_`. */                                         \
  X(Underscore)                                                  \
  /* The name of `vptr`. */                                      \
  X(Vptr)

// The ID of a name. A name is either a string or a special name such as
// `self`, `Self`, or `base`.
struct NameId : public IdBase<NameId> {
  static constexpr llvm::StringLiteral Label = "name";

  // names().GetFormatted() is used for diagnostics.
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  // An enum of special names.
  enum class SpecialNameId : uint8_t {
#define CARBON_SPECIAL_NAME_ID_FOR_ENUM(Name) Name,
    CARBON_SPECIAL_NAME_ID(CARBON_SPECIAL_NAME_ID_FOR_ENUM)
#undef CARBON_SPECIAL_NAME_ID_FOR_ENUM
  };

  // For each SpecialNameId, provide a matching `NameId` instance for
  // convenience.
#define CARBON_SPECIAL_NAME_ID_FOR_DECL(Name) static const NameId Name;
  CARBON_SPECIAL_NAME_ID(CARBON_SPECIAL_NAME_ID_FOR_DECL)
#undef CARBON_SPECIAL_NAME_ID_FOR_DECL

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

  // Expose special names for `switch`.
  constexpr auto AsSpecialNameId() const -> std::optional<SpecialNameId> {
    if (index >= NoneIndex) {
      return std::nullopt;
    }
    return static_cast<SpecialNameId>(NoneIndex - 1 - index);
  }

  auto Print(llvm::raw_ostream& out) const -> void;
};

// Define the special `static const NameId` values.
#define CARBON_SPECIAL_NAME_ID_FOR_DEF(Name) \
  constexpr NameId NameId::Name =            \
      NameId(NoneIndex - 1 - static_cast<int>(NameId::SpecialNameId::Name));
CARBON_SPECIAL_NAME_ID(CARBON_SPECIAL_NAME_ID_FOR_DEF)
#undef CARBON_SPECIAL_NAME_ID_FOR_DEF

// Count non-index values, including `None` and special names.
#define CARBON_SPECIAL_NAME_ID_FOR_COUNT(...) +1
constexpr int NameId::NonIndexValueCount =
    1 CARBON_SPECIAL_NAME_ID(CARBON_SPECIAL_NAME_ID_FOR_COUNT);
#undef CARBON_SPECIAL_NAME_ID_FOR_COUNT

// The ID of a `NameScope`.
struct NameScopeId : public IdBase<NameScopeId> {
  static constexpr llvm::StringLiteral Label = "name_scope";

  // The package (or file) name scope, guaranteed to be the first added.
  static const NameScopeId Package;

  using IdBase::IdBase;
};

constexpr NameScopeId NameScopeId::Package = NameScopeId(0);

// The ID of an `InstId` block.
struct InstBlockId : public IdBase<InstBlockId> {
  static constexpr llvm::StringLiteral Label = "inst_block";

  // The canonical empty block, reused to avoid allocating empty vectors. Always
  // the 0-index block.
  static const InstBlockId Empty;

  // Exported instructions. Empty until the File is fully checked; intermediate
  // state is in the Check::Context.
  static const InstBlockId Exports;

  // Instructions produced through import logic. Empty until the File is fully
  // checked; intermediate state is in the Check::Context.
  static const InstBlockId Imports;

  // Global declaration initialization instructions. Empty if none are present.
  // Otherwise, __global_init function will be generated and this block will
  // be inserted into it.
  static const InstBlockId GlobalInit;

  // An ID for unreachable code.
  static const InstBlockId Unreachable;

  using IdBase::IdBase;
  auto Print(llvm::raw_ostream& out) const -> void;
};

constexpr InstBlockId InstBlockId::Empty = InstBlockId(0);
constexpr InstBlockId InstBlockId::Exports = InstBlockId(1);
constexpr InstBlockId InstBlockId::Imports = InstBlockId(2);
constexpr InstBlockId InstBlockId::GlobalInit = InstBlockId(3);
constexpr InstBlockId InstBlockId::Unreachable = InstBlockId(NoneIndex - 1);

// Contains either an `InstBlockId` value, an error value, or
// `InstBlockId::None`.
//
// Error values are treated as values, though they are not representable as an
// `InstBlockId` (unlike for the singleton error `InstId`).
class InstBlockIdOrError {
 public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  InstBlockIdOrError(InstBlockId inst_block_id)
      : InstBlockIdOrError(inst_block_id, false) {}

  static auto MakeError() -> InstBlockIdOrError {
    return {InstBlockId::None, true};
  }

  // Returns whether this class contains either an InstBlockId (other than
  // `None`) or an error.
  //
  // An error is treated as a value (as same for the singleton error `InstId`),
  // but it can not actually be materialized as an error value outside of this
  // class.
  auto has_value() const -> bool {
    return has_error_value() || inst_block_id_.has_value();
  }

  // Returns whether this class contains an error value.
  auto has_error_value() const -> bool { return error_; }

  // Returns the id of a non-empty inst block, or `None` if `has_value()` is
  // false.
  //
  // Only valid to call if `has_error_value()` is false.
  auto inst_block_id() const -> InstBlockId {
    CARBON_CHECK(!has_error_value());
    return inst_block_id_;
  }

 private:
  InstBlockIdOrError(InstBlockId inst_block_id, bool error)
      : inst_block_id_(inst_block_id), error_(error) {}

  InstBlockId inst_block_id_;
  bool error_;
};

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

// The ID of an `ExprRegion`.
// TODO: Move this out of sem_ir and into check, if we don't wind up using it
// in the SemIR for expression patterns.
struct ExprRegionId : public IdBase<ExprRegionId> {
  static constexpr llvm::StringLiteral Label = "region";

  using IdBase::IdBase;
};

// The ID of a `StructTypeField` block.
struct StructTypeFieldsId : public IdBase<StructTypeFieldsId> {
  static constexpr llvm::StringLiteral Label = "struct_type_fields";

  // The canonical empty block, reused to avoid allocating empty vectors. Always
  // the 0-index block.
  static const StructTypeFieldsId Empty;

  using IdBase::IdBase;
};

constexpr StructTypeFieldsId StructTypeFieldsId::Empty = StructTypeFieldsId(0);

// The ID of a `CustomLayout` block.
struct CustomLayoutId : public IdBase<CustomLayoutId> {
  static constexpr llvm::StringLiteral Label = "custom_layout";

  // The canonical empty block. This is never used, but needed by
  // BlockValueStore.
  static const CustomLayoutId Empty;

  // The index in a custom layout of the overall size field.
  static constexpr int SizeIndex = 0;
  // The index in a custom layout of the overall alignment field.
  static constexpr int AlignIndex = 1;
  // The index in a custom layout of the offset of the first struct field.
  static constexpr int FirstFieldIndex = 2;

  using IdBase::IdBase;
};

constexpr CustomLayoutId CustomLayoutId::Empty = CustomLayoutId(0);

// The ID of a type.
struct TypeId : public IdBase<TypeId> {
  static constexpr llvm::StringLiteral Label = "type";

  // `StringifyConstantInst` is used for diagnostics. However, where possible,
  // an `InstId` describing how the type was written should be preferred, using
  // `InstIdAsType` or `TypeOfInstId` as the diagnostic argument type.
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

  using IdBase::IdBase;

  // Returns the ID of the type corresponding to the constant `const_id`, which
  // must be of type `type`. As an exception, the type `Error` is of type
  // `Error`.
  static constexpr auto ForTypeConstant(ConstantId const_id) -> TypeId {
    return TypeId(const_id.index);
  }

  // Returns the constant ID that defines the type.
  auto AsConstantId() const -> ConstantId { return ConstantId(index); }

  // Returns whether this represents a symbolic type. Requires has_value.
  auto is_symbolic() const -> bool { return AsConstantId().is_symbolic(); }
  // Returns whether this represents a concrete type. Requires has_value.
  auto is_concrete() const -> bool { return AsConstantId().is_concrete(); }

  auto Print(llvm::raw_ostream& out) const -> void;
};

// The ID of a `clang::SourceLocation`.
struct ClangSourceLocId : public IdBase<ClangSourceLocId> {
  static constexpr llvm::StringLiteral Label = "clang_source_loc";

  using IdBase::IdBase;
};

// An index for element access, for structs, tuples, and classes.
struct ElementIndex : public IndexBase<ElementIndex> {
  static constexpr llvm::StringLiteral Label = "element";
  using IndexBase::IndexBase;
};

// The ID of a library name. This is either a string literal or `default`.
struct LibraryNameId : public IdBase<LibraryNameId> {
  static constexpr llvm::StringLiteral Label = "library_name";
  using DiagnosticType = Diagnostics::TypeInfo<std::string>;

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

constexpr LibraryNameId LibraryNameId::Default = LibraryNameId(NoneIndex - 1);
constexpr LibraryNameId LibraryNameId::Error = LibraryNameId(NoneIndex - 2);

// The ID of an `ImportIRInst`.
struct ImportIRInstId : public IdBase<ImportIRInstId> {
  static constexpr llvm::StringLiteral Label = "import_ir_inst";

  // The maximum ID, non-inclusive. This is constrained to fit inside LocId.
  static constexpr int Max =
      -(std::numeric_limits<int32_t>::min() + 2 * Parse::NodeId::Max + 1);

  constexpr explicit ImportIRInstId(int32_t index) : IdBase(index) {
    CARBON_DCHECK(index < Max, "Index out of range: {0}", index);
  }
};

// A SemIR location used as the location of instructions. This contains either a
// InstId, NodeId, ImportIRInstId, or None. The intent is that any of these can
// indicate the source of an instruction, and also be used to associate a line
// in diagnostics.
//
// The structure is:
// - None: The standard NoneIndex for all Id types, -1.
// - InstId: Positive values including zero; a full 31 bits.
//   - [0, 1 << 31)
// - NodeId: Negative values starting after None; the 24 bit NodeId range.
//   - [-2, -2 - (1 << 24))
// - Desugared NodeId: Another 24 bit NodeId range.
//   - [-2 - (1 << 24), -2 - (1 << 25))
// - ImportIRInstId: Remaining negative values; after NodeId, fills out negative
//   values.
//   - [-2 - (1 << 25), -(1 << 31)]
//
// For desugaring, use `InstStore::GetLocIdForDesugaring()`.
struct LocId : public IdBase<LocId> {
  // The contained index kind.
  enum class Kind {
    None,
    ImportIRInstId,
    InstId,
    NodeId,
  };

  static constexpr llvm::StringLiteral Label = "loc";

  using IdBase::IdBase;

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(ImportIRInstId import_ir_inst_id)
      : IdBase(import_ir_inst_id.has_value()
                   ? FirstImportIRInstId - import_ir_inst_id.index
                   : NoneIndex) {}

  explicit constexpr LocId(InstId inst_id) : IdBase(inst_id.index) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(Parse::NoneNodeId /*none*/) : IdBase(NoneIndex) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr LocId(Parse::NodeId node_id)
      : IdBase(FirstNodeId - node_id.index) {}

  // Forms an equivalent LocId for a desugared location. Prefer calling
  // `InstStore::GetLocIdForDesugaring`.
  auto AsDesugared() const -> LocId {
    // This should only be called for NodeId or ImportIRInstId (i.e. canonical
    // locations), but we only set the flag for NodeId.
    CARBON_CHECK(kind() != Kind::InstId, "Use InstStore::GetDesugaredLocId");
    if (index <= FirstNodeId && index > FirstDesugaredNodeId) {
      return LocId(index - Parse::NodeId::Max);
    }
    return *this;
  }

  // Returns the kind of the `LocId`.
  auto kind() const -> Kind {
    if (!has_value()) {
      return Kind::None;
    }
    if (index >= 0) {
      return Kind::InstId;
    }
    if (index <= FirstImportIRInstId) {
      return Kind::ImportIRInstId;
    }
    return Kind::NodeId;
  }

  // Returns true if the location corresponds to desugared instructions.
  // Requires a non-`InstId` location.
  auto is_desugared() const -> bool {
    return index <= FirstDesugaredNodeId && index > FirstImportIRInstId;
  }

  // Returns the equivalent `ImportIRInstId` when `kind()` matches or is `None`.
  // Note that the returned `ImportIRInstId` only identifies a location; it is
  // not correct to interpret it as the instruction from which another
  // instruction was imported. Use `InstStore::GetImportSource` for that.
  auto import_ir_inst_id() const -> ImportIRInstId {
    if (!has_value()) {
      return ImportIRInstId::None;
    }
    CARBON_CHECK(kind() == Kind::ImportIRInstId, "{0}", index);
    return ImportIRInstId(FirstImportIRInstId - index);
  }

  // Returns the equivalent `InstId` when `kind()` matches or is `None`.
  auto inst_id() const -> InstId {
    CARBON_CHECK(kind() == Kind::None || kind() == Kind::InstId, "{0}", index);
    return InstId(index);
  }

  // Returns the equivalent `NodeId` when `kind()` matches or is `None`.
  auto node_id() const -> Parse::NodeId {
    if (!has_value()) {
      return Parse::NodeId::None;
    }
    CARBON_CHECK(kind() == Kind::NodeId, "{0}", index);
    if (index <= FirstDesugaredNodeId) {
      return Parse::NodeId(FirstDesugaredNodeId - index);
    } else {
      return Parse::NodeId(FirstNodeId - index);
    }
  }

  auto Print(llvm::raw_ostream& out) const -> void;

 private:
  // The value of the 0 index for each of `NodeId` and `ImportIRInstId`.
  static constexpr int32_t FirstNodeId = NoneIndex - 1;
  static constexpr int32_t FirstDesugaredNodeId =
      FirstNodeId - Parse::NodeId::Max;
  static constexpr int32_t FirstImportIRInstId =
      FirstDesugaredNodeId - Parse::NodeId::Max;
};

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
