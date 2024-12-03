// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_FILE_H_

#include "common/error.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/base/int.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/entity_name.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/interface.h"
#include "toolchain/sem_ir/name.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/singleton_insts.h"
#include "toolchain/sem_ir/struct_type_field.h"
#include "toolchain/sem_ir/type.h"
#include "toolchain/sem_ir/type_info.h"

namespace Carbon::SemIR {

// Provides semantic analysis on a Parse::Tree.
class File : public Printable<File> {
 public:
  // Starts a new file for Check::CheckParseTree.
  explicit File(CheckIRId check_ir_id,
                const std::optional<Parse::Tree::PackagingDecl>& packaging_decl,
                SharedValueStores& value_stores, std::string filename);

  File(const File&) = delete;
  auto operator=(const File&) -> File& = delete;

  // Verifies that invariants of the semantics IR hold.
  auto Verify() const -> ErrorOr<Success>;

  // Prints the full IR. Allow omitting singletons so that changes to the list
  // of singletons won't churn golden test file content.
  auto Print(llvm::raw_ostream& out, bool include_singletons = false) const
      -> void {
    Yaml::Print(out, OutputYaml(include_singletons));
  }
  auto OutputYaml(bool include_singletons) const -> Yaml::OutputMapping;

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void;

  // Returns array bound value from the bound instruction.
  auto GetArrayBoundValue(InstId bound_id) const -> uint64_t {
    return ints().Get(insts().GetAs<IntValue>(bound_id).int_id).getZExtValue();
  }

  // Gets the pointee type of the given type, which must be a pointer type.
  auto GetPointeeType(TypeId pointer_id) const -> TypeId {
    return types().GetAs<PointerType>(pointer_id).pointee_id;
  }

  auto check_ir_id() const -> CheckIRId { return check_ir_id_; }
  auto package_id() const -> IdentifierId { return package_id_; }
  auto library_id() const -> SemIR::LibraryNameId { return library_id_; }

  // Directly expose SharedValueStores members.
  auto identifiers() -> SharedValueStores::IdentifierStore& {
    return value_stores_->identifiers();
  }
  auto identifiers() const -> const SharedValueStores::IdentifierStore& {
    return value_stores_->identifiers();
  }
  auto ints() -> SharedValueStores::IntStore& { return value_stores_->ints(); }
  auto ints() const -> const SharedValueStores::IntStore& {
    return value_stores_->ints();
  }
  auto reals() -> SharedValueStores::RealStore& {
    return value_stores_->reals();
  }
  auto reals() const -> const SharedValueStores::RealStore& {
    return value_stores_->reals();
  }
  auto floats() -> SharedValueStores::FloatStore& {
    return value_stores_->floats();
  }
  auto floats() const -> const SharedValueStores::FloatStore& {
    return value_stores_->floats();
  }
  auto string_literal_values() -> SharedValueStores::StringLiteralStore& {
    return value_stores_->string_literal_values();
  }
  auto string_literal_values() const
      -> const SharedValueStores::StringLiteralStore& {
    return value_stores_->string_literal_values();
  }

  auto entity_names() -> EntityNameStore& { return entity_names_; }
  auto entity_names() const -> const EntityNameStore& { return entity_names_; }
  auto functions() -> ValueStore<FunctionId>& { return functions_; }
  auto functions() const -> const ValueStore<FunctionId>& { return functions_; }
  auto classes() -> ValueStore<ClassId>& { return classes_; }
  auto classes() const -> const ValueStore<ClassId>& { return classes_; }
  auto interfaces() -> ValueStore<InterfaceId>& { return interfaces_; }
  auto interfaces() const -> const ValueStore<InterfaceId>& {
    return interfaces_;
  }
  auto facet_types() -> CanonicalValueStore<FacetTypeId>& {
    return facet_types_;
  }
  auto facet_types() const -> const CanonicalValueStore<FacetTypeId>& {
    return facet_types_;
  }
  auto impls() -> ImplStore& { return impls_; }
  auto impls() const -> const ImplStore& { return impls_; }
  auto generics() -> GenericStore& { return generics_; }
  auto generics() const -> const GenericStore& { return generics_; }
  auto specifics() -> SpecificStore& { return specifics_; }
  auto specifics() const -> const SpecificStore& { return specifics_; }
  auto import_irs() -> ValueStore<ImportIRId>& { return import_irs_; }
  auto import_irs() const -> const ValueStore<ImportIRId>& {
    return import_irs_;
  }
  auto import_ir_insts() -> ValueStore<ImportIRInstId>& {
    return import_ir_insts_;
  }
  auto import_ir_insts() const -> const ValueStore<ImportIRInstId>& {
    return import_ir_insts_;
  }
  auto names() const -> NameStoreWrapper {
    return NameStoreWrapper(&identifiers());
  }
  auto name_scopes() -> NameScopeStore& { return name_scopes_; }
  auto name_scopes() const -> const NameScopeStore& { return name_scopes_; }
  auto struct_type_fields() -> StructTypeFieldsStore& {
    return struct_type_fields_;
  }
  auto struct_type_fields() const -> const StructTypeFieldsStore& {
    return struct_type_fields_;
  }
  auto types() -> TypeStore& { return types_; }
  auto types() const -> const TypeStore& { return types_; }
  auto type_blocks() -> BlockValueStore<TypeBlockId>& { return type_blocks_; }
  auto type_blocks() const -> const BlockValueStore<TypeBlockId>& {
    return type_blocks_;
  }
  auto insts() -> InstStore& { return insts_; }
  auto insts() const -> const InstStore& { return insts_; }
  auto constant_values() -> ConstantValueStore& { return constant_values_; }
  auto constant_values() const -> const ConstantValueStore& {
    return constant_values_;
  }
  auto inst_blocks() -> InstBlockStore& { return inst_blocks_; }
  auto inst_blocks() const -> const InstBlockStore& { return inst_blocks_; }
  auto constants() -> ConstantStore& { return constants_; }
  auto constants() const -> const ConstantStore& { return constants_; }

  auto top_inst_block_id() const -> InstBlockId { return top_inst_block_id_; }
  auto set_top_inst_block_id(InstBlockId block_id) -> void {
    top_inst_block_id_ = block_id;
  }
  auto global_ctor_id() const -> FunctionId { return global_ctor_id_; }
  auto set_global_ctor_id(FunctionId function_id) -> void {
    global_ctor_id_ = function_id;
  }

  // Returns true if there were errors creating the semantics IR.
  auto has_errors() const -> bool { return has_errors_; }
  auto set_has_errors(bool has_errors) -> void { has_errors_ = has_errors; }

  auto filename() const -> llvm::StringRef { return filename_; }

 private:
  // True if parts of the IR may be invalid.
  bool has_errors_ = false;

  // The file's ID.
  CheckIRId check_ir_id_;

  // The file's package.
  IdentifierId package_id_ = IdentifierId::Invalid;

  // The file's library.
  LibraryNameId library_id_ = LibraryNameId::Invalid;

  // Shared, compile-scoped values.
  SharedValueStores* value_stores_;

  // Slab allocator, used to allocate instruction and type blocks.
  llvm::BumpPtrAllocator allocator_;

  // The associated filename.
  // TODO: If SemIR starts linking back to tokens, reuse its filename.
  std::string filename_;

  // Storage for EntityNames.
  EntityNameStore entity_names_;

  // Storage for callable objects.
  ValueStore<FunctionId> functions_;

  // Storage for classes.
  ValueStore<ClassId> classes_;

  // Storage for interfaces.
  ValueStore<InterfaceId> interfaces_;

  // Storage for facet types.
  CanonicalValueStore<FacetTypeId> facet_types_;

  // Storage for impls.
  ImplStore impls_;

  // Storage for generics.
  GenericStore generics_;

  // Storage for specifics.
  SpecificStore specifics_;

  // Related IRs. There are some fixed entries at the start; see ImportIRId.
  ValueStore<ImportIRId> import_irs_;

  // Related IR instructions. These are created for LocIds for instructions
  // that are import-related.
  ValueStore<ImportIRInstId> import_ir_insts_;

  // Type blocks within the IR. These reference entries in types_. Storage for
  // the data is provided by allocator_.
  BlockValueStore<TypeBlockId> type_blocks_;

  // All instructions. The first entries will always be the singleton
  // instructions.
  InstStore insts_;

  // Storage for name scopes.
  NameScopeStore name_scopes_;

  // Constant values for instructions.
  ConstantValueStore constant_values_;

  // Instruction blocks within the IR. These reference entries in
  // insts_. Storage for the data is provided by allocator_.
  InstBlockStore inst_blocks_;

  // The top instruction block ID.
  InstBlockId top_inst_block_id_ = InstBlockId::Invalid;

  // The global constructor function id.
  FunctionId global_ctor_id_ = FunctionId::Invalid;

  // Storage for instructions that represent computed global constants, such as
  // types.
  ConstantStore constants_;

  // Storage for StructTypeField lists.
  StructTypeFieldsStore struct_type_fields_ = StructTypeFieldsStore(allocator_);

  // Descriptions of types used in this file.
  TypeStore types_ = TypeStore(this);
};

// The expression category of a sem_ir instruction. See /docs/design/values.md
// for details.
enum class ExprCategory : int8_t {
  // This instruction does not correspond to an expression, and as such has no
  // category.
  NotExpr,
  // The category of this instruction is not known due to an error.
  Error,
  // This instruction represents a value expression.
  Value,
  // This instruction represents a durable reference expression, that denotes an
  // object that outlives the current full expression context.
  DurableRef,
  // This instruction represents an ephemeral reference expression, that denotes
  // an
  // object that does not outlive the current full expression context.
  EphemeralRef,
  // This instruction represents an initializing expression, that describes how
  // to
  // initialize an object.
  Initializing,
  // This instruction represents a syntactic combination of expressions that are
  // permitted to have different expression categories. This is used for tuple
  // and struct literals, where the subexpressions for different elements can
  // have different categories.
  Mixed,
  Last = Mixed
};

// Returns the expression category for an instruction.
auto GetExprCategory(const File& file, InstId inst_id) -> ExprCategory;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FILE_H_
