// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_FILE_H_
#define CARBON_TOOLCHAIN_SEM_IR_FILE_H_

#include "clang/Frontend/ASTUnit.h"
#include "common/error.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/base/canonical_value_store.h"
#include "toolchain/base/int.h"
#include "toolchain/base/relational_value_store.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/associated_constant.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/entity_name.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/import_cpp.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/interface.h"
#include "toolchain/sem_ir/name.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/singleton_insts.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/struct_type_field.h"
#include "toolchain/sem_ir/type.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/vtable.h"

namespace Carbon::SemIR {

// An expression that may contain control flow, represented as a
// single-entry/single-exit region. `block_ids` are the blocks that are part of
// evaluation of the expression, and `result_id` represents the result of
// evaluating the expression. `block_ids` consists of all blocks that are
// dominated by `block_ids.front()` and post-dominated by `block_ids.back()`,
// and should be in lexical order. `result_id` will be in `block_ids.back()` or
// some block that dominates it.
//
// `block_ids` cannot be empty. If it has a single element, then the region
// should be used via a `SpliceBlock` inst. Otherwise, the region should be used
// by branching to the entry block, and the last inst in the exit block will
// likewise be a branch.
struct ExprRegion {
  llvm::SmallVector<InstBlockId> block_ids;
  InstId result_id;
};

using ExprRegionStore = ValueStore<ExprRegionId, ExprRegion>;

using CustomLayoutStore = BlockValueStore<CustomLayoutId, uint64_t>;

// Provides semantic analysis on a Parse::Tree.
class File : public Printable<File> {
 public:
  using IdentifiedFacetTypeStore =
      RelationalValueStore<FacetTypeId, IdentifiedFacetTypeId,
                           IdentifiedFacetType>;

  // Starts a new file for Check::CheckParseTree.
  explicit File(const Parse::Tree* parse_tree, CheckIRId check_ir_id,
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
  // TODO: Move this function elsewhere.
  auto GetArrayBoundValue(InstId bound_id) const -> std::optional<uint64_t> {
    if (auto bound = insts().TryGetAs<IntValue>(
            constant_values().GetConstantInstId(bound_id))) {
      return ints().Get(bound->int_id).getZExtValue();
    }
    return std::nullopt;
  }

  // Gets the pointee type of the given type, which must be a pointer type.
  // TODO: Move this function elsewhere.
  auto GetPointeeType(TypeId pointer_id) const -> TypeId {
    return types().GetTypeIdForTypeInstId(
        types().GetAs<PointerType>(pointer_id).pointee_id);
  }

  // Returns true if this file is an `impl`.
  auto is_impl() -> bool {
    return import_irs().Get(ImportIRId::ApiForImpl).sem_ir != nullptr;
  }

  auto check_ir_id() const -> CheckIRId { return check_ir_id_; }
  auto package_id() const -> PackageNameId { return package_id_; }
  auto library_id() const -> LibraryNameId { return library_id_; }

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
  auto functions() -> FunctionStore& { return functions_; }
  auto functions() const -> const FunctionStore& { return functions_; }
  auto classes() -> ClassStore& { return classes_; }
  auto classes() const -> const ClassStore& { return classes_; }
  auto interfaces() -> InterfaceStore& { return interfaces_; }
  auto interfaces() const -> const InterfaceStore& { return interfaces_; }
  auto associated_constants() -> AssociatedConstantStore& {
    return associated_constants_;
  }
  auto associated_constants() const -> const AssociatedConstantStore& {
    return associated_constants_;
  }
  // TODO: Rename these to `facet_type_infos`.
  auto facet_types() -> FacetTypeInfoStore& { return facet_types_; }
  auto facet_types() const -> const FacetTypeInfoStore& { return facet_types_; }
  auto identified_facet_types() -> IdentifiedFacetTypeStore& {
    return identified_facet_types_;
  }
  auto identified_facet_types() const -> const IdentifiedFacetTypeStore& {
    return identified_facet_types_;
  }
  auto impls() -> ImplStore& { return impls_; }
  auto impls() const -> const ImplStore& { return impls_; }
  auto specific_interfaces() -> SpecificInterfaceStore& {
    return specific_interfaces_;
  }
  auto specific_interfaces() const -> const SpecificInterfaceStore& {
    return specific_interfaces_;
  }
  auto generics() -> GenericStore& { return generics_; }
  auto generics() const -> const GenericStore& { return generics_; }
  auto specifics() -> SpecificStore& { return specifics_; }
  auto specifics() const -> const SpecificStore& { return specifics_; }
  auto import_irs() -> ImportIRStore& { return import_irs_; }
  auto import_irs() const -> const ImportIRStore& { return import_irs_; }
  auto import_ir_insts() -> ImportIRInstStore& { return import_ir_insts_; }
  auto import_ir_insts() const -> const ImportIRInstStore& {
    return import_ir_insts_;
  }
  auto import_cpps() -> ImportCppStore& { return import_cpps_; }
  auto import_cpps() const -> const ImportCppStore& { return import_cpps_; }
  auto cpp_ast() -> clang::ASTUnit* { return cpp_ast_; }
  auto cpp_ast() const -> const clang::ASTUnit* { return cpp_ast_; }
  // TODO: When the AST can be created before creating `File`, initialize the
  // pointer in the constructor and remove this function. This is part of
  // https://github.com/carbon-language/carbon-lang/issues/4666
  auto set_cpp_ast(clang::ASTUnit* cpp_ast) -> void { cpp_ast_ = cpp_ast; }
  auto clang_decls() -> ClangDeclStore& { return clang_decls_; }
  auto clang_decls() const -> const ClangDeclStore& { return clang_decls_; }
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
  auto custom_layouts() -> CustomLayoutStore& { return custom_layouts_; }
  auto custom_layouts() const -> const CustomLayoutStore& {
    return custom_layouts_;
  }
  auto types() -> TypeStore& { return types_; }
  auto types() const -> const TypeStore& { return types_; }
  auto insts() -> InstStore& { return insts_; }
  auto insts() const -> const InstStore& { return insts_; }
  auto vtables() -> VtableStore& { return vtables_; }
  auto vtables() const -> const VtableStore& { return vtables_; }
  auto constant_values() -> ConstantValueStore& { return constant_values_; }
  auto constant_values() const -> const ConstantValueStore& {
    return constant_values_;
  }
  auto inst_blocks() -> InstBlockStore& { return inst_blocks_; }
  auto inst_blocks() const -> const InstBlockStore& { return inst_blocks_; }
  auto constants() -> ConstantStore& { return constants_; }
  auto constants() const -> const ConstantStore& { return constants_; }

  auto expr_regions() -> ExprRegionStore& { return expr_regions_; }
  auto expr_regions() const -> const ExprRegionStore& { return expr_regions_; }

  using ClangSourceLocStore =
      ValueStore<ClangSourceLocId, clang::SourceLocation>;
  auto clang_source_locs() -> ClangSourceLocStore& {
    return clang_source_locs_;
  }
  auto clang_source_locs() const -> const ClangSourceLocStore& {
    return clang_source_locs_;
  }

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

  auto parse_tree() const -> const Parse::Tree& { return *parse_tree_; }

 private:
  const Parse::Tree* parse_tree_;

  // True if parts of the IR may be invalid.
  bool has_errors_ = false;

  // The file's ID.
  CheckIRId check_ir_id_;

  // The file's package.
  PackageNameId package_id_ = PackageNameId::None;

  // The file's library.
  LibraryNameId library_id_ = LibraryNameId::None;

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
  FunctionStore functions_;

  // Storage for classes.
  ClassStore classes_;

  // Storage for interfaces.
  InterfaceStore interfaces_;

  // Storage for associated constants.
  AssociatedConstantStore associated_constants_;

  // Storage for facet types.
  FacetTypeInfoStore facet_types_;

  // Storage for identified facet types.
  IdentifiedFacetTypeStore identified_facet_types_;

  // Storage for impls.
  ImplStore impls_;

  // Storage for specific interfaces, which are an individual unit of impl
  // lookup for a single interface.
  SpecificInterfaceStore specific_interfaces_;

  // Storage for generics.
  GenericStore generics_;

  // Storage for specifics.
  SpecificStore specifics_;

  // Related IRs. There are some fixed entries at the start; see ImportIRId.
  ImportIRStore import_irs_;

  // Related IR instructions. These are created for LocIds for instructions
  // that are import-related.
  ImportIRInstStore import_ir_insts_;

  // List of Cpp imports.
  ImportCppStore import_cpps_;

  // The Clang AST to use when looking up `Cpp` names. Null if there are no
  // `Cpp` imports.
  clang::ASTUnit* cpp_ast_ = nullptr;

  // Clang AST declarations pointing to the AST and their mapped Carbon
  // instructions. When calling `Lookup()`, `inst_id` is ignored. `Add()` will
  // not add multiple entries with the same `decl` and different `inst_id`.
  ClangDeclStore clang_decls_;

  // All instructions. The first entries will always be the singleton
  // instructions.
  InstStore insts_ = InstStore(this);

  VtableStore vtables_;

  // Storage for name scopes.
  NameScopeStore name_scopes_ = NameScopeStore(this);

  // Constant values for instructions.
  ConstantValueStore constant_values_;

  // Instruction blocks within the IR. These reference entries in
  // insts_. Storage for the data is provided by allocator_.
  InstBlockStore inst_blocks_;

  // The top instruction block ID.
  InstBlockId top_inst_block_id_ = InstBlockId::None;

  // The global constructor function id.
  FunctionId global_ctor_id_ = FunctionId::None;

  // Storage for instructions that represent computed global constants, such as
  // types.
  ConstantStore constants_;

  // Storage for StructTypeField lists.
  StructTypeFieldsStore struct_type_fields_ = StructTypeFieldsStore(allocator_);

  // Storage for custom layouts.
  CustomLayoutStore custom_layouts_ = CustomLayoutStore(allocator_);

  // Descriptions of types used in this file.
  TypeStore types_ = TypeStore(this);

  // Single-entry/single-exit regions that are referenced as units, e.g. because
  // they represent expressions.
  ExprRegionStore expr_regions_;

  // C++ source locations for C++ interop.
  ClangSourceLocStore clang_source_locs_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FILE_H_
