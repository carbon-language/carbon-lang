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
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/constant.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/import_ir.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/interface.h"
#include "toolchain/sem_ir/name.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/type.h"
#include "toolchain/sem_ir/type_info.h"

namespace Carbon::SemIR {

struct BindNameInfo : public Printable<BindNameInfo> {
  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{name: " << name_id << ", enclosing_scope: " << enclosing_scope_id
        << "}";
  }

  // The name.
  NameId name_id;
  // The enclosing scope.
  NameScopeId enclosing_scope_id;
};

class File;

// Provides semantic analysis on a Parse::Tree.
class File : public Printable<File> {
 public:
  // Produces a file for the builtins.
  explicit File(SharedValueStores& value_stores);

  // Starts a new file for Check::CheckParseTree. Builtins are required.
  explicit File(SharedValueStores& value_stores, std::string filename,
                const File* builtins);

  File(const File&) = delete;
  auto operator=(const File&) -> File& = delete;

  // Verifies that invariants of the semantics IR hold.
  auto Verify() const -> ErrorOr<Success>;

  // Prints the full IR. Allow omitting builtins so that unrelated changes are
  // less likely to alter test golden files.
  // TODO: In the future, the things to print may change, for example by adding
  // preludes. We may then want the ability to omit other things similar to
  // builtins.
  auto Print(llvm::raw_ostream& out, bool include_builtins = false) const
      -> void {
    Yaml::Print(out, OutputYaml(include_builtins));
  }
  auto OutputYaml(bool include_builtins) const -> Yaml::OutputMapping;

  // Returns array bound value from the bound instruction.
  auto GetArrayBoundValue(InstId bound_id) const -> uint64_t {
    return ints()
        .Get(insts().GetAs<IntLiteral>(bound_id).int_id)
        .getZExtValue();
  }

  // Marks a type as complete, and sets its value representation.
  auto CompleteType(TypeId object_type_id, ValueRepr value_repr) -> void {
    if (object_type_id.index < 0) {
      // We already know our builtin types are complete.
      return;
    }
    CARBON_CHECK(types().Get(object_type_id).value_repr.kind ==
                 ValueRepr::Unknown)
        << "Type " << object_type_id << " completed more than once";
    types().Get(object_type_id).value_repr = value_repr;
    complete_types_.push_back(object_type_id);
  }

  // Gets the pointee type of the given type, which must be a pointer type.
  auto GetPointeeType(TypeId pointer_id) const -> TypeId {
    return types().GetAs<PointerType>(pointer_id).pointee_id;
  }

  // Produces a string version of a type.
  auto StringifyType(TypeId type_id) const -> std::string;

  // Same as `StringifyType`, but starting with an instruction representing a
  // type expression rather than a canonical type.
  auto StringifyTypeExpr(InstId outer_inst_id) const -> std::string;

  // Directly expose SharedValueStores members.
  auto identifiers() -> StringStoreWrapper<IdentifierId>& {
    return value_stores_->identifiers();
  }
  auto identifiers() const -> const StringStoreWrapper<IdentifierId>& {
    return value_stores_->identifiers();
  }
  auto ints() -> ValueStore<IntId>& { return value_stores_->ints(); }
  auto ints() const -> const ValueStore<IntId>& {
    return value_stores_->ints();
  }
  auto reals() -> ValueStore<RealId>& { return value_stores_->reals(); }
  auto reals() const -> const ValueStore<RealId>& {
    return value_stores_->reals();
  }
  auto string_literal_values() -> StringStoreWrapper<StringLiteralValueId>& {
    return value_stores_->string_literal_values();
  }
  auto string_literal_values() const
      -> const StringStoreWrapper<StringLiteralValueId>& {
    return value_stores_->string_literal_values();
  }

  auto bind_names() -> ValueStore<BindNameId>& { return bind_names_; }
  auto bind_names() const -> const ValueStore<BindNameId>& {
    return bind_names_;
  }
  auto functions() -> ValueStore<FunctionId>& { return functions_; }
  auto functions() const -> const ValueStore<FunctionId>& { return functions_; }
  auto classes() -> ValueStore<ClassId>& { return classes_; }
  auto classes() const -> const ValueStore<ClassId>& { return classes_; }
  auto interfaces() -> ValueStore<InterfaceId>& { return interfaces_; }
  auto interfaces() const -> const ValueStore<InterfaceId>& {
    return interfaces_;
  }
  auto impls() -> ImplStore& { return impls_; }
  auto impls() const -> const ImplStore& { return impls_; }
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

  // A list of types that were completed in this file, in the order in which
  // they were completed. Earlier types in this list cannot contain instances of
  // later types.
  auto complete_types() const -> llvm::ArrayRef<TypeId> {
    return complete_types_;
  }

  auto top_inst_block_id() const -> InstBlockId { return top_inst_block_id_; }
  auto set_top_inst_block_id(InstBlockId block_id) -> void {
    top_inst_block_id_ = block_id;
  }

  // Returns true if there were errors creating the semantics IR.
  auto has_errors() const -> bool { return has_errors_; }
  auto set_has_errors(bool has_errors) -> void { has_errors_ = has_errors; }

  auto filename() const -> llvm::StringRef { return filename_; }

 private:
  // Common File initialization.
  explicit File(SharedValueStores& value_stores, std::string filename,
                const File* builtins, llvm::function_ref<void()> init_builtins);

  bool has_errors_ = false;

  // Shared, compile-scoped values.
  SharedValueStores* value_stores_;

  // Slab allocator, used to allocate instruction and type blocks.
  llvm::BumpPtrAllocator allocator_;

  // The associated filename.
  // TODO: If SemIR starts linking back to tokens, reuse its filename.
  std::string filename_;

  // Storage for bind names.
  ValueStore<BindNameId> bind_names_;

  // Storage for callable objects.
  ValueStore<FunctionId> functions_;

  // Storage for classes.
  ValueStore<ClassId> classes_;

  // Storage for interfaces.
  ValueStore<InterfaceId> interfaces_;

  // Storage for impls.
  ImplStore impls_;

  // Related IRs. There are some fixed entries at the start; see ImportIRId.
  ValueStore<ImportIRId> import_irs_;

  // Related IR instructions. These are created for LocIds for instructions
  // that are import-related.
  ValueStore<ImportIRInstId> import_ir_insts_;

  // Storage for name scopes.
  NameScopeStore name_scopes_;

  // Type blocks within the IR. These reference entries in types_. Storage for
  // the data is provided by allocator_.
  BlockValueStore<TypeBlockId> type_blocks_;

  // All instructions. The first entries will always be ImportRefs to builtins,
  // at indices matching BuiltinKind ordering.
  InstStore insts_;

  // Constant values for instructions.
  ConstantValueStore constant_values_;

  // Instruction blocks within the IR. These reference entries in
  // insts_. Storage for the data is provided by allocator_.
  InstBlockStore inst_blocks_;

  // The top instruction block ID.
  InstBlockId top_inst_block_id_ = InstBlockId::Invalid;

  // Storage for instructions that represent computed global constants, such as
  // types.
  ConstantStore constants_;

  // Descriptions of types used in this file.
  TypeStore types_ = TypeStore(&insts_);

  // Types that were completed in this file.
  llvm::SmallVector<TypeId> complete_types_;
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

// Returns information about the value representation to use for a type.
inline auto GetValueRepr(const File& file, TypeId type_id) -> ValueRepr {
  return file.types().GetValueRepr(type_id);
}

// The initializing representation to use when returning by value.
struct InitRepr {
  enum Kind : int8_t {
    // The type has no initializing representation. This is used for empty
    // types, where no initialization is necessary.
    None,
    // An initializing expression produces an object representation by value,
    // which is copied into the initialized object.
    ByCopy,
    // An initializing expression takes a location as input, which is
    // initialized as a side effect of evaluating the expression.
    InPlace,
    // TODO: Consider adding a kind where the expression takes an advisory
    // location and returns a value plus an indicator of whether the location
    // was actually initialized.
  };
  // The kind of initializing representation used by this type.
  Kind kind;

  // Returns whether a return slot is used when returning this type.
  auto has_return_slot() const -> bool { return kind == InPlace; }
};

// Returns information about the initializing representation to use for a type.
auto GetInitRepr(const File& file, TypeId type_id) -> InitRepr;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_FILE_H_
