// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/file.h"

#include <optional>
#include <string>
#include <utility>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"
#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/block_value_store_impl.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/shared_value_stores.h"
#include "toolchain/base/value_store_impl.h"
#include "toolchain/base/yaml.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

File::File(const Parse::Tree* parse_tree, CheckIRId check_ir_id,
           const std::optional<Parse::Tree::PackagingDecl>& packaging_decl,
           SharedValueStores& value_stores, std::string filename)
    : parse_tree_(parse_tree),
      check_ir_id_(check_ir_id),
      package_id_(packaging_decl ? packaging_decl->names.package_id
                                 : PackageNameId::None),
      library_id_(packaging_decl ? LibraryNameId::ForStringLiteralValueId(
                                       packaging_decl->names.library_id)
                                 : LibraryNameId::Default),
      value_stores_(&value_stores),
      filename_(std::move(filename)),
      entity_names_(check_ir_id),
      cpp_global_vars_(check_ir_id),
      functions_(check_ir_id),
      cpp_overload_sets_(check_ir_id),
      thunks_(check_ir_id),
      classes_(check_ir_id),
      fields_(check_ir_id),
      interfaces_(check_ir_id),
      named_constraints_(check_ir_id),
      require_impls_(check_ir_id),
      // 1 reserved id for `RequireImplsBlockId::Empty`.
      require_impls_blocks_(allocator_, check_ir_id, 1),
      associated_constants_(check_ir_id),
      facet_types_(check_ir_id),
      identified_facet_types_(check_ir_id),
      impls_(*this),
      specific_interfaces_(check_ir_id),
      generics_(check_ir_id),
      specifics_(check_ir_id),
      // The `2` prevents adding a tag for the global ids
      // `ImportIRId::{ApiForImpl,Cpp}`.
      import_irs_(check_ir_id, 2),
      clang_decls_(check_ir_id),
      clang_decl_signatures_(check_ir_id),
      // The `+1` prevents adding a tag to the global `NameSpace::PackageInstId`
      // instruction. It's not a "singleton" instruction, but it's a unique
      // instruction id that comes right after the singletons.
      insts_(this, SingletonInstKinds.size() + 1),
      vtables_(check_ir_id),
      constant_values_(ConstantId::NotConstant, &insts_),
      inst_blocks_(allocator_, check_ir_id),
      constants_(this),
      // 1 reserved id for `StructTypeFieldsId::Empty`.
      struct_type_fields_(allocator_, check_ir_id, 1),
      // 1 reserved id for `CustomLayoutId::Empty`.
      custom_layouts_(allocator_, check_ir_id, 1),
      expr_regions_(check_ir_id),
      clang_source_locs_(check_ir_id),
      bundles_(allocator_, check_ir_id) {
  // `type`, `form`, and the error type are both complete & concrete types.
  // TODO: This duplicates the code in `check/type_completion.cpp`. Consider
  // requiring these types to be complete from Check initialization instead,
  // and consider doing this automatically for all singleton types.
  types_.SetComplete(
      TypeType::TypeId,
      {.value_repr = {.kind = ValueRepr::Copy, .type_id = TypeType::TypeId},
       .object_layout = SemIR::ObjectLayout::Empty()});
  types_.SetComplete(
      FormType::TypeId,
      {.value_repr = {.kind = ValueRepr::Copy, .type_id = FormType::TypeId},
       .object_layout = SemIR::ObjectLayout::Empty()});
  types_.SetComplete(
      ErrorInst::TypeId,
      {.value_repr = {.kind = ValueRepr::Copy, .type_id = ErrorInst::TypeId},
       .object_layout = SemIR::ObjectLayout::Empty()});
  types_.SetComplete(
      InstType::TypeId,
      {.value_repr = {.kind = ValueRepr::Copy, .type_id = InstType::TypeId},
       .object_layout = SemIR::ObjectLayout::Empty()});

  insts_.Reserve(SingletonInstKinds.size());
  for (auto kind : SingletonInstKinds) {
    auto inst_id =
        insts_.AddInNoBlock(LocIdAndInst::NoLoc(Inst::MakeSingleton(kind)));
    constant_values_.Set(inst_id, ConstantId::ForConcreteConstant(inst_id));
  }
}

File::~File() = default;

auto File::Verify() const -> ErrorOr<Success> {
  // Invariants don't necessarily hold for invalid IR.
  if (has_errors_) {
    return Success();
  }

  // Check that every code block has a terminator sequence that appears at the
  // end of the block.
  for (const Function& function : functions_.values()) {
    for (InstBlockId block_id : function.body_block_ids) {
      TerminatorKind prior_kind = TerminatorKind::NotTerminator;
      for (InstId inst_id : inst_blocks().Get(block_id)) {
        TerminatorKind inst_kind =
            insts().Get(inst_id).kind().terminator_kind();
        if (prior_kind == TerminatorKind::Terminator) {
          return Error(llvm::formatv("Inst {0} in block {1} follows terminator",
                                     inst_id, block_id));
        }
        if (prior_kind > inst_kind) {
          return Error(
              llvm::formatv("Non-terminator inst {0} in block {1} follows "
                            "terminator sequence",
                            inst_id, block_id));
        }
        prior_kind = inst_kind;
      }
      if (prior_kind != TerminatorKind::Terminator) {
        return Error(llvm::formatv("No terminator in block {0}", block_id));
      }
    }
  }

  // TODO: Check that an instruction only references other instructions that are
  // either global or that dominate it.
  return Success();
}

auto File::OutputYaml(bool include_singletons) const -> Yaml::OutputMapping {
  return Yaml::OutputMapping([this, include_singletons](
                                 Yaml::OutputMapping::Map map) {
    map.Add("filename", filename_);
    map.Add(
        "sem_ir", Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
          map.Add("names", names().OutputYaml());
          map.Add("import_irs", import_irs_.OutputYaml());
          map.Add("import_ir_insts", import_ir_insts_.OutputYaml());
          map.Add("clang_decls", clang_decls_.OutputYaml());
          map.Add("clang_decl_signatures", clang_decl_signatures_.OutputYaml());
          map.Add("name_scopes", name_scopes_.OutputYaml());
          map.Add("entity_names", entity_names_.OutputYaml());
          map.Add("cpp_global_vars", cpp_global_vars_.OutputYaml());
          map.Add("functions", functions_.OutputYaml());
          map.Add("classes", classes_.OutputYaml());
          map.Add("interfaces", interfaces_.OutputYaml());
          map.Add("associated_constants", associated_constants_.OutputYaml());
          map.Add("impls", impls_.OutputYaml());
          map.Add("generics", generics_.OutputYaml());
          map.Add("specifics", specifics_.OutputYaml());
          map.Add("specific_interfaces", specific_interfaces_.OutputYaml());
          map.Add("struct_type_fields", struct_type_fields_.OutputYaml());
          map.Add("types", types_.OutputYaml());
          map.Add("facet_types", facet_types_.OutputYaml());
          map.Add("insts",
                  Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                    for (auto [id, inst] : insts_.enumerate()) {
                      inst.CacheBundleDebugKinds(bundles_);
                      if (!include_singletons && IsSingletonInstId(id)) {
                        continue;
                      }
                      map.Add(PrintToString(id), Yaml::OutputScalar(inst));
                    }
                  }));
          map.Add("bundles", bundles_.OutputYaml());
          map.Add("constant_values",
                  constant_values_.OutputYaml(include_singletons));
          map.Add("inst_blocks", inst_blocks_.OutputYaml());
          map.Add("value_stores", value_stores_->OutputYaml());
        }));
  });
}

auto File::CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
    -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "allocator_"), allocator_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "entity_names_"),
                    entity_names_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "cpp_global_vars_"),
                    cpp_global_vars_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "functions_"), functions_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "thunks_"), thunks_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "classes_"), classes_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "interfaces_"), interfaces_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "impls_"), impls_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "generics_"), generics_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "specifics_"), specifics_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "import_irs_"), import_irs_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "import_ir_insts_"),
                    import_ir_insts_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "clang_decls_"), clang_decls_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "clang_decl_signatures_"),
                    clang_decl_signatures_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "struct_type_fields_"),
                    struct_type_fields_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "insts_"), insts_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "name_scopes_"), name_scopes_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "constant_values_"),
                    constant_values_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "inst_blocks_"), inst_blocks_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "constants_"), constants_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "types_"), types_);
}

auto File::set_cpp_file(std::unique_ptr<SemIR::CppFile> cpp_file) -> void {
  cpp_file_ = std::move(cpp_file);
}

auto File::AppendCppMangledTypeName(ClassId class_id,
                                    llvm::raw_ostream& out) const -> bool {
  // Only classes imported *from* C++ need this. A C++ scope records the Clang
  // declaration it came from; note that a Carbon class exported *to* C++ also
  // has an associated Clang declaration, but its scope is not a C++ scope and
  // it is already uniquely identified by its name and parent scope.
  auto scope_id = classes().Get(class_id).scope_id;
  if (!scope_id.has_value()) {
    return false;
  }
  const auto& scope = name_scopes().Get(scope_id);
  if (!scope.is_cpp_scope()) {
    return false;
  }
  auto clang_decl_id = scope.clang_decl_context_id();
  if (!clang_decl_id.has_value()) {
    return false;
  }
  // A C++ class's scope always maps to a Clang tag declaration.
  auto* tag_decl =
      clang::cast<clang::TagDecl>(clang_decls().Get(clang_decl_id).key.decl);
  cpp_file_->mangle_context().mangleCanonicalTypeName(
      cpp_file_->ast_context().getCanonicalTagType(tag_decl), out);
  return true;
}

}  // namespace Carbon::SemIR

namespace Carbon {
template class ValueStore<SemIR::ExprRegionId, SemIR::ExprRegion,
                          Tag<SemIR::CheckIRId>>;
template class ValueStore<SemIR::ClangSourceLocId, clang::SourceLocation,
                          Tag<SemIR::CheckIRId>>;
template class ValueStore<SemIR::CustomLayoutId,
                          llvm::MutableArrayRef<SemIR::ObjectSize>,
                          Tag<SemIR::CheckIRId>>;
template class BlockValueStore<SemIR::CustomLayoutId, SemIR::ObjectSize,
                               Tag<SemIR::CheckIRId>>;
template class BlockValueStore<SemIR::RawBundleId, SemIR::AnyRawId,
                               Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
