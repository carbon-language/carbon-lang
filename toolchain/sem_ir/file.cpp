// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/file.h"

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/shared_value_stores.h"
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
      impls_(*this),
      type_blocks_(allocator_),
      constant_values_(ConstantId::NotConstant),
      inst_blocks_(allocator_),
      constants_(this) {
  // `type` and the error type are both complete & concrete types.
  types_.SetComplete(TypeType::SingletonTypeId,
                     {.value_repr = {.kind = ValueRepr::Copy,
                                     .type_id = TypeType::SingletonTypeId}});
  types_.SetComplete(ErrorInst::SingletonTypeId,
                     {.value_repr = {.kind = ValueRepr::Copy,
                                     .type_id = ErrorInst::SingletonTypeId}});

  insts_.Reserve(SingletonInstKinds.size());
  for (auto kind : SingletonInstKinds) {
    auto inst_id =
        insts_.AddInNoBlock(LocIdAndInst::NoLoc(Inst::MakeSingleton(kind)));
    constant_values_.Set(inst_id,
                         SemIR::ConstantId::ForConcreteConstant(inst_id));
  }
}

auto File::Verify() const -> ErrorOr<Success> {
  // Invariants don't necessarily hold for invalid IR.
  if (has_errors_) {
    return Success();
  }

  // Check that every code block has a terminator sequence that appears at the
  // end of the block.
  for (const Function& function : functions_.array_ref()) {
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
          map.Add("import_irs", import_irs_.OutputYaml());
          map.Add("import_ir_insts", import_ir_insts_.OutputYaml());
          map.Add("name_scopes", name_scopes_.OutputYaml());
          map.Add("entity_names", entity_names_.OutputYaml());
          map.Add("functions", functions_.OutputYaml());
          map.Add("classes", classes_.OutputYaml());
          map.Add("generics", generics_.OutputYaml());
          map.Add("specifics", specifics_.OutputYaml());
          map.Add("struct_type_fields", struct_type_fields_.OutputYaml());
          map.Add("types", types_.OutputYaml());
          map.Add("type_blocks", type_blocks_.OutputYaml());
          map.Add("insts",
                  Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                    for (auto [id, inst] : insts_.enumerate()) {
                      if (!include_singletons && IsSingletonInstId(id)) {
                        continue;
                      }
                      map.Add(PrintToString(id), Yaml::OutputScalar(inst));
                    }
                  }));
          map.Add("constant_values",
                  Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                    for (auto [id, _] : insts_.enumerate()) {
                      if (!include_singletons && IsSingletonInstId(id)) {
                        continue;
                      }
                      auto value = constant_values_.Get(id);
                      if (!value.has_value() || value.is_constant()) {
                        map.Add(PrintToString(id), Yaml::OutputScalar(value));
                      }
                    }
                  }));
          map.Add(
              "symbolic_constants",
              Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                for (const auto& [i, symbolic] :
                     llvm::enumerate(constant_values().symbolic_constants())) {
                  map.Add(
                      PrintToString(ConstantId::ForSymbolicConstantIndex(i)),
                      Yaml::OutputScalar(symbolic));
                }
              }));
          map.Add("inst_blocks", inst_blocks_.OutputYaml());
        }));
  });
}

auto File::CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
    -> void {
  mem_usage.Collect(MemUsage::ConcatLabel(label, "allocator_"), allocator_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "entity_names_"),
                    entity_names_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "functions_"), functions_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "classes_"), classes_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "interfaces_"), interfaces_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "impls_"), impls_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "generics_"), generics_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "specifics_"), specifics_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "import_irs_"), import_irs_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "import_ir_insts_"),
                    import_ir_insts_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "struct_type_fields_"),
                    struct_type_fields_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "type_blocks_"), type_blocks_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "insts_"), insts_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "name_scopes_"), name_scopes_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "constant_values_"),
                    constant_values_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "inst_blocks_"), inst_blocks_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "constants_"), constants_);
  mem_usage.Collect(MemUsage::ConcatLabel(label, "types_"), types_);
}

auto GetExprCategory(const File& file, InstId inst_id) -> ExprCategory {
  const File* ir = &file;

  // The overall expression category if the current instruction is a value
  // expression.
  ExprCategory value_category = ExprCategory::Value;

  while (true) {
    auto untyped_inst = ir->insts().Get(inst_id);
    CARBON_KIND_SWITCH(untyped_inst) {
      case AdaptDecl::Kind:
      case AddrPattern::Kind:
      case Assign::Kind:
      case BaseDecl::Kind:
      case BindingPattern::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case FieldDecl::Kind:
      case FunctionDecl::Kind:
      case ImplDecl::Kind:
      case NameBindingDecl::Kind:
      case Namespace::Kind:
      case OutParamPattern::Kind:
      case RefParamPattern::Kind:
      case RequirementEquivalent::Kind:
      case RequirementImpls::Kind:
      case RequirementRewrite::Kind:
      case Return::Kind:
      case ReturnSlotPattern::Kind:
      case ReturnExpr::Kind:
      case TuplePattern::Kind:
      case VarPattern::Kind:
      case Vtable::Kind:
        return ExprCategory::NotExpr;

      case ImportRefUnloaded::Kind:
      case ImportRefLoaded::Kind: {
        auto import_ir_inst = ir->import_ir_insts().Get(
            untyped_inst.As<SemIR::AnyImportRef>().import_ir_inst_id);
        ir = ir->import_irs().Get(import_ir_inst.ir_id).sem_ir;
        inst_id = import_ir_inst.inst_id;
        continue;
      }

      case CARBON_KIND(AsCompatible inst): {
        inst_id = inst.source_id;
        continue;
      }

      case CARBON_KIND(BindAlias inst): {
        inst_id = inst.value_id;
        continue;
      }
      case CARBON_KIND(ExportDecl inst): {
        inst_id = inst.value_id;
        continue;
      }
      case CARBON_KIND(NameRef inst): {
        inst_id = inst.value_id;
        continue;
      }

      case CARBON_KIND(Converted inst): {
        inst_id = inst.result_id;
        continue;
      }

      case CARBON_KIND(SpecificConstant inst): {
        inst_id = inst.inst_id;
        continue;
      }

      case AccessMemberAction::Kind:
      case AddrOf::Kind:
      case ArrayType::Kind:
      case AssociatedConstantDecl::Kind:
      case AssociatedEntity::Kind:
      case AssociatedEntityType::Kind:
      case AutoType::Kind:
      case BindSymbolicName::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case BoolType::Kind:
      case BoundMethod::Kind:
      case BoundMethodType::Kind:
      case ClassDecl::Kind:
      case ClassType::Kind:
      case CompleteTypeWitness::Kind:
      case ConstType::Kind:
      case ConvertToValueAction::Kind:
      case FacetAccessType::Kind:
      case FacetAccessWitness::Kind:
      case FacetType::Kind:
      case FacetValue::Kind:
      case FloatLiteral::Kind:
      case FloatType::Kind:
      case FunctionType::Kind:
      case FunctionTypeWithSelfType::Kind:
      case GenericClassType::Kind:
      case GenericInterfaceType::Kind:
      case ImplWitness::Kind:
      case ImplWitnessAccess::Kind:
      case ImportCppDecl::Kind:
      case ImportDecl::Kind:
      case InstType::Kind:
      case InstValue::Kind:
      case IntLiteralType::Kind:
      case IntType::Kind:
      case IntValue::Kind:
      case InterfaceDecl::Kind:
      case LegacyFloatType::Kind:
      case NamespaceType::Kind:
      case PointerType::Kind:
      case RefineTypeAction::Kind:
      case RequireCompleteType::Kind:
      case SpecificFunction::Kind:
      case SpecificFunctionType::Kind:
      case SpecificImplFunction::Kind:
      case StringLiteral::Kind:
      case StringType::Kind:
      case StructType::Kind:
      case StructValue::Kind:
      case SymbolicBindingPattern::Kind:
      case TupleType::Kind:
      case TupleValue::Kind:
      case TypeOfInst::Kind:
      case TypeType::Kind:
      case UnaryOperatorNot::Kind:
      case UnboundElementType::Kind:
      case ValueOfInitializer::Kind:
      case ValueParam::Kind:
      case ValueParamPattern::Kind:
      case VtableType::Kind:
      case WhereExpr::Kind:
      case WitnessType::Kind:
        return value_category;

      case ErrorInst::Kind:
        return ExprCategory::Error;

      case CARBON_KIND(BindName inst): {
        // TODO: Don't rely on value_id for expression category, since it may
        // not be valid yet. This workaround only works because we don't support
        // `var` in function signatures yet.
        if (!inst.value_id.has_value()) {
          return value_category;
        }
        inst_id = inst.value_id;
        continue;
      }

      case CARBON_KIND(ArrayIndex inst): {
        inst_id = inst.array_id;
        continue;
      }

      case VtablePtr::Kind:
        return ExprCategory::EphemeralRef;

      case CARBON_KIND(ClassElementAccess inst): {
        inst_id = inst.base_id;
        // A value of class type is a pointer to an object representation.
        // Therefore, if the base is a value, the result is an ephemeral
        // reference.
        value_category = ExprCategory::EphemeralRef;
        continue;
      }

      case CARBON_KIND(StructAccess inst): {
        inst_id = inst.struct_id;
        continue;
      }

      case CARBON_KIND(TupleAccess inst): {
        inst_id = inst.tuple_id;
        continue;
      }

      case CARBON_KIND(SpliceBlock inst): {
        inst_id = inst.result_id;
        continue;
      }

      case SpliceInst::Kind:
        // TODO: Add ExprCategory::Dependent.
        return value_category;

      case StructLiteral::Kind:
      case TupleLiteral::Kind:
        return ExprCategory::Mixed;

      case ArrayInit::Kind:
      case Call::Kind:
      case InitializeFrom::Kind:
      case ClassInit::Kind:
      case StructInit::Kind:
      case TupleInit::Kind:
        return ExprCategory::Initializing;

      case Deref::Kind:
      case VarStorage::Kind:
      case ReturnSlot::Kind:
        return ExprCategory::DurableRef;

      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case ValueAsRef::Kind:
        return ExprCategory::EphemeralRef;

      case OutParam::Kind:
      case RefParam::Kind:
        // TODO: Consider introducing a separate category for OutParam:
        // unlike other DurableRefs, it permits initialization.
        return ExprCategory::DurableRef;
    }
  }
}

}  // namespace Carbon::SemIR
