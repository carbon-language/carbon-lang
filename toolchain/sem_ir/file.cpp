// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/file.h"

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_store.h"
#include "toolchain/base/yaml.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/builtin_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

auto Function::GetParamFromParamRefId(const File& sem_ir, InstId param_ref_id)
    -> std::pair<InstId, Param> {
  auto ref = sem_ir.insts().Get(param_ref_id);

  if (auto addr_pattern = ref.TryAs<SemIR::AddrPattern>()) {
    param_ref_id = addr_pattern->inner_id;
    ref = sem_ir.insts().Get(param_ref_id);
  }

  if (auto bind_name = ref.TryAs<SemIR::AnyBindName>()) {
    param_ref_id = bind_name->value_id;
    ref = sem_ir.insts().Get(param_ref_id);
  }

  return {param_ref_id, ref.As<SemIR::Param>()};
}

auto ValueRepr::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: ";
  switch (kind) {
    case Unknown:
      out << "unknown";
      break;
    case None:
      out << "none";
      break;
    case Copy:
      out << "copy";
      break;
    case Pointer:
      out << "pointer";
      break;
    case Custom:
      out << "custom";
      break;
  }
  out << ", type: " << type_id << "}";
}

auto TypeInfo::Print(llvm::raw_ostream& out) const -> void {
  out << "{constant: " << constant_id << ", value_rep: " << value_repr << "}";
}

File::File(SharedValueStores& value_stores, std::string filename,
           const File* builtins, llvm::function_ref<void()> init_builtins)
    : value_stores_(&value_stores),
      filename_(std::move(filename)),
      type_blocks_(allocator_),
      constant_values_(ConstantId::NotConstant),
      inst_blocks_(allocator_),
      constants_(*this, allocator_) {
  CARBON_CHECK(builtins != nullptr);
  auto builtins_id =
      import_irs_.Add({.node_id = Parse::NodeId::Invalid, .sem_ir = builtins});
  CARBON_CHECK(builtins_id == ImportIRId::Builtins)
      << "Builtins must be the first IR";

  auto api_placeholder_id =
      import_irs_.Add({.node_id = Parse::NodeId::Invalid, .sem_ir = nullptr});
  CARBON_CHECK(api_placeholder_id == ImportIRId::ApiForImpl)
      << "ApiForImpl must be the second IR";

  insts_.Reserve(BuiltinKind::ValidCount);
  init_builtins();
  CARBON_CHECK(insts_.size() == BuiltinKind::ValidCount)
      << "Builtins should produce " << BuiltinKind::ValidCount
      << " insts, actual: " << insts_.size();
  for (auto i : llvm::seq(BuiltinKind::ValidCount)) {
    auto builtin_id = SemIR::InstId(i);
    constant_values_.Set(builtin_id,
                         SemIR::ConstantId::ForTemplateConstant(builtin_id));
  }
}

File::File(SharedValueStores& value_stores)
    : File(value_stores, "<builtins>", this, [&]() {
// Error uses a self-referential type so that it's not accidentally treated as
// a normal type. Every other builtin is a type, including the
// self-referential TypeType.
#define CARBON_SEM_IR_BUILTIN_KIND(Name, ...)                              \
  insts_.AddInNoBlock(                                                     \
      {Builtin{BuiltinKind::Name == BuiltinKind::Error ? TypeId::Error     \
                                                       : TypeId::TypeType, \
               BuiltinKind::Name}});
#include "toolchain/sem_ir/builtin_kind.def"
      }) {
}

File::File(SharedValueStores& value_stores, std::string filename,
           const File* builtins)
    : File(value_stores, filename, builtins, [&]() {
        for (auto [i, inst] : llvm::enumerate(builtins->insts_.array_ref())) {
          // We can reuse the type_id from the builtin IR's inst because they're
          // special-cased values.
          auto import_ir_inst_id = import_ir_insts_.Add(
              {.ir_id = ImportIRId::Builtins, .inst_id = SemIR::InstId(i)});
          insts_.AddInNoBlock(
              ImportRefLoaded{inst.type_id(), import_ir_inst_id});
        }
      }) {}

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

auto File::OutputYaml(bool include_builtins) const -> Yaml::OutputMapping {
  return Yaml::OutputMapping([this,
                              include_builtins](Yaml::OutputMapping::Map map) {
    map.Add("filename", filename_);
    map.Add(
        "sem_ir", Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
          map.Add("import_irs_size", Yaml::OutputScalar(import_irs_.size()));
          map.Add("name_scopes", name_scopes_.OutputYaml());
          map.Add("bind_names", bind_names_.OutputYaml());
          map.Add("functions", functions_.OutputYaml());
          map.Add("classes", classes_.OutputYaml());
          map.Add("types", types_.OutputYaml());
          map.Add("type_blocks", type_blocks_.OutputYaml());
          map.Add("insts",
                  Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                    int start = include_builtins ? 0 : BuiltinKind::ValidCount;
                    for (int i : llvm::seq(start, insts_.size())) {
                      auto id = InstId(i);
                      map.Add(PrintToString(id),
                              Yaml::OutputScalar(insts_.Get(id)));
                    }
                  }));
          map.Add("constant_values",
                  Yaml::OutputMapping([&](Yaml::OutputMapping::Map map) {
                    int start = include_builtins ? 0 : BuiltinKind::ValidCount;
                    for (int i : llvm::seq(start, insts_.size())) {
                      auto id = InstId(i);
                      auto value = constant_values_.Get(id);
                      if (!value.is_valid() || value.is_constant()) {
                        map.Add(PrintToString(id), Yaml::OutputScalar(value));
                      }
                    }
                  }));
          map.Add("inst_blocks", inst_blocks_.OutputYaml());
        }));
  });
}

// Map an instruction kind representing a type into an integer describing the
// precedence of that type's syntax. Higher numbers correspond to higher
// precedence.
static auto GetTypePrecedence(InstKind kind) -> int {
  switch (kind) {
    case ArrayType::Kind:
    case AssociatedEntityType::Kind:
    case BindAlias::Kind:
    case BindSymbolicName::Kind:
    case Builtin::Kind:
    case ClassType::Kind:
    case FacetTypeAccess::Kind:
    case ImportRefLoaded::Kind:
    case ImportRefUsed::Kind:
    case InterfaceType::Kind:
    case IntType::Kind:
    case NameRef::Kind:
    case StructType::Kind:
    case TupleType::Kind:
    case UnboundElementType::Kind:
      return 0;
    case ConstType::Kind:
      return -1;
    case PointerType::Kind:
      return -2;

    case AdaptDecl::Kind:
    case AddrOf::Kind:
    case AddrPattern::Kind:
    case ArrayIndex::Kind:
    case ArrayInit::Kind:
    case AsCompatible::Kind:
    case Assign::Kind:
    case AssociatedConstantDecl::Kind:
    case AssociatedEntity::Kind:
    case BaseDecl::Kind:
    case BindName::Kind:
    case BindValue::Kind:
    case BlockArg::Kind:
    case BoolLiteral::Kind:
    case BoundMethod::Kind:
    case Branch::Kind:
    case BranchIf::Kind:
    case BranchWithArg::Kind:
    case Call::Kind:
    case ClassDecl::Kind:
    case ClassElementAccess::Kind:
    case ClassInit::Kind:
    case Converted::Kind:
    case Deref::Kind:
    case FieldDecl::Kind:
    case FunctionDecl::Kind:
    case ImplDecl::Kind:
    case ImportRefUnloaded::Kind:
    case InitializeFrom::Kind:
    case InterfaceDecl::Kind:
    case InterfaceWitness::Kind:
    case InterfaceWitnessAccess::Kind:
    case IntLiteral::Kind:
    case Namespace::Kind:
    case Param::Kind:
    case RealLiteral::Kind:
    case Return::Kind:
    case ReturnExpr::Kind:
    case SpliceBlock::Kind:
    case StringLiteral::Kind:
    case StructAccess::Kind:
    case StructTypeField::Kind:
    case StructLiteral::Kind:
    case StructInit::Kind:
    case StructValue::Kind:
    case Temporary::Kind:
    case TemporaryStorage::Kind:
    case TupleAccess::Kind:
    case TupleIndex::Kind:
    case TupleLiteral::Kind:
    case TupleInit::Kind:
    case TupleValue::Kind:
    case UnaryOperatorNot::Kind:
    case ValueAsRef::Kind:
    case ValueOfInitializer::Kind:
    case VarStorage::Kind:
      CARBON_FATAL() << "GetTypePrecedence for non-type inst kind " << kind;
  }
}

// Implements File::StringifyTypeExpr. Static to prevent accidental use of
// member functions while traversing IRs.
static auto StringifyTypeExprImpl(const SemIR::File& outer_sem_ir,
                                  InstId outer_inst_id) {
  std::string str;
  llvm::raw_string_ostream out(str);

  struct Step {
    // The instruction's file.
    const File& sem_ir;
    // The instruction to print.
    InstId inst_id;
    // The index into inst_id to print. Not used by all types.
    int index = 0;

    auto Next() const -> Step {
      return {.sem_ir = sem_ir, .inst_id = inst_id, .index = index + 1};
    }
  };
  llvm::SmallVector<Step> steps = {
      Step{.sem_ir = outer_sem_ir, .inst_id = outer_inst_id}};

  while (!steps.empty()) {
    auto step = steps.pop_back_val();
    if (!step.inst_id.is_valid()) {
      out << "<invalid type>";
      continue;
    }

    // Builtins have designated labels.
    if (step.inst_id.is_builtin()) {
      out << step.inst_id.builtin_kind().label();
      continue;
    }

    const auto& sem_ir = step.sem_ir;
    // Helper for instructions with the current sem_ir.
    auto push_inst_id = [&](InstId inst_id) {
      steps.push_back({.sem_ir = sem_ir, .inst_id = inst_id});
    };

    auto untyped_inst = sem_ir.insts().Get(step.inst_id);
    CARBON_KIND_SWITCH(untyped_inst) {
      case CARBON_KIND(ArrayType inst): {
        if (step.index == 0) {
          out << "[";
          steps.push_back(step.Next());
          push_inst_id(sem_ir.types().GetInstId(inst.element_type_id));
        } else if (step.index == 1) {
          out << "; " << sem_ir.GetArrayBoundValue(inst.bound_id) << "]";
        }
        break;
      }
      case CARBON_KIND(AssociatedEntityType inst): {
        if (step.index == 0) {
          out << "<associated ";
          steps.push_back(step.Next());
          push_inst_id(sem_ir.types().GetInstId(inst.entity_type_id));
        } else {
          auto interface_name_id =
              sem_ir.interfaces().Get(inst.interface_id).name_id;
          out << " in " << sem_ir.names().GetFormatted(interface_name_id)
              << ">";
        }
        break;
      }
      case BindAlias::Kind:
      case BindSymbolicName::Kind: {
        auto name_id = untyped_inst.As<AnyBindName>().bind_name_id;
        out << sem_ir.names().GetFormatted(
            sem_ir.bind_names().Get(name_id).name_id);
        break;
      }
      case CARBON_KIND(ClassType inst): {
        auto class_name_id = sem_ir.classes().Get(inst.class_id).name_id;
        out << sem_ir.names().GetFormatted(class_name_id);
        break;
      }
      case CARBON_KIND(ConstType inst): {
        if (step.index == 0) {
          out << "const ";

          // Add parentheses if required.
          auto inner_type_inst_id = sem_ir.types().GetInstId(inst.inner_id);
          if (GetTypePrecedence(sem_ir.insts().Get(inner_type_inst_id).kind()) <
              GetTypePrecedence(SemIR::ConstType::Kind)) {
            out << "(";
            steps.push_back(step.Next());
          }

          push_inst_id(inner_type_inst_id);
        } else if (step.index == 1) {
          out << ")";
        }
        break;
      }
      case CARBON_KIND(FacetTypeAccess inst): {
        // Print `T as type` as simply `T`.
        push_inst_id(inst.facet_id);
        break;
      }
      case ImportRefLoaded::Kind:
      case ImportRefUsed::Kind: {
        auto import_ir_inst = sem_ir.import_ir_insts().Get(
            untyped_inst.As<AnyImportRef>().import_ir_inst_id);
        steps.push_back(
            {.sem_ir = *sem_ir.import_irs().Get(import_ir_inst.ir_id).sem_ir,
             .inst_id = import_ir_inst.inst_id});
        break;
      }
      case CARBON_KIND(InterfaceType inst): {
        auto interface_name_id =
            sem_ir.interfaces().Get(inst.interface_id).name_id;
        out << sem_ir.names().GetFormatted(interface_name_id);
        break;
      }
      case CARBON_KIND(IntType inst): {
        if (step.index == 1) {
          out << ")";
        } else if (auto width_value =
                       sem_ir.insts().TryGetAs<IntLiteral>(inst.bit_width_id)) {
          out << (inst.int_kind.is_signed() ? "i" : "u");
          sem_ir.ints().Get(width_value->int_id).print(out, /*isSigned=*/false);
        } else {
          out << (inst.int_kind.is_signed() ? "Core.Int(" : "Core.UInt(");
          steps.push_back(step.Next());
          push_inst_id(inst.bit_width_id);
        }
        break;
      }
      case CARBON_KIND(NameRef inst): {
        out << sem_ir.names().GetFormatted(inst.name_id);
        break;
      }
      case CARBON_KIND(PointerType inst): {
        if (step.index == 0) {
          steps.push_back(step.Next());
          push_inst_id(sem_ir.types().GetInstId(inst.pointee_id));
        } else if (step.index == 1) {
          out << "*";
        }
        break;
      }
      case CARBON_KIND(StructType inst): {
        auto refs = sem_ir.inst_blocks().Get(inst.fields_id);
        if (refs.empty()) {
          out << "{}";
          break;
        } else if (step.index == 0) {
          out << "{";
        } else if (step.index < static_cast<int>(refs.size())) {
          out << ", ";
        } else {
          out << "}";
          break;
        }

        steps.push_back(step.Next());
        push_inst_id(refs[step.index]);
        break;
      }
      case CARBON_KIND(StructTypeField inst): {
        out << "." << sem_ir.names().GetFormatted(inst.name_id) << ": ";
        push_inst_id(sem_ir.types().GetInstId(inst.field_type_id));
        break;
      }
      case CARBON_KIND(TupleType inst): {
        auto refs = sem_ir.type_blocks().Get(inst.elements_id);
        if (refs.empty()) {
          out << "()";
          break;
        } else if (step.index == 0) {
          out << "(";
        } else if (step.index < static_cast<int>(refs.size())) {
          out << ", ";
        } else {
          // A tuple of one element has a comma to disambiguate from an
          // expression.
          if (step.index == 1) {
            out << ",";
          }
          out << ")";
          break;
        }
        steps.push_back(step.Next());
        push_inst_id(sem_ir.types().GetInstId(refs[step.index]));
        break;
      }
      case CARBON_KIND(UnboundElementType inst): {
        if (step.index == 0) {
          out << "<unbound element of class ";
          steps.push_back(step.Next());
          push_inst_id(sem_ir.types().GetInstId(inst.class_type_id));
        } else {
          out << ">";
        }
        break;
      }
      case AdaptDecl::Kind:
      case AddrOf::Kind:
      case AddrPattern::Kind:
      case ArrayIndex::Kind:
      case ArrayInit::Kind:
      case AsCompatible::Kind:
      case Assign::Kind:
      case AssociatedConstantDecl::Kind:
      case AssociatedEntity::Kind:
      case BaseDecl::Kind:
      case BindName::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case BoundMethod::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case Builtin::Kind:
      case Call::Kind:
      case ClassDecl::Kind:
      case ClassElementAccess::Kind:
      case ClassInit::Kind:
      case Converted::Kind:
      case Deref::Kind:
      case FieldDecl::Kind:
      case FunctionDecl::Kind:
      case ImplDecl::Kind:
      case ImportRefUnloaded::Kind:
      case InitializeFrom::Kind:
      case InterfaceDecl::Kind:
      case InterfaceWitness::Kind:
      case InterfaceWitnessAccess::Kind:
      case IntLiteral::Kind:
      case Namespace::Kind:
      case Param::Kind:
      case RealLiteral::Kind:
      case Return::Kind:
      case ReturnExpr::Kind:
      case SpliceBlock::Kind:
      case StringLiteral::Kind:
      case StructAccess::Kind:
      case StructLiteral::Kind:
      case StructInit::Kind:
      case StructValue::Kind:
      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case TupleAccess::Kind:
      case TupleIndex::Kind:
      case TupleLiteral::Kind:
      case TupleInit::Kind:
      case TupleValue::Kind:
      case UnaryOperatorNot::Kind:
      case ValueAsRef::Kind:
      case ValueOfInitializer::Kind:
      case VarStorage::Kind:
        // We don't need to handle stringification for instructions that don't
        // show up in errors, but make it clear what's going on so that it's
        // clearer when stringification is needed.
        out << "<cannot stringify " << step.inst_id << ">";
        break;
    }
  }

  return str;
}

auto File::StringifyType(TypeId type_id) const -> std::string {
  return StringifyTypeExprImpl(*this, types().GetInstId(type_id));
}

auto File::StringifyTypeExpr(InstId outer_inst_id) const -> std::string {
  return StringifyTypeExprImpl(*this, outer_inst_id);
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
      case Assign::Kind:
      case BaseDecl::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case FieldDecl::Kind:
      case FunctionDecl::Kind:
      case ImplDecl::Kind:
      case ImportRefUnloaded::Kind:
      case Namespace::Kind:
      case Return::Kind:
      case ReturnExpr::Kind:
      case StructTypeField::Kind:
        return ExprCategory::NotExpr;

      case ImportRefLoaded::Kind:
      case ImportRefUsed::Kind: {
        auto import_ir_inst = ir->import_ir_insts().Get(
            untyped_inst.As<AnyImportRef>().import_ir_inst_id);
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
      case CARBON_KIND(NameRef inst): {
        inst_id = inst.value_id;
        continue;
      }

      case CARBON_KIND(Converted inst): {
        inst_id = inst.result_id;
        continue;
      }

      case AddrOf::Kind:
      case AddrPattern::Kind:
      case ArrayType::Kind:
      case AssociatedConstantDecl::Kind:
      case AssociatedEntity::Kind:
      case AssociatedEntityType::Kind:
      case BindSymbolicName::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case BoundMethod::Kind:
      case ClassDecl::Kind:
      case ClassType::Kind:
      case ConstType::Kind:
      case FacetTypeAccess::Kind:
      case InterfaceDecl::Kind:
      case InterfaceType::Kind:
      case InterfaceWitness::Kind:
      case InterfaceWitnessAccess::Kind:
      case IntLiteral::Kind:
      case IntType::Kind:
      case Param::Kind:
      case PointerType::Kind:
      case RealLiteral::Kind:
      case StringLiteral::Kind:
      case StructValue::Kind:
      case StructType::Kind:
      case TupleValue::Kind:
      case TupleType::Kind:
      case UnaryOperatorNot::Kind:
      case UnboundElementType::Kind:
      case ValueOfInitializer::Kind:
        return value_category;

      case CARBON_KIND(Builtin inst): {
        if (inst.builtin_kind == BuiltinKind::Error) {
          return ExprCategory::Error;
        }
        return value_category;
      }

      case CARBON_KIND(BindName inst): {
        inst_id = inst.value_id;
        continue;
      }

      case CARBON_KIND(ArrayIndex inst): {
        inst_id = inst.array_id;
        continue;
      }

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

      case CARBON_KIND(TupleIndex inst): {
        inst_id = inst.tuple_id;
        continue;
      }

      case CARBON_KIND(SpliceBlock inst): {
        inst_id = inst.result_id;
        continue;
      }

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
        return ExprCategory::DurableRef;

      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case ValueAsRef::Kind:
        return ExprCategory::EphemeralRef;
    }
  }
}

auto GetInitRepr(const File& file, TypeId type_id) -> InitRepr {
  auto value_rep = GetValueRepr(file, type_id);
  switch (value_rep.kind) {
    case ValueRepr::None:
      return {.kind = InitRepr::None};

    case ValueRepr::Copy:
      // TODO: Use in-place initialization for types that have non-trivial
      // destructive move.
      return {.kind = InitRepr::ByCopy};

    case ValueRepr::Pointer:
    case ValueRepr::Custom:
      return {.kind = InitRepr::InPlace};

    case ValueRepr::Unknown:
      CARBON_FATAL()
          << "Attempting to perform initialization of incomplete type";
  }
}

}  // namespace Carbon::SemIR
