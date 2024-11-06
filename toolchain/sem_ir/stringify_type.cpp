// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/stringify_type.h"

#include "toolchain/base/kind_switch.h"

namespace Carbon::SemIR {

// Map an instruction kind representing a type into an integer describing the
// precedence of that type's syntax. Higher numbers correspond to higher
// precedence.
static auto GetTypePrecedence(InstKind kind) -> int {
  CARBON_CHECK(kind.is_type() != InstIsType::Never,
               "Only called for kinds which can define a type.");
  if (kind == ConstType::Kind) {
    return -1;
  }
  if (kind == PointerType::Kind) {
    return -2;
  }
  return 0;
}

auto StringifyTypeExpr(const SemIR::File& outer_sem_ir, InstId outer_inst_id)
    -> std::string {
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
      out << step.inst_id.builtin_inst_kind().label();
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
        } else if (step.index == 1) {
          out << " in ";
          steps.push_back(step.Next());
          push_inst_id(sem_ir.types().GetInstId(inst.interface_type_id));
        } else {
          out << ">";
        }
        break;
      }
      case BindAlias::Kind:
      case BindSymbolicName::Kind:
      case ExportDecl::Kind: {
        auto name_id =
            untyped_inst.As<AnyBindNameOrExportDecl>().entity_name_id;
        out << sem_ir.names().GetFormatted(
            sem_ir.entity_names().Get(name_id).name_id);
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
      case CARBON_KIND(FacetType inst): {
        if (step.index == 0) {
          out << "<facet type ";
          steps.push_back(step.Next());
          FacetTypeInfo facet_type_info =
              sem_ir.facet_types().Get(inst.facet_type_id);
          // TODO: Also output restrictions from
          // facet_type_info.requirement_block_id.
          TypeId type_id = facet_type_info.base_facet_type_id;
          push_inst_id(sem_ir.types().GetInstId(type_id));
        } else {
          out << ">";
        }
        break;
      }
      case CARBON_KIND(FacetTypeAccess inst): {
        // Print `T as type` as simply `T`.
        push_inst_id(inst.facet_id);
        break;
      }
      case CARBON_KIND(FloatType inst): {
        // TODO: Is this okay?
        if (step.index == 1) {
          out << ")";
        } else if (auto width_value =
                       sem_ir.insts().TryGetAs<IntValue>(inst.bit_width_id)) {
          out << "f";
          sem_ir.ints().Get(width_value->int_id).print(out, /*isSigned=*/false);
        } else {
          out << "Core.Float(";
          steps.push_back(step.Next());
          push_inst_id(inst.bit_width_id);
        }
        break;
      }
      case CARBON_KIND(FunctionType inst): {
        auto fn_name_id = sem_ir.functions().Get(inst.function_id).name_id;
        out << "<type of " << sem_ir.names().GetFormatted(fn_name_id) << ">";
        break;
      }
      case CARBON_KIND(GenericClassType inst): {
        auto class_name_id = sem_ir.classes().Get(inst.class_id).name_id;
        out << "<type of " << sem_ir.names().GetFormatted(class_name_id) << ">";
        break;
      }
      case CARBON_KIND(GenericInterfaceType inst): {
        auto interface_name_id =
            sem_ir.interfaces().Get(inst.interface_id).name_id;
        out << "<type of " << sem_ir.names().GetFormatted(interface_name_id)
            << ">";
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
                       sem_ir.insts().TryGetAs<IntValue>(inst.bit_width_id)) {
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
      case CARBON_KIND(WhereExpr inst): {
        if (step.index == 0) {
          out << "<where restriction on ";
          steps.push_back(step.Next());
          TypeId type_id = sem_ir.insts().Get(inst.period_self_id).type_id();
          push_inst_id(sem_ir.types().GetInstId(type_id));
          // TODO: Also output restrictions from the inst block
          // inst.requirements_id.
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
      case BindingPattern::Kind:
      case BindName::Kind:
      case BindValue::Kind:
      case BlockArg::Kind:
      case BoolLiteral::Kind:
      case BoundMethod::Kind:
      case Branch::Kind:
      case BranchIf::Kind:
      case BranchWithArg::Kind:
      case BuiltinInst::Kind:
      case Call::Kind:
      case ClassDecl::Kind:
      case ClassElementAccess::Kind:
      case ClassInit::Kind:
      case CompleteTypeWitness::Kind:
      case Converted::Kind:
      case Deref::Kind:
      case FieldDecl::Kind:
      case FloatLiteral::Kind:
      case FunctionDecl::Kind:
      case ImplDecl::Kind:
      case ImportDecl::Kind:
      case ImportRefLoaded::Kind:
      case ImportRefUnloaded::Kind:
      case InitializeFrom::Kind:
      case SpecificConstant::Kind:
      case InterfaceDecl::Kind:
      case InterfaceWitness::Kind:
      case InterfaceWitnessAccess::Kind:
      case IntValue::Kind:
      case Namespace::Kind:
      case OutParam::Kind:
      case OutParamPattern::Kind:
      case RequirementEquivalent::Kind:
      case RequirementImpls::Kind:
      case RequirementRewrite::Kind:
      case Return::Kind:
      case ReturnExpr::Kind:
      case ReturnSlot::Kind:
      case ReturnSlotPattern::Kind:
      case SpecificFunction::Kind:
      case SpliceBlock::Kind:
      case StringLiteral::Kind:
      case StructAccess::Kind:
      case StructLiteral::Kind:
      case StructInit::Kind:
      case StructValue::Kind:
      case SymbolicBindingPattern::Kind:
      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case TupleAccess::Kind:
      case TupleLiteral::Kind:
      case TupleInit::Kind:
      case TupleValue::Kind:
      case UnaryOperatorNot::Kind:
      case ValueAsRef::Kind:
      case ValueOfInitializer::Kind:
      case ValueParam::Kind:
      case ValueParamPattern::Kind:
      case VarStorage::Kind:
        // We don't know how to print this instruction, but it might have a
        // constant value that we can print.
        auto const_inst_id =
            sem_ir.constant_values().GetConstantInstId(step.inst_id);
        if (const_inst_id.is_valid() && const_inst_id != step.inst_id) {
          push_inst_id(const_inst_id);
          break;
        }

        // We don't need to handle stringification for instructions that don't
        // show up in errors, but make it clear what's going on so that it's
        // clearer when stringification is needed.
        out << "<cannot stringify " << step.inst_id << ">";
        break;
    }
  }

  return str;
}

}  // namespace Carbon::SemIR
