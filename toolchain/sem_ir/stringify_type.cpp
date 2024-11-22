// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/stringify_type.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entity_with_params_base.h"

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
    enum Kind : uint8_t {
      Inst,
      FixedString,
    };
    // The kind of step to perform.
    Kind kind;
    union {
      // The instruction to print, when kind is Inst.
      InstId inst_id;
      // The fixed string to print, when kind is FixedString.
      const char* fixed_string;
    };
    // The index within the current step. Not used by all kinds of step.
    int index = 0;

    auto Next() const -> Step {
      Step next = *this;
      ++next.index;
      return next;
    }
  };
  llvm::SmallVector<Step> steps = {Step{
      .sem_ir = outer_sem_ir, .kind = Step::Inst, .inst_id = outer_inst_id}};

  auto push_string = [&](const char* string) {
    steps.push_back({.sem_ir = outer_sem_ir,
                     .kind = Step::FixedString,
                     .fixed_string = string});
  };

  while (!steps.empty()) {
    auto step = steps.pop_back_val();

    if (step.kind == Step::FixedString) {
      out << step.fixed_string;
      continue;
    }

    CARBON_CHECK(step.kind == Step::Inst);
    if (!step.inst_id.is_valid()) {
      out << "<invalid type>";
      continue;
    }

    const auto& sem_ir = step.sem_ir;
    // Helper for instructions with the current sem_ir.
    auto push_inst_id = [&](InstId inst_id) {
      steps.push_back(
          {.sem_ir = sem_ir, .kind = Step::Inst, .inst_id = inst_id});
    };

    auto push_specific_id = [&](const EntityWithParamsBase& entity,
                                SpecificId specific_id) {
      if (!entity.param_patterns_id.is_valid()) {
        return;
      }
      int num_params =
          sem_ir.inst_blocks().Get(entity.param_patterns_id).size();
      if (!num_params) {
        out << "()";
        return;
      }
      if (!specific_id.is_valid()) {
        // The name of the generic was used within the generic itself.
        // TODO: Should we print the names of the generic parameters in this
        // case?
        return;
      }
      out << "(";
      const auto& specific = sem_ir.specifics().Get(specific_id);
      auto args =
          sem_ir.inst_blocks().Get(specific.args_id).take_back(num_params);
      bool last = true;
      for (auto arg : llvm::reverse(args)) {
        push_string(last ? ")" : ", ");
        push_inst_id(arg);
        last = false;
      }
    };

    auto untyped_inst = sem_ir.insts().Get(step.inst_id);
    CARBON_KIND_SWITCH(untyped_inst) {
      case SemIR::AutoType::Kind:
      case SemIR::BoolType::Kind:
      case SemIR::BoundMethodType::Kind:
      case SemIR::ErrorInst::Kind:
      case SemIR::IntLiteralType::Kind:
      case SemIR::LegacyFloatType::Kind:
      case SemIR::NamespaceType::Kind:
      case SemIR::SpecificFunctionType::Kind:
      case SemIR::StringType::Kind:
      case SemIR::TypeType::Kind:
      case SemIR::VtableType::Kind:
      case SemIR::WitnessType::Kind: {
        // Builtin instructions use their IR name as a label.
        out << untyped_inst.kind().ir_name();
        break;
      }
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
        const auto& class_info = sem_ir.classes().Get(inst.class_id);
        out << sem_ir.names().GetFormatted(class_info.name_id);
        push_specific_id(class_info, inst.specific_id);
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
      case CARBON_KIND(FacetAccessType inst): {
        // Given `T:! I`, print `T as type` as simply `T`.
        push_inst_id(inst.facet_value_inst_id);
        break;
      }
      case CARBON_KIND(FacetType inst): {
        const FacetTypeInfo& facet_type_info =
            sem_ir.facet_types().Get(inst.facet_type_id);
        if (facet_type_info.impls_constraints.empty()) {
          out << "type";
        } else {
          const auto& impls = facet_type_info.impls_constraints[step.index];
          const auto& interface_info =
              sem_ir.interfaces().Get(impls.interface_id);
          out << sem_ir.names().GetFormatted(interface_info.name_id);
          push_specific_id(interface_info, impls.specific_id);
          if (step.index + 1 <
              static_cast<int>(facet_type_info.impls_constraints.size())) {
            steps.push_back(step.Next());
            push_string(" & ");
          }
        }
        // TODO: Also output other restrictions from facet_type_info.
        if (step.index + 1 >=
                static_cast<int>(facet_type_info.impls_constraints.size()) &&
            facet_type_info.requirement_block_id.is_valid()) {
          out << " where...";
        }
        break;
      }
      case CARBON_KIND(FacetValue inst): {
        // No need to output the witness.
        push_inst_id(sem_ir.types().GetInstId(inst.type_id));
        push_string(" as ");
        push_inst_id(inst.type_inst_id);
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
        auto fields = sem_ir.struct_type_fields().Get(inst.fields_id);
        if (fields.empty()) {
          out << "{}";
          break;
        }

        if (step.index >= static_cast<int>(fields.size())) {
          out << "}";
          break;
        }

        const auto& field = fields[step.index];
        out << (step.index == 0 ? "{" : ", ") << "."
            << sem_ir.names().GetFormatted(field.name_id) << ": ";
        steps.push_back(step.Next());
        push_inst_id(sem_ir.types().GetInstId(field.type_id));
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
      case BindName::Kind:
      case BindValue::Kind:
      case BindingPattern::Kind:
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
      case IntValue::Kind:
      case InterfaceDecl::Kind:
      case InterfaceWitness::Kind:
      case InterfaceWitnessAccess::Kind:
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
      case SpecificConstant::Kind:
      case SpecificFunction::Kind:
      case SpliceBlock::Kind:
      case StringLiteral::Kind:
      case StructAccess::Kind:
      case StructInit::Kind:
      case StructLiteral::Kind:
      case StructValue::Kind:
      case SymbolicBindingPattern::Kind:
      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case TupleAccess::Kind:
      case TupleInit::Kind:
      case TupleLiteral::Kind:
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
        out << "<cannot stringify " << step.inst_id << " kind "
            << untyped_inst.kind() << ">";
        break;
    }
  }

  return str;
}

}  // namespace Carbon::SemIR
