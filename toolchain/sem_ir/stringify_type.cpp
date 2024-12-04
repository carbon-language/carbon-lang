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

namespace {
class StepStack {
 public:
  enum Kind : uint8_t {
    Inst,
    FixedString,
    ArrayBound,
    Name,
  };
  struct Step {
    // The kind of step to perform.
    Kind kind;
    union {
      // The instruction to print, when kind is Inst.
      InstId inst_id;
      // The fixed string to print, when kind is FixedString.
      const char* fixed_string;
      // The array bound to print, when kind is ArrayBound.
      InstId bound_id;
      // The name to print, when kind is Name.
      NameId name_id;
    };
  };

  explicit StepStack(const SemIR::File* file, InstId outer_inst_id)
      : sem_ir(file) {
    steps.push_back({.kind = Inst, .inst_id = outer_inst_id});
  }

  auto PushInstId(InstId inst_id) -> void {
    steps.push_back({.kind = Inst, .inst_id = inst_id});
  }
  auto PushString(const char* string) -> void {
    steps.push_back({.kind = FixedString, .fixed_string = string});
  }
  auto PushArrayBound(InstId bound_id) -> void {
    steps.push_back({.kind = ArrayBound, .bound_id = bound_id});
  }
  auto PushNameId(NameId name_id) -> void {
    steps.push_back({.kind = Name, .name_id = name_id});
  }
  auto PushTypeId(TypeId type_id) -> void {
    PushInstId(sem_ir->types().GetInstId(type_id));
  }
  auto PushSpecificId(const EntityWithParamsBase& entity,
                      SpecificId specific_id) -> void {
    if (!entity.param_patterns_id.is_valid()) {
      return;
    }
    int num_params = sem_ir->inst_blocks().Get(entity.param_patterns_id).size();
    if (!num_params) {
      PushString("()");
      return;
    }
    if (!specific_id.is_valid()) {
      // The name of the generic was used within the generic itself.
      // TODO: Should we print the names of the generic parameters in this
      // case?
      return;
    }
    const auto& specific = sem_ir->specifics().Get(specific_id);
    auto args =
        sem_ir->inst_blocks().Get(specific.args_id).take_back(num_params);
    bool last = true;
    for (auto arg : llvm::reverse(args)) {
      PushString(last ? ")" : ", ");
      PushInstId(arg);
      last = false;
    }
    PushString("(");
  }

  auto empty() const -> bool { return steps.empty(); }
  auto Pop() -> Step { return steps.pop_back_val(); }

 private:
  const SemIR::File* sem_ir;
  llvm::SmallVector<Step> steps;
};
}  // namespace

auto StringifyTypeExpr(const SemIR::File& sem_ir, InstId outer_inst_id)
    -> std::string {
  std::string str;
  llvm::raw_string_ostream out(str);

  // Note: Since this is a stack, work is resolved in the reverse order from the
  // order pushed.
  StepStack step_stack(&sem_ir, outer_inst_id);

  while (!step_stack.empty()) {
    auto step = step_stack.Pop();

    switch (step.kind) {
      case StepStack::FixedString:
        out << step.fixed_string;
        continue;
      case StepStack::ArrayBound:
        out << sem_ir.GetArrayBoundValue(step.bound_id);
        continue;
      case StepStack::Name:
        out << sem_ir.names().GetFormatted(step.name_id);
        continue;
      case StepStack::Inst:
        if (!step.inst_id.is_valid()) {
          out << "<invalid type>";
          continue;
        }
        // Fall through to the rest of the function.
    }

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
        out << "[";
        step_stack.PushString("]");
        step_stack.PushArrayBound(inst.bound_id);
        step_stack.PushString("; ");
        step_stack.PushTypeId(inst.element_type_id);
        break;
      }
      case CARBON_KIND(AssociatedEntityType inst): {
        out << "<associated ";
        step_stack.PushString(">");
        step_stack.PushTypeId(inst.interface_type_id);
        step_stack.PushString(" in ");
        step_stack.PushTypeId(inst.entity_type_id);
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
        step_stack.PushSpecificId(class_info, inst.specific_id);
        break;
      }
      case CARBON_KIND(ConstType inst): {
        out << "const ";

        // Add parentheses if required.
        auto inner_type_inst_id = sem_ir.types().GetInstId(inst.inner_id);
        if (GetTypePrecedence(sem_ir.insts().Get(inner_type_inst_id).kind()) <
            GetTypePrecedence(SemIR::ConstType::Kind)) {
          out << "(";
          step_stack.PushString(")");
        }

        step_stack.PushInstId(inner_type_inst_id);
        break;
      }
      case CARBON_KIND(FacetAccessType inst): {
        // Given `T:! I`, print `T as type` as simply `T`.
        step_stack.PushInstId(inst.facet_value_inst_id);
        break;
      }
      case CARBON_KIND(FacetAccessWitness inst): {
        out << "<witness for ";
        step_stack.PushString(">");
        step_stack.PushInstId(inst.facet_value_inst_id);
        break;
      }
      case CARBON_KIND(FacetType inst): {
        const FacetTypeInfo& facet_type_info =
            sem_ir.facet_types().Get(inst.facet_type_id);
        // Output `where` restrictions.
        bool some_where = false;
        if (facet_type_info.other_requirements) {
          step_stack.PushString("...");
          some_where = true;
        }
        for (auto rewrite :
             llvm::reverse(facet_type_info.rewrite_constraints)) {
          if (some_where) {
            step_stack.PushString(" and");
          }
          step_stack.PushInstId(
              sem_ir.constant_values().GetInstId(rewrite.rhs_const_id));
          step_stack.PushString(" = ");
          step_stack.PushInstId(
              sem_ir.constant_values().GetInstId(rewrite.lhs_const_id));
          step_stack.PushString(" ");
          some_where = true;
        }
        // TODO: Other restrictions from facet_type_info.
        if (some_where) {
          step_stack.PushString(" where");
        }

        // Output interface requirements.
        if (facet_type_info.impls_constraints.empty()) {
          step_stack.PushString("type");
          break;
        }
        for (auto index : llvm::reverse(
                 llvm::seq(facet_type_info.impls_constraints.size()))) {
          const auto& impls = facet_type_info.impls_constraints[index];
          const auto& interface_info =
              sem_ir.interfaces().Get(impls.interface_id);
          step_stack.PushSpecificId(interface_info, impls.specific_id);
          step_stack.PushNameId(interface_info.name_id);
          if (index > 0) {
            step_stack.PushString(" & ");
          }
        }
        break;
      }
      case CARBON_KIND(FacetValue inst): {
        // No need to output the witness.
        step_stack.PushTypeId(inst.type_id);
        step_stack.PushString(" as ");
        step_stack.PushInstId(inst.type_inst_id);
        break;
      }
      case CARBON_KIND(FloatType inst): {
        // TODO: Is this okay?
        if (auto width_value =
                sem_ir.insts().TryGetAs<IntValue>(inst.bit_width_id)) {
          out << "f";
          sem_ir.ints().Get(width_value->int_id).print(out, /*isSigned=*/false);
        } else {
          out << "Core.Float(";
          step_stack.PushString(")");
          step_stack.PushInstId(inst.bit_width_id);
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
      case CARBON_KIND(ImportRefUnloaded inst): {
        if (inst.entity_name_id.is_valid()) {
          auto name_id = sem_ir.entity_names().Get(inst.entity_name_id).name_id;
          out << sem_ir.names().GetFormatted(name_id);
        } else {
          out << "<import ref unloaded invalid entity name>";
        }
        break;
      }
      case CARBON_KIND(IntType inst): {
        if (auto width_value =
                sem_ir.insts().TryGetAs<IntValue>(inst.bit_width_id)) {
          out << (inst.int_kind.is_signed() ? "i" : "u");
          sem_ir.ints().Get(width_value->int_id).print(out, /*isSigned=*/false);
        } else {
          out << (inst.int_kind.is_signed() ? "Core.Int(" : "Core.UInt(");
          step_stack.PushString(")");
          step_stack.PushInstId(inst.bit_width_id);
        }
        break;
      }
      case CARBON_KIND(InterfaceWitnessAccess inst): {
        auto witness_inst_id =
            sem_ir.constant_values().GetConstantInstId(inst.witness_id);
        auto witness =
            sem_ir.insts().GetAs<FacetAccessWitness>(witness_inst_id);
        auto witness_type_id =
            sem_ir.insts().Get(witness.facet_value_inst_id).type_id();
        auto facet_type = sem_ir.types().GetAs<FacetType>(witness_type_id);
        step_stack.PushString(")");
        // TODO: Support != 1 interface better.
        if (auto impls_constraint = sem_ir.facet_types()
                                        .Get(facet_type.facet_type_id)
                                        .TryAsSingleInterface()) {
          const auto& interface =
              sem_ir.interfaces().Get(impls_constraint->interface_id);
          auto entities =
              sem_ir.inst_blocks().Get(interface.associated_entities_id);
          size_t index = inst.index.index;
          CARBON_CHECK(index < entities.size(), "Access out of bounds.");
          auto entity_inst_id = entities[index];
          if (auto associated_const =
                  sem_ir.insts().TryGetAs<AssociatedConstantDecl>(
                      entity_inst_id)) {
            step_stack.PushNameId(associated_const->name_id);
          } else if (auto function_decl = sem_ir.insts().TryGetAs<FunctionDecl>(
                         entity_inst_id)) {
            const auto& function =
                sem_ir.functions().Get(function_decl->function_id);
            step_stack.PushNameId(function.name_id);
          } else {
            step_stack.PushInstId(entity_inst_id);
          }
          step_stack.PushString(".");
          step_stack.PushNameId(interface.name_id);
          step_stack.PushString(".(");
        } else {
          step_stack.PushTypeId(witness_type_id);
          step_stack.PushString(".(TODO: ");
        }

        bool period_self = false;
        if (auto sym_name = sem_ir.insts().TryGetAs<BindSymbolicName>(
                witness.facet_value_inst_id)) {
          auto name_id =
              sem_ir.entity_names().Get(sym_name->entity_name_id).name_id;
          period_self = (name_id == SemIR::NameId::PeriodSelf);
        }
        if (!period_self) {
          step_stack.PushInstId(witness.facet_value_inst_id);
        }
        break;
      }
      case CARBON_KIND(IntValue inst): {
        sem_ir.ints().Get(inst.int_id).print(out, /*isSigned=*/true);
        break;
      }
      case CARBON_KIND(NameRef inst): {
        out << sem_ir.names().GetFormatted(inst.name_id);
        break;
      }
      case CARBON_KIND(PointerType inst): {
        step_stack.PushString("*");
        step_stack.PushTypeId(inst.pointee_id);
        break;
      }
      case CARBON_KIND(StructType inst): {
        auto fields = sem_ir.struct_type_fields().Get(inst.fields_id);
        if (fields.empty()) {
          out << "{}";
          break;
        }
        out << "{";
        step_stack.PushString("}");
        for (auto index : llvm::reverse(llvm::seq(fields.size()))) {
          const auto& field = fields[index];
          step_stack.PushTypeId(field.type_id);
          step_stack.PushString(": ");
          step_stack.PushNameId(field.name_id);
          step_stack.PushString(".");
          if (index > 0) {
            step_stack.PushString(", ");
          }
        }
        break;
      }
      case CARBON_KIND(TupleType inst): {
        auto refs = sem_ir.type_blocks().Get(inst.elements_id);
        if (refs.empty()) {
          out << "()";
          break;
        }
        out << "(";
        step_stack.PushString(")");
        // A tuple of one element has a comma to disambiguate from an
        // expression.
        if (refs.size() == 1) {
          step_stack.PushString(",");
        }
        for (auto i : llvm::reverse(llvm::seq(refs.size()))) {
          step_stack.PushTypeId(refs[i]);
          if (i > 0) {
            step_stack.PushString(", ");
          }
        }
        break;
      }
      case CARBON_KIND(UnboundElementType inst): {
        out << "<unbound element of class ";
        step_stack.PushString(">");
        step_stack.PushTypeId(inst.class_type_id);
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
      case InitializeFrom::Kind:
      case InterfaceDecl::Kind:
      case InterfaceWitness::Kind:
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
      case WhereExpr::Kind:
        // We don't know how to print this instruction, but it might have a
        // constant value that we can print.
        auto const_inst_id =
            sem_ir.constant_values().GetConstantInstId(step.inst_id);
        if (const_inst_id.is_valid() && const_inst_id != step.inst_id) {
          step_stack.PushInstId(const_inst_id);
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
