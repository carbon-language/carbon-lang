// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/stringify_type.h"

#include "common/raw_string_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/typed_insts.h"

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
// Contains the stack of steps for `StringifyTypeExpr`.
class StepStack {
 public:
  // An individual step in the stack, which stringifies some component of a type
  // name.
  struct Step {
    // The kind of step to perform.
    enum Kind : uint8_t {
      Inst,
      FixedString,
      Name,
      Index,
    };

    Kind kind;

    union {
      // The instruction to print, when kind is Inst.
      InstId inst_id;
      // The fixed string to print, when kind is FixedString.
      const char* fixed_string;
      // The name to print, when kind is Name.
      NameId name_id;
      // The element index to print, when kind is Index.
      ElementIndex element_index;
    };
  };

  // Starts a new stack, which always contains the first instruction to
  // stringify.
  explicit StepStack(const SemIR::File* file, InstId outer_inst_id)
      : sem_ir_(file) {
    PushInstId(outer_inst_id);
  }

  // These push basic entries onto the stack.
  auto PushInstId(InstId inst_id) -> void {
    steps_.push_back({.kind = Step::Inst, .inst_id = inst_id});
  }
  auto PushString(const char* string) -> void {
    steps_.push_back({.kind = Step::FixedString, .fixed_string = string});
  }
  auto PushNameId(NameId name_id) -> void {
    steps_.push_back({.kind = Step::Name, .name_id = name_id});
  }
  auto PushElementIndex(ElementIndex element_index) -> void {
    steps_.push_back({.kind = Step::Index, .element_index = element_index});
  }

  // Pushes all components of a qualified name (`A.B.C`) onto the stack.
  auto PushQualifiedName(NameScopeId name_scope_id, NameId name_id) -> void {
    PushNameId(name_id);
    while (name_scope_id.has_value() && name_scope_id != NameScopeId::Package) {
      const auto& name_scope = sem_ir_->name_scopes().Get(name_scope_id);
      // TODO: Decide how to print unnamed scopes.
      if (name_scope.name_id().has_value()) {
        PushString(".");
        // TODO: For a generic scope, pass a SpecificId to this function and
        // include the relevant arguments.
        PushNameId(name_scope.name_id());
      }
      name_scope_id = name_scope.parent_scope_id();
    }
  }

  // Pushes a specific's entity name onto the stack, such as `A.B(T)`.
  auto PushEntityName(const EntityWithParamsBase& entity,
                      SpecificId specific_id) -> void {
    PushSpecificId(entity, specific_id);
    PushQualifiedName(entity.parent_scope_id, entity.name_id);
  }

  // Pushes a entity name onto the stack, such as `A.B`.
  auto PushEntityName(EntityNameId entity_name_id) -> void {
    const auto& entity_name = sem_ir_->entity_names().Get(entity_name_id);
    PushQualifiedName(entity_name.parent_scope_id, entity_name.name_id);
  }

  // Pushes an instruction by its TypeId.
  auto PushTypeId(TypeId type_id) -> void {
    PushInstId(sem_ir_->types().GetInstId(type_id));
  }

  auto empty() const -> bool { return steps_.empty(); }
  auto Pop() -> Step { return steps_.pop_back_val(); }

 private:
  // Handles the generic portion of a specific entity name, such as `(T)` in
  // `A.B(T)`.
  auto PushSpecificId(const EntityWithParamsBase& entity,
                      SpecificId specific_id) -> void {
    if (!entity.param_patterns_id.has_value()) {
      return;
    }
    int num_params =
        sem_ir_->inst_blocks().Get(entity.param_patterns_id).size();
    if (!num_params) {
      PushString("()");
      return;
    }
    if (!specific_id.has_value()) {
      // The name of the generic was used within the generic itself.
      // TODO: Should we print the names of the generic parameters in this
      // case?
      return;
    }
    const auto& specific = sem_ir_->specifics().Get(specific_id);
    auto args =
        sem_ir_->inst_blocks().Get(specific.args_id).take_back(num_params);
    bool last = true;
    for (auto arg : llvm::reverse(args)) {
      PushString(last ? ")" : ", ");
      PushInstId(arg);
      last = false;
    }
    PushString("(");
  }

  const SemIR::File* sem_ir_;
  // Remaining steps to take.
  llvm::SmallVector<Step> steps_;
};
}  // namespace

auto StringifyTypeExpr(const SemIR::File& sem_ir, InstId outer_inst_id)
    -> std::string {
  RawStringOstream out;

  // Note: Since this is a stack, work is resolved in the reverse order from the
  // order pushed.
  StepStack step_stack(&sem_ir, outer_inst_id);

  while (!step_stack.empty()) {
    auto step = step_stack.Pop();

    switch (step.kind) {
      case StepStack::Step::FixedString:
        out << step.fixed_string;
        continue;
      case StepStack::Step::Index:
        out << step.element_index.index;
        continue;
      case StepStack::Step::Name:
        out << sem_ir.names().GetFormatted(step.name_id);
        continue;
      case StepStack::Step::Inst:
        if (!step.inst_id.has_value()) {
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
      case SemIR::Vtable::Kind:
      case SemIR::WitnessType::Kind: {
        // Singleton instructions use their IR name as a label.
        out << untyped_inst.kind().ir_name();
        break;
      }
      case CARBON_KIND(ArrayType inst): {
        out << "[";
        step_stack.PushString("]");
        step_stack.PushInstId(inst.bound_id);
        step_stack.PushString("; ");
        step_stack.PushTypeId(inst.element_type_id);
        break;
      }
      case CARBON_KIND(AssociatedConstantDecl inst): {
        const auto& assoc_const =
            sem_ir.associated_constants().Get(inst.assoc_const_id);
        step_stack.PushQualifiedName(assoc_const.parent_scope_id,
                                     assoc_const.name_id);
        break;
      }
      case CARBON_KIND(AssociatedEntityType inst): {
        out << "<associated entity in ";
        step_stack.PushString(">");
        step_stack.PushTypeId(inst.interface_type_id);
        break;
      }
      case BindAlias::Kind:
      case BindSymbolicName::Kind:
      case ExportDecl::Kind: {
        auto name_id =
            untyped_inst.As<AnyBindNameOrExportDecl>().entity_name_id;
        step_stack.PushEntityName(name_id);
        break;
      }
      case CARBON_KIND(ClassType inst): {
        const auto& class_info = sem_ir.classes().Get(inst.class_id);
        if (auto literal_info = NumericTypeLiteralInfo::ForType(sem_ir, inst);
            literal_info.is_valid()) {
          literal_info.PrintLiteral(sem_ir, out);
          break;
        }
        step_stack.PushEntityName(class_info, inst.specific_id);
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
        step_stack.PushElementIndex(inst.index);
        step_stack.PushString(", interface ");
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
          step_stack.PushEntityName(interface_info, impls.specific_id);
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
        const auto& fn = sem_ir.functions().Get(inst.function_id);
        out << "<type of ";
        step_stack.PushString(">");
        step_stack.PushQualifiedName(fn.parent_scope_id, fn.name_id);
        break;
      }
      case CARBON_KIND(FunctionTypeWithSelfType inst): {
        out << "<type of ";
        step_stack.PushString(">");
        step_stack.PushInstId(inst.self_id);
        step_stack.PushString(" in ");
        if (auto fn_inst = sem_ir.insts().TryGetAs<FunctionType>(
                inst.interface_function_type_id)) {
          const auto& fn = sem_ir.functions().Get(fn_inst->function_id);
          step_stack.PushQualifiedName(fn.parent_scope_id, fn.name_id);
        } else {
          step_stack.PushInstId(inst.interface_function_type_id);
        }
        break;
      }
      case CARBON_KIND(GenericClassType inst): {
        const auto& class_info = sem_ir.classes().Get(inst.class_id);
        out << "<type of ";
        step_stack.PushString(">");
        step_stack.PushQualifiedName(class_info.parent_scope_id,
                                     class_info.name_id);
        break;
      }
      case CARBON_KIND(GenericInterfaceType inst): {
        const auto& interface = sem_ir.interfaces().Get(inst.interface_id);
        out << "<type of ";
        step_stack.PushString(">");
        step_stack.PushQualifiedName(interface.parent_scope_id,
                                     interface.name_id);
        break;
      }
      case CARBON_KIND(ImplWitnessAccess inst): {
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
            step_stack.PushNameId(sem_ir.associated_constants()
                                      .Get(associated_const->assoc_const_id)
                                      .name_id);
          } else if (auto function_decl = sem_ir.insts().TryGetAs<FunctionDecl>(
                         entity_inst_id)) {
            const auto& function =
                sem_ir.functions().Get(function_decl->function_id);
            step_stack.PushNameId(function.name_id);
          } else {
            step_stack.PushInstId(entity_inst_id);
          }
          step_stack.PushString(".");
          step_stack.PushEntityName(interface, impls_constraint->specific_id);
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
      case CARBON_KIND(ImportRefUnloaded inst): {
        if (inst.entity_name_id.has_value()) {
          step_stack.PushEntityName(inst.entity_name_id);
        } else {
          out << "<import ref unloaded invalid entity name>";
        }
        break;
      }
      case CARBON_KIND(IntType inst): {
        out << "<builtin ";
        step_stack.PushString(">");
        if (auto width_value =
                sem_ir.insts().TryGetAs<IntValue>(inst.bit_width_id)) {
          out << (inst.int_kind.is_signed() ? "i" : "u");
          sem_ir.ints().Get(width_value->int_id).print(out, /*isSigned=*/false);
        } else {
          out << (inst.int_kind.is_signed() ? "Int(" : "UInt(");
          step_stack.PushString(")");
          step_stack.PushInstId(inst.bit_width_id);
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
      case CARBON_KIND(Namespace inst): {
        const auto& name_scope = sem_ir.name_scopes().Get(inst.name_scope_id);
        step_stack.PushQualifiedName(name_scope.parent_scope_id(),
                                     name_scope.name_id());
        break;
      }
      case CARBON_KIND(PointerType inst): {
        step_stack.PushString("*");
        step_stack.PushTypeId(inst.pointee_id);
        break;
      }
      case CARBON_KIND(SpecificFunction inst): {
        auto callee = SemIR::GetCalleeFunction(sem_ir, inst.callee_id);
        if (callee.function_id.has_value()) {
          step_stack.PushEntityName(sem_ir.functions().Get(callee.function_id),
                                    inst.specific_id);
        } else {
          step_stack.PushString("<invalid specific function>");
        }
        break;
      }
      case CARBON_KIND(SpecificImplFunction inst): {
        auto callee = SemIR::GetCalleeFunction(sem_ir, inst.callee_id);
        if (callee.function_id.has_value()) {
          // TODO: The specific_id here is for the interface member, but the
          // entity we're passing is the impl member. This might result in
          // strange output once we render specific arguments properly.
          step_stack.PushEntityName(sem_ir.functions().Get(callee.function_id),
                                    inst.specific_id);
        } else {
          step_stack.PushString("<invalid specific function>");
        }
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
      case CARBON_KIND(StructValue inst): {
        auto field_values = sem_ir.inst_blocks().Get(inst.elements_id);
        if (field_values.empty()) {
          out << "{}";
          break;
        }
        auto struct_type = sem_ir.types().GetAs<StructType>(
            sem_ir.types().GetObjectRepr(inst.type_id));
        auto fields = sem_ir.struct_type_fields().Get(struct_type.fields_id);
        if (fields.size() != field_values.size()) {
          out << "{<struct value type length mismatch>}";
          break;
        }
        out << "{";
        step_stack.PushString("}");
        for (auto index : llvm::reverse(llvm::seq(fields.size()))) {
          SemIR::InstId value_inst_id = field_values[index];
          step_stack.PushInstId(value_inst_id);
          step_stack.PushString(" = ");
          step_stack.PushNameId(fields[index].name_id);
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
      case CARBON_KIND(TupleValue inst): {
        auto refs = sem_ir.inst_blocks().Get(inst.elements_id);
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
          step_stack.PushInstId(refs[i]);
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
      case VtablePtr::Kind: {
        out << "<vtable ptr>";
        break;
      }
      case AdaptDecl::Kind:
      case AddrOf::Kind:
      case AddrPattern::Kind:
      case ArrayIndex::Kind:
      case ArrayInit::Kind:
      case AsCompatible::Kind:
      case Assign::Kind:
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
      case ImplWitness::Kind:
      case ImportCppDecl::Kind:
      case ImportDecl::Kind:
      case ImportRefLoaded::Kind:
      case InitializeFrom::Kind:
      case InterfaceDecl::Kind:
      case NameBindingDecl::Kind:
      case OutParam::Kind:
      case OutParamPattern::Kind:
      case RefParam::Kind:
      case RefParamPattern::Kind:
      case RequireCompleteType::Kind:
      case RequirementEquivalent::Kind:
      case RequirementImpls::Kind:
      case RequirementRewrite::Kind:
      case Return::Kind:
      case ReturnExpr::Kind:
      case ReturnSlot::Kind:
      case ReturnSlotPattern::Kind:
      case SpecificConstant::Kind:
      case SpliceBlock::Kind:
      case StringLiteral::Kind:
      case StructAccess::Kind:
      case StructInit::Kind:
      case StructLiteral::Kind:
      case SymbolicBindingPattern::Kind:
      case Temporary::Kind:
      case TemporaryStorage::Kind:
      case TupleAccess::Kind:
      case TupleInit::Kind:
      case TupleLiteral::Kind:
      case TuplePattern::Kind:
      case UnaryOperatorNot::Kind:
      case ValueAsRef::Kind:
      case ValueOfInitializer::Kind:
      case ValueParam::Kind:
      case ValueParamPattern::Kind:
      case VarPattern::Kind:
      case VarStorage::Kind:
      case WhereExpr::Kind:
        // We don't know how to print this instruction, but it might have a
        // constant value that we can print.
        auto const_inst_id =
            sem_ir.constant_values().GetConstantInstId(step.inst_id);
        if (const_inst_id.has_value() && const_inst_id != step.inst_id) {
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

  return out.TakeStr();
}

}  // namespace Carbon::SemIR
