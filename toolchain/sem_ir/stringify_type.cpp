// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/stringify_type.h"

#include "common/raw_string_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/singleton_insts.h"
#include "toolchain/sem_ir/struct_type_field.h"
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

// Provides `StringifyTypeInst` overloads for each instruction.
class Stringifier {
 public:
  explicit Stringifier(const SemIR::File* sem_ir, StepStack* step_stack,
                       llvm::raw_ostream* out)
      : sem_ir_(sem_ir), step_stack_(step_stack), out_(out) {}

  // By default try to print a constant, but otherwise may fail to
  // stringify.
  auto StringifyTypeInstDefault(SemIR::InstId inst_id, Inst inst) -> void {
    // We don't know how to print this instruction, but it might have a
    // constant value that we can print.
    auto const_inst_id = sem_ir_->constant_values().GetConstantInstId(inst_id);
    if (const_inst_id.has_value() && const_inst_id != inst_id) {
      step_stack_->PushInstId(const_inst_id);
      return;
    }

    // We don't need to handle stringification for instructions that don't
    // show up in errors, but make it clear what's going on so that it's
    // clearer when stringification is needed.
    *out_ << "<cannot stringify " << inst_id << ": " << inst << ">";
  }

  template <typename InstT>
  auto StringifyTypeInst(SemIR::InstId inst_id, InstT inst) -> void {
    // This doesn't use requires so that more specific overloads are chosen when
    // provided.
    static_assert(InstT::Kind.is_type() != InstIsType::Always ||
                      std::same_as<InstT, WhereExpr>,
                  "Types should have a dedicated overload");
    StringifyTypeInstDefault(inst_id, inst);
  }

  // Singleton instructions use their IR name as a label.
  template <typename InstT>
    requires(IsSingletonInstKind(InstT::Kind))
  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, InstT /*inst*/) -> void {
    *out_ << InstT::Kind.ir_name();
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, ArrayType inst) -> void {
    *out_ << "[";
    step_stack_->PushString("]");
    step_stack_->PushInstId(inst.bound_id);
    step_stack_->PushString("; ");
    step_stack_->PushTypeId(inst.element_type_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, AssociatedConstantDecl inst)
      -> void {
    const auto& assoc_const =
        sem_ir_->associated_constants().Get(inst.assoc_const_id);
    step_stack_->PushQualifiedName(assoc_const.parent_scope_id,
                                   assoc_const.name_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, AssociatedEntityType inst)
      -> void {
    *out_ << "<associated entity in ";
    step_stack_->PushString(">");
    step_stack_->PushTypeId(inst.interface_type_id);
  }

  template <typename InstT>
    requires(std::same_as<InstT, BindAlias> ||
             std::same_as<InstT, BindSymbolicName> ||
             std::same_as<InstT, ExportDecl>)
  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, InstT inst) -> void {
    auto name_id = inst.entity_name_id;
    step_stack_->PushEntityName(name_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, ClassType inst) -> void {
    const auto& class_info = sem_ir_->classes().Get(inst.class_id);
    if (auto literal_info = NumericTypeLiteralInfo::ForType(*sem_ir_, inst);
        literal_info.is_valid()) {
      literal_info.PrintLiteral(*sem_ir_, *out_);
      return;
    }
    step_stack_->PushEntityName(class_info, inst.specific_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, ConstType inst) -> void {
    *out_ << "const ";

    // Add parentheses if required.
    auto inner_type_inst_id = sem_ir_->types().GetInstId(inst.inner_id);
    if (GetTypePrecedence(sem_ir_->insts().Get(inner_type_inst_id).kind()) <
        GetTypePrecedence(SemIR::ConstType::Kind)) {
      *out_ << "(";
      step_stack_->PushString(")");
    }

    step_stack_->PushInstId(inner_type_inst_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, FacetAccessType inst)
      -> void {
    // Given `T:! I`, print `T as type` as simply `T`.
    step_stack_->PushInstId(inst.facet_value_inst_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, FacetAccessWitness inst)
      -> void {
    *out_ << "<witness for ";
    step_stack_->PushString(">");
    step_stack_->PushElementIndex(inst.index);
    step_stack_->PushString(", interface ");
    step_stack_->PushInstId(inst.facet_value_inst_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, FacetType inst) -> void {
    const FacetTypeInfo& facet_type_info =
        sem_ir_->facet_types().Get(inst.facet_type_id);
    // Output `where` restrictions.
    bool some_where = false;
    if (facet_type_info.other_requirements) {
      step_stack_->PushString("...");
      some_where = true;
    }
    for (auto rewrite : llvm::reverse(facet_type_info.rewrite_constraints)) {
      if (some_where) {
        step_stack_->PushString(" and");
      }
      step_stack_->PushInstId(
          sem_ir_->constant_values().GetInstId(rewrite.rhs_const_id));
      step_stack_->PushString(" = ");
      step_stack_->PushInstId(
          sem_ir_->constant_values().GetInstId(rewrite.lhs_const_id));
      step_stack_->PushString(" ");
      some_where = true;
    }
    // TODO: Other restrictions from facet_type_info.
    if (some_where) {
      step_stack_->PushString(" where");
    }

    // Output interface requirements.
    if (facet_type_info.impls_constraints.empty()) {
      step_stack_->PushString("type");
      return;
    }
    for (auto index :
         llvm::reverse(llvm::seq(facet_type_info.impls_constraints.size()))) {
      const auto& impls = facet_type_info.impls_constraints[index];
      const auto& interface_info =
          sem_ir_->interfaces().Get(impls.interface_id);
      step_stack_->PushEntityName(interface_info, impls.specific_id);
      if (index > 0) {
        step_stack_->PushString(" & ");
      }
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, FacetValue inst) -> void {
    // No need to output the witness.
    step_stack_->PushTypeId(inst.type_id);
    step_stack_->PushString(" as ");
    step_stack_->PushInstId(inst.type_inst_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, FloatType inst) -> void {
    // TODO: Is this okay?
    if (auto width_value =
            sem_ir_->insts().TryGetAs<IntValue>(inst.bit_width_id)) {
      *out_ << "f";
      sem_ir_->ints().Get(width_value->int_id).print(*out_, /*isSigned=*/false);
    } else {
      *out_ << "Core.Float(";
      step_stack_->PushString(")");
      step_stack_->PushInstId(inst.bit_width_id);
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, FunctionType inst) -> void {
    const auto& fn = sem_ir_->functions().Get(inst.function_id);
    *out_ << "<type of ";
    step_stack_->PushString(">");
    step_stack_->PushQualifiedName(fn.parent_scope_id, fn.name_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/,
                         FunctionTypeWithSelfType inst) -> void {
    *out_ << "<type of ";
    step_stack_->PushString(">");
    step_stack_->PushInstId(inst.self_id);
    step_stack_->PushString(" in ");
    if (auto fn_inst = sem_ir_->insts().TryGetAs<FunctionType>(
            inst.interface_function_type_id)) {
      const auto& fn = sem_ir_->functions().Get(fn_inst->function_id);
      step_stack_->PushQualifiedName(fn.parent_scope_id, fn.name_id);
    } else {
      step_stack_->PushInstId(inst.interface_function_type_id);
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, GenericClassType inst)
      -> void {
    const auto& class_info = sem_ir_->classes().Get(inst.class_id);
    *out_ << "<type of ";
    step_stack_->PushString(">");
    step_stack_->PushQualifiedName(class_info.parent_scope_id,
                                   class_info.name_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, GenericInterfaceType inst)
      -> void {
    const auto& interface = sem_ir_->interfaces().Get(inst.interface_id);
    *out_ << "<type of ";
    step_stack_->PushString(">");
    step_stack_->PushQualifiedName(interface.parent_scope_id,
                                   interface.name_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, ImplWitnessAccess inst)
      -> void {
    auto witness_inst_id =
        sem_ir_->constant_values().GetConstantInstId(inst.witness_id);
    auto witness = sem_ir_->insts().GetAs<FacetAccessWitness>(witness_inst_id);
    auto witness_type_id =
        sem_ir_->insts().Get(witness.facet_value_inst_id).type_id();
    auto facet_type = sem_ir_->types().GetAs<FacetType>(witness_type_id);
    step_stack_->PushString(")");
    // TODO: Support != 1 interface better.
    if (auto impls_constraint = sem_ir_->facet_types()
                                    .Get(facet_type.facet_type_id)
                                    .TryAsSingleInterface()) {
      const auto& interface =
          sem_ir_->interfaces().Get(impls_constraint->interface_id);
      auto entities =
          sem_ir_->inst_blocks().Get(interface.associated_entities_id);
      size_t index = inst.index.index;
      CARBON_CHECK(index < entities.size(), "Access out of bounds.");
      auto entity_inst_id = entities[index];
      if (auto associated_const =
              sem_ir_->insts().TryGetAs<AssociatedConstantDecl>(
                  entity_inst_id)) {
        step_stack_->PushNameId(sem_ir_->associated_constants()
                                    .Get(associated_const->assoc_const_id)
                                    .name_id);
      } else if (auto function_decl =
                     sem_ir_->insts().TryGetAs<FunctionDecl>(entity_inst_id)) {
        const auto& function =
            sem_ir_->functions().Get(function_decl->function_id);
        step_stack_->PushNameId(function.name_id);
      } else {
        step_stack_->PushInstId(entity_inst_id);
      }
      step_stack_->PushString(".");
      step_stack_->PushEntityName(interface, impls_constraint->specific_id);
      step_stack_->PushString(".(");
    } else {
      step_stack_->PushTypeId(witness_type_id);
      step_stack_->PushString(".(TODO: ");
    }

    bool period_self = false;
    if (auto sym_name = sem_ir_->insts().TryGetAs<BindSymbolicName>(
            witness.facet_value_inst_id)) {
      auto name_id =
          sem_ir_->entity_names().Get(sym_name->entity_name_id).name_id;
      period_self = (name_id == SemIR::NameId::PeriodSelf);
    }
    if (!period_self) {
      step_stack_->PushInstId(witness.facet_value_inst_id);
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, ImportRefUnloaded inst)
      -> void {
    if (inst.entity_name_id.has_value()) {
      step_stack_->PushEntityName(inst.entity_name_id);
    } else {
      *out_ << "<import ref unloaded invalid entity name>";
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, IntType inst) -> void {
    *out_ << "<builtin ";
    step_stack_->PushString(">");
    if (auto width_value =
            sem_ir_->insts().TryGetAs<IntValue>(inst.bit_width_id)) {
      *out_ << (inst.int_kind.is_signed() ? "i" : "u");
      sem_ir_->ints().Get(width_value->int_id).print(*out_, /*isSigned=*/false);
    } else {
      *out_ << (inst.int_kind.is_signed() ? "Int(" : "UInt(");
      step_stack_->PushString(")");
      step_stack_->PushInstId(inst.bit_width_id);
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, IntValue inst) -> void {
    sem_ir_->ints().Get(inst.int_id).print(*out_, /*isSigned=*/true);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, NameRef inst) -> void {
    *out_ << sem_ir_->names().GetFormatted(inst.name_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, Namespace inst) -> void {
    const auto& name_scope = sem_ir_->name_scopes().Get(inst.name_scope_id);
    step_stack_->PushQualifiedName(name_scope.parent_scope_id(),
                                   name_scope.name_id());
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, PointerType inst) -> void {
    step_stack_->PushString("*");
    step_stack_->PushTypeId(inst.pointee_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, SpecificFunction inst)
      -> void {
    auto callee = SemIR::GetCalleeFunction(*sem_ir_, inst.callee_id);
    if (callee.function_id.has_value()) {
      step_stack_->PushEntityName(sem_ir_->functions().Get(callee.function_id),
                                  inst.specific_id);
    } else {
      step_stack_->PushString("<invalid specific function>");
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, SpecificImplFunction inst)
      -> void {
    auto callee = SemIR::GetCalleeFunction(*sem_ir_, inst.callee_id);
    if (callee.function_id.has_value()) {
      // TODO: The specific_id here is for the interface member, but the
      // entity we're passing is the impl member. This might result in
      // strange output once we render specific arguments properly.
      step_stack_->PushEntityName(sem_ir_->functions().Get(callee.function_id),
                                  inst.specific_id);
    } else {
      step_stack_->PushString("<invalid specific function>");
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, StructType inst) -> void {
    auto fields = sem_ir_->struct_type_fields().Get(inst.fields_id);
    if (fields.empty()) {
      *out_ << "{}";
      return;
    }
    *out_ << "{";
    step_stack_->PushString("}");
    for (auto index : llvm::reverse(llvm::seq(fields.size()))) {
      const auto& field = fields[index];
      step_stack_->PushTypeId(field.type_id);
      step_stack_->PushString(": ");
      step_stack_->PushNameId(field.name_id);
      step_stack_->PushString(".");
      if (index > 0) {
        step_stack_->PushString(", ");
      }
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, StructValue inst) -> void {
    auto field_values = sem_ir_->inst_blocks().Get(inst.elements_id);
    if (field_values.empty()) {
      *out_ << "{}";
      return;
    }
    auto struct_type = sem_ir_->types().GetAs<StructType>(
        sem_ir_->types().GetObjectRepr(inst.type_id));
    auto fields = sem_ir_->struct_type_fields().Get(struct_type.fields_id);
    if (fields.size() != field_values.size()) {
      *out_ << "{<struct value type length mismatch>}";
      return;
    }
    *out_ << "{";
    step_stack_->PushString("}");
    for (auto index : llvm::reverse(llvm::seq(fields.size()))) {
      SemIR::InstId value_inst_id = field_values[index];
      step_stack_->PushInstId(value_inst_id);
      step_stack_->PushString(" = ");
      step_stack_->PushNameId(fields[index].name_id);
      step_stack_->PushString(".");
      if (index > 0) {
        step_stack_->PushString(", ");
      }
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, TupleType inst) -> void {
    auto refs = sem_ir_->type_blocks().Get(inst.elements_id);
    if (refs.empty()) {
      *out_ << "()";
      return;
    }
    *out_ << "(";
    step_stack_->PushString(")");
    // A tuple of one element has a comma to disambiguate from an
    // expression.
    if (refs.size() == 1) {
      step_stack_->PushString(",");
    }
    for (auto i : llvm::reverse(llvm::seq(refs.size()))) {
      step_stack_->PushTypeId(refs[i]);
      if (i > 0) {
        step_stack_->PushString(", ");
      }
    }
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, TupleValue inst) -> void {
    auto refs = sem_ir_->inst_blocks().Get(inst.elements_id);
    if (refs.empty()) {
      *out_ << "()";
      return;
    }
    *out_ << "(";
    step_stack_->PushString(")");
    // A tuple of one element has a comma to disambiguate from an
    // expression.
    if (refs.size() == 1) {
      step_stack_->PushString(",");
    }
    for (auto i : llvm::reverse(llvm::seq(refs.size()))) {
      step_stack_->PushInstId(refs[i]);
      if (i > 0) {
        step_stack_->PushString(", ");
      }
    }
  }

  auto StringifyTypeInst(SemIR::InstId inst_id, TypeOfInst /*inst*/) -> void {
    // Print the constant value if we've already computed the inst.
    auto const_inst_id = sem_ir_->constant_values().GetConstantInstId(inst_id);
    if (const_inst_id.has_value() && const_inst_id != inst_id) {
      step_stack_->PushInstId(const_inst_id);
      return;
    }
    *out_ << "<dependent type>";
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, UnboundElementType inst)
      -> void {
    *out_ << "<unbound element of class ";
    step_stack_->PushString(">");
    step_stack_->PushTypeId(inst.class_type_id);
  }

  auto StringifyTypeInst(SemIR::InstId /*inst_id*/, VtablePtr /*inst*/)
      -> void {
    *out_ << "<vtable ptr>";
  }

 private:
  const SemIR::File* sem_ir_;
  StepStack* step_stack_;
  llvm::raw_ostream* out_;
};

}  // namespace

auto StringifyTypeExpr(const SemIR::File& sem_ir, InstId outer_inst_id)
    -> std::string {
  RawStringOstream out;

  // Note: Since this is a stack, work is resolved in the reverse order from the
  // order pushed.
  StepStack step_stack(&sem_ir, outer_inst_id);
  Stringifier stringifier(&sem_ir, &step_stack, &out);

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
#define CARBON_SEM_IR_INST_KIND(InstT)                       \
  case CARBON_KIND(InstT typed_inst): {                      \
    stringifier.StringifyTypeInst(step.inst_id, typed_inst); \
    break;                                                   \
  }
#include "toolchain/sem_ir/inst_kind.def"
    }
  }

  return out.TakeStr();
}

}  // namespace Carbon::SemIR
