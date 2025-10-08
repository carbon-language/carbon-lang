// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/stringify.h"

#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "common/concepts.h"
#include "common/raw_string_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/singleton_insts.h"
#include "toolchain/sem_ir/struct_type_field.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Map an instruction kind representing an expression into an integer describing
// the precedence of that expression's syntax. Higher numbers correspond to
// higher precedence.
static auto GetPrecedence(InstKind kind) -> int {
  if (kind == ConstType::Kind) {
    return -1;
  }
  if (kind == PointerType::Kind) {
    return -2;
  }
  // TODO: Handle other kinds of expressions with precedence.
  return 0;
}

namespace {

// Contains the stack of steps for `Stringify`.
//
// Note that when pushing items onto the stack, they're printed in the reverse
// order of when they were pushed. All reference lifetimes must match the
// lifetime of `Stringify`.
class StepStack {
 public:
  // An individual step in the stack, which stringifies some component of a type
  // name.
  using Step = std::variant<InstId, llvm::StringRef, NameId, ElementIndex>;

  // Support `Push` for a qualified name. e.g., `A.B.C`.
  using QualifiedNameItem = std::pair<NameScopeId, NameId>;

  // Support `Push` for a qualified entity name. e.g., `A.B.C`.
  using EntityNameItem = std::pair<const EntityWithParamsBase&, SpecificId>;

  // The full set of things which can be pushed, including all members of
  // `Step`.
  using PushItem =
      std::variant<InstId, llvm::StringRef, NameId, ElementIndex,
                   QualifiedNameItem, EntityNameItem, EntityNameId, TypeId,
                   SpecificInterface, llvm::ListSeparator*>;

  // Starts a new stack, which always contains the first instruction to
  // stringify.
  explicit StepStack(const File* file) : sem_ir_(file) {}

  // These push basic entries onto the stack.
  auto PushInstId(InstId inst_id) -> void { steps_.push_back(inst_id); }
  auto PushString(llvm::StringRef string) -> void { steps_.push_back(string); }
  auto PushNameId(NameId name_id) -> void { steps_.push_back(name_id); }
  auto PushElementIndex(ElementIndex element_index) -> void {
    steps_.push_back(element_index);
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
  auto PushEntityNameId(EntityNameId entity_name_id) -> void {
    const auto& entity_name = sem_ir_->entity_names().Get(entity_name_id);
    PushQualifiedName(entity_name.parent_scope_id, entity_name.name_id);
  }

  // Pushes an instruction by its TypeId.
  auto PushTypeId(TypeId type_id) -> void {
    PushInstId(sem_ir_->types().GetInstId(type_id));
  }

  // Pushes a specific interface.
  auto PushSpecificInterface(SpecificInterface specific_interface) -> void {
    PushEntityName(sem_ir_->interfaces().Get(specific_interface.interface_id),
                   specific_interface.specific_id);
  }

  // Pushes a sequence of items onto the stack. This handles reversal, such that
  // the caller can pass items in print order instead of stack order.
  //
  // Note that with `ListSeparator`, the object's reference isn't stored, but
  // the separator `StringRef` will be. That should be a constant though, so is
  // safe.
  auto PushArray(llvm::ArrayRef<PushItem> items) -> void {
    for (auto item : llvm::reverse(items)) {
      CARBON_KIND_SWITCH(item) {
        case CARBON_KIND(InstId inst_id): {
          PushInstId(inst_id);
          break;
        }
        case CARBON_KIND(llvm::StringRef string): {
          PushString(string);
          break;
        }
        case CARBON_KIND(NameId name_id): {
          PushNameId(name_id);
          break;
        }
        case CARBON_KIND(ElementIndex element_index): {
          PushElementIndex(element_index);
          break;
        }
        case CARBON_KIND(QualifiedNameItem qualified_name): {
          PushQualifiedName(qualified_name.first, qualified_name.second);
          break;
        }
        case CARBON_KIND(EntityNameItem entity_name): {
          PushEntityName(entity_name.first, entity_name.second);
          break;
        }
        case CARBON_KIND(EntityNameId entity_name_id): {
          PushEntityNameId(entity_name_id);
          break;
        }
        case CARBON_KIND(TypeId type_id): {
          PushTypeId(type_id);
          break;
        }
        case CARBON_KIND(SpecificInterface specific_interface): {
          PushSpecificInterface(specific_interface);
          break;
        }
        case CARBON_KIND(llvm::ListSeparator * sep): {
          PushString(*sep);
          break;
        }
      }
    }
  }

  // Wraps `PushArray` without requiring `{}` for arguments.
  template <typename... T>
  auto Push(T... items) -> void {
    PushArray({items...});
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

  const File* sem_ir_;
  // Remaining steps to take.
  llvm::SmallVector<Step> steps_;
};

// Provides `StringifyInst` overloads for each instruction.
class Stringifier {
 public:
  explicit Stringifier(const File* sem_ir, StepStack* step_stack,
                       llvm::raw_ostream* out)
      : sem_ir_(sem_ir), step_stack_(step_stack), out_(out) {}

  // By default try to print a constant, but otherwise may fail to
  // stringify.
  auto StringifyInstDefault(InstId inst_id, Inst inst) -> void {
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
  auto StringifyInst(InstId inst_id, InstT inst) -> void {
    // This doesn't use requires so that more specific overloads are chosen when
    // provided.
    static_assert(InstT::Kind.is_type() != InstIsType::Always ||
                      std::same_as<InstT, WhereExpr>,
                  "Types should have a dedicated overload");
    // TODO: We should have Stringify support for all types where
    // InstT::Kind.constant_kind() is neither Never nor Indirect.
    StringifyInstDefault(inst_id, inst);
  }

  // Singleton instructions use their IR name as a label.
  template <typename InstT>
    requires(IsSingletonInstKind(InstT::Kind))
  auto StringifyInst(InstId /*inst_id*/, InstT /*inst*/) -> void {
    *out_ << InstT::Kind.ir_name();
  }

  auto StringifyInst(InstId /*inst_id*/, ArrayType inst) -> void {
    *out_ << "array(";
    step_stack_->Push(inst.element_type_inst_id, ", ", inst.bound_id, ")");
  }

  auto StringifyInst(InstId /*inst_id*/, AssociatedConstantDecl inst) -> void {
    const auto& assoc_const =
        sem_ir_->associated_constants().Get(inst.assoc_const_id);
    step_stack_->PushQualifiedName(assoc_const.parent_scope_id,
                                   assoc_const.name_id);
  }

  auto StringifyInst(InstId /*inst_id*/, AssociatedEntityType inst) -> void {
    *out_ << "<associated entity in ";
    step_stack_->Push(">");
    step_stack_->PushSpecificInterface(
        SpecificInterface{inst.interface_id, inst.interface_specific_id});
  }

  auto StringifyInst(InstId /*inst_id*/, BoolLiteral inst) -> void {
    step_stack_->Push(inst.value.ToBool() ? "true" : "false");
  }

  template <typename InstT>
    requires(SameAsOneOf<InstT, BindAlias, BindSymbolicName, ExportDecl>)
  auto StringifyInst(InstId /*inst_id*/, InstT inst) -> void {
    step_stack_->PushEntityNameId(inst.entity_name_id);
  }

  auto StringifyInst(InstId /*inst_id*/, ClassType inst) -> void {
    const auto& class_info = sem_ir_->classes().Get(inst.class_id);
    if (auto literal_info = TypeLiteralInfo::ForType(*sem_ir_, inst);
        literal_info.is_valid()) {
      literal_info.PrintLiteral(*sem_ir_, *out_);
      return;
    }
    step_stack_->PushEntityName(class_info, inst.specific_id);
  }

  auto StringifyInst(InstId /*inst_id*/, ConstType inst) -> void {
    *out_ << "const ";

    // Add parentheses if required.
    if (GetPrecedence(sem_ir_->insts().Get(inst.inner_id).kind()) <
        GetPrecedence(ConstType::Kind)) {
      *out_ << "(";
      // Note the `inst.inner_id` ends up here.
      step_stack_->PushString(")");
    }

    step_stack_->PushInstId(inst.inner_id);
  }

  auto StringifyInst(InstId /*inst_id*/, CustomLayoutType inst) -> void {
    auto layout = sem_ir_->custom_layouts().Get(inst.layout_id);
    *out_ << "<size " << layout[CustomLayoutId::SizeIndex] << ", align "
          << layout[CustomLayoutId::AlignIndex] << ">";
  }

  auto StringifyInst(InstId /*inst_id*/, FacetAccessType inst) -> void {
    // Given `T:! I`, print `T as type` as simply `T`.
    step_stack_->PushInstId(inst.facet_value_inst_id);
  }

  auto StringifyInst(InstId /*inst_id*/, FacetType inst) -> void {
    const FacetTypeInfo& facet_type_info =
        sem_ir_->facet_types().Get(inst.facet_type_id);
    // Output `where` restrictions.
    bool some_where = false;
    if (facet_type_info.other_requirements) {
      step_stack_->PushString("...");
      some_where = true;
    }
    if (facet_type_info.builtin_constraint_mask.HasAnyOf(
            SemIR::BuiltinConstraintMask::TypeCanDestroy)) {
      if (some_where) {
        step_stack_->PushString(" and");
      }
      step_stack_->PushString(" .Self impls Core.CanDestroy");
      some_where = true;
    }
    for (auto rewrite : llvm::reverse(facet_type_info.rewrite_constraints)) {
      if (some_where) {
        step_stack_->PushString(" and");
      }
      step_stack_->Push(" ", rewrite.lhs_id, " = ", rewrite.rhs_id);
      some_where = true;
    }
    if (!facet_type_info.self_impls_constraints.empty()) {
      if (some_where) {
        step_stack_->PushString(" and");
      }
      llvm::ListSeparator sep(" & ");
      for (auto impls : llvm::reverse(facet_type_info.self_impls_constraints)) {
        step_stack_->Push(impls, &sep);
      }
      step_stack_->PushString(" .Self impls ");
      some_where = true;
    }
    // TODO: Other restrictions from facet_type_info.
    if (some_where) {
      step_stack_->PushString(" where");
    }

    // Output extend interface requirements.
    if (facet_type_info.extend_constraints.empty()) {
      step_stack_->PushString("type");
      return;
    }
    llvm::ListSeparator sep(" & ");
    for (auto impls : llvm::reverse(facet_type_info.extend_constraints)) {
      step_stack_->Push(impls, &sep);
    }
  }

  auto StringifyInst(InstId /*inst_id*/, FacetValue inst) -> void {
    // No need to output the witness.
    step_stack_->Push(inst.type_inst_id, " as ", inst.type_id);
  }

  auto StringifyInst(InstId /*inst_id*/, FloatType inst) -> void {
    *out_ << "<builtin ";
    step_stack_->PushString(">");
    if (auto width_value =
            sem_ir_->insts().TryGetAs<IntValue>(inst.bit_width_id)) {
      *out_ << "f";
      sem_ir_->ints().Get(width_value->int_id).print(*out_, /*isSigned=*/false);
    } else {
      *out_ << "Core.Float(";
      step_stack_->Push(inst.bit_width_id, ")");
    }
  }

  auto StringifyInst(InstId /*inst_id*/, CppOverloadSetType inst) -> void {
    const auto& overload_set =
        sem_ir_->cpp_overload_sets().Get(inst.overload_set_id);
    *out_ << "<type of ";
    step_stack_->Push(StepStack::QualifiedNameItem{overload_set.parent_scope_id,
                                                   overload_set.name_id},
                      ">");
  }

  auto StringifyInst(InstId /*inst_id*/, FunctionType inst) -> void {
    const auto& fn = sem_ir_->functions().Get(inst.function_id);
    *out_ << "<type of ";
    step_stack_->Push(
        StepStack::QualifiedNameItem{fn.parent_scope_id, fn.name_id}, ">");
  }

  auto StringifyInst(InstId /*inst_id*/, FunctionTypeWithSelfType inst)
      -> void {
    StepStack::PushItem fn_name = InstId::None;
    if (auto fn_inst = sem_ir_->insts().TryGetAs<FunctionType>(
            inst.interface_function_type_id)) {
      const auto& fn = sem_ir_->functions().Get(fn_inst->function_id);
      fn_name = StepStack::QualifiedNameItem(fn.parent_scope_id, fn.name_id);
    } else {
      fn_name = inst.interface_function_type_id;
    }

    *out_ << "<type of ";
    step_stack_->Push(fn_name, " in ", inst.self_id, ">");
  }

  auto StringifyInst(InstId /*inst_id*/, GenericClassType inst) -> void {
    const auto& class_info = sem_ir_->classes().Get(inst.class_id);
    *out_ << "<type of ";
    step_stack_->Push(StepStack::QualifiedNameItem{class_info.parent_scope_id,
                                                   class_info.name_id},
                      ">");
  }

  auto StringifyInst(InstId /*inst_id*/, GenericInterfaceType inst) -> void {
    const auto& interface = sem_ir_->interfaces().Get(inst.interface_id);
    *out_ << "<type of ";
    step_stack_->Push(StepStack::QualifiedNameItem{interface.parent_scope_id,
                                                   interface.name_id},
                      ">");
  }

  // Determine the specific interface that an impl witness instruction provides
  // an implementation of.
  // TODO: Should we track this in the type?
  auto TryGetSpecificInterfaceForImplWitness(InstId impl_witness_id)
      -> std::optional<SpecificInterface> {
    if (auto lookup =
            sem_ir_->insts().TryGetAs<LookupImplWitness>(impl_witness_id)) {
      return sem_ir_->specific_interfaces().Get(
          lookup->query_specific_interface_id);
    }

    // TODO: Handle ImplWitness.
    return std::nullopt;
  }

  auto StringifyInst(InstId /*inst_id*/, ImplWitnessAccess inst) -> void {
    auto witness_inst_id =
        sem_ir_->constant_values().GetConstantInstId(inst.witness_id);
    auto lookup = sem_ir_->insts().GetAs<LookupImplWitness>(witness_inst_id);
    auto specific_interface =
        sem_ir_->specific_interfaces().Get(lookup.query_specific_interface_id);
    const auto& interface =
        sem_ir_->interfaces().Get(specific_interface.interface_id);
    if (!interface.associated_entities_id.has_value()) {
      step_stack_->Push(".(TODO: element ", inst.index, " in incomplete ",
                        witness_inst_id, ")");
    } else {
      auto entities =
          sem_ir_->inst_blocks().Get(interface.associated_entities_id);
      size_t index = inst.index.index;
      CARBON_CHECK(index < entities.size(), "Access out of bounds.");
      auto entity_inst_id = entities[index];
      step_stack_->PushString(")");
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
      step_stack_->Push(
          ".(",
          StepStack::EntityNameItem{interface, specific_interface.specific_id},
          ".");
    }

    if (auto lookup =
            sem_ir_->insts().TryGetAs<LookupImplWitness>(witness_inst_id)) {
      bool period_self = false;
      if (auto sym_name = sem_ir_->insts().TryGetAs<BindSymbolicName>(
              lookup->query_self_inst_id)) {
        auto name_id =
            sem_ir_->entity_names().Get(sym_name->entity_name_id).name_id;
        period_self = (name_id == NameId::PeriodSelf);
      }
      if (!period_self) {
        step_stack_->PushInstId(lookup->query_self_inst_id);
      }
    } else {
      // TODO: Omit parens if not needed for precedence.
      step_stack_->Push("(", witness_inst_id, ")");
    }
  }

  auto StringifyInst(InstId /*inst_id*/, ImportRefUnloaded inst) -> void {
    if (inst.entity_name_id.has_value()) {
      step_stack_->PushEntityNameId(inst.entity_name_id);
    } else {
      *out_ << "<import ref unloaded invalid entity name>";
    }
  }

  auto StringifyInst(InstId /*inst_id*/, IntType inst) -> void {
    *out_ << "<builtin ";
    step_stack_->PushString(">");
    if (auto width_value =
            sem_ir_->insts().TryGetAs<IntValue>(inst.bit_width_id)) {
      *out_ << (inst.int_kind.is_signed() ? "i" : "u");
      sem_ir_->ints().Get(width_value->int_id).print(*out_, /*isSigned=*/false);
    } else {
      *out_ << (inst.int_kind.is_signed() ? "Int(" : "UInt(");
      step_stack_->Push(inst.bit_width_id, ")");
    }
  }

  auto StringifyInst(InstId /*inst_id*/, IntValue inst) -> void {
    sem_ir_->ints().Get(inst.int_id).print(*out_, /*isSigned=*/true);
  }

  auto StringifyInst(InstId /*inst_id*/, LookupImplWitness inst) -> void {
    step_stack_->Push(
        inst.query_self_inst_id, " as ",
        sem_ir_->specific_interfaces().Get(inst.query_specific_interface_id));
  }

  auto StringifyInst(InstId /*inst_id*/, MaybeUnformedType inst) -> void {
    step_stack_->Push("<builtin MaybeUnformed(", inst.inner_id, ")>");
  }

  auto StringifyInst(InstId /*inst_id*/, NameRef inst) -> void {
    *out_ << sem_ir_->names().GetFormatted(inst.name_id);
  }

  auto StringifyInst(InstId /*inst_id*/, Namespace inst) -> void {
    const auto& name_scope = sem_ir_->name_scopes().Get(inst.name_scope_id);
    step_stack_->PushQualifiedName(name_scope.parent_scope_id(),
                                   name_scope.name_id());
  }

  auto StringifyInst(InstId /*inst_id*/, PartialType inst) -> void {
    *out_ << "partial ";

    step_stack_->PushInstId(inst.inner_id);
  }

  auto StringifyInst(InstId /*inst_id*/, PatternType inst) -> void {
    *out_ << "<pattern for ";
    step_stack_->Push(inst.scrutinee_type_inst_id, ">");
  }

  auto StringifyInst(InstId /*inst_id*/, PointerType inst) -> void {
    step_stack_->Push(inst.pointee_id, "*");
  }

  auto StringifyInst(InstId /*inst_id*/, SpecificFunction inst) -> void {
    auto callee = GetCallee(*sem_ir_, inst.callee_id);
    if (auto* fn = std::get_if<CalleeFunction>(&callee)) {
      step_stack_->PushEntityName(sem_ir_->functions().Get(fn->function_id),
                                  inst.specific_id);
      return;
    }
    step_stack_->PushString("<invalid specific function>");
  }

  auto StringifyInst(InstId /*inst_id*/, SpecificImplFunction inst) -> void {
    auto callee = GetCallee(*sem_ir_, inst.callee_id);
    if (auto* fn = std::get_if<CalleeFunction>(&callee)) {
      // TODO: The specific_id here is for the interface member, but the
      // entity we're passing is the impl member. This might result in
      // strange output once we render specific arguments properly.
      step_stack_->PushEntityName(sem_ir_->functions().Get(fn->function_id),
                                  inst.specific_id);
      return;
    }
    step_stack_->PushString("<invalid specific function>");
  }

  auto StringifyInst(InstId /*inst_id*/, StructType inst) -> void {
    auto fields = sem_ir_->struct_type_fields().Get(inst.fields_id);
    if (fields.empty()) {
      *out_ << "{}";
      return;
    }
    *out_ << "{";
    step_stack_->PushString("}");
    llvm::ListSeparator sep;
    for (auto field : llvm::reverse(fields)) {
      step_stack_->Push(".", field.name_id, ": ", field.type_inst_id, &sep);
    }
  }

  auto StringifyInst(InstId /*inst_id*/, StructValue inst) -> void {
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
    llvm::ListSeparator sep;
    for (auto [field, value_inst_id] :
         llvm::reverse(llvm::zip(fields, field_values))) {
      step_stack_->Push(".", field.name_id, " = ", value_inst_id, &sep);
    }
  }

  auto StringifyInst(InstId /*inst_id*/, SymbolicBindingType inst) -> void {
    step_stack_->PushEntityNameId(inst.entity_name_id);
  }

  auto StringifyInst(InstId /*inst_id*/, TupleType inst) -> void {
    auto refs = sem_ir_->inst_blocks().Get(inst.type_elements_id);
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
    llvm::ListSeparator sep;
    for (auto ref : llvm::reverse(refs)) {
      step_stack_->Push(ref, &sep);
    }
  }

  auto StringifyInst(InstId /*inst_id*/, TupleValue inst) -> void {
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
    llvm::ListSeparator sep;
    for (auto ref : llvm::reverse(refs)) {
      step_stack_->Push(ref, &sep);
    }
  }

  auto StringifyInst(InstId inst_id, TypeOfInst /*inst*/) -> void {
    // Print the constant value if we've already computed the inst.
    auto const_inst_id = sem_ir_->constant_values().GetConstantInstId(inst_id);
    if (const_inst_id.has_value() && const_inst_id != inst_id) {
      step_stack_->PushInstId(const_inst_id);
      return;
    }
    *out_ << "<dependent type>";
  }

  auto StringifyInst(InstId /*inst_id*/, UnboundElementType inst) -> void {
    *out_ << "<unbound element of class ";
    step_stack_->Push(inst.class_type_inst_id, ">");
  }

  auto StringifyInst(InstId /*inst_id*/, VtablePtr /*inst*/) -> void {
    *out_ << "<vtable ptr>";
  }

 private:
  const File* sem_ir_;
  StepStack* step_stack_;
  llvm::raw_ostream* out_;
};

}  // namespace

// NOLINTNEXTLINE(readability-function-size)
static auto Stringify(const File& sem_ir, StepStack& step_stack)
    -> std::string {
  RawStringOstream out;

  Stringifier stringifier(&sem_ir, &step_stack, &out);

  while (!step_stack.empty()) {
    CARBON_KIND_SWITCH(step_stack.Pop()) {
      case CARBON_KIND(InstId inst_id): {
        if (!inst_id.has_value()) {
          out << "<invalid>";
          break;
        }
        auto untyped_inst = sem_ir.insts().Get(inst_id);
        CARBON_KIND_SWITCH(untyped_inst) {
#define CARBON_SEM_IR_INST_KIND(InstT)              \
  case CARBON_KIND(InstT typed_inst): {             \
    stringifier.StringifyInst(inst_id, typed_inst); \
    break;                                          \
  }
#include "toolchain/sem_ir/inst_kind.def"
        }
        break;
      }
      case CARBON_KIND(llvm::StringRef string):
        out << string;
        break;
      case CARBON_KIND(NameId name_id):
        out << sem_ir.names().GetFormatted(name_id);
        break;
      case CARBON_KIND(ElementIndex element_index):
        out << element_index.index;
        break;
    }
  }

  return out.TakeStr();
}

auto StringifyConstantInst(const File& sem_ir, InstId outer_inst_id)
    -> std::string {
  StepStack step_stack(&sem_ir);
  step_stack.PushInstId(outer_inst_id);
  return Stringify(sem_ir, step_stack);
}

auto StringifySpecific(const File& sem_ir, SpecificId specific_id)
    -> std::string {
  StepStack step_stack(&sem_ir);

  const auto& specific = sem_ir.specifics().Get(specific_id);
  const auto& generic = sem_ir.generics().Get(specific.generic_id);
  auto decl = sem_ir.insts().Get(generic.decl_id);
  CARBON_KIND_SWITCH(decl) {
    case CARBON_KIND(ClassDecl class_decl): {
      // Print `Core.Int(N)` as `iN`.
      // TODO: This duplicates work done in StringifyInst for ClassType.
      const auto& class_info = sem_ir.classes().Get(class_decl.class_id);
      if (auto literal_info = TypeLiteralInfo::ForType(
              sem_ir, ClassType{.type_id = TypeType::TypeId,
                                .class_id = class_decl.class_id,
                                .specific_id = specific_id});
          literal_info.is_valid()) {
        RawStringOstream out;
        literal_info.PrintLiteral(sem_ir, out);
        return out.TakeStr();
      }
      step_stack.PushEntityName(class_info, specific_id);
      break;
    }
    case CARBON_KIND(FunctionDecl function_decl): {
      step_stack.PushEntityName(
          sem_ir.functions().Get(function_decl.function_id), specific_id);
      break;
    }
    case CARBON_KIND(ImplDecl impl_decl): {
      step_stack.PushEntityName(sem_ir.impls().Get(impl_decl.impl_id),
                                specific_id);
      break;
    }
    case CARBON_KIND(InterfaceDecl interface_decl): {
      step_stack.PushEntityName(
          sem_ir.interfaces().Get(interface_decl.interface_id), specific_id);
      break;
    }
    default: {
      // TODO: Include the specific arguments here.
      step_stack.PushInstId(generic.decl_id);
      break;
    }
  }
  return Stringify(sem_ir, step_stack);
}

auto StringifySpecificInterface(const File& sem_ir,
                                SpecificInterface specific_interface)
    -> std::string {
  if (specific_interface.specific_id.has_value()) {
    return StringifySpecific(sem_ir, specific_interface.specific_id);
  } else {
    auto name_id =
        sem_ir.interfaces().Get(specific_interface.interface_id).name_id;
    return sem_ir.names().GetFormatted(name_id).str();
  }
}

}  // namespace Carbon::SemIR
