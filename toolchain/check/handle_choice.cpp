// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/name_component.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::ChoiceIntroducerId node_id)
    -> bool {
  // Create an instruction block to hold the instructions created as part of the
  // choice signature, such as generic parameters.
  context.inst_block_stack().Push();
  // Push the bracketing node.
  context.node_stack().Push(node_id);
  // The choice's name follows.
  context.decl_name_stack().PushScopeAndStartName();
  // This choice is potentially generic.
  StartGenericDefinition(context);
  return true;
}

auto HandleParseNode(Context& context, Parse::ChoiceDefinitionStartId node_id)
    -> bool {
  auto name = PopNameComponent(context);
  auto name_context = context.decl_name_stack().FinishName(name);
  context.node_stack()
      .PopAndDiscardSoloNodeId<Parse::NodeKind::ChoiceIntroducer>();

  auto decl_block_id = context.inst_block_stack().Pop();

  // Choices create a ClassId, since they ultimately turn into a class with
  // methods and some builtin impls.
  // TODO: Should we make a separate SemIR::ChoiceId type?
  auto class_decl =
      SemIR::ClassDecl{.type_id = SemIR::TypeType::SingletonTypeId,
                       .class_id = SemIR::ClassId::Invalid,
                       .decl_block_id = decl_block_id};
  auto class_decl_id =
      context.AddPlaceholderInst(SemIR::LocIdAndInst(node_id, class_decl));

  context.decl_name_stack().AddNameOrDiagnose(name_context, class_decl_id,
                                              SemIR::AccessKind::Public);

  // An inst block for the body of the choice.
  context.inst_block_stack().Push();
  auto body_block_id = context.inst_block_stack().PeekOrAdd();

  SemIR::Class class_info = {
      name_context.MakeEntityWithParamsBase(name, class_decl_id,
                                            /*is_extern=*/false,
                                            SemIR::LibraryNameId::Invalid),
      {// `.self_type_id` depends on the ClassType, so is set below.
       .self_type_id = SemIR::TypeId::Invalid,
       .inheritance_kind = SemIR::ClassFields::Final,
       // TODO: Handle the case where there's control flow in the alternatives.
       // For example:
       //
       //   choice C {
       //     Alt(if true then i32 else f64),
       //   }
       //
       // We may need to track a list of instruction blocks here, as we do for a
       // function.
       .body_block_id = body_block_id}};

  // This call finishes the GenericDecl, after which we can use the `Self`
  // specific.
  class_info.generic_id = BuildGenericDecl(context, class_decl_id);
  auto self_specific_id =
      context.generics().GetSelfSpecific(class_info.generic_id);

  class_info.definition_id = class_decl_id;
  class_info.scope_id = context.name_scopes().Add(
      class_decl_id, SemIR::NameId::Invalid, class_info.parent_scope_id);
  class_decl.class_id = context.classes().Add(class_info);
  if (class_info.has_parameters()) {
    class_decl.type_id = context.GetGenericClassType(
        class_decl.class_id, context.scope_stack().PeekSpecificId());
  }

  context.ReplaceInstBeforeConstantUse(class_decl_id, class_decl);

  // We had to construct the `ClassId` from `Class` in order to build the `Self`
  // type below. But it needs to be written back to the `Class` in the
  // ValueStore, not the local variable. This gives a mutable reference to the
  // `Class` in the ValueStore.
  SemIR::Class& mut_class_info = context.classes().Get(class_decl.class_id);
  // Build the `Self` type using the resulting type constant.
  mut_class_info.self_type_id = context.GetTypeIdForTypeConstant(
      TryEvalInst(context, SemIR::InstId::Invalid,
                  SemIR::ClassType{.type_id = SemIR::TypeType::SingletonTypeId,
                                   .class_id = class_decl.class_id,
                                   .specific_id = self_specific_id}));

  // Enter the choice scope.
  context.scope_stack().Push(class_decl_id, class_info.scope_id,
                             self_specific_id);
  StartGenericDefinition(context);

  // TODO: Do Choice types have a `Self` name they can use in the alternatives?
  // It's unsized/recursive but it could be used in pointers like Box<Self>.
  // context.name_scopes().AddRequiredName(
  //     class_info.scope_id, SemIR::NameId::SelfTFype,
  //     context.types().GetInstId(class_info.self_type_id));

  // Mark the beginning of the choice body.
  context.node_stack().Push(node_id, class_decl.class_id);

  CARBON_CHECK(context.choice_deferred_bindings().empty(),
               "Alternatives left behind in choice_deferred_bindings: {0}",
               context.choice_deferred_bindings().size());

  // FIXME: Can we skip these if the choice type is empty? Is that allowed?
  // Set up blocks and scopes for the first alternative.
  context.decl_name_stack().PushScopeAndStartName();
  // context.inst_block_stack().Push();
  // context.pattern_block_stack().Push();
  //  context.param_and_arg_refs_stack().Push();
  return true;
}

static auto AddChoiceAlternative(Context& context, Parse::NodeId node_id)
    -> void {
  auto name_component = PopNameComponent(context);

  auto name_context = context.decl_name_stack().FinishName(name_component);
  context.decl_name_stack().PopScope();
  // Note, there is nothing like a ChoiceAlternativeIntroducer node, so no parse
  // node to pop here.

  if (name_component.param_patterns_id == SemIR::InstBlockId::Empty) {
    context.TODO(name_component.params_loc_id,
                 "Make this an error: Empty () found, should be omitted");
  }

  context.choice_deferred_bindings().push_back(
      {node_id, name_component, name_context});

  // FIXME: Avoid opening a scope if this is the last alternative?
  context.decl_name_stack().PushScopeAndStartName();
  // context.inst_block_stack().Push();
  // context.pattern_block_stack().Push();
  //  context.param_and_arg_refs_stack().Push();
}

struct Alternative {
  SemIR::TypeId struct_type_id;
  llvm::SmallVector</* FIXME: What type? */ int> params;
};

static auto MakeAlternatives(
    Context& context,
    llvm::ArrayRef<Context::ChoiceDeferredBinding> deferred_bindings)
    -> llvm::SmallVector<Alternative> {
  llvm::SmallVector<Alternative> alternatives;

  for (auto binding : deferred_bindings) {
    llvm::SmallVector<SemIR::StructTypeField> struct_type_fields;

    if (binding.name_component.param_patterns_id.is_valid()) {
      // FIXME: Add the alternative's params to `struct_type_fields`.
      context.TODO(binding.node_id,
                   "Alternative with parameters not yet handled");
    }

    context.struct_type_fields_stack().PushArray();

    if (binding.name_component.param_patterns_id.is_valid()) {
      context.TODO(binding.node_id, "params for alternative not yet supported");
      // TODO: context.struct_type_fields_stack().AppendToTop({.name_id =
      // name_id, .type_id = value_type_id})
    }

    auto struct_type_id =
        context.GetStructType(context.struct_type_fields().AddCanonical(
            context.struct_type_fields_stack().PeekArray()));
    context.struct_type_fields_stack().PopArray();

    alternatives.push_back({.struct_type_id = struct_type_id, .params = {}});
  }

  return alternatives;
}

static auto MakeDiscriminant(Context& context, Parse::NodeId node_id,
                             int num_alternative_bits) -> SemIR::TypeId {
  // TODO: Try to store the bits into padding or invalid states of the various
  // alternative structs when possible.
  return MakeIntType(context, node_id, SemIR::IntKind::Unsigned,
                     context.ints().Add(num_alternative_bits));
}

static auto MakeStorageTuple(Context& context, llvm::ArrayRef<Alternative> alts)
    -> SemIR::TypeId {
  llvm::SmallVector<SemIR::TypeId> tuple_element_types;
  for (const auto& alt : alts) {
    tuple_element_types.push_back(alt.struct_type_id);
  }
  return context.GetTupleType(tuple_element_types);
}

static auto MakeLetBinding(Context& context, SemIR::TypeId self_type_id,
                           std::optional<SemIR::TypeId> discriminant_type_id,
                           SemIR::TypeId storage_type_id, int index,
                           int num_alternatives,
                           const Context::ChoiceDeferredBinding& binding,
                           const Alternative& alt) -> void {
  // ===== Types =====

  // An empty struct for inactive alternatives.
  auto empty_struct_value = SemIR::StructLiteral{
      .type_id = context.GetStructType(SemIR::StructTypeFieldsId::Empty),
      .elements_id = SemIR::InstBlockId::Empty,
  };

  // The alternative value struct.
  llvm::SmallVector<SemIR::StructTypeField> alt_fields;
  if (!alt.params.empty()) {
    // TODO: add alt.params types to `alt_fields`.
    context.TODO(SemIR::LocId::Invalid, "no support for parameters yet");
  }
  auto alt_fields_id = context.struct_type_fields().AddCanonical(alt_fields);
  auto alt_struct_type_id = context.GetStructType(alt_fields_id);

  // Struct literal type to hold the fields of Self for initialization.
  llvm::SmallVector<SemIR::StructTypeField, 2> self_fields;
  if (discriminant_type_id) {
    self_fields.push_back({
        .name_id = SemIR::NameId::ChoiceDiscriminant,
        .type_id = *discriminant_type_id,
    });
  }
  self_fields.push_back({
      .name_id = SemIR::NameId::ChoiceStorage,
      .type_id = storage_type_id,
  });
  auto self_struct_type_id = context.GetStructType(
      context.struct_type_fields().AddCanonical(self_fields));

  // ===== Values =====

  std::optional<SemIR::InstId> discriminant_value_id;
  if (discriminant_type_id) {
    // FIXME: Move to class' inst block.
    discriminant_value_id = ConvertToValueOfType(
        context, binding.node_id,
        MakeIntLiteral(context, binding.node_id, context.ints().Add(index)),
        *discriminant_type_id);
  }

  llvm::SmallVector<SemIR::InstId> storage_tuple_value_ids;
  for (int i = 0; i < num_alternatives; ++i) {
    if (i == index) {
      llvm::SmallVector<SemIR::InstId> alt_fields_values;
      if (!alt.params.empty()) {
        // TODO: Construct alt.params values and add to `alt_fields_values`.
        context.TODO(binding.node_id, "TODO: alt.params values here");
      }

      auto alt_struct_elements_id = [&] {
        context.inst_block_stack().Push();
        for (auto id : alt_fields_values) {
          context.inst_block_stack().AddInstId(id);
        }
        return context.inst_block_stack().Pop();
      }();
      auto alt_struct_value_id = ConvertToValueOfType(
          context, binding.node_id,
          context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
              binding.node_id,
              SemIR::StructLiteral{
                  .type_id = alt_struct_type_id,
                  .elements_id = alt_struct_elements_id,
              })),
          alt_struct_type_id);
      storage_tuple_value_ids.push_back(alt_struct_value_id);
    } else {
      auto alt_struct_value_id =
          context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
              binding.node_id, empty_struct_value));
      storage_tuple_value_ids.push_back(alt_struct_value_id);
    }
  }

  auto storage_value_id = ConvertToValueOfType(
      context, binding.node_id,
      context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
          binding.node_id,
          SemIR::TupleLiteral{
              .type_id = storage_type_id,
              .elements_id =
                  [&] {
                    context.inst_block_stack().Push();
                    for (auto id : storage_tuple_value_ids) {
                      context.inst_block_stack().AddInstId(id);
                    }
                    return context.inst_block_stack().Pop();
                  }(),
          })),
      storage_type_id);

  auto self_value_id = ConvertToValueOfType(
      context, binding.node_id,
      context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
          binding.node_id,
          SemIR::StructLiteral{
              .type_id = self_struct_type_id,
              .elements_id =
                  [&] {
                    context.inst_block_stack().Push();
                    if (discriminant_value_id) {
                      context.inst_block_stack().AddInstId(
                          *discriminant_value_id);
                    }
                    context.inst_block_stack().AddInstId(storage_value_id);
                    return context.inst_block_stack().Pop();
                  }(),
          })),
      self_type_id);

  auto entity_name_id = context.entity_names().Add(
      {.name_id = binding.name_component.name_id,
       .parent_scope_id = context.scope_stack().PeekNameScopeId(),
       .bind_index = SemIR::CompileTimeBindIndex::Invalid});
  auto bind_name_id = context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
      binding.node_id, SemIR::BindName{
                           .type_id = self_type_id,
                           .entity_name_id = entity_name_id,
                           .value_id = self_value_id,
                       }));
  context.decl_name_stack().AddNameOrDiagnose(
      binding.name_context, bind_name_id, SemIR::AccessKind::Public);
}

static auto MakeFunctionBinding(
    Context& context, SemIR::TypeId self_type_id,
    std::optional<SemIR::TypeId> discriminant_type_id,
    SemIR::TypeId storage_type_id, int index,
    const Context::ChoiceDeferredBinding& binding, const Alternative& alt)
    -> void {
  (void)context;
  (void)self_type_id;
  (void)discriminant_type_id;
  (void)storage_type_id;
  (void)index;
  (void)binding;
  (void)alt;

#if 0
  auto decl_block_id = context.inst_block_stack()
                           .PeekOrAdd();  // context.inst_block_stack().Pop();
  auto self_specific_id = context.scope_stack().PeekSpecificId();

  auto function_decl = SemIR::FunctionDecl{
      SemIR::TypeId::Invalid, SemIR::FunctionId::Invalid, decl_block_id};
  auto decl_id = context.AddPlaceholderInst(
      SemIR::LocIdAndInst::UncheckedLoc(binding.node_id, function_decl));

  auto function_info = SemIR::Function{
      {binding.name_context.MakeEntityWithParamsBase(binding.name_component, decl_id, /*is_extern=*/false,
                                             SemIR::LibraryNameId::Invalid)},
      {.return_slot_pattern_id = binding.name_component.return_slot_pattern_id,
       .virtual_modifier = SemIR::FunctionFields::VirtualModifier::None}};
  function_info.definition_id = decl_id;

  {
    context.inst_block_stack().Push();

    {
      context.inst_block_stack().Push();
      // FIXME: Add the parameter values.
    }
    auto alt_values_block_id = context.inst_block_stack().Pop();

    auto alt_struct_value = SemIR::StructLiteral{
        .type_id = alt.struct_type_id,
        .elements_id = alt_values_block_id,
    };
    auto alt_struct_value_id = context.AddInst(
        SemIR::LocIdAndInst::UncheckedLoc(binding.node_id, alt_struct_value));
    auto return_expr = SemIR::ReturnExpr{
        .expr_id = alt_struct_value_id,
        .dest_id = SemIR::InstId::Invalid,
    };
    context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(binding.node_id, return_expr));

    auto body_block_id = context.inst_block_stack().Pop();
    function_info.body_block_ids = {body_block_id};
  }

  function_decl.function_id = context.functions().Add(function_info);
  function_decl.type_id =
      context.GetFunctionType(function_decl.function_id, self_specific_id);

  // Write the function ID into the FunctionDecl.
  context.ReplaceInstBeforeConstantUse(decl_id, function_decl);

  // FIXME: How do we add an alternative? We have a name, where are its
  // parameters if it has any? We need to add a method to the choice class.

  // auto params_block_id = context.param_and_arg_refs_stack().Pop();

  context.decl_name_stack().AddNameOrDiagnose(binding.name_context, decl_id,
                                              SemIR::AccessKind::Public);
#endif
}

auto HandleParseNode(Context& context, Parse::ChoiceDefinitionId node_id)
    -> bool {
  // The last alternative may optionally not have a comma after it, in which
  // case we get here after the last alternative.
  if (!context.node_stack().PeekIs<Parse::NodeKind::ChoiceDefinitionStart>()) {
    AddChoiceAlternative(context, node_id);
  }

  // Cleanup/abandon scopes set up for the next alternative, as there are no
  // more.
  context.decl_name_stack().AbortName();
  context.decl_name_stack().PopScope();
  // context.inst_block_stack().Pop();
  // context.pattern_block_stack().Pop();
  //  context.param_and_arg_refs_stack().Pop();

  auto class_id =
      context.node_stack().Pop<Parse::NodeKind::ChoiceDefinitionStart>();

  int num_alternatives = context.choice_deferred_bindings().size();
  int num_alternative_bits = [&] {
    if (num_alternatives > 1) {
      return static_cast<int>(ceil(log2(num_alternatives)));
    } else {
      return 0;
    }
  }();

  llvm::SmallVector<Alternative> alternatives;
  std::optional<SemIR::TypeId> discriminant_type_id;
  if (num_alternatives == 0) {
    // An empty choice is not constructible (which can be a useful type). We
    // don't need to add any members to it.
  } else {
    alternatives =
        MakeAlternatives(context, context.choice_deferred_bindings());
    if (num_alternative_bits > 0) {
      discriminant_type_id =
          MakeDiscriminant(context, node_id, num_alternative_bits);
    }
  }
  SemIR::TypeId storage_type_inst_id = MakeStorageTuple(context, alternatives);

  // TODO: Change to a custom discriminated union storage instead of using
  // struct representation and a tuple.
  llvm::SmallVector<SemIR::StructTypeField> struct_type_fields;
  if (discriminant_type_id) {
    struct_type_fields.push_back({
        .name_id = SemIR::NameId::ChoiceDiscriminant,
        .type_id = *discriminant_type_id,
    });
  }
  struct_type_fields.push_back({
      .name_id = SemIR::NameId::ChoiceStorage,
      .type_id = storage_type_inst_id,
  });
  auto fields_id =
      context.struct_type_fields().AddCanonical(struct_type_fields);
  auto choice_witness_id = context.AddInst<SemIR::CompleteTypeWitness>(
      node_id,
      {.type_id = context.GetSingletonType(SemIR::WitnessType::SingletonInstId),
       .object_repr_id = context.GetStructType(fields_id)});
  // Note: avoid storing a reference to the returned ClassInfo, it may be
  // invalidated by other type constructions.
  context.classes().Get(class_id).complete_type_witness_id = choice_witness_id;

  auto self_type_id = context.classes().Get(class_id).self_type_id;

  for (auto [index, pair] : llvm::enumerate(
           llvm::zip(context.choice_deferred_bindings(), alternatives))) {
    auto [binding, alt] = pair;
    // If there's no params, then we make a let binding with the name of the
    // alternative. Otherwise, we make a function.
    if (alt.params.empty()) {
      MakeLetBinding(context, self_type_id, discriminant_type_id,
                     storage_type_inst_id, index, num_alternatives, binding,
                     alt);
    } else {
      MakeFunctionBinding(context, self_type_id, discriminant_type_id,
                          storage_type_inst_id, index, binding, alt);
    }
  }

  // The scopes and blocks for the choice itself.
  context.inst_block_stack().Pop();
  context.decl_name_stack().PopScope();

  FinishGenericDefinition(context, context.classes().Get(class_id).generic_id);

  context.choice_deferred_bindings().clear();
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::ChoiceAlternativeListCommaId node_id) -> bool {
  AddChoiceAlternative(context, node_id);
  return true;
}

}  // namespace Carbon::Check
