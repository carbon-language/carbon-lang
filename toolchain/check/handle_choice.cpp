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
  // There's no modifiers on a choice, but this informs how to typecheck any
  // generic binding pattern.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Choice>();
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
  context.decl_introducer_state_stack().Pop<Lex::TokenKind::Choice>();

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
  SemIR::Class& mut_class = context.classes().Get(class_decl.class_id);
  // Build the `Self` type using the resulting type constant.
  mut_class.self_type_id = context.GetTypeIdForTypeConstant(
      TryEvalInst(context, SemIR::InstId::Invalid,
                  SemIR::ClassType{.type_id = SemIR::TypeType::SingletonTypeId,
                                   .class_id = class_decl.class_id,
                                   .specific_id = self_specific_id}));

  // Enter the choice scope.
  context.scope_stack().Push(class_decl_id, class_info.scope_id,
                             self_specific_id);
  // Checking the binding pattern for an alternative requires a non-empty stack.
  // FIXME: Choice is incorrect as we're not parsing the pattern for a choice
  // name, but there's no Lex token that's a decl introducer that we could
  // safely use here. Is there a better way to communicate to
  // `HandleAnyBindingPattern` that we're checking a choice alternative?
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Choice>();
  StartGenericDefinition(context);

  // TODO: Do Choice types have a `Self` name they can use in the alternatives?
  // context.name_scopes().AddRequiredName(
  //     class_info.scope_id, SemIR::NameId::SelfType,
  //     context.types().GetInstId(class_info.self_type_id));

  // Mark the beginning of the choice body.
  context.node_stack().Push(node_id, class_decl.class_id);

  CARBON_CHECK(context.choice_deferred_bindings().empty(),
               "Alternatives left behind in choice_deferred_bindings: {0}",
               context.choice_deferred_bindings().size());
  return true;
}

static auto AddChoiceAlternative(Context& context, Parse::NodeId node_id)
    -> void {
  // Note, there is nothing like a ChoiceAlternativeIntroducer node, so no parse
  // node to pop here.
  auto name_component = PopNameComponent(context);
  if (name_component.param_patterns_id == SemIR::InstBlockId::Empty) {
    CARBON_DIAGNOSTIC(ChoiceAlternativeEmptyParams, Error,
                      "choice alternative has empty parameter list");
    CARBON_DIAGNOSTIC(ChoiceAlternativeEmptyParamsNote, Note,
                      "remove the empty `()`");
    context.emitter()
        .Build(name_component.params_loc_id, ChoiceAlternativeEmptyParams)
        .Note(name_component.params_loc_id, ChoiceAlternativeEmptyParamsNote)
        .Emit();
    return;
  }
  if (name_component.param_patterns_id.is_valid()) {
    context.TODO(name_component.params_loc_id,
                 "choice alternatives with parameters are not yet supported");
    return;
  }
  context.choice_deferred_bindings().push_back({node_id, name_component});
}

static auto MakeLetBinding(Context& context, SemIR::TypeId self_type_id,
                           SemIR::NameScopeId choice_name_scope_id,
                           SemIR::TypeId discriminant_type_id, int index,
                           int num_alternative_bits,
                           const Context::ChoiceDeferredBinding& binding)
    -> void {
  llvm::SmallVector<SemIR::StructTypeField, 2> self_fields;
  self_fields.push_back({
      .name_id = SemIR::NameId::ChoiceDiscriminant,
      .type_id = discriminant_type_id,
  });

  SemIR::InstId discriminant_value_id = [&] {
    if (num_alternative_bits == 0) {
      return context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
          binding.node_id, SemIR::TupleLiteral{
                               .type_id = context.GetTupleType({}),
                               .elements_id = SemIR::InstBlockId::Empty,
                           }));
    } else {
      return MakeIntLiteral(context, binding.node_id,
                            context.ints().Add(index));
    }
  }();
  discriminant_value_id = ConvertToValueOfType(
      context, binding.node_id, discriminant_value_id, discriminant_type_id);

  auto self_struct_type_id = context.GetStructType(
      context.struct_type_fields().AddCanonical(self_fields));

  auto self_value_id = ConvertToValueOfType(
      context, binding.node_id,
      context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
          binding.node_id,
          SemIR::StructLiteral{
              .type_id = self_struct_type_id,
              .elements_id =
                  [&] {
                    context.inst_block_stack().Push();
                    context.inst_block_stack().AddInstId(discriminant_value_id);
                    return context.inst_block_stack().Pop();
                  }(),
          })),
      self_type_id);

  auto entity_name_id = context.entity_names().Add(
      {.name_id = binding.name_component.name_id,
       .parent_scope_id = choice_name_scope_id,
       .bind_index = SemIR::CompileTimeBindIndex::Invalid});
  auto bind_name_id = context.AddInst(SemIR::LocIdAndInst::UncheckedLoc(
      binding.node_id, SemIR::BindName{
                           .type_id = self_type_id,
                           .entity_name_id = entity_name_id,
                           .value_id = self_value_id,
                       }));
  context.name_scopes()
      .Get(choice_name_scope_id)
      .AddRequired({.name_id = binding.name_component.name_id,
                    .inst_id = bind_name_id,
                    .access_kind = SemIR::AccessKind::Public});
}

auto HandleParseNode(Context& context, Parse::ChoiceDefinitionId node_id)
    -> bool {
  // The last alternative may optionally not have a comma after it, in which
  // case we get here after the last alternative.
  if (!context.node_stack().PeekIs(Parse::NodeKind::ChoiceDefinitionStart)) {
    AddChoiceAlternative(context, node_id);
  }

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

  SemIR::TypeId discriminant_type_id = [&] {
    if (num_alternative_bits == 0) {
      // An empty choice is not constructible (which can be a useful type). We
      // always add an empty tuple as a field to make it not constructible
      // directly.
      // TODO: This can be done in a nicer way without adding an empty field.
      return context.GetTupleType({});
    } else {
      return MakeIntType(context, node_id, SemIR::IntKind::Unsigned,
                         context.ints().Add(num_alternative_bits));
    }
  }();

  llvm::SmallVector<SemIR::StructTypeField> struct_type_fields;
  struct_type_fields.push_back({
      .name_id = SemIR::NameId::ChoiceDiscriminant,
      .type_id = discriminant_type_id,
  });
  auto fields_id =
      context.struct_type_fields().AddCanonical(struct_type_fields);
  auto choice_witness_id = context.AddInst<SemIR::CompleteTypeWitness>(
      node_id,
      {.type_id = context.GetSingletonType(SemIR::WitnessType::SingletonInstId),
       .object_repr_id = context.GetStructType(fields_id)});
  // Note: avoid storing a reference to the returned Class, since it may be
  // invalidated by other type constructions.
  context.classes().Get(class_id).complete_type_witness_id = choice_witness_id;

  auto self_type_id = context.classes().Get(class_id).self_type_id;
  auto name_scope_id = context.classes().Get(class_id).scope_id;

  for (auto [index, binding] :
       llvm::enumerate(context.choice_deferred_bindings())) {
    MakeLetBinding(context, self_type_id, name_scope_id, discriminant_type_id,
                   index, num_alternative_bits, binding);
  }

  // The scopes and blocks for the choice itself.
  context.inst_block_stack().Pop();
  context.decl_introducer_state_stack().Pop<Lex::TokenKind::Choice>();
  context.scope_stack().Pop();
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
