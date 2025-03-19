// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/literal.h"
#include "toolchain/check/name_component.h"
#include "toolchain/check/type.h"
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
  auto class_decl =
      SemIR::ClassDecl{.type_id = SemIR::TypeType::SingletonTypeId,
                       .class_id = SemIR::ClassId::None,
                       .decl_block_id = decl_block_id};
  auto class_decl_id =
      AddPlaceholderInst(context, SemIR::LocIdAndInst(node_id, class_decl));

  context.decl_name_stack().AddNameOrDiagnose(name_context, class_decl_id,
                                              SemIR::AccessKind::Public);

  // An inst block for the body of the choice.
  context.inst_block_stack().Push();
  auto body_block_id = context.inst_block_stack().PeekOrAdd();

  SemIR::Class class_info = {
      name_context.MakeEntityWithParamsBase(name, class_decl_id,
                                            /*is_extern=*/false,
                                            SemIR::LibraryNameId::None),
      {// `.self_type_id` depends on the ClassType, so is set below.
       .self_type_id = SemIR::TypeId::None,
       .inheritance_kind = SemIR::ClassFields::Final,
       // TODO: Handle the case where there's control flow in the alternatives.
       // For example:
       //
       //   choice C {
       //     Alt(x: if true then i32 else f64),
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
      class_decl_id, SemIR::NameId::None, class_info.parent_scope_id);
  class_decl.class_id = context.classes().Add(class_info);
  if (class_info.has_parameters()) {
    class_decl.type_id = GetGenericClassType(
        context, class_decl.class_id, context.scope_stack().PeekSpecificId());
  }

  ReplaceInstBeforeConstantUse(context, class_decl_id, class_decl);

  // We had to construct the `ClassId` from `Class` in order to build the `Self`
  // type below. But it needs to be written back to the `Class` in the
  // ValueStore, not the local variable. This gives a mutable reference to the
  // `Class` in the ValueStore.
  SemIR::Class& mut_class = context.classes().Get(class_decl.class_id);
  // Build the `Self` type using the resulting type constant.
  auto self_type_id = context.types().GetTypeIdForTypeConstantId(
      TryEvalInst(context, SemIR::InstId::None,
                  SemIR::ClassType{.type_id = SemIR::TypeType::SingletonTypeId,
                                   .class_id = class_decl.class_id,
                                   .specific_id = self_specific_id}));
  mut_class.self_type_id = self_type_id;

  // Enter the choice scope.
  context.scope_stack().Push(class_decl_id, class_info.scope_id,
                             self_specific_id);
  // Checking the binding pattern for an alternative requires a non-empty stack.
  // We reuse the Choice token even though we're now checking an alternative
  // inside the Choice, since there's no better token to use.
  //
  //  TODO: The token here is _not_ `Choice` though, we shouldn't need to use
  //  that here. Either remove the need for a token or find a token (a new
  //  introducer?) for the alternative to name.
  context.decl_introducer_state_stack().Push<Lex::TokenKind::Choice>();
  StartGenericDefinition(context);

  context.name_scopes().AddRequiredName(
      class_info.scope_id, SemIR::NameId::SelfType,
      context.types().GetInstId(self_type_id));

  // Mark the beginning of the choice body.
  context.node_stack().Push(node_id, class_decl.class_id);

  CARBON_CHECK(context.choice_deferred_bindings().empty(),
               "Alternatives left behind in choice_deferred_bindings: {0}",
               context.choice_deferred_bindings().size());
  return true;
}

static auto AddChoiceAlternative(
    Context& context, Parse::NodeIdOneOf<Parse::ChoiceAlternativeListCommaId,
                                         Parse::ChoiceDefinitionId>
                          node_id) -> void {
  // Note, there is nothing like a ChoiceAlternativeIntroducer node, so no parse
  // node to pop here.
  auto name_component = PopNameComponent(context);
  if (name_component.param_patterns_id == SemIR::InstBlockId::Empty) {
    // Treat an empty parameter list the same as no parameter list.
    //
    // TODO: The current design suggests that we want Foo() to result in a
    // member function `ChoiceType.Foo()`, and `Foo` to result in a member
    // constant `ChoiceType.Foo`, but that only one of the two is allowed in a
    // single choice type. See
    // https://github.com/carbon-language/carbon-lang/blob/trunk/docs/design/sum_types.md#user-defined-sum-types.
    // For now they are not treated differently and both resolve to a member
    // constant.
    context.TODO(name_component.params_loc_id,
                 "empty parameter list should make a member function");
    name_component.param_patterns_id = SemIR::InstBlockId::None;
  }
  if (name_component.param_patterns_id.has_value()) {
    context.TODO(name_component.params_loc_id,
                 "choice alternatives with parameters are not yet supported");
    return;
  }
  context.choice_deferred_bindings().push_back({node_id, name_component});
}

// Info about the Choice type, used to construct each alternative member of the
// class representing the Choice.
struct ChoiceInfo {
  // The `Self` type.
  SemIR::TypeId self_type_id;
  // The scope of the class for adding the alternatives to.
  SemIR::NameScopeId name_scope_id;
  // A struct type with the same fields as `Self`. Used to construct `Self`.
  SemIR::TypeId self_struct_type_id;
  // The type of the discriminant value.
  SemIR::TypeId discriminant_type_id;
  int num_alternative_bits;
};

// Builds a `let` binding for an alternative without parameters as a member of
// the resulting class for the Choice definition. If the alternative was `Alt`
// then the binding will be like:
// ```
//   let Alt: ChoiceType = <ChoiceType with Alt selected>;
// ```
static auto MakeLetBinding(Context& context, const ChoiceInfo& choice_info,
                           int alternative_index,
                           const Context::ChoiceDeferredBinding& binding)
    -> void {
  SemIR::InstId discriminant_value_id = [&] {
    if (choice_info.num_alternative_bits == 0) {
      return AddInst(context, SemIR::LocIdAndInst(
                                  binding.node_id,
                                  SemIR::TupleLiteral{
                                      .type_id = GetTupleType(context, {}),
                                      .elements_id = SemIR::InstBlockId::Empty,
                                  }));
    } else {
      return MakeIntLiteral(context, binding.node_id,
                            context.ints().Add(alternative_index));
    }
  }();
  discriminant_value_id =
      ConvertToValueOfType(context, binding.node_id, discriminant_value_id,
                           choice_info.discriminant_type_id);

  auto self_value_id = ConvertToValueOfType(
      context, binding.node_id,
      AddInst(context, SemIR::LocIdAndInst(
                           binding.node_id,
                           SemIR::StructLiteral{
                               .type_id = choice_info.self_struct_type_id,
                               .elements_id =
                                   [&] {
                                     context.inst_block_stack().Push();
                                     context.inst_block_stack().AddInstId(
                                         discriminant_value_id);
                                     return context.inst_block_stack().Pop();
                                   }(),
                           })),
      choice_info.self_type_id);

  auto entity_name_id = context.entity_names().Add(
      {.name_id = binding.name_component.name_id,
       .parent_scope_id = choice_info.name_scope_id});
  auto bind_name_id = AddInst(
      context, SemIR::LocIdAndInst(binding.node_id,
                                   SemIR::BindName{
                                       .type_id = choice_info.self_type_id,
                                       .entity_name_id = entity_name_id,
                                       .value_id = self_value_id,
                                   }));
  context.name_scopes()
      .Get(choice_info.name_scope_id)
      .AddRequired({.name_id = binding.name_component.name_id,
                    .result = SemIR::ScopeLookupResult::MakeFound(
                        bind_name_id, SemIR::AccessKind::Public)});
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
      // Even though there's no bits needed, we add an empty field. We want to
      // prevent constructing the Choice from an empty struct literal instead of
      // going through an alternative. And in the case there is no alternative,
      // then there's no way to construct the Choice (which can be a useful
      // type).
      //
      // TODO: Find a way to produce a better diagnostic, and not require an
      // empty field.
      return GetTupleType(context, {});
    } else {
      return MakeIntType(context, node_id, SemIR::IntKind::Unsigned,
                         context.ints().Add(num_alternative_bits));
    }
  }();

  llvm::SmallVector<SemIR::StructTypeField, 1> struct_type_fields;
  struct_type_fields.push_back({
      .name_id = SemIR::NameId::ChoiceDiscriminant,
      .type_id = discriminant_type_id,
  });
  auto fields_id =
      context.struct_type_fields().AddCanonical(struct_type_fields);
  auto choice_witness_id =
      AddInst(context, node_id,
              SemIR::CompleteTypeWitness{
                  .type_id = GetSingletonType(
                      context, SemIR::WitnessType::SingletonInstId),
                  .object_repr_id = GetStructType(context, fields_id)});
  // Note: avoid storing a reference to the returned Class, since it may be
  // invalidated by other type constructions.
  context.classes().Get(class_id).complete_type_witness_id = choice_witness_id;

  auto self_type_id = context.classes().Get(class_id).self_type_id;
  auto name_scope_id = context.classes().Get(class_id).scope_id;

  auto self_struct_type_id = GetStructType(
      context, context.struct_type_fields().AddCanonical(struct_type_fields));

  for (auto [i, deferred_binding] :
       llvm::enumerate(context.choice_deferred_bindings())) {
    MakeLetBinding(context,
                   ChoiceInfo{.self_type_id = self_type_id,
                              .name_scope_id = name_scope_id,
                              .self_struct_type_id = self_struct_type_id,
                              .discriminant_type_id = discriminant_type_id,
                              .num_alternative_bits = num_alternative_bits},
                   i, deferred_binding);
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
