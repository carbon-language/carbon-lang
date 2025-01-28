// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/return.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

static auto HandleAnyBindingPattern(Context& context, Parse::NodeId node_id,
                                    Parse::NodeKind node_kind) -> bool {
  // TODO: split this into smaller, more focused functions.
  auto [type_node, parsed_type_id] = context.node_stack().PopExprWithNodeId();
  auto [cast_type_inst_id, cast_type_id] =
      ExprAsType(context, type_node, parsed_type_id);

  // TODO: Handle `_` bindings.

  SemIR::ExprRegionId type_expr_region_id =
      context.EndSubpatternAsExpr(cast_type_inst_id);

  // Every other kind of pattern binding has a name.
  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  // Determine whether we're handling an associated constant. These share the
  // syntax for a compile-time binding, but don't behave like other compile-time
  // bindings.
  // TODO: Consider using a different parse node kind to make this easier.
  bool is_associated_constant = false;
  bool is_generic = node_kind == Parse::NodeKind::CompileTimeBindingPattern;
  if (is_generic) {
    auto inst_id = context.scope_stack().PeekInstId();
    is_associated_constant = inst_id.has_value() &&
                             context.insts().Is<SemIR::InterfaceDecl>(inst_id);
  }

  bool needs_compile_time_binding = is_generic && !is_associated_constant;

  const DeclIntroducerState& introducer =
      context.decl_introducer_state_stack().innermost();

  auto make_binding_pattern = [&]() -> SemIR::InstId {
    auto bind_id = SemIR::InstId::None;
    auto binding_pattern_id = SemIR::InstId::None;
    // TODO: Eventually the name will need to support associations with other
    // scopes, but right now we don't support qualified names here.
    auto entity_name_id = context.entity_names().Add(
        {.name_id = name_id,
         .parent_scope_id = context.scope_stack().PeekNameScopeId(),
         // TODO: Don't allocate a compile-time binding index for an associated
         // constant declaration.
         .bind_index = needs_compile_time_binding
                           ? context.scope_stack().AddCompileTimeBinding()
                           : SemIR::CompileTimeBindIndex::None});
    if (is_generic) {
      // TODO: Create a `BindTemplateName` instead inside a `template` pattern.
      bind_id = context.AddInstInNoBlock(SemIR::LocIdAndInst(
          name_node, SemIR::BindSymbolicName{.type_id = cast_type_id,
                                             .entity_name_id = entity_name_id,
                                             .value_id = SemIR::InstId::None}));
      binding_pattern_id =
          context.AddPatternInst<SemIR::SymbolicBindingPattern>(
              name_node,
              {.type_id = cast_type_id, .entity_name_id = entity_name_id});
    } else {
      bind_id = context.AddInstInNoBlock(SemIR::LocIdAndInst(
          name_node, SemIR::BindName{.type_id = cast_type_id,
                                     .entity_name_id = entity_name_id,
                                     .value_id = SemIR::InstId::None}));
      binding_pattern_id = context.AddPatternInst<SemIR::BindingPattern>(
          name_node,
          {.type_id = cast_type_id, .entity_name_id = entity_name_id});
    }

    // Add name to lookup immediately, so it can be used in the rest of the
    // enclosing pattern.
    if (needs_compile_time_binding) {
      context.scope_stack().PushCompileTimeBinding(bind_id);
    }
    auto name_context =
        context.decl_name_stack().MakeUnqualifiedName(node_id, name_id);
    context.decl_name_stack().AddNameOrDiagnose(
        name_context, bind_id, introducer.modifier_set.GetAccessKind());
    context.full_pattern_stack().AddBindName(name_id);

    bool inserted = context.bind_name_map()
                        .Insert(binding_pattern_id,
                                {.bind_name_id = bind_id,
                                 .type_expr_region_id = type_expr_region_id})
                        .is_inserted();
    CARBON_CHECK(inserted);
    return binding_pattern_id;
  };

  // A `self` binding can only appear in an implicit parameter list.
  if (name_id == SemIR::NameId::SelfValue &&
      !context.node_stack().PeekIs(Parse::NodeKind::ImplicitParamListStart)) {
    CARBON_DIAGNOSTIC(
        SelfOutsideImplicitParamList, Error,
        "`self` can only be declared in an implicit parameter list");
    context.emitter().Emit(node_id, SelfOutsideImplicitParamList);
  }

  // A `var` binding in a class scope declares a field, not a true binding,
  // so we handle it separately.
  if (auto parent_class_decl = context.GetCurrentScopeAs<SemIR::ClassDecl>();
      parent_class_decl.has_value() &&
      node_kind == Parse::NodeKind::VarBindingPattern) {
    cast_type_id = context.AsConcreteType(
        cast_type_id, type_node,
        [&] {
          CARBON_DIAGNOSTIC(IncompleteTypeInFieldDecl, Error,
                            "field has incomplete type {0}", SemIR::TypeId);
          return context.emitter().Build(type_node, IncompleteTypeInFieldDecl,
                                         cast_type_id);
        },
        [&] {
          CARBON_DIAGNOSTIC(AbstractTypeInFieldDecl, Error,
                            "field has abstract type {0}", SemIR::TypeId);
          return context.emitter().Build(type_node, AbstractTypeInFieldDecl,
                                         cast_type_id);
        });
    auto binding_id =
        is_generic
            ? Parse::NodeId::None
            : context.parse_tree().As<Parse::VarBindingPatternId>(node_id);
    auto& class_info = context.classes().Get(parent_class_decl->class_id);
    auto field_type_id =
        context.GetUnboundElementType(class_info.self_type_id, cast_type_id);
    auto field_id = context.AddInst<SemIR::FieldDecl>(
        binding_id, {.type_id = field_type_id,
                     .name_id = name_id,
                     .index = SemIR::ElementIndex::None});
    context.field_decls_stack().AppendToTop(field_id);

    context.node_stack().Push(node_id, field_id);
    auto name_context =
        context.decl_name_stack().MakeUnqualifiedName(node_id, name_id);
    context.decl_name_stack().AddNameOrDiagnose(
        name_context, field_id, introducer.modifier_set.GetAccessKind());
    return true;
  }

  // A binding in an interface scope declares an associated constant, not a
  // true binding, so we handle it separately.
  if (auto parent_interface_decl =
          context.GetCurrentScopeAs<SemIR::InterfaceDecl>();
      parent_interface_decl.has_value() && is_generic) {
    cast_type_id = context.AsCompleteType(cast_type_id, type_node, [&] {
      CARBON_DIAGNOSTIC(IncompleteTypeInAssociatedDecl, Error,
                        "associated constant has incomplete type {0}",
                        SemIR::TypeId);
      return context.emitter().Build(type_node, IncompleteTypeInAssociatedDecl,
                                     cast_type_id);
    });

    SemIR::AssociatedConstantDecl assoc_const_decl = {
        .type_id = cast_type_id,
        .assoc_const_id = SemIR::AssociatedConstantId::None,
        .decl_block_id = SemIR::InstBlockId::None};
    auto decl_id = context.AddPlaceholderInstInNoBlock(SemIR::LocIdAndInst(
        context.parse_tree().As<Parse::CompileTimeBindingPatternId>(node_id),
        assoc_const_decl));
    assoc_const_decl.assoc_const_id = context.associated_constants().Add(
        {.name_id = name_id,
         .parent_scope_id = context.scope_stack().PeekNameScopeId(),
         .decl_id = decl_id,
         .generic_id = SemIR::GenericId::None,
         .default_value_id = SemIR::InstId::None});
    context.ReplaceInstBeforeConstantUse(decl_id, assoc_const_decl);

    context.node_stack().Push(node_id, decl_id);
    return true;
  }

  // Allocate an instruction of the appropriate kind, linked to the name for
  // error locations.
  switch (context.full_pattern_stack().CurrentKind()) {
    case FullPatternStack::Kind::ImplicitParamList:
    case FullPatternStack::Kind::ExplicitParamList: {
      // Parameters can have incomplete types in a function declaration, but not
      // in a function definition. We don't know which kind we have here.
      // TODO: A tuple pattern can appear in other places than function
      // parameters.
      auto param_pattern_id = SemIR::InstId::None;
      bool had_error = false;
      switch (introducer.kind) {
        case Lex::TokenKind::Fn: {
          if (context.full_pattern_stack().CurrentKind() ==
                  FullPatternStack::Kind::ImplicitParamList &&
              !(is_generic || name_id == SemIR::NameId::SelfValue)) {
            CARBON_DIAGNOSTIC(
                ImplictParamMustBeConstant, Error,
                "implicit parameters of functions must be constant or `self`");
            context.emitter().Emit(node_id, ImplictParamMustBeConstant);
            had_error = true;
          }
          break;
        }
        case Lex::TokenKind::Class:
        case Lex::TokenKind::Impl:
        case Lex::TokenKind::Interface: {
          if (name_id == SemIR::NameId::SelfValue) {
            CARBON_DIAGNOSTIC(SelfParameterNotAllowed, Error,
                              "`self` parameter only allowed on functions");
            context.emitter().Emit(node_id, SelfParameterNotAllowed);
            had_error = true;
          } else if (!is_generic) {
            CARBON_DIAGNOSTIC(GenericParamMustBeConstant, Error,
                              "parameters of generic types must be constant");
            context.emitter().Emit(node_id, GenericParamMustBeConstant);
            had_error = true;
          }
          break;
        }
        default:
          break;
      }
      if (had_error) {
        context.AddNameToLookup(name_id, SemIR::ErrorInst::SingletonInstId);
        // Replace the parameter with `ErrorInst` so that we don't try
        // constructing a generic based on it.
        param_pattern_id = SemIR::ErrorInst::SingletonInstId;
      } else {
        auto pattern_inst_id = make_binding_pattern();
        param_pattern_id = context.AddPatternInst<SemIR::ValueParamPattern>(
            node_id,
            {
                .type_id = context.insts().Get(pattern_inst_id).type_id(),
                .subpattern_id = pattern_inst_id,
                .runtime_index = is_generic ? SemIR::RuntimeParamIndex::None
                                            : SemIR::RuntimeParamIndex::Unknown,
            });
      }
      context.node_stack().Push(node_id, param_pattern_id);
      break;
    }

    case FullPatternStack::Kind::NameBindingDecl: {
      auto incomplete_diagnoser = [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInBindingDecl, Error,
                          "binding pattern has incomplete type {0} in name "
                          "binding declaration",
                          InstIdAsType);
        return context.emitter().Build(type_node, IncompleteTypeInBindingDecl,
                                       cast_type_inst_id);
      };
      if (node_kind == Parse::NodeKind::VarBindingPattern) {
        cast_type_id = context.AsConcreteType(
            cast_type_id, type_node, incomplete_diagnoser, [&] {
              CARBON_DIAGNOSTIC(
                  AbstractTypeInVarPattern, Error,
                  "binding pattern has abstract type {0} in `var` "
                  "pattern",
                  SemIR::TypeId);
              return context.emitter().Build(
                  type_node, AbstractTypeInVarPattern, cast_type_id);
            });
      } else {
        cast_type_id = context.AsCompleteType(cast_type_id, type_node,
                                              incomplete_diagnoser);
      }
      auto binding_pattern_id = make_binding_pattern();
      if (node_kind == Parse::NodeKind::VarBindingPattern) {
        CARBON_CHECK(!is_generic);

        if (introducer.modifier_set.HasAnyOf(KeywordModifierSet::Returned)) {
          // TODO: Should we check this for the `var` as a whole, rather than
          // for the name binding?
          auto bind_id = context.bind_name_map()
                             .Lookup(binding_pattern_id)
                             .value()
                             .bind_name_id;
          RegisterReturnedVar(context,
                              introducer.modifier_node_id(ModifierOrder::Decl),
                              type_node, cast_type_id, bind_id);
        }
      }
      context.node_stack().Push(node_id, binding_pattern_id);
      break;
    }
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::LetBindingPatternId node_id)
    -> bool {
  return HandleAnyBindingPattern(context, node_id,
                                 Parse::NodeKind::LetBindingPattern);
}

auto HandleParseNode(Context& context, Parse::VarBindingPatternId node_id)
    -> bool {
  return HandleAnyBindingPattern(context, node_id,
                                 Parse::NodeKind::VarBindingPattern);
}

auto HandleParseNode(Context& context,
                     Parse::CompileTimeBindingPatternId node_id) -> bool {
  auto node_kind = Parse::NodeKind::CompileTimeBindingPattern;
  if (context.decl_introducer_state_stack().innermost().kind ==
      Lex::TokenKind::Let) {
    // Disallow `let` outside of function and interface definitions.
    // TODO: Find a less brittle way of doing this. A `scope_inst_id` of `None`
    // can represent a block scope, but is also used for other kinds of scopes
    // that aren't necessarily part of an interface or function decl.
    auto scope_inst_id = context.scope_stack().PeekInstId();
    if (scope_inst_id.has_value()) {
      auto scope_inst = context.insts().Get(scope_inst_id);
      if (!scope_inst.Is<SemIR::InterfaceDecl>() &&
          !scope_inst.Is<SemIR::FunctionDecl>()) {
        context.TODO(
            node_id,
            "`let` compile time binding outside function or interface");
        node_kind = Parse::NodeKind::LetBindingPattern;
      }
    }
  }

  return HandleAnyBindingPattern(context, node_id, node_kind);
}

auto HandleParseNode(Context& context, Parse::AddrId node_id) -> bool {
  auto param_pattern_id = context.node_stack().PopPattern();
  if (SemIR::Function::GetNameFromPatternId(
          context.sem_ir(), param_pattern_id) == SemIR::NameId::SelfValue) {
    auto pointer_type = context.types().TryGetAs<SemIR::PointerType>(
        context.insts().Get(param_pattern_id).type_id());
    if (pointer_type) {
      auto addr_pattern_id = context.AddPatternInst<SemIR::AddrPattern>(
          node_id, {.type_id = SemIR::AutoType::SingletonTypeId,
                    .inner_id = param_pattern_id});
      context.node_stack().Push(node_id, addr_pattern_id);
    } else {
      CARBON_DIAGNOSTIC(
          AddrOnNonPointerType, Error,
          "`addr` can only be applied to a binding with a pointer type");
      context.emitter().Emit(node_id, AddrOnNonPointerType);
      context.node_stack().Push(node_id, param_pattern_id);
    }
  } else {
    CARBON_DIAGNOSTIC(AddrOnNonSelfParam, Error,
                      "`addr` can only be applied to a `self` parameter");
    context.emitter().Emit(TokenOnly(node_id), AddrOnNonSelfParam);
    context.node_stack().Push(node_id, param_pattern_id);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::TemplateId node_id) -> bool {
  return context.TODO(node_id, "HandleTemplate");
}

}  // namespace Carbon::Check
