// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/action.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/period_self.h"
#include "toolchain/check/return.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/check/unused.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::UnderscoreNameId node_id)
    -> bool {
  context.node_stack().Push(node_id, SemIR::NameId::Underscore);
  return true;
}

// Returns the `InstKind` corresponding to the pattern's `NodeKind`.
static auto GetLeafBindingPatternInstKind(Parse::NodeKind node_kind,
                                          bool is_ref) -> SemIR::InstKind {
  switch (node_kind) {
    case Parse::NodeKind::CompileTimeBindingPattern:
      return SemIR::InstKind::SymbolicBindingPattern;
    case Parse::NodeKind::LetBindingPattern:
      return is_ref ? SemIR::InstKind::RefBindingPattern
                    : SemIR::InstKind::ValueBindingPattern;
    case Parse::NodeKind::VarBindingPattern:
      return SemIR::InstKind::RefBindingPattern;
    default:
      CARBON_FATAL("Unexpected node kind: {0}", node_kind);
  }
}

// Returns true if a parameter is valid in the given `introducer_kind`.
static auto IsValidParamForIntroducer(Context& context, Parse::NodeId node_id,
                                      SemIR::NameId name_id,
                                      Lex::TokenKind introducer_kind,
                                      bool is_generic) -> bool {
  switch (introducer_kind) {
    case Lex::TokenKind::Fn: {
      // `self` in the implicit parameter list is diagnosed separately (see
      // `SelfInImplicitParamList`), so skip it here to avoid a redundant
      // diagnostic.
      if (context.full_pattern_stack().CurrentKind() ==
              FullPatternStack::Kind::ImplicitParamList &&
          !(is_generic || name_id == SemIR::NameId::SelfValue)) {
        CARBON_DIAGNOSTIC(ImplictParamMustBeConstant, Error,
                          "implicit parameters of functions must be constant");
        context.emitter().Emit(node_id, ImplictParamMustBeConstant);
        return false;
      }
      // Parameters can have incomplete types in a function declaration, but not
      // in a function definition. We don't know which kind we have here, so
      // don't validate it.
      return true;
    }
    case Lex::TokenKind::Choice:
      if (context.scope_stack().PeekInstId().has_value()) {
        // We are building a pattern for a choice alternative, not the
        // choice type itself.

        // Implicit param lists are prevented during parse.
        CARBON_CHECK(context.full_pattern_stack().CurrentKind() !=
                         FullPatternStack::Kind::ImplicitParamList,
                     "choice alternative with implicit parameters");
        // Don't fall through to the `Class` logic for choice alternatives.
        return true;
      }
      [[fallthrough]];
    case Lex::TokenKind::Class:
    case Lex::TokenKind::Impl:
    case Lex::TokenKind::Interface: {
      if (name_id == SemIR::NameId::SelfValue) {
        CARBON_DIAGNOSTIC(SelfParameterNotAllowed, Error,
                          "`self` parameter only allowed on functions");
        context.emitter().Emit(node_id, SelfParameterNotAllowed);
        return false;
      }
      if (!is_generic) {
        CARBON_DIAGNOSTIC(GenericParamMustBeConstant, Error,
                          "parameters of generic types must be constant");
        context.emitter().Emit(node_id, GenericParamMustBeConstant);
        return false;
      }
      return true;
    }
    default:
      return true;
  }
}

namespace {
// Information about the expression in the type position of a binding pattern,
// i.e. the position following the `:`/`:?`/`:!` separator. Note that this
// expression may be interpreted as a type or a form, depending on the binding
// kind.
struct BindingPatternTypeInfo {
  // The parse node representing the expression. For a `self` binding with an
  // omitted type this is the binding pattern node itself, since there is no
  // separate type expression.
  Parse::NodeId node_id;
  // The inst representing the converted value of that expression. For a `:?`
  // binding the expression is converted to type `Core.Form`; otherwise it is
  // converted to type `type`.
  SemIR::InstId inst_id;
  // For a `:?` binding this is the type component of the form denoted by
  // `inst_id`. Otherwise this is the type denoted by `inst_id`.
  SemIR::TypeId type_component_id;
};
}  // namespace

// Handle the type position of a binding pattern. For a `self` binding with an
// omitted type, `self_type_inst_id` is the synthesized `Self` type expression
// and there is no type expression on the node stack to pop.
static auto HandleAnyBindingPatternType(Context& context,
                                        Parse::NodeId binding_node_id,
                                        Parse::NodeKind node_kind,
                                        SemIR::InstId self_type_inst_id)
    -> BindingPatternTypeInfo {
  if (self_type_inst_id.has_value()) {
    auto as_type = ExprAsType(context, binding_node_id, self_type_inst_id);
    return {.node_id = binding_node_id,
            .inst_id = as_type.inst_id,
            .type_component_id = as_type.type_id};
  }

  auto [node_id, original_inst_id] = context.node_stack().PopExprWithNodeId();

  if (node_kind == Parse::FormBindingPattern::Kind) {
    auto as_form = FormExprAsForm(context, node_id, original_inst_id);
    return {.node_id = node_id,
            .inst_id = as_form.form_inst_id,
            .type_component_id = as_form.type_component_id};
  } else {
    auto as_type = ExprAsType(context, node_id, original_inst_id);
    return {.node_id = node_id,
            .inst_id = as_type.inst_id,
            .type_component_id = as_type.type_id};
  }
}

// TODO: make this function shorter by factoring pieces out.
static auto HandleAnyBindingPattern(
    Context& context, Parse::NodeId node_id, Parse::NodeKind node_kind,
    bool is_unused = false,
    SemIR::InstId self_type_inst_id = SemIR::InstId::None) -> bool {
  auto type_expr = HandleAnyBindingPatternType(context, node_id, node_kind,
                                               self_type_inst_id);
  if (context.types()
          .GetAsInst(type_expr.type_component_id)
          .Is<SemIR::TypeComponentOf>()) {
    return context.TODO(node_id, "Support symbolic form bindings");
  }

  SemIR::ExprRegionId type_expr_region_id =
      ConsumeSubpatternExpr(context, type_expr.inst_id);

  // The name in a generic binding may be wrapped in `template`.
  bool is_generic = node_kind == Parse::NodeKind::CompileTimeBindingPattern;
  bool is_template =
      context.node_stack()
          .PopAndDiscardSoloNodeIdIf<Parse::NodeKind::TemplateBindingName>();
  // A non-generic template binding is diagnosed by the parser.
  is_template &= is_generic;

  // The name in a runtime binding may be wrapped in `ref`.
  bool is_ref =
      context.node_stack()
          .PopAndDiscardSoloNodeIdIf<Parse::NodeKind::RefBindingName>();

  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  const DeclIntroducerState& introducer =
      context.decl_introducer_state_stack().innermost();

  auto form_id = node_kind == Parse::FormBindingPattern::Kind
                     ? type_expr.inst_id
                     : SemIR::InstId::None;

  // Adds a binding pattern for `node_id`, with the given kind and subpattern,
  // and adds its name to the current context. The subpattern must not be
  // provided unless the kind is `FormBindingPattern`.
  auto make_binding_pattern = [&](SemIR::InstKind kind,
                                  SemIR::InstId subpattern_id =
                                      SemIR::InstId::None) -> SemIR::InstId {
    // TODO: Eventually the name will need to support associations with other
    // scopes, but right now we don't support qualified names here.
    auto phase = BindingPhase::Runtime;
    if (kind == SemIR::SymbolicBindingPattern::Kind) {
      phase = is_template ? BindingPhase::Template : BindingPhase::Symbolic;
    }
    auto binding = AddBindingPattern(
        context, node_id, type_expr_region_id, type_expr.type_component_id,
        {.kind = kind,
         .type_id = GetPatternType(context, type_expr.type_component_id),
         .entity_name_id =
             AddBindingEntityName(context, name_id, form_id, is_unused, phase),
         .subpattern_id = subpattern_id});

    // TODO: If `is_generic`, then `binding.bind_id is a SymbolicBinding. Subst
    // the `.Self` of type `type` in the `cast_type_id` type (a `FacetType`)
    // with the `binding.bind_id` itself, and build a new pattern with that.
    // This is kind of cyclical. So we need to reuse the EntityNameId, which
    // will also reuse the CompileTimeBinding for the new SymbolicBinding.

    if (name_id != SemIR::NameId::Underscore) {
      // Add name to lookup immediately, so it can be used in the rest of the
      // enclosing pattern.
      auto name_context =
          context.decl_name_stack().MakeUnqualifiedName(name_node, name_id);
      context.decl_name_stack().AddNameOrDiagnose(
          name_context, binding.bind_id,
          introducer.modifier_set.GetAccessKind());
      context.full_pattern_stack().AddBindName(name_id);
    }

    return binding.pattern_id;
  };

  auto abstract_diagnostic_context = [&](auto& builder) {
    CARBON_DIAGNOSTIC(AbstractTypeInVarPattern, Context,
                      "binding pattern has abstract type {0} in `var` "
                      "pattern",
                      SemIR::TypeId);
    builder.Context(type_expr.node_id, AbstractTypeInVarPattern,
                    type_expr.type_component_id);
  };

  // A `self` binding must be the first parameter in the explicit parameter
  // list (see proposal #7016). Here we can reject `self` in the implicit
  // parameter list or outside any parameter list; that it must be *first* in
  // the explicit list is checked once the full list is known (see
  // `BuildFunctionDecl`).
  if (name_id == SemIR::NameId::SelfValue) {
    switch (context.full_pattern_stack().CurrentKind()) {
      case FullPatternStack::Kind::ExplicitParamList:
        break;
      case FullPatternStack::Kind::ImplicitParamList: {
        CARBON_DIAGNOSTIC(
            SelfInImplicitParamList, Error,
            "`self` must be declared in the explicit parameter list");
        context.emitter().Emit(node_id, SelfInImplicitParamList);
        break;
      }
      default: {
        CARBON_DIAGNOSTIC(SelfOutsideParamList, Error,
                          "`self` can only be declared in a parameter list");
        context.emitter().Emit(node_id, SelfOutsideParamList);
        break;
      }
    }
  }

  if (node_kind == Parse::NodeKind::CompileTimeBindingPattern &&
      introducer.kind == Lex::TokenKind::Let) {
    // TODO: We should re-evaluate the contents of the eval block in a
    // synthesized specific to form these values, in order to propagate the
    // values.
    return context.TODO(node_id,
                        "local `let :!` bindings are currently unsupported");
  }

  // Allocate an instruction of the appropriate kind, linked to the name for
  // error locations.
  switch (context.full_pattern_stack().CurrentKind()) {
    case FullPatternStack::Kind::ImplicitParamList:
    case FullPatternStack::Kind::ExplicitParamList: {
      if (!IsValidParamForIntroducer(context, node_id, name_id, introducer.kind,
                                     is_generic)) {
        if (name_id != SemIR::NameId::Underscore) {
          AddNameToLookup(context, name_id, SemIR::ErrorInst::InstId);
        }
        // Replace the parameter with `ErrorInst` so that we don't try
        // constructing a generic based on it.
        context.node_stack().Push(node_id, SemIR::ErrorInst::InstId);
        break;
      }

      // Using `AsConcreteType` here causes `fn F[var self: Self]();`
      // to fail since `Self` is an incomplete type.
      if (node_kind == Parse::NodeKind::VarBindingPattern) {
        auto [unqualified_type_id, qualifiers] =
            context.types().GetUnqualifiedTypeAndQualifiers(
                type_expr.type_component_id);
        if ((qualifiers & SemIR::TypeQualifiers::Partial) !=
                SemIR::TypeQualifiers::Partial &&
            context.types().Is<SemIR::ClassType>(unqualified_type_id)) {
          auto class_type =
              context.types().GetAs<SemIR::ClassType>(unqualified_type_id);
          auto& class_info = context.classes().Get(class_type.class_id);
          if (class_info.inheritance_kind ==
              SemIR::Class::InheritanceKind::Abstract) {
            Diagnostics::ContextScope scope(&context.emitter(),
                                            abstract_diagnostic_context);
            DiagnoseAbstractClass(context, class_type.class_id,
                                  /*direct_use=*/true);
            type_expr.type_component_id = SemIR::ErrorInst::TypeId;
          }
        }
      }

      auto result_inst_id = SemIR::InstId::None;
      switch (node_kind) {
        // A binding pattern in a function signature is a `Call` parameter
        // unless it's nested inside a `var` pattern (because then the
        // enclosing `var` pattern is), or it's a compile-time binding pattern
        // (because then it's not passed to the `Call` inst).
        case Parse::NodeKind::LetBindingPattern:
        case Parse::NodeKind::FormBindingPattern: {
          auto param_pattern_id = SemIR::InstId::None;
          auto pattern_type_id =
              GetPatternType(context, type_expr.type_component_id);
          if (is_ref) {
            param_pattern_id = AddInst<SemIR::RefParamPattern>(
                context, node_id,
                {.type_id = pattern_type_id, .pretty_name_id = name_id});
          } else if (node_kind == Parse::NodeKind::FormBindingPattern) {
            auto pattern_type_inst_id =
                context.types().GetTypeInstId(pattern_type_id);
            param_pattern_id = HandleAction<SemIR::FormParamPatternAction>(
                context,
                context.parse_tree()
                    .As<Parse::NodeIdForKind<
                        Parse::NodeKind::FormBindingPattern>>(node_id),
                pattern_type_inst_id,
                {.type_id = SemIR::InstType::TypeId,
                 .form_id = form_id,
                 .pretty_name_id = name_id});
          } else {
            param_pattern_id = AddInst<SemIR::ValueParamPattern>(
                context, node_id,
                {.type_id = pattern_type_id, .pretty_name_id = name_id});
          }
          if (param_pattern_id == SemIR::ErrorInst::InstId) {
            result_inst_id = SemIR::ErrorInst::InstId;
            break;
          }
          result_inst_id = make_binding_pattern(
              SemIR::WrapperBindingPattern::Kind, param_pattern_id);
          break;
        }
        case Parse::NodeKind::VarBindingPattern:
          result_inst_id = make_binding_pattern(SemIR::RefBindingPattern::Kind);
          break;
        case Parse::NodeKind::CompileTimeBindingPattern:
          result_inst_id =
              make_binding_pattern(SemIR::SymbolicBindingPattern::Kind);
          break;
        default:
          CARBON_FATAL("Unexpected node kind {0}", node_kind);
      }
      context.node_stack().Push(node_id, result_inst_id);
      break;
    }

    case FullPatternStack::Kind::NameBindingDecl:
    case FullPatternStack::Kind::ClassScopeVarDecl: {
      if (node_kind == Parse::NodeKind::FormBindingPattern) {
        return context.TODO(node_id, "support local form bindings");
      }
      auto incomplete_diagnostic_context = [&](auto& builder) {
        CARBON_DIAGNOSTIC(IncompleteTypeInBindingDecl, Context,
                          "binding pattern has incomplete type {0} in name "
                          "binding declaration",
                          InstIdAsType);
        builder.Context(type_expr.node_id, IncompleteTypeInBindingDecl,
                        type_expr.inst_id);
      };
      if (node_kind == Parse::NodeKind::VarBindingPattern) {
        if (!RequireConcreteType(
                context, type_expr.type_component_id, type_expr.node_id,
                incomplete_diagnostic_context, abstract_diagnostic_context)) {
          type_expr.type_component_id = SemIR::ErrorInst::TypeId;
        }
      } else {
        if (!RequireCompleteType(context, type_expr.type_component_id,
                                 type_expr.node_id,
                                 incomplete_diagnostic_context)) {
          type_expr.type_component_id = SemIR::ErrorInst::TypeId;
        }
      }

      auto binding_pattern_id = make_binding_pattern(
          GetLeafBindingPatternInstKind(node_kind, is_ref));
      if (node_kind == Parse::NodeKind::VarBindingPattern) {
        CARBON_CHECK(!is_generic);

        if (introducer.modifier_set.HasAnyOf(KeywordModifierSet::Returned)) {
          // TODO: Should we check this for the `var` as a whole, rather than
          // for the name binding?
          auto bind_id = context.bind_name_map()
                             .Lookup(binding_pattern_id)
                             .value()
                             .bind_name_id;
          RegisterReturnedVar(
              context, introducer.modifier_node_id(ModifierOrder::Decl),
              type_expr.node_id, type_expr.type_component_id, bind_id, name_id);
        }
      }
      context.node_stack().Push(node_id, binding_pattern_id);
      break;
    }

    case FullPatternStack::Kind::NotInEitherParamList:
      CARBON_FATAL("Unreachable");
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::LetBindingPatternId node_id)
    -> bool {
  return HandleAnyBindingPattern(context, node_id,
                                 Parse::NodeKind::LetBindingPattern);
}

auto HandleParseNode(Context& context, Parse::SelfBindingPatternId node_id)
    -> bool {
  // A `self` binding with an omitted type behaves like `self: Self`. There is
  // no type expression in the parse tree, so synthesize a reference to `Self`
  // into the current subpattern region and feed it in as the binding's type.
  auto self_type =
      LookupUnqualifiedName(context, node_id, SemIR::NameId::SelfType);
  auto self_type_inst_id = BuildNameRef(
      context, node_id, SemIR::NameId::SelfType,
      self_type.scope_result.target_inst_id(), self_type.specific_id);
  return HandleAnyBindingPattern(context, node_id,
                                 Parse::NodeKind::LetBindingPattern,
                                 /*is_unused=*/false, self_type_inst_id);
}

auto HandleParseNode(Context& context, Parse::VarBindingPatternId node_id)
    -> bool {
  return HandleAnyBindingPattern(context, node_id,
                                 Parse::NodeKind::VarBindingPattern);
}

auto HandleParseNode(Context& context, Parse::FormBindingPatternId node_id)
    -> bool {
  return HandleAnyBindingPattern(context, node_id,
                                 Parse::NodeKind::FormBindingPattern);
}

auto HandleParseNode(Context& context,
                     Parse::CompileTimeBindingPatternStartId node_id) -> bool {
  // Make a scope to contain the `.Self` facet value for use in the type of the
  // compile time binding. This is popped when handling the
  // CompileTimeBindingPatternId.
  context.scope_stack().PushForSameRegion();
  MakePeriodSelfFacetValue(context, node_id, GetEmptyFacetType(context));
  return true;
}

auto HandleParseNode(Context& context,
                     Parse::CompileTimeBindingPatternId node_id) -> bool {
  // Pop the `.Self` facet value name introduced by the
  // CompileTimeBindingPatternStart.
  context.scope_stack().Pop(/*check_unused=*/true);

  auto node_kind = Parse::NodeKind::CompileTimeBindingPattern;
  const DeclIntroducerState& introducer =
      context.decl_introducer_state_stack().innermost();
  if (introducer.kind == Lex::TokenKind::Let) {
    // Disallow `let` outside of function and interface definitions.
    // TODO: Find a less brittle way of doing this. A `scope_inst_id` of `None`
    // can represent a block scope, but is also used for other kinds of scopes
    // that aren't necessarily part of a function decl.
    // We don't need to check if the scope is an interface here as this is
    // already caught in the parse phase by the separated associated constant
    // logic.
    auto scope_inst_id = context.scope_stack().PeekInstId();
    if (scope_inst_id.has_value()) {
      auto scope_inst = context.insts().Get(scope_inst_id);
      if (!scope_inst.Is<SemIR::FunctionDecl>()) {
        context.TODO(
            node_id,
            "`let` compile time binding outside function or interface");
        node_kind = Parse::NodeKind::LetBindingPattern;
      }
    }
  }

  return HandleAnyBindingPattern(context, node_id, node_kind);
}

auto HandleParseNode(Context& context,
                     Parse::AssociatedConstantNameAndTypeId node_id) -> bool {
  auto [type_node, parsed_type_id] = context.node_stack().PopExprWithNodeId();
  auto [cast_type_inst_id, cast_type_id] =
      ExprAsType(context, type_node, parsed_type_id);

  auto region_id = ConsumeSubpatternExpr(context, cast_type_inst_id);
  // TODO: Should we be tracking this somewhere?
  (void)region_id;

  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  if (name_id == SemIR::NameId::Underscore) {
    // The action item here may be to document this as not allowed, and
    // add a proper diagnostic.
    context.TODO(node_id, "_ used as associated constant name");
  }

  SemIR::AssociatedConstantDecl assoc_const_decl = {
      .type_id = cast_type_id,
      .assoc_const_id = SemIR::AssociatedConstantId::None,
      .decl_block_id = SemIR::InstBlockId::None};
  auto decl_id =
      AddPlaceholderInstInNoBlock(context, node_id, assoc_const_decl);
  assoc_const_decl.assoc_const_id = context.associated_constants().Add(
      {.name_id = name_id,
       .parent_scope_id = context.scope_stack().PeekNameScopeId(),
       .decl_id = decl_id,
       .default_value_id = SemIR::InstId::None});
  ReplaceInstBeforeConstantUse(context, decl_id, assoc_const_decl);

  context.node_stack().Push(node_id, decl_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::RefBindingNameId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  return true;
}

auto HandleParseNode(Context& context, Parse::TemplateBindingNameId node_id)
    -> bool {
  context.node_stack().Push(node_id);
  return true;
}

// Within a pattern with an unused modifier, sets the is_unused on all
// entity names and also returns whether any names were found. The result
// is needed to emit a diagnostic when the unused modifier is
// unnecessary.
static auto MarkPatternUnused(Context& context, SemIR::InstId inst_id) -> bool {
  bool found_name = false;
  llvm::SmallVector<SemIR::InstId> worklist;
  worklist.push_back(inst_id);
  while (!worklist.empty()) {
    auto current_inst_id = worklist.pop_back_val();
    auto inst = context.insts().Get(current_inst_id);
    CARBON_KIND_SWITCH(inst) {
      case CARBON_KIND_ANY(SemIR::AnyBindingPattern, bind): {
        auto& name = context.entity_names().Get(bind.entity_name_id);
        name.is_unused = true;
        // We treat `_` as not marking the pattern as unused for the purpose of
        // deciding whether to issue a warning for `unused` on a pattern that
        // doesn't contain any bindings. `_` is implicitly unused, so marking it
        // `unused` is redundant but harmless.
        if (name.name_id != SemIR::NameId::Underscore) {
          found_name = true;
        }
        break;
      }
      case CARBON_KIND_ANY(SemIR::AnyVarPattern, var): {
        worklist.push_back(var.subpattern_id);
        break;
      }
      case CARBON_KIND(SemIR::TuplePattern tuple): {
        for (auto elem_id : context.inst_blocks().Get(tuple.elements_id)) {
          worklist.push_back(elem_id);
        }
        break;
      }
      default:
        break;
    }
  }
  return found_name;
}

auto HandleParseNode(Context& context, Parse::UnusedPatternId node_id) -> bool {
  auto [child_node, child_inst_id] =
      context.node_stack().PopPatternWithNodeId();
  if (!MarkPatternUnused(context, child_inst_id)) {
    CARBON_DIAGNOSTIC(UnusedPatternNoBindings, Warning,
                      "`unused` modifier on pattern without bindings");
    context.emitter().Emit(node_id, UnusedPatternNoBindings);
  }
  context.node_stack().Push(node_id, child_inst_id);
  return true;
}

}  // namespace Carbon::Check
