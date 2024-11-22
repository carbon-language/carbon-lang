// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/return.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

static auto HandleAnyBindingPattern(Context& context, Parse::NodeId node_id,
                                    bool is_generic) -> bool {
  auto [type_node, parsed_type_id] = context.node_stack().PopExprWithNodeId();
  auto [cast_type_inst_id, cast_type_id] =
      ExprAsType(context, type_node, parsed_type_id);

  // TODO: Handle `_` bindings.

  // Every other kind of pattern binding has a name.
  auto [name_node, name_id] = context.node_stack().PopNameWithNodeId();

  // Determine whether we're handling an associated constant. These share the
  // syntax for a compile-time binding, but don't behave like other compile-time
  // bindings.
  // TODO: Consider using a different parse node kind to make this easier.
  bool is_associated_constant = false;
  if (is_generic) {
    auto inst_id = context.scope_stack().PeekInstId();
    is_associated_constant =
        inst_id.is_valid() && context.insts().Is<SemIR::InterfaceDecl>(inst_id);
  }

  bool needs_compile_time_binding = is_generic && !is_associated_constant;

  // Create the appropriate kind of binding for this pattern.
  auto make_bind_name = [&](SemIR::TypeId type_id,
                            SemIR::InstId value_id) -> SemIR::LocIdAndInst {
    // TODO: Eventually the name will need to support associations with other
    // scopes, but right now we don't support qualified names here.
    auto entity_name_id = context.entity_names().Add(
        {.name_id = name_id,
         .parent_scope_id = context.scope_stack().PeekNameScopeId(),
         // TODO: Don't allocate a compile-time binding index for an associated
         // constant declaration.
         .bind_index = needs_compile_time_binding
                           ? context.scope_stack().AddCompileTimeBinding()
                           : SemIR::CompileTimeBindIndex::Invalid});
    if (is_generic) {
      // TODO: Create a `BindTemplateName` instead inside a `template` pattern.
      return SemIR::LocIdAndInst(
          name_node, SemIR::BindSymbolicName{.type_id = type_id,
                                             .entity_name_id = entity_name_id,
                                             .value_id = value_id});
    } else {
      return SemIR::LocIdAndInst(
          name_node, SemIR::BindName{.type_id = type_id,
                                     .entity_name_id = entity_name_id,
                                     .value_id = value_id});
    }
  };

  // Push the binding onto the node stack and, if necessary, onto the scope
  // stack.
  auto push_bind_name = [&](SemIR::InstId bind_id) {
    context.node_stack().Push(node_id, bind_id);
    if (needs_compile_time_binding) {
      context.scope_stack().PushCompileTimeBinding(bind_id);
    }
  };

  // A `self` binding can only appear in an implicit parameter list.
  if (name_id == SemIR::NameId::SelfValue &&
      !context.node_stack().PeekIs<Parse::NodeKind::ImplicitParamListStart>()) {
    CARBON_DIAGNOSTIC(
        SelfOutsideImplicitParamList, Error,
        "`self` can only be declared in an implicit parameter list");
    context.emitter().Emit(node_id, SelfOutsideImplicitParamList);
  }

  // Allocate an instruction of the appropriate kind, linked to the name for
  // error locations.
  // TODO: The node stack is a fragile way of getting context information.
  // Get this information from somewhere else.
  switch (auto context_node_kind = context.node_stack().PeekNodeKind()) {
    case Parse::NodeKind::ReturnedModifier:
    case Parse::NodeKind::VariableIntroducer: {
      if (is_generic) {
        CARBON_DIAGNOSTIC(
            CompileTimeBindingInVarDecl, Error,
            "`var` declaration cannot declare a compile-time binding");
        context.emitter().Emit(type_node, CompileTimeBindingInVarDecl);
        // Prevent lambda helpers from creating a compile time binding.
        needs_compile_time_binding = false;
      }
      auto binding_id =
          is_generic
              ? Parse::NodeId::Invalid
              : context.parse_tree().As<Parse::BindingPatternId>(node_id);

      // A `var` declaration at class scope introduces a field.
      auto parent_class_decl = context.GetCurrentScopeAs<SemIR::ClassDecl>();
      cast_type_id = context.AsCompleteType(
          cast_type_id,
          [&] {
            CARBON_DIAGNOSTIC(IncompleteTypeInVarDecl, Error,
                              "{0:field|variable} has incomplete type {1}",
                              BoolAsSelect, SemIR::TypeId);
            return context.emitter().Build(type_node, IncompleteTypeInVarDecl,
                                           parent_class_decl.has_value(),
                                           cast_type_id);
          },
          [&] {
            CARBON_DIAGNOSTIC(AbstractTypeInVarDecl, Error,
                              "{0:field|variable} has abstract type {1}",
                              BoolAsSelect, SemIR::TypeId);
            return context.emitter().Build(type_node, AbstractTypeInVarDecl,
                                           parent_class_decl.has_value(),
                                           cast_type_id);
          });
      if (parent_class_decl) {
        CARBON_CHECK(context_node_kind == Parse::NodeKind::VariableIntroducer,
                     "`returned var` at class scope");
        auto& class_info = context.classes().Get(parent_class_decl->class_id);
        auto field_type_id = context.GetUnboundElementType(
            class_info.self_type_id, cast_type_id);
        auto field_id = context.AddInst<SemIR::FieldDecl>(
            binding_id,
            {.type_id = field_type_id,
             .name_id = name_id,
             .index = SemIR::ElementIndex(
                 context.struct_type_fields_stack().PeekArray().size())});

        // Add a corresponding field to the object representation of the class.
        context.struct_type_fields_stack().AppendToTop(
            {.name_id = name_id, .type_id = cast_type_id});
        context.node_stack().Push(node_id, field_id);
        break;
      }

      SemIR::InstId value_id = SemIR::InstId::Invalid;
      if (context_node_kind == Parse::NodeKind::ReturnedModifier) {
        // TODO: Should we check this for the `var` as a whole, rather than for
        // the name binding?
        value_id =
            CheckReturnedVar(context, context.node_stack().PeekNodeId(),
                             name_node, name_id, type_node, cast_type_id);
      } else {
        value_id = context.AddInst<SemIR::VarStorage>(
            name_node, {.type_id = cast_type_id, .name_id = name_id});
      }
      auto bind_id = context.AddInst(make_bind_name(cast_type_id, value_id));
      push_bind_name(bind_id);

      if (context_node_kind == Parse::NodeKind::ReturnedModifier) {
        RegisterReturnedVar(context, bind_id);
      }
      break;
    }

    case Parse::NodeKind::ImplicitParamListStart:
    case Parse::NodeKind::TuplePatternStart: {
      // Parameters can have incomplete types in a function declaration, but not
      // in a function definition. We don't know which kind we have here.
      // TODO: A tuple pattern can appear in other places than function
      // parameters.
      auto param_pattern_id = SemIR::InstId::Invalid;
      bool had_error = false;
      switch (context.decl_introducer_state_stack().innermost().kind) {
        case Lex::TokenKind::Fn: {
          if (context_node_kind == Parse::NodeKind::ImplicitParamListStart &&
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
        context.AddNameToLookup(name_id, SemIR::InstId::BuiltinErrorInst);
        // Replace the parameter with an invalid instruction so that we don't
        // try constructing a generic based on it.
        param_pattern_id = SemIR::InstId::BuiltinErrorInst;
      } else {
        auto bind_id = context.AddInstInNoBlock(
            make_bind_name(cast_type_id, SemIR::InstId::Invalid));
        if (needs_compile_time_binding) {
          context.scope_stack().PushCompileTimeBinding(bind_id);
        }
        // TODO: Bindings should come into scope immediately in other contexts
        // too.
        context.AddNameToLookup(name_id, bind_id);
        auto entity_name_id =
            context.insts().GetAs<SemIR::AnyBindName>(bind_id).entity_name_id;
        bool inserted = context.bind_name_cache()
                            .Insert(entity_name_id, bind_id)
                            .is_inserted();
        CARBON_CHECK(inserted);
        auto pattern_inst_id = SemIR::InstId::Invalid;
        if (is_generic) {
          pattern_inst_id =
              context.AddPatternInst<SemIR::SymbolicBindingPattern>(
                  name_node,
                  {.type_id = cast_type_id, .entity_name_id = entity_name_id});
        } else {
          pattern_inst_id = context.AddPatternInst<SemIR::BindingPattern>(
              name_node,
              {.type_id = cast_type_id, .entity_name_id = entity_name_id});
        }
        param_pattern_id = context.AddPatternInst<SemIR::ValueParamPattern>(
            node_id,
            {
                .type_id = context.insts().Get(pattern_inst_id).type_id(),
                .subpattern_id = pattern_inst_id,
                .runtime_index = is_generic ? SemIR::RuntimeParamIndex::Invalid
                                            : SemIR::RuntimeParamIndex::Unknown,
            });
      }
      context.node_stack().Push(node_id, param_pattern_id);

      // TODO: Use the pattern insts to generate the pattern-match insts
      // at the end of the full pattern, instead of eagerly generating them
      // here.
      break;
    }

    case Parse::NodeKind::LetIntroducer: {
      cast_type_id = context.AsCompleteType(cast_type_id, [&] {
        CARBON_DIAGNOSTIC(IncompleteTypeInLetDecl, Error,
                          "`let` binding has incomplete type {0}",
                          InstIdAsType);
        return context.emitter().Build(type_node, IncompleteTypeInLetDecl,
                                       cast_type_inst_id);
      });
      // Create the instruction, but don't add it to a block until after we've
      // formed its initializer.
      // TODO: For general pattern parsing, we'll need to create a block to hold
      // the `let` pattern before we see the initializer.
      auto bind_id = context.AddPlaceholderInstInNoBlock(
          make_bind_name(cast_type_id, SemIR::InstId::Invalid));
      push_bind_name(bind_id);
      break;
    }

    default:
      CARBON_FATAL("Found a pattern binding in unexpected context {0}",
                   context_node_kind);
  }
  return true;
}

auto HandleParseNode(Context& context, Parse::BindingPatternId node_id)
    -> bool {
  return HandleAnyBindingPattern(context, node_id, /*is_generic=*/false);
}

auto HandleParseNode(Context& context,
                     Parse::CompileTimeBindingPatternId node_id) -> bool {
  bool is_generic = true;
  if (context.decl_introducer_state_stack().innermost().kind ==
      Lex::TokenKind::Let) {
    // Disallow `let` outside of function and interface definitions.
    // TODO: Find a less brittle way of doing this. An invalid scope_inst_id
    // can represent a block scope, but is also used for other kinds of scopes
    // that aren't necessarily part of an interface or function decl.
    auto scope_inst_id = context.scope_stack().PeekInstId();
    if (scope_inst_id.is_valid()) {
      auto scope_inst = context.insts().Get(scope_inst_id);
      if (!scope_inst.Is<SemIR::InterfaceDecl>() &&
          !scope_inst.Is<SemIR::FunctionDecl>()) {
        context.TODO(
            node_id,
            "`let` compile time binding outside function or interface");
        is_generic = false;
      }
    }
  }

  return HandleAnyBindingPattern(context, node_id, is_generic);
}

auto HandleParseNode(Context& context, Parse::AddrId node_id) -> bool {
  auto param_pattern_id = context.node_stack().PopPattern();
  if (SemIR::Function::GetNameFromPatternId(
          context.sem_ir(), param_pattern_id) == SemIR::NameId::SelfValue) {
    auto pointer_type = context.types().TryGetAs<SemIR::PointerType>(
        context.insts().Get(param_pattern_id).type_id());
    if (pointer_type) {
      auto addr_pattern_id = context.AddPatternInst<SemIR::AddrPattern>(
          node_id,
          {.type_id = SemIR::TypeId::AutoType, .inner_id = param_pattern_id});
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
