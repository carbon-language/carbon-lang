// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/expression_checker.h"

#include "explorer/interpreter/interpreter.h"

namespace Carbon {

using ::llvm::cast;
using ::llvm::dyn_cast;
using ::llvm::isa;

auto ExpressionChecker::TypeCheckExp(Nonnull<Expression*> e,
                                     const ImplScope& impl_scope)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "checking " << ExpressionKindDesc(e->kind()) << " "
                    << *e;
    **trace_stream_ << "\n";
  }
  if (e->is_type_checked()) {
    if (trace_stream_) {
      **trace_stream_ << "expression has already been type-checked\n";
    }
    return Success();
  }
  switch (e->kind()) {
    case ExpressionKind::InstantiateImpl:
    case ExpressionKind::ValueLiteral:
      CARBON_FATAL() << "attempting to type check node " << *e
                     << " generated during type checking";
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&index.object(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&index.offset(), impl_scope));
      const Value& object_type = index.object().static_type();
      switch (object_type.kind()) {
        case Value::Kind::TupleValue: {
          const auto& tuple_type = cast<TupleValue>(object_type);
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(index.offset().source_loc(), "tuple index",
                              arena_->New<IntType>(),
                              &index.offset().static_type(), impl_scope));
          CARBON_ASSIGN_OR_RETURN(
              auto offset_value,
              InterpExp(&index.offset(), arena_, trace_stream_));
          int i = cast<IntValue>(*offset_value).value();
          if (i < 0 || i >= static_cast<int>(tuple_type.elements().size())) {
            return CompilationError(e->source_loc())
                   << "index " << i << " is out of range for type "
                   << tuple_type;
          }
          index.set_static_type(tuple_type.elements()[i]);
          index.set_value_category(index.object().value_category());
          return Success();
        }
        case Value::Kind::StaticArrayType: {
          CARBON_RETURN_IF_ERROR(
              ExpectExactType(index.offset().source_loc(), "array index",
                              arena_->New<IntType>(),
                              &index.offset().static_type(), impl_scope));
          index.set_static_type(
              &cast<StaticArrayType>(object_type).element_type());
          index.set_value_category(index.object().value_category());
          return Success();
        }
        default:
          return CompilationError(e->source_loc()) << "expected a tuple";
      }
    }
    case ExpressionKind::TupleLiteral: {
      std::vector<Nonnull<const Value*>> arg_types;
      for (auto* arg : cast<TupleLiteral>(*e).fields()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(arg, impl_scope));
        CARBON_RETURN_IF_ERROR(
            ExpectIsConcreteType(arg->source_loc(), &arg->static_type()));
        arg_types.push_back(&arg->static_type());
      }
      e->set_static_type(arena_->New<TupleValue>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StructLiteral: {
      std::vector<NamedValue> arg_types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(&arg.expression(), impl_scope));
        CARBON_RETURN_IF_ERROR(ExpectIsConcreteType(
            arg.expression().source_loc(), &arg.expression().static_type()));
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      e->set_static_type(arena_->New<StructType>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StructTypeLiteral: {
      auto& struct_type = cast<StructTypeLiteral>(*e);
      for (auto& arg : struct_type.fields()) {
        CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(&arg.expression(), impl_scope));
      }
      if (struct_type.fields().empty()) {
        // `{}` is the type of `{}`, just as `()` is the type of `()`.
        // This applies only if there are no fields, because (unlike with
        // tuples) non-empty struct types are syntactically disjoint
        // from non-empty struct values.
        struct_type.set_static_type(arena_->New<StructType>());
      } else {
        struct_type.set_static_type(arena_->New<TypeType>());
      }
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::SimpleMemberAccessExpression: {
      auto& access = cast<SimpleMemberAccessExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&access.object(), impl_scope));
      const Value& object_type = access.object().static_type();
      switch (object_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(object_type);
          for (const auto& field : struct_type.fields()) {
            if (access.member_name() == field.name) {
              access.set_member(Member(&field));
              access.set_static_type(field.value);
              access.set_value_category(access.object().value_category());
              return Success();
            }
          }
          return CompilationError(access.source_loc())
                 << "struct " << struct_type << " does not have a field named "
                 << access.member_name();
        }
        case Value::Kind::NominalClassType: {
          const auto& t_class = cast<NominalClassType>(object_type);
          if (std::optional<Nonnull<const Declaration*>> member = FindMember(
                  access.member_name(), t_class.declaration().members());
              member.has_value()) {
            Nonnull<const Value*> field_type =
                Substitute(t_class.type_args(), &(*member)->static_type());
            access.set_member(Member(member.value()));
            access.set_static_type(field_type);
            switch ((*member)->kind()) {
              case DeclarationKind::VariableDeclaration:
                access.set_value_category(access.object().value_category());
                break;
              case DeclarationKind::FunctionDeclaration: {
                auto func_decl = cast<FunctionDeclaration>(*member);
                if (func_decl->is_method() && func_decl->me_pattern().kind() ==
                                                  PatternKind::AddrPattern) {
                  access.set_is_field_addr_me_method();
                  Nonnull<const Value*> me_type =
                      Substitute(t_class.type_args(),
                                 &func_decl->me_pattern().static_type());
                  CARBON_RETURN_IF_ERROR(ExpectType(
                      e->source_loc(), "method access, receiver type", me_type,
                      &access.object().static_type(), impl_scope));
                  if (access.object().value_category() != ValueCategory::Var) {
                    return CompilationError(e->source_loc())
                           << "method " << access.member_name()
                           << " requires its receiver to be an lvalue";
                  }
                }
                access.set_value_category(ValueCategory::Let);
                break;
              }
              default:
                CARBON_FATAL() << "member " << access.member_name()
                               << " is not a field or method";
                break;
            }
            return Success();
          } else {
            return CompilationError(e->source_loc())
                   << "class " << t_class.declaration().name()
                   << " does not have a field named " << access.member_name();
          }
        }
        case Value::Kind::VariableType: {
          // This case handles access to a method on a receiver whose type
          // is a type variable. For example, `x.foo` where the type of
          // `x` is `T` and `foo` and `T` implements an interface that
          // includes `foo`.
          const Value& typeof_var =
              cast<VariableType>(object_type).binding().static_type();
          CARBON_ASSIGN_OR_RETURN(
              ConstraintLookupResult result,
              LookupInConstraint(e->source_loc(), "member access", &typeof_var,
                                 access.member_name()));

          const Value& member_type = result.member->static_type();
          BindingMap binding_map = result.interface->args();
          binding_map[result.interface->declaration().self()] = &object_type;
          Nonnull<const Value*> inst_member_type =
              Substitute(binding_map, &member_type);
          access.set_member(Member(result.member));
          access.set_found_in_interface(result.interface);
          access.set_static_type(inst_member_type);

          CARBON_ASSIGN_OR_RETURN(
              Nonnull<Expression*> impl,
              impl_scope.Resolve(result.interface, &object_type,
                                 e->source_loc(), *this));
          access.set_impl(impl);
          return Success();
        }
        case Value::Kind::InterfaceType:
        case Value::Kind::ConstraintType: {
          // This case handles access to a class function from a constrained
          // type variable. If `T` is a type variable and `foo` is a class
          // function in an interface implemented by `T`, then `T.foo` accesses
          // the `foo` class function of `T`.
          //
          // TODO: Per the language rules, we are supposed to also perform
          // lookup into `type` and report an ambiguity if the name is found in
          // both places.
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> type,
              InterpExp(&access.object(), arena_, trace_stream_));
          CARBON_ASSIGN_OR_RETURN(
              ConstraintLookupResult result,
              LookupInConstraint(e->source_loc(), "member access", &object_type,
                                 access.member_name()));
          CARBON_ASSIGN_OR_RETURN(Nonnull<Expression*> impl,
                                  impl_scope.Resolve(result.interface, type,
                                                     e->source_loc(), *this));
          access.set_member(Member(result.member));
          access.set_impl(impl);
          access.set_found_in_interface(result.interface);

          bool is_instance_member;
          switch (result.member->kind()) {
            case DeclarationKind::FunctionDeclaration:
              is_instance_member =
                  cast<FunctionDeclaration>(*result.member).is_method();
              break;
            case DeclarationKind::AssociatedConstantDeclaration:
              is_instance_member = false;
              break;
            default:
              CARBON_FATAL()
                  << "unexpected kind for interface member " << *result.member;
              break;
          }

          if (is_instance_member) {
            // This is a member name denoting an instance member.
            // TODO: Consider setting the static type of all instance member
            // declarations to be member name types, rather than special-casing
            // member accesses that name them.
            access.set_static_type(
                arena_->New<TypeOfMemberName>(Member(result.member)));
            access.set_value_category(ValueCategory::Let);
          } else {
            // This is a non-instance member whose value is found directly via
            // the witness table, such as a non-method function or an
            // associated constant.
            const Value& member_type = result.member->static_type();
            BindingMap binding_map = result.interface->args();
            binding_map[result.interface->declaration().self()] = type;
            Nonnull<const Value*> inst_member_type =
                Substitute(binding_map, &member_type);
            access.set_static_type(inst_member_type);
            access.set_value_category(ValueCategory::Let);
          }
          return Success();
        }
        case Value::Kind::TypeType:
        case Value::Kind::TypeOfChoiceType:
        case Value::Kind::TypeOfClassType:
        case Value::Kind::TypeOfConstraintType:
        case Value::Kind::TypeOfInterfaceType: {
          // This is member access into an unconstrained type. Evaluate it and
          // perform lookup in the result.
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> type,
              InterpExp(&access.object(), arena_, trace_stream_));
          switch (type->kind()) {
            case Value::Kind::StructType: {
              for (const auto& field : cast<StructType>(type)->fields()) {
                if (access.member_name() == field.name) {
                  access.set_member(Member(&field));
                  access.set_static_type(
                      arena_->New<TypeOfMemberName>(Member(&field)));
                  access.set_value_category(ValueCategory::Let);
                  return Success();
                }
              }
              return CompilationError(access.source_loc())
                     << "struct " << *type << " does not have a field named "
                     << " does not have a field named " << access.member_name();
            }
            case Value::Kind::ChoiceType: {
              const ChoiceType& choice = cast<ChoiceType>(*type);
              std::optional<Nonnull<const Value*>> parameter_types =
                  choice.FindAlternative(access.member_name());
              if (!parameter_types.has_value()) {
                return CompilationError(e->source_loc())
                       << "choice " << choice.name()
                       << " does not have an alternative named "
                       << access.member_name();
              }
              Nonnull<const Value*> substituted_parameter_type =
                  *parameter_types;
              if (choice.IsParameterized()) {
                substituted_parameter_type =
                    Substitute(choice.type_args(), *parameter_types);
              }
              Nonnull<const Value*> type = arena_->New<FunctionType>(
                  substituted_parameter_type, llvm::None, &choice, llvm::None,
                  llvm::None);
              // TODO: Should there be a Declaration corresponding to each
              // choice type alternative?
              access.set_member(Member(arena_->New<NamedValue>(
                  NamedValue{access.member_name(), type})));
              access.set_static_type(type);
              access.set_value_category(ValueCategory::Let);
              return Success();
            }
            case Value::Kind::NominalClassType: {
              const NominalClassType& class_type =
                  cast<NominalClassType>(*type);
              if (std::optional<Nonnull<const Declaration*>> member =
                      FindMember(access.member_name(),
                                 class_type.declaration().members());
                  member.has_value()) {
                access.set_member(Member(member.value()));
                switch ((*member)->kind()) {
                  case DeclarationKind::FunctionDeclaration: {
                    const auto& func = cast<FunctionDeclaration>(*member);
                    if (func->is_method()) {
                      break;
                    }
                    Nonnull<const Value*> field_type = Substitute(
                        class_type.type_args(), &(*member)->static_type());
                    access.set_static_type(field_type);
                    access.set_value_category(ValueCategory::Let);
                    return Success();
                  }
                  default:
                    break;
                }
                access.set_static_type(
                    arena_->New<TypeOfMemberName>(Member(*member)));
                access.set_value_category(ValueCategory::Let);
                return Success();
              } else {
                return CompilationError(access.source_loc())
                       << class_type << " does not have a member named "
                       << access.member_name();
              }
            }
            case Value::Kind::InterfaceType:
            case Value::Kind::ConstraintType: {
              CARBON_ASSIGN_OR_RETURN(
                  ConstraintLookupResult result,
                  LookupInConstraint(e->source_loc(), "member access", type,
                                     access.member_name()));
              access.set_member(Member(result.member));
              access.set_found_in_interface(result.interface);
              access.set_static_type(
                  arena_->New<TypeOfMemberName>(Member(result.member)));
              access.set_value_category(ValueCategory::Let);
              return Success();
            }
            default:
              return CompilationError(access.source_loc())
                     << "unsupported member access into type " << *type;
          }
        }
        default:
          return CompilationError(e->source_loc())
                 << "member access, unexpected " << object_type << " in " << *e;
      }
    }
    case ExpressionKind::CompoundMemberAccessExpression: {
      auto& access = cast<CompoundMemberAccessExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&access.object(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&access.path(), impl_scope));
      if (!isa<TypeOfMemberName>(access.path().static_type())) {
        return CompilationError(e->source_loc())
               << "expected name of instance member or interface member in "
                  "compound member access, found "
               << access.path().static_type();
      }

      // Evaluate the member name expression to determine which member we're
      // accessing.
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> member_name_value,
                              InterpExp(&access.path(), arena_, trace_stream_));
      const auto& member_name = cast<MemberName>(*member_name_value);
      access.set_member(&member_name);

      bool has_instance = true;
      std::optional<Nonnull<const Value*>> base_type = member_name.base_type();
      if (!base_type.has_value()) {
        if (IsTypeOfType(&access.object().static_type())) {
          // This is `Type.(member_name)`, where `member_name` doesn't specify
          // a type. This access doesn't perform instance binding.
          CARBON_ASSIGN_OR_RETURN(
              base_type, InterpExp(&access.object(), arena_, trace_stream_));
          has_instance = false;
        } else {
          // This is `value.(member_name)`, where `member_name` doesn't specify
          // a type. The member will be found in the type of `value`, or in a
          // corresponding `impl` if `member_name` is an interface member.
          base_type = &access.object().static_type();
        }
      } else {
        // This is `value.(member_name)`, where `member_name` specifies a type.
        // `value` is implicitly converted to that type.
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> converted_object,
            ImplicitlyConvert("compound member access", impl_scope,
                              &access.object(), *base_type));
        access.set_object(converted_object);
      }

      // Perform impl selection if necessary.
      if (std::optional<Nonnull<const Value*>> iface =
              member_name.interface()) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<Expression*> impl,
            impl_scope.Resolve(*iface, *base_type, e->source_loc(), *this));
        access.set_impl(impl);
      }

      auto SubstituteIntoMemberType = [&]() {
        Nonnull<const Value*> member_type = &member_name.member().type();
        if (member_name.interface()) {
          Nonnull<const InterfaceType*> iface_type = *member_name.interface();
          BindingMap binding_map = iface_type->args();
          binding_map[iface_type->declaration().self()] = *base_type;
          return Substitute(binding_map, member_type);
        }
        if (auto* class_type = dyn_cast<NominalClassType>(base_type.value())) {
          return Substitute(class_type->type_args(), member_type);
        }
        return member_type;
      };

      switch (std::optional<Nonnull<const Declaration*>> decl =
                  member_name.member().declaration();
              decl ? decl.value()->kind()
                   : DeclarationKind::VariableDeclaration) {
        case DeclarationKind::VariableDeclaration:
          if (has_instance) {
            access.set_static_type(SubstituteIntoMemberType());
            access.set_value_category(access.object().value_category());
            return Success();
          }
          break;
        case DeclarationKind::FunctionDeclaration: {
          bool is_method = cast<FunctionDeclaration>(*decl.value()).is_method();
          if (has_instance || !is_method) {
            // This should not be possible: the name of a static member
            // function should have function type not member name type.
            CARBON_CHECK(!has_instance || is_method ||
                         !member_name.base_type().has_value())
                << "vacuous compound member access";
            access.set_static_type(SubstituteIntoMemberType());
            access.set_value_category(ValueCategory::Let);
            return Success();
          }
          break;
        }
        case DeclarationKind::AssociatedConstantDeclaration:
          access.set_static_type(SubstituteIntoMemberType());
          access.set_value_category(access.object().value_category());
          return Success();
        default:
          CARBON_FATAL() << "member " << member_name
                         << " is not a field or method";
          break;
      }

      access.set_static_type(
          arena_->New<TypeOfMemberName>(member_name.member()));
      access.set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::IdentifierExpression: {
      auto& ident = cast<IdentifierExpression>(*e);
      if (ident.value_node().base().kind() ==
          AstNodeKind::FunctionDeclaration) {
        const auto& function =
            cast<FunctionDeclaration>(ident.value_node().base());
        if (!function.has_static_type()) {
          CARBON_CHECK(function.return_term().is_auto());
          return CompilationError(ident.source_loc())
                 << "Function calls itself, but has a deduced return type";
        }
      }
      ident.set_static_type(&ident.value_node().static_type());
      ident.set_value_category(ident.value_node().value_category());
      return Success();
    }
    case ExpressionKind::DotSelfExpression: {
      auto& dot_self = cast<DotSelfExpression>(*e);
      if (dot_self.self_binding().is_type_checked()) {
        dot_self.set_static_type(&dot_self.self_binding().static_type());
      } else {
        dot_self.set_static_type(arena_->New<TypeType>());
        dot_self.self_binding().set_named_as_type_via_dot_self();
      }
      dot_self.set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::IntLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<IntType>());
      return Success();
    case ExpressionKind::BoolLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<BoolType>());
      return Success();
    case ExpressionKind::OperatorExpression: {
      auto& op = cast<OperatorExpression>(*e);
      std::vector<Nonnull<const Value*>> ts;
      for (Nonnull<Expression*> argument : op.arguments()) {
        CARBON_RETURN_IF_ERROR(TypeCheckExp(argument, impl_scope));
        ts.push_back(&argument->static_type());
      }

      auto handle_unary_operator =
          [&](Builtins::Builtin builtin) -> ErrorOr<Success> {
        ErrorOr<Nonnull<Expression*>> result = BuildBuiltinMethodCall(
            impl_scope, op.arguments()[0], BuiltinInterfaceName{builtin},
            BuiltinMethodCall{"Op"});
        if (!result.ok()) {
          // We couldn't find a matching `impl`.
          return CompilationError(e->source_loc())
                 << "type error in `" << ToString(op.op()) << "`:\n"
                 << result.error().message();
        }
        op.set_rewritten_form(*result);
        return Success();
      };

      auto handle_binary_operator =
          [&](Builtins::Builtin builtin) -> ErrorOr<Success> {
        ErrorOr<Nonnull<Expression*>> result = BuildBuiltinMethodCall(
            impl_scope, op.arguments()[0], BuiltinInterfaceName{builtin, ts[1]},
            BuiltinMethodCall{"Op", {op.arguments()[1]}});
        if (!result.ok()) {
          // We couldn't find a matching `impl`.
          return CompilationError(e->source_loc())
                 << "type error in `" << ToString(op.op()) << "`:\n"
                 << result.error().message();
        }
        op.set_rewritten_form(*result);
        return Success();
      };

      auto handle_binary_arithmetic =
          [&](Builtins::Builtin builtin) -> ErrorOr<Success> {
        // Handle a built-in operator first.
        // TODO: Replace this with an intrinsic.
        if (isa<IntType>(ts[0]) && isa<IntType>(ts[1]) &&
            IsSameType(ts[0], ts[1], impl_scope)) {
          op.set_static_type(ts[0]);
          op.set_value_category(ValueCategory::Let);
          return Success();
        }

        // Now try an overloaded operator.
        return handle_binary_operator(builtin);
      };

      switch (op.op()) {
        case Operator::Neg: {
          // Handle a built-in negation first.
          // TODO: Replace this with an intrinsic.
          if (isa<IntType>(ts[0])) {
            op.set_static_type(arena_->New<IntType>());
            op.set_value_category(ValueCategory::Let);
            return Success();
          }
          // Now try an overloaded negation.
          return handle_unary_operator(Builtins::Negate);
        }
        case Operator::Add:
          return handle_binary_arithmetic(Builtins::AddWith);
        case Operator::Sub:
          return handle_binary_arithmetic(Builtins::SubWith);
        case Operator::Mul:
          return handle_binary_arithmetic(Builtins::MulWith);
        case Operator::Mod:
          return handle_binary_arithmetic(Builtins::ModWith);
        case Operator::BitwiseAnd:
          // `&` between type-of-types performs constraint combination.
          // TODO: Should this be done via an intrinsic?
          if (IsTypeOfType(ts[0]) && IsTypeOfType(ts[1])) {
            std::optional<Nonnull<const ConstraintType*>> constraints[2];
            for (int i : {0, 1}) {
              if (auto* iface_type_type =
                      dyn_cast<TypeOfInterfaceType>(ts[i])) {
                constraints[i] = MakeConstraintForInterface(
                    e->source_loc(), &iface_type_type->interface_type());
              } else if (auto* constraint_type_type =
                             dyn_cast<TypeOfConstraintType>(ts[i])) {
                constraints[i] = &constraint_type_type->constraint_type();
              } else {
                return CompilationError(op.arguments()[i]->source_loc())
                       << "argument to " << ToString(op.op())
                       << " should be a constraint, found `" << *ts[i] << "`";
              }
            }
            op.set_static_type(
                arena_->New<TypeOfConstraintType>(CombineConstraints(
                    e->source_loc(), {*constraints[0], *constraints[1]})));
            op.set_value_category(ValueCategory::Let);
            return Success();
          }
          return handle_binary_operator(Builtins::BitAndWith);
        case Operator::BitwiseOr:
          return handle_binary_operator(Builtins::BitOrWith);
        case Operator::BitwiseXor:
          return handle_binary_operator(Builtins::BitXorWith);
        case Operator::BitShiftLeft:
          return handle_binary_operator(Builtins::LeftShiftWith);
        case Operator::BitShiftRight:
          return handle_binary_operator(Builtins::RightShiftWith);
        case Operator::Complement:
          return handle_unary_operator(Builtins::BitComplement);
        case Operator::And:
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "&&(1)",
                                                 arena_->New<BoolType>(), ts[0],
                                                 impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "&&(2)",
                                                 arena_->New<BoolType>(), ts[1],
                                                 impl_scope));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Or:
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "||(1)",
                                                 arena_->New<BoolType>(), ts[0],
                                                 impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "||(2)",
                                                 arena_->New<BoolType>(), ts[1],
                                                 impl_scope));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Not:
          CARBON_RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "!",
                                                 arena_->New<BoolType>(), ts[0],
                                                 impl_scope));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::Eq: {
          ErrorOr<Nonnull<Expression*>> converted = BuildBuiltinMethodCall(
              impl_scope, op.arguments()[0],
              BuiltinInterfaceName{Builtins::EqWith, ts[1]},
              BuiltinMethodCall{"Equal", op.arguments()[1]});
          if (!converted.ok()) {
            // We couldn't find a matching `impl`.
            return CompilationError(e->source_loc())
                   << *ts[0] << " is not equality comparable with " << *ts[1]
                   << " (" << converted.error().message() << ")";
          }
          op.set_rewritten_form(*converted);
          return Success();
        }
        case Operator::Less: {
          ErrorOr<Nonnull<Expression*>> converted = BuildBuiltinMethodCall(
              impl_scope, op.arguments()[0],
              BuiltinInterfaceName{Builtins::LessWith, ts[1]},
              BuiltinMethodCall{"Less", op.arguments()[1]});
          if (!converted.ok()) {
            // We couldn't find a matching `impl`.
            return CompilationError(e->source_loc())
                   << *ts[0] << " is not less comparable with " << *ts[1]
                   << " (" << converted.error().message() << ")";
          }
          op.set_rewritten_form(*converted);
          return Success();
        }
        case Operator::LessEq: {
          ErrorOr<Nonnull<Expression*>> converted = BuildBuiltinMethodCall(
              impl_scope, op.arguments()[0],
              BuiltinInterfaceName{Builtins::LessEqWith, ts[1]},
              BuiltinMethodCall{"LessEq", op.arguments()[1]});
          if (!converted.ok()) {
            // We couldn't find a matching `impl`.
            return CompilationError(e->source_loc())
                   << *ts[0] << " is not less equal comparable with " << *ts[1]
                   << " (" << converted.error().message() << ")";
          }
          op.set_rewritten_form(*converted);
          return Success();
        }
        case Operator::GreaterEq: {
          ErrorOr<Nonnull<Expression*>> converted = BuildBuiltinMethodCall(
              impl_scope, op.arguments()[0],
              BuiltinInterfaceName{Builtins::GreaterEqWith, ts[1]},
              BuiltinMethodCall{"GreaterEq", op.arguments()[1]});
          if (!converted.ok()) {
            // We couldn't find a matching `impl`.
            return CompilationError(e->source_loc())
                   << *ts[0] << " is not greater equal comparable with "
                   << *ts[1] << " (" << converted.error().message() << ")";
          }
          op.set_rewritten_form(*converted);
          return Success();
        }
        case Operator::Greater: {
          ErrorOr<Nonnull<Expression*>> converted = BuildBuiltinMethodCall(
              impl_scope, op.arguments()[0],
              BuiltinInterfaceName{Builtins::GreaterWith, ts[1]},
              BuiltinMethodCall{"Greater", op.arguments()[1]});
          if (!converted.ok()) {
            // We couldn't find a matching `impl`.
            return CompilationError(e->source_loc())
                   << *ts[0] << " is not greater comparable with " << *ts[1]
                   << " (" << converted.error().message() << ")";
          }
          op.set_rewritten_form(*converted);
          return Success();
        }
        case Operator::Deref:
          CARBON_RETURN_IF_ERROR(
              ExpectPointerType(e->source_loc(), "*", ts[0]));
          op.set_static_type(&cast<PointerType>(*ts[0]).type());
          op.set_value_category(ValueCategory::Var);
          return Success();
        case Operator::Ptr:
          CARBON_RETURN_IF_ERROR(ExpectType(e->source_loc(), "*",
                                            arena_->New<TypeType>(), ts[0],
                                            impl_scope));
          op.set_static_type(arena_->New<TypeType>());
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::AddressOf:
          if (op.arguments()[0]->value_category() != ValueCategory::Var) {
            return CompilationError(op.arguments()[0]->source_loc())
                   << "Argument to " << ToString(op.op())
                   << " should be an lvalue.";
          }
          op.set_static_type(arena_->New<PointerType>(ts[0]));
          op.set_value_category(ValueCategory::Let);
          return Success();
        case Operator::As: {
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> type,
              InterpExp(op.arguments()[1], arena_, trace_stream_));
          CARBON_RETURN_IF_ERROR(
              ExpectIsConcreteType(op.arguments()[1]->source_loc(), type));
          ErrorOr<Nonnull<Expression*>> converted =
              BuildBuiltinMethodCall(impl_scope, op.arguments()[0],
                                     BuiltinInterfaceName{Builtins::As, type},
                                     BuiltinMethodCall{"Convert"});
          if (!converted.ok()) {
            // We couldn't find a matching `impl`.
            return CompilationError(e->source_loc())
                   << "type error in `as`: `" << *ts[0]
                   << "` is not explicitly convertible to `" << *type << "`:\n"
                   << converted.error().message();
          }
          op.set_rewritten_form(*converted);
          return Success();
        }
      }
      break;
    }
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&call.function(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&call.argument(), impl_scope));
      switch (call.function().static_type().kind()) {
        case Value::Kind::FunctionType: {
          const auto& fun_t = cast<FunctionType>(call.function().static_type());
          if (trace_stream_) {
            **trace_stream_
                << "checking call to function of type " << fun_t
                << "\nwith arguments of type: " << call.argument().static_type()
                << "\n";
          }
          CARBON_RETURN_IF_ERROR(DeduceCallBindings(
              call, &fun_t.parameters(), fun_t.generic_parameters(),
              fun_t.deduced_bindings(), fun_t.impl_bindings(), impl_scope));
          const BindingMap& generic_bindings = call.deduced_args();

          // Substitute into the return type to determine the type of the call
          // expression.
          Nonnull<const Value*> return_type =
              Substitute(generic_bindings, &fun_t.return_type());
          call.set_static_type(return_type);
          call.set_value_category(ValueCategory::Let);
          return Success();
        }
        case Value::Kind::TypeOfParameterizedEntityName: {
          // This case handles the application of a parameterized class or
          // interface to a set of arguments, such as Point(i32) or
          // AddWith(i32).
          const ParameterizedEntityName& param_name =
              cast<TypeOfParameterizedEntityName>(call.function().static_type())
                  .name();

          // Collect the top-level generic parameters and their constraints.
          std::vector<FunctionType::GenericParameter> generic_parameters;
          std::vector<Nonnull<const ImplBinding*>> impl_bindings;
          llvm::ArrayRef<Nonnull<const Pattern*>> params =
              param_name.params().fields();
          for (size_t i = 0; i != params.size(); ++i) {
            // TODO: Should we disallow all other kinds of top-level params?
            if (auto* binding = dyn_cast<GenericBinding>(params[i])) {
              generic_parameters.push_back({i, binding});
              if (binding->impl_binding().has_value()) {
                impl_bindings.push_back(*binding->impl_binding());
              }
            }
          }

          CARBON_RETURN_IF_ERROR(DeduceCallBindings(
              call, &param_name.params().static_type(), generic_parameters,
              /*deduced_bindings=*/llvm::None, impl_bindings, impl_scope));
          Nonnull<const Bindings*> bindings =
              arena_->New<Bindings>(call.deduced_args(), Bindings::NoWitnesses);

          const Declaration& decl = param_name.declaration();
          switch (decl.kind()) {
            case DeclarationKind::ClassDeclaration: {
              Nonnull<NominalClassType*> inst_class_type =
                  arena_->New<NominalClassType>(&cast<ClassDeclaration>(decl),
                                                bindings);
              call.set_static_type(
                  arena_->New<TypeOfClassType>(inst_class_type));
              call.set_value_category(ValueCategory::Let);
              break;
            }
            case DeclarationKind::InterfaceDeclaration: {
              Nonnull<InterfaceType*> inst_iface_type =
                  arena_->New<InterfaceType>(&cast<InterfaceDeclaration>(decl),
                                             bindings);
              call.set_static_type(
                  arena_->New<TypeOfInterfaceType>(inst_iface_type));
              call.set_value_category(ValueCategory::Let);
              break;
            }
            case DeclarationKind::ChoiceDeclaration: {
              Nonnull<ChoiceType*> ct = arena_->New<ChoiceType>(
                  cast<ChoiceDeclaration>(&decl), bindings);
              Nonnull<TypeOfChoiceType*> inst_choice_type =
                  arena_->New<TypeOfChoiceType>(ct);
              call.set_static_type(inst_choice_type);
              call.set_value_category(ValueCategory::Let);
              break;
            }
            default:
              CARBON_FATAL()
                  << "unknown type of ParameterizedEntityName for " << decl;
          }
          return Success();
        }
        case Value::Kind::TypeOfChoiceType:
        default: {
          return CompilationError(e->source_loc())
                 << "in call `" << *e
                 << "`, expected callee to be a function, found `"
                 << call.function().static_type() << "`";
        }
      }
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fn = cast<FunctionTypeLiteral>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(&fn.parameter(), impl_scope));
      CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(&fn.return_type(), impl_scope));
      fn.set_static_type(arena_->New<TypeType>());
      fn.set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::StringLiteral:
      e->set_static_type(arena_->New<StringType>());
      e->set_value_category(ValueCategory::Let);
      return Success();
    case ExpressionKind::IntrinsicExpression: {
      auto& intrinsic_exp = cast<IntrinsicExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&intrinsic_exp.args(), impl_scope));
      const auto& args = intrinsic_exp.args().fields();
      switch (cast<IntrinsicExpression>(*e).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          // TODO: Remove Print special casing once we have variadics or
          // overloads. Here, that's the name Print instead of __intrinsic_print
          // in errors.
          if (args.size() < 1 || args.size() > 2) {
            return CompilationError(e->source_loc())
                   << "Print takes 1 or 2 arguments, received " << args.size();
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "Print argument 0", arena_->New<StringType>(),
              &args[0]->static_type(), impl_scope));
          if (args.size() >= 2) {
            CARBON_RETURN_IF_ERROR(ExpectExactType(
                e->source_loc(), "Print argument 1", arena_->New<IntType>(),
                &args[1]->static_type(), impl_scope));
          }
          e->set_static_type(TupleValue::Empty());
          e->set_value_category(ValueCategory::Let);
          return Success();
        case IntrinsicExpression::Intrinsic::Alloc: {
          if (args.size() != 1) {
            return CompilationError(e->source_loc())
                   << "__intrinsic_new takes 1 argument";
          }
          auto arg_type = &args[0]->static_type();
          e->set_static_type(arena_->New<PointerType>(arg_type));
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::Dealloc: {
          if (args.size() != 1) {
            return CompilationError(e->source_loc())
                   << "__intrinsic_new takes 1 argument";
          }
          auto arg_type = &args[0]->static_type();
          CARBON_RETURN_IF_ERROR(
              ExpectPointerType(e->source_loc(), "*", arg_type));
          e->set_static_type(TupleValue::Empty());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::Rand: {
          if (args.size() != 2) {
            return CompilationError(e->source_loc())
                   << "Rand takes 2 arguments, received " << args.size();
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "Rand argument 0", arena_->New<IntType>(),
              &args[0]->static_type(), impl_scope));

          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "Rand argument 1", arena_->New<IntType>(),
              &args[1]->static_type(), impl_scope));

          e->set_static_type(arena_->New<IntType>());

          return Success();
        }
        case IntrinsicExpression::Intrinsic::IntEq: {
          if (args.size() != 2) {
            return CompilationError(e->source_loc())
                   << "__intrinsic_int_eq takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_int_eq argument 1",
              arena_->New<IntType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_int_eq argument 2",
              arena_->New<IntType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<BoolType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::IntCompare: {
          if (args.size() != 2) {
            return CompilationError(e->source_loc())
                   << "__intrinsic_int_compare takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_int_compare argument 1",
              arena_->New<IntType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_int_compare argument 2",
              arena_->New<IntType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<IntType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::StrEq: {
          if (args.size() != 2) {
            return CompilationError(e->source_loc())
                   << "__intrinsic_str_eq takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_str_eq argument 1",
              arena_->New<StringType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_str_eq argument 2",
              arena_->New<StringType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<BoolType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::StrCompare: {
          if (args.size() != 2) {
            return CompilationError(e->source_loc())
                   << "__intrinsic_str_compare takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_str_compare argument 1",
              arena_->New<StringType>(), &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "__intrinsic_str_compare argument 2",
              arena_->New<StringType>(), &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<IntType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        }
        case IntrinsicExpression::Intrinsic::IntBitComplement:
          if (args.size() != 1) {
            return CompilationError(e->source_loc())
                   << intrinsic_exp.name() << " takes 1 argument";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "complement argument", arena_->New<IntType>(),
              &args[0]->static_type(), impl_scope));
          e->set_static_type(arena_->New<IntType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
        case IntrinsicExpression::Intrinsic::IntBitAnd:
        case IntrinsicExpression::Intrinsic::IntBitOr:
        case IntrinsicExpression::Intrinsic::IntBitXor:
        case IntrinsicExpression::Intrinsic::IntLeftShift:
        case IntrinsicExpression::Intrinsic::IntRightShift:
          if (args.size() != 2) {
            return CompilationError(e->source_loc())
                   << intrinsic_exp.name() << " takes 2 arguments";
          }
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "argument 1", arena_->New<IntType>(),
              &args[0]->static_type(), impl_scope));
          CARBON_RETURN_IF_ERROR(ExpectExactType(
              e->source_loc(), "argument 2", arena_->New<IntType>(),
              &args[1]->static_type(), impl_scope));
          e->set_static_type(arena_->New<IntType>());
          e->set_value_category(ValueCategory::Let);
          return Success();
      }
    }
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<TypeType>());
      return Success();
    case ExpressionKind::IfExpression: {
      auto& if_expr = cast<IfExpression>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckExp(&if_expr.condition(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<Expression*> converted_condition,
          ImplicitlyConvert("condition of `if`", impl_scope,
                            &if_expr.condition(), arena_->New<BoolType>()));
      if_expr.set_condition(converted_condition);

      // TODO: Compute the common type and convert both operands to it.
      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&if_expr.then_expression(), impl_scope));
      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&if_expr.else_expression(), impl_scope));
      CARBON_RETURN_IF_ERROR(ExpectExactType(
          e->source_loc(), "expression of `if` expression",
          &if_expr.then_expression().static_type(),
          &if_expr.else_expression().static_type(), impl_scope));
      e->set_static_type(&if_expr.then_expression().static_type());
      e->set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::WhereExpression: {
      auto& where = cast<WhereExpression>(*e);
      ImplScope inner_impl_scope;
      inner_impl_scope.AddParent(&impl_scope);
      CARBON_RETURN_IF_ERROR(TypeCheckPattern(&where.self_binding(),
                                              std::nullopt, inner_impl_scope,
                                              ValueCategory::Let));
      for (Nonnull<WhereClause*> clause : where.clauses()) {
        CARBON_RETURN_IF_ERROR(TypeCheckWhereClause(clause, inner_impl_scope));
      }

      std::optional<Nonnull<const ConstraintType*>> base;
      const Value& base_type = where.self_binding().static_type();
      if (auto* constraint_type = dyn_cast<ConstraintType>(&base_type)) {
        base = constraint_type;
      } else if (auto* interface_type = dyn_cast<InterfaceType>(&base_type)) {
        base = MakeConstraintForInterface(e->source_loc(), interface_type);
      } else if (isa<TypeType>(base_type)) {
        // Start with an unconstrained type.
      } else {
        return CompilationError(e->source_loc())
               << "expected constraint as first operand of `where` expression, "
               << "found " << base_type;
      }

      // Start with the given constraint, if any.
      ConstraintTypeBuilder builder(&where.self_binding());
      if (base) {
        BindingMap map;
        map[(*base)->self_binding()] = builder.GetSelfType(arena_);
        builder.Add(cast<ConstraintType>(Substitute(map, *base)));
      }

      // Apply the `where` clauses.
      for (Nonnull<const WhereClause*> clause : where.clauses()) {
        switch (clause->kind()) {
          case WhereClauseKind::IsWhereClause: {
            const auto& is_clause = cast<IsWhereClause>(*clause);
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> type,
                InterpExp(&is_clause.type(), arena_, trace_stream_));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> constraint,
                InterpExp(&is_clause.constraint(), arena_, trace_stream_));
            if (auto* interface = dyn_cast<InterfaceType>(constraint)) {
              // `where X is Y` produces an `impl` constraint.
              builder.AddImplConstraint({.type = type, .interface = interface});
            } else if (auto* constraint_type =
                           dyn_cast<ConstraintType>(constraint)) {
              // Transform `where .B is (C where .D is E)` into
              // `where .B is C and .B.D is E` then add all the resulting
              // constraints.
              BindingMap map;
              map[constraint_type->self_binding()] = type;
              builder.Add(cast<ConstraintType>(Substitute(map, constraint)));
            } else {
              return CompilationError(is_clause.constraint().source_loc())
                     << "expression after `is` does not resolve to a "
                        "constraint, found value "
                     << *constraint << " of type "
                     << is_clause.constraint().static_type();
            }
            break;
          }
          case WhereClauseKind::EqualsWhereClause: {
            const auto& equals_clause = cast<EqualsWhereClause>(*clause);
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> lhs,
                InterpExp(&equals_clause.lhs(), arena_, trace_stream_));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> rhs,
                InterpExp(&equals_clause.rhs(), arena_, trace_stream_));
            if (!ValueEqual(lhs, rhs, std::nullopt)) {
              builder.AddEqualityConstraint({.values = {lhs, rhs}});
            }
            break;
          }
        }
      }

      where.set_static_type(
          arena_->New<TypeOfConstraintType>(std::move(builder).Build(arena_)));
      where.set_value_category(ValueCategory::Let);
      return Success();
    }
    case ExpressionKind::UnimplementedExpression:
      CARBON_FATAL() << "Unimplemented: " << *e;
    case ExpressionKind::ArrayTypeLiteral: {
      auto& array_literal = cast<ArrayTypeLiteral>(*e);
      CARBON_RETURN_IF_ERROR(TypeCheckTypeExp(
          &array_literal.element_type_expression(), impl_scope));

      CARBON_RETURN_IF_ERROR(
          TypeCheckExp(&array_literal.size_expression(), impl_scope));
      CARBON_RETURN_IF_ERROR(ExpectExactType(
          array_literal.size_expression().source_loc(), "array size",
          arena_->New<IntType>(),
          &array_literal.size_expression().static_type(), impl_scope));
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> size_value,
          InterpExp(&array_literal.size_expression(), arena_, trace_stream_));
      if (cast<IntValue>(size_value)->value() < 0) {
        return CompilationError(array_literal.size_expression().source_loc())
               << "Array size cannot be negative";
      }
      array_literal.set_static_type(arena_->New<TypeType>());
      array_literal.set_value_category(ValueCategory::Let);
      return Success();
    }
  }
}

}  // namespace Carbon
