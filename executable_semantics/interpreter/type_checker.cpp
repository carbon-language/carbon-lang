// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/type_checker.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "executable_semantics/interpreter/impl_scope.h"
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {

static void SetValue(Nonnull<Pattern*> pattern, Nonnull<const Value*> value) {
  // TODO: find some way to CHECK that `value` is identical to pattern->value(),
  // if it's already set. Unclear if `ValueEqual` is suitable, because it
  // currently focuses more on "real" values, and disallows the pseudo-values
  // like `BindingPlaceholderValue` that we get in pattern evaluation.
  if (!pattern->has_value()) {
    pattern->set_value(value);
  }
}

static void ExpectExactType(SourceLocation source_loc,
                            const std::string& context,
                            Nonnull<const Value*> expected,
                            Nonnull<const Value*> actual) {
  if (!TypeEqual(expected, actual)) {
    FATAL_COMPILATION_ERROR(source_loc) << "type error in " << context << "\n"
                                        << "expected: " << *expected << "\n"
                                        << "actual: " << *actual;
  }
}

static void ExpectPointerType(SourceLocation source_loc,
                              const std::string& context,
                              Nonnull<const Value*> actual) {
  if (actual->kind() != Value::Kind::PointerType) {
    FATAL_COMPILATION_ERROR(source_loc) << "type error in " << context << "\n"
                                        << "expected a pointer type\n"
                                        << "actual: " << *actual;
  }
}

// Returns whether *value represents a concrete type, as opposed to a
// type pattern or a non-type value.
static auto IsConcreteType(Nonnull<const Value*> value) -> bool {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::BoolValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
      return false;
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::InterfaceType:
    case Value::Kind::Witness:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
      return true;
    case Value::Kind::AutoType:
      // `auto` isn't a concrete type, it's a pattern that matches types.
      return false;
    case Value::Kind::TupleValue:
      for (Nonnull<const Value*> field : cast<TupleValue>(*value).elements()) {
        if (!IsConcreteType(field)) {
          return false;
        }
      }
      return true;
  }
}

void TypeChecker::ExpectIsConcreteType(SourceLocation source_loc,
                                       Nonnull<const Value*> value) {
  if (!IsConcreteType(value)) {
    FATAL_COMPILATION_ERROR(source_loc)
        << "Expected a type, but got " << *value;
  }
}

auto TypeChecker::FieldTypesImplicitlyConvertible(
    llvm::ArrayRef<NamedValue> source_fields,
    llvm::ArrayRef<NamedValue> destination_fields) {
  if (source_fields.size() != destination_fields.size()) {
    return false;
  }
  for (const auto& source_field : source_fields) {
    auto it = std::find_if(destination_fields.begin(), destination_fields.end(),
                           [&](const NamedValue& field) {
                             return field.name == source_field.name;
                           });
    if (it == destination_fields.end() ||
        !IsImplicitlyConvertible(source_field.value, it->value)) {
      return false;
    }
  }
  return true;
}

auto TypeChecker::FieldTypes(const NominalClassType& class_type)
    -> std::vector<NamedValue> {
  std::vector<NamedValue> field_types;
  for (Nonnull<Declaration*> m : class_type.declaration().members()) {
    switch (m->kind()) {
      case DeclarationKind::VariableDeclaration: {
        const auto& var = cast<VariableDeclaration>(*m);
        Nonnull<const Value*> field_type =
            Substitute(class_type.type_args(), &var.binding().static_type());
        field_types.push_back(
            {.name = var.binding().name(), .value = field_type});
        break;
      }
      default:
        break;
    }
  }
  return field_types;
}

auto TypeChecker::IsImplicitlyConvertible(Nonnull<const Value*> source,
                                          Nonnull<const Value*> destination)
    -> bool {
  CHECK(IsConcreteType(source));
  CHECK(IsConcreteType(destination));
  if (TypeEqual(source, destination)) {
    return true;
  }
  switch (source->kind()) {
    case Value::Kind::StructType:
      switch (destination->kind()) {
        case Value::Kind::StructType:
          return FieldTypesImplicitlyConvertible(
              cast<StructType>(*source).fields(),
              cast<StructType>(*destination).fields());
        case Value::Kind::NominalClassType:
          return FieldTypesImplicitlyConvertible(
              cast<StructType>(*source).fields(),
              FieldTypes(cast<NominalClassType>(*destination)));
        default:
          return false;
      }
    case Value::Kind::TupleValue:
      switch (destination->kind()) {
        case Value::Kind::TupleValue: {
          const std::vector<Nonnull<const Value*>>& source_elements =
              cast<TupleValue>(*source).elements();
          const std::vector<Nonnull<const Value*>>& destination_elements =
              cast<TupleValue>(*destination).elements();
          if (source_elements.size() != destination_elements.size()) {
            return false;
          }
          for (size_t i = 0; i < source_elements.size(); ++i) {
            if (!IsImplicitlyConvertible(source_elements[i],
                                         destination_elements[i])) {
              return false;
            }
          }
          return true;
        }
        default:
          return false;
      }
    case Value::Kind::TypeType:
      switch (destination->kind()) {
        case Value::Kind::InterfaceType:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

void TypeChecker::ExpectType(SourceLocation source_loc,
                             const std::string& context,
                             Nonnull<const Value*> expected,
                             Nonnull<const Value*> actual) {
  if (!IsImplicitlyConvertible(actual, expected)) {
    FATAL_COMPILATION_ERROR(source_loc)
        << "type error in " << context << ": "
        << "'" << *actual << "' is not implicitly convertible to '" << *expected
        << "'";
  }
}

void TypeChecker::ArgumentDeduction(SourceLocation source_loc,
                                    BindingMap& deduced,
                                    Nonnull<const Value*> param_type,
                                    Nonnull<const Value*> arg_type) {
  switch (param_type->kind()) {
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*param_type);
      auto [it, success] = deduced.insert({&var_type.binding(), arg_type});
      if (!success) {
        // TODO: can we allow implicit conversions here?
        ExpectExactType(source_loc, "argument deduction", it->second, arg_type);
      }
      return;
    }
    case Value::Kind::TupleValue: {
      if (arg_type->kind() != Value::Kind::TupleValue) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "type error in argument deduction\n"
            << "expected: " << *param_type << "\n"
            << "actual: " << *arg_type;
      }
      const auto& param_tup = cast<TupleValue>(*param_type);
      const auto& arg_tup = cast<TupleValue>(*arg_type);
      if (param_tup.elements().size() != arg_tup.elements().size()) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "mismatch in tuple sizes, expected "
            << param_tup.elements().size() << " but got "
            << arg_tup.elements().size();
      }
      for (size_t i = 0; i < param_tup.elements().size(); ++i) {
        ArgumentDeduction(source_loc, deduced, param_tup.elements()[i],
                          arg_tup.elements()[i]);
      }
      return;
    }
    case Value::Kind::StructType: {
      if (arg_type->kind() != Value::Kind::StructType) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "type error in argument deduction\n"
            << "expected: " << *param_type << "\n"
            << "actual: " << *arg_type;
      }
      const auto& param_struct = cast<StructType>(*param_type);
      const auto& arg_struct = cast<StructType>(*arg_type);
      if (param_struct.fields().size() != arg_struct.fields().size()) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "mismatch in struct field counts, expected "
            << param_struct.fields().size() << " but got "
            << arg_struct.fields().size();
      }
      for (size_t i = 0; i < param_struct.fields().size(); ++i) {
        if (param_struct.fields()[i].name != arg_struct.fields()[i].name) {
          FATAL_COMPILATION_ERROR(source_loc)
              << "mismatch in field names, " << param_struct.fields()[i].name
              << " != " << arg_struct.fields()[i].name;
        }
        ArgumentDeduction(source_loc, deduced, param_struct.fields()[i].value,
                          arg_struct.fields()[i].value);
      }
      return;
    }
    case Value::Kind::FunctionType: {
      if (arg_type->kind() != Value::Kind::FunctionType) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "type error in argument deduction\n"
            << "expected: " << *param_type << "\n"
            << "actual: " << *arg_type;
      }
      const auto& param_fn = cast<FunctionType>(*param_type);
      const auto& arg_fn = cast<FunctionType>(*arg_type);
      // TODO: handle situation when arg has deduced parameters.
      ArgumentDeduction(source_loc, deduced, &param_fn.parameters(),
                        &arg_fn.parameters());
      ArgumentDeduction(source_loc, deduced, &param_fn.return_type(),
                        &arg_fn.return_type());
      return;
    }
    case Value::Kind::PointerType: {
      if (arg_type->kind() != Value::Kind::PointerType) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "type error in argument deduction\n"
            << "expected: " << *param_type << "\n"
            << "actual: " << *arg_type;
      }
      ArgumentDeduction(source_loc, deduced,
                        &cast<PointerType>(*param_type).type(),
                        &cast<PointerType>(*arg_type).type());
      return;
    }
    // Nothing to do in the case for `auto`.
    case Value::Kind::AutoType: {
      return;
    }
    case Value::Kind::NominalClassType: {
      const auto& param_class_type = cast<NominalClassType>(*param_type);
      switch (arg_type->kind()) {
        case Value::Kind::NominalClassType: {
          const auto& arg_class_type = cast<NominalClassType>(*arg_type);
          if (param_class_type.declaration().name() ==
              arg_class_type.declaration().name()) {
            for (const auto& [ty, param_ty] : param_class_type.type_args()) {
              ArgumentDeduction(source_loc, deduced, param_ty,
                                arg_class_type.type_args().at(ty));
            }
            return;
          }
          break;
        }
        default:
          break;
      }
      FATAL_COMPILATION_ERROR(source_loc)
          << "type error in argument deduction\n"
          << "expected: " << *param_type << "\n"
          << "actual: " << *arg_type;
    }
    // For the following cases, we check for type convertability.
    case Value::Kind::ContinuationType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ChoiceType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
      ExpectType(source_loc, "argument deduction", param_type, arg_type);
      return;
    // The rest of these cases should never happen.
    case Value::Kind::Witness:
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
      FATAL() << "In ArgumentDeduction: expected type, not value "
              << *param_type;
  }
}

auto TypeChecker::Substitute(
    const std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>& dict,
    Nonnull<const Value*> type) -> Nonnull<const Value*> {
  switch (type->kind()) {
    case Value::Kind::VariableType: {
      auto it = dict.find(&cast<VariableType>(*type).binding());
      if (it == dict.end()) {
        return type;
      } else {
        return it->second;
      }
    }
    case Value::Kind::TupleValue: {
      std::vector<Nonnull<const Value*>> elts;
      for (const auto& elt : cast<TupleValue>(*type).elements()) {
        elts.push_back(Substitute(dict, elt));
      }
      return arena_->New<TupleValue>(elts);
    }
    case Value::Kind::StructType: {
      std::vector<NamedValue> fields;
      for (const auto& [name, value] : cast<StructType>(*type).fields()) {
        auto new_type = Substitute(dict, value);
        fields.push_back({name, new_type});
      }
      return arena_->New<StructType>(std::move(fields));
    }
    case Value::Kind::FunctionType: {
      const auto& fn_type = cast<FunctionType>(*type);
      auto param = Substitute(dict, &fn_type.parameters());
      auto ret = Substitute(dict, &fn_type.return_type());
      return arena_->New<FunctionType>(
          std::vector<Nonnull<const GenericBinding*>>(), param, ret,
          std::vector<Nonnull<const ImplBinding*>>());
    }
    case Value::Kind::PointerType: {
      return arena_->New<PointerType>(
          Substitute(dict, &cast<PointerType>(*type).type()));
    }
    case Value::Kind::NominalClassType: {
      const auto& class_type = cast<NominalClassType>(*type);
      BindingMap type_args;
      for (const auto& [name, value] : class_type.type_args()) {
        type_args[name] = Substitute(dict, value);
      }
      return arena_->New<NominalClassType>(&class_type.declaration(),
                                           type_args);
    }
    case Value::Kind::AutoType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
      return type;
    // The rest of these cases should never happen.
    case Value::Kind::Witness:
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
    case Value::Kind::LValue:
    case Value::Kind::StructValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringValue:
      FATAL() << "In Substitute: expected type, not value " << *type;
  }
}

void TypeChecker::TypeCheckExp(Nonnull<Expression*> e, ImplScope& impl_scope) {
  if (trace_) {
    llvm::outs() << "checking expression " << *e;
    llvm::outs() << "\nconstants: ";
    PrintConstants(llvm::outs());
    llvm::outs() << "\n";
  }
  switch (e->kind()) {
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(*e);
      TypeCheckExp(&index.aggregate(), impl_scope);
      const Value& aggregate_type = index.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::TupleValue: {
          const auto& tuple_type = cast<TupleValue>(aggregate_type);
          int i = cast<IntValue>(*InterpExp(&index.offset(), arena_, trace_))
                      .value();
          if (i < 0 || i >= static_cast<int>(tuple_type.elements().size())) {
            FATAL_COMPILATION_ERROR(e->source_loc())
                << "index " << i << " is out of range for type " << tuple_type;
          }
          index.set_static_type(tuple_type.elements()[i]);
          index.set_value_category(index.aggregate().value_category());
          return;
        }
        default:
          FATAL_COMPILATION_ERROR(e->source_loc()) << "expected a tuple";
      }
    }
    case ExpressionKind::TupleLiteral: {
      std::vector<Nonnull<const Value*>> arg_types;
      for (auto& arg : cast<TupleLiteral>(*e).fields()) {
        TypeCheckExp(arg, impl_scope);
        arg_types.push_back(&arg->static_type());
      }
      e->set_static_type(arena_->New<TupleValue>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return;
    }
    case ExpressionKind::StructLiteral: {
      std::vector<NamedValue> arg_types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        TypeCheckExp(&arg.expression(), impl_scope);
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      e->set_static_type(arena_->New<StructType>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return;
    }
    case ExpressionKind::StructTypeLiteral: {
      auto& struct_type = cast<StructTypeLiteral>(*e);
      for (auto& arg : struct_type.fields()) {
        TypeCheckExp(&arg.expression(), impl_scope);
        ExpectIsConcreteType(arg.expression().source_loc(),
                             InterpExp(&arg.expression(), arena_, trace_));
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
      return;
    }
    case ExpressionKind::FieldAccessExpression: {
      auto& access = cast<FieldAccessExpression>(*e);
      TypeCheckExp(&access.aggregate(), impl_scope);
      const Value& aggregate_type = access.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(aggregate_type);
          for (const auto& [field_name, field_type] : struct_type.fields()) {
            if (access.field() == field_name) {
              access.set_static_type(field_type);
              access.set_value_category(access.aggregate().value_category());
              return;
            }
          }
          FATAL_COMPILATION_ERROR(access.source_loc())
              << "struct " << struct_type << " does not have a field named "
              << access.field();
        }
        case Value::Kind::NominalClassType: {
          const auto& t_class = cast<NominalClassType>(aggregate_type);
          if (std::optional<Nonnull<const Declaration*>> member =
                  FindMember(access.field(), t_class.declaration().members());
              member.has_value()) {
            Nonnull<const Value*> field_type =
                Substitute(t_class.type_args(), &(*member)->static_type());
            access.set_static_type(field_type);
            switch ((*member)->kind()) {
              case DeclarationKind::VariableDeclaration:
                access.set_value_category(access.aggregate().value_category());
                break;
              case DeclarationKind::FunctionDeclaration:
                access.set_value_category(ValueCategory::Let);
                break;
              default:
                FATAL() << "member " << access.field()
                        << " is not a field or method";
                break;
            }
            return;
          } else {
            FATAL_COMPILATION_ERROR(e->source_loc())
                << "class " << t_class.declaration().name()
                << " does not have a field named " << access.field();
          }
        }
        case Value::Kind::TypeOfChoiceType: {
          const ChoiceType& choice =
              cast<TypeOfChoiceType>(aggregate_type).choice_type();
          std::optional<Nonnull<const Value*>> parameter_types =
              choice.FindAlternative(access.field());
          if (!parameter_types.has_value()) {
            FATAL_COMPILATION_ERROR(e->source_loc())
                << "choice " << choice.name() << " does not have a field named "
                << access.field();
          }
          access.set_static_type(arena_->New<FunctionType>(
              std::vector<Nonnull<const GenericBinding*>>(), *parameter_types,
              &aggregate_type, std::vector<Nonnull<const ImplBinding*>>()));
          access.set_value_category(ValueCategory::Let);
          return;
        }
        case Value::Kind::TypeOfClassType: {
          const NominalClassType& class_type =
              cast<TypeOfClassType>(aggregate_type).class_type();
          if (std::optional<Nonnull<const Declaration*>> member = FindMember(
                  access.field(), class_type.declaration().members());
              member.has_value()) {
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
                return;
              }
              default:
                break;
            }
            FATAL_COMPILATION_ERROR(access.source_loc())
                << access.field() << " is not a class function";
          } else {
            FATAL_COMPILATION_ERROR(access.source_loc())
                << class_type << " does not have a class function named "
                << access.field();
          }
        }
        case Value::Kind::VariableType: {
          // This case handles access to a method on a receiver whose type
          // is a type variable. For example, `x.foo` where the type of
          // `x` is `T` and `foo` and `T` implements an interface that
          // includes `foo`.
          const VariableType& var_type = cast<VariableType>(aggregate_type);
          const Value& typeof_var = var_type.binding().static_type();
          switch (typeof_var.kind()) {
            case Value::Kind::InterfaceType: {
              const InterfaceType& iface_type = cast<InterfaceType>(typeof_var);
              const InterfaceDeclaration& iface_decl = iface_type.declaration();
              if (std::optional<Nonnull<const Declaration*>> member =
                      FindMember(access.field(), iface_decl.members());
                  member.has_value()) {
                const Value& member_type = (*member)->static_type();
                std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>
                    self_map;
                self_map[iface_decl.self()] = &var_type;
                Nonnull<const Value*> inst_member_type =
                    Substitute(self_map, &member_type);
                access.set_static_type(inst_member_type);
                CHECK(var_type.binding().impl_binding().has_value());
                access.set_impl(*var_type.binding().impl_binding());
                return;
              } else {
                FATAL_COMPILATION_ERROR(e->source_loc())
                    << "field access, " << access.field() << " not in "
                    << iface_decl.name();
              }
              break;
            }
            default:
              FATAL_COMPILATION_ERROR(e->source_loc())
                  << "field access, unexpected " << aggregate_type
                  << " of non-interface type " << typeof_var << " in " << *e;
              break;
          }
          break;
        }
        case Value::Kind::InterfaceType: {
          // This case handles access to a class function from a type variable.
          // If `T` is a type variable and `foo` is a class function in an
          // interface implemented by `T`, then `T.foo` accesses the `foo` class
          // function of `T`.
          const VariableType& var_type = cast<VariableType>(
              *InterpExp(&access.aggregate(), arena_, trace_));
          const InterfaceType& iface_type = cast<InterfaceType>(aggregate_type);
          const InterfaceDeclaration& iface_decl = iface_type.declaration();
          if (std::optional<Nonnull<const Declaration*>> member =
                  FindMember(access.field(), iface_decl.members());
              member.has_value()) {
            const Value& member_type = (*member)->static_type();
            std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>
                self_map;
            self_map[iface_decl.self()] = &var_type;
            Nonnull<const Value*> inst_member_type =
                Substitute(self_map, &member_type);
            access.set_static_type(inst_member_type);
            CHECK(var_type.binding().impl_binding().has_value());
            access.set_impl(*var_type.binding().impl_binding());
            return;
          } else {
            FATAL_COMPILATION_ERROR(e->source_loc())
                << "field access, " << access.field() << " not in "
                << iface_decl.name();
          }
          break;
        }
        default:
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "field access, unexpected " << aggregate_type << " in " << *e;
      }
    }
    case ExpressionKind::IdentifierExpression: {
      auto& ident = cast<IdentifierExpression>(*e);
      if (ident.value_node().base().kind() ==
          AstNodeKind::FunctionDeclaration) {
        const auto& function =
            cast<FunctionDeclaration>(ident.value_node().base());
        if (!function.has_static_type()) {
          CHECK(function.return_term().is_auto());
          FATAL_COMPILATION_ERROR(ident.source_loc())
              << "Function calls itself, but has a deduced return type";
        }
      }
      ident.set_static_type(&ident.value_node().static_type());
      ident.set_value_category(ident.value_node().value_category());
      return;
    }
    case ExpressionKind::IntLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<IntType>());
      return;
    case ExpressionKind::BoolLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<BoolType>());
      return;
    case ExpressionKind::PrimitiveOperatorExpression: {
      auto& op = cast<PrimitiveOperatorExpression>(*e);
      std::vector<Nonnull<const Value*>> ts;
      for (Nonnull<Expression*> argument : op.arguments()) {
        TypeCheckExp(argument, impl_scope);
        ts.push_back(&argument->static_type());
      }
      switch (op.op()) {
        case Operator::Neg:
          ExpectExactType(e->source_loc(), "negation", arena_->New<IntType>(),
                          ts[0]);
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Add:
          ExpectExactType(e->source_loc(), "addition(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "addition(2)",
                          arena_->New<IntType>(), ts[1]);
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Sub:
          ExpectExactType(e->source_loc(), "subtraction(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "subtraction(2)",
                          arena_->New<IntType>(), ts[1]);
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Mul:
          ExpectExactType(e->source_loc(), "multiplication(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "multiplication(2)",
                          arena_->New<IntType>(), ts[1]);
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::And:
          ExpectExactType(e->source_loc(), "&&(1)", arena_->New<BoolType>(),
                          ts[0]);
          ExpectExactType(e->source_loc(), "&&(2)", arena_->New<BoolType>(),
                          ts[1]);
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Or:
          ExpectExactType(e->source_loc(), "||(1)", arena_->New<BoolType>(),
                          ts[0]);
          ExpectExactType(e->source_loc(), "||(2)", arena_->New<BoolType>(),
                          ts[1]);
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Not:
          ExpectExactType(e->source_loc(), "!", arena_->New<BoolType>(), ts[0]);
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Eq:
          ExpectExactType(e->source_loc(), "==", ts[0], ts[1]);
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Deref:
          ExpectPointerType(e->source_loc(), "*", ts[0]);
          op.set_static_type(&cast<PointerType>(*ts[0]).type());
          op.set_value_category(ValueCategory::Var);
          return;
        case Operator::Ptr:
          ExpectExactType(e->source_loc(), "*", arena_->New<TypeType>(), ts[0]);
          op.set_static_type(arena_->New<TypeType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::AddressOf:
          if (op.arguments()[0]->value_category() != ValueCategory::Var) {
            FATAL_COMPILATION_ERROR(op.arguments()[0]->source_loc())
                << "Argument to " << ToString(op.op())
                << " should be an lvalue.";
          }
          op.set_static_type(arena_->New<PointerType>(ts[0]));
          op.set_value_category(ValueCategory::Let);
          return;
      }
      break;
    }
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(*e);
      TypeCheckExp(&call.function(), impl_scope);
      TypeCheckExp(&call.argument(), impl_scope);
      switch (call.function().static_type().kind()) {
        case Value::Kind::FunctionType: {
          const auto& fun_t = cast<FunctionType>(call.function().static_type());
          Nonnull<const Value*> parameters = &fun_t.parameters();
          Nonnull<const Value*> return_type = &fun_t.return_type();
          if (!fun_t.deduced().empty()) {
            BindingMap deduced_type_args;
            ArgumentDeduction(e->source_loc(), deduced_type_args, parameters,
                              &call.argument().static_type());
            call.set_deduced_args(deduced_type_args);
            for (Nonnull<const GenericBinding*> deduced_param :
                 fun_t.deduced()) {
              // TODO: change the following to a CHECK once the real checking
              // has been added to the type checking of function signatures.
              if (auto it = deduced_type_args.find(deduced_param);
                  it == deduced_type_args.end()) {
                FATAL_COMPILATION_ERROR(e->source_loc())
                    << "could not deduce type argument for type parameter "
                    << deduced_param->name() << "\n"
                    << "in " << call;
              }
            }
            parameters = Substitute(deduced_type_args, parameters);
            return_type = Substitute(deduced_type_args, return_type);

            // Find impls for all the impl bindings of the function
            std::map<Nonnull<const ImplBinding*>, ValueNodeView> impls;
            for (Nonnull<const ImplBinding*> impl_binding :
                 fun_t.impl_bindings()) {
              switch (impl_binding->interface()->kind()) {
                case Value::Kind::InterfaceType: {
                  ValueNodeView impl = impl_scope.Resolve(
                      impl_binding->interface(),
                      deduced_type_args[impl_binding->type_var()],
                      e->source_loc());
                  impls.emplace(impl_binding, impl);
                  break;
                }
                case Value::Kind::TypeType:
                  break;
                default:
                  FATAL_COMPILATION_ERROR(e->source_loc())
                      << "unexpected type of deduced parameter "
                      << *impl_binding->interface();
              }
            }
            call.set_impls(impls);
          } else {  // No deduced parameters.
                    // Check that the argument types are convertible to the
                    // parameter types
            ExpectType(e->source_loc(), "call", parameters,
                       &call.argument().static_type());
          }
          call.set_static_type(return_type);
          call.set_value_category(ValueCategory::Let);
          return;
        }
        case Value::Kind::TypeOfClassType: {
          // This case handles the application of a generic class to
          // a type argument, such as Point(i32).
          const ClassDeclaration& class_decl =
              cast<TypeOfClassType>(call.function().static_type())
                  .class_type()
                  .declaration();
          BindingMap generic_args;
          if (class_decl.type_params().has_value()) {
            if (trace_)
              llvm::outs() << "pattern matching type params and args ";
            CHECK(PatternMatch(&(*class_decl.type_params())->value(),
                               InterpExp(&call.argument(), arena_, trace_),
                               call.source_loc(), std::nullopt, generic_args));
          }
          // Find impls for all the impl bindings of the class
          std::map<Nonnull<const ImplBinding*>, ValueNodeView> impls;
          for (const auto& [binding, val] : generic_args) {
            if (binding->impl_binding().has_value()) {
              Nonnull<const ImplBinding*> impl_binding =
                  *binding->impl_binding();
              switch (impl_binding->interface()->kind()) {
                case Value::Kind::InterfaceType: {
                  ValueNodeView impl = impl_scope.Resolve(
                      impl_binding->interface(), generic_args[binding],
                      call.source_loc());
                  impls.emplace(impl_binding, impl);
                  break;
                }
                case Value::Kind::TypeType:
                  break;
                default:
                  FATAL_COMPILATION_ERROR(e->source_loc())
                      << "unexpected type of deduced parameter "
                      << *impl_binding->interface();
              }
            }
          }  // for generic_args
          Nonnull<NominalClassType*> class_type =
              arena_->New<NominalClassType>(&class_decl, generic_args, impls);
          call.set_impls(impls);
          call.set_static_type(class_type);
          call.set_value_category(ValueCategory::Let);
          return;
        }
        default: {
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "in call, expected a function\n"
              << *e << "\nnot an operator of type "
              << call.function().static_type() << "\n";
        }
      }
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fn = cast<FunctionTypeLiteral>(*e);
      ExpectIsConcreteType(fn.parameter().source_loc(),
                           InterpExp(&fn.parameter(), arena_, trace_));
      ExpectIsConcreteType(fn.return_type().source_loc(),
                           InterpExp(&fn.return_type(), arena_, trace_));
      fn.set_static_type(arena_->New<TypeType>());
      fn.set_value_category(ValueCategory::Let);
      return;
    }
    case ExpressionKind::StringLiteral:
      e->set_static_type(arena_->New<StringType>());
      e->set_value_category(ValueCategory::Let);
      return;
    case ExpressionKind::IntrinsicExpression: {
      auto& intrinsic_exp = cast<IntrinsicExpression>(*e);
      TypeCheckExp(&intrinsic_exp.args(), impl_scope);
      switch (cast<IntrinsicExpression>(*e).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          if (intrinsic_exp.args().fields().size() != 1) {
            FATAL_COMPILATION_ERROR(e->source_loc())
                << "__intrinsic_print takes 1 argument";
          }
          ExpectType(e->source_loc(), "__intrinsic_print argument",
                     arena_->New<StringType>(),
                     &intrinsic_exp.args().fields()[0]->static_type());
          e->set_static_type(TupleValue::Empty());
          e->set_value_category(ValueCategory::Let);
          return;
      }
    }
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<TypeType>());
      return;
    case ExpressionKind::UnimplementedExpression:
      FATAL() << "Unimplemented: " << *e;
  }
}

void TypeChecker::PatternImpls(Nonnull<Pattern*> p, ImplScope& impl_scope) {
  switch (p->kind()) {
    case PatternKind::GenericBinding: {
      auto& binding = cast<GenericBinding>(*p);
      CHECK(binding.impl_binding().has_value());
      Nonnull<const ImplBinding*> impl_binding = *binding.impl_binding();
      impl_scope.Add(impl_binding->interface(),
                     *impl_binding->type_var()->compile_time_value(),
                     impl_binding);
      return;
    }
    case PatternKind::TuplePattern: {
      auto& tuple = cast<TuplePattern>(*p);
      for (size_t i = 0; i < tuple.fields().size(); ++i) {
        Nonnull<Pattern*> field = tuple.fields()[i];
        PatternImpls(field, impl_scope);
      }
      return;
    }
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(*p);
      PatternImpls(&alternative.arguments(), impl_scope);
      return;
    }
    case PatternKind::ExpressionPattern:
    case PatternKind::AutoPattern:
    case PatternKind::BindingPattern:
      return;
  }
}

void TypeChecker::TypeCheckPattern(
    Nonnull<Pattern*> p, std::optional<Nonnull<const Value*>> expected,
    ImplScope& impl_scope) {
  if (trace_) {
    llvm::outs() << "checking pattern " << *p;
    if (expected) {
      llvm::outs() << ", expecting " << **expected;
    }
    llvm::outs() << "\nconstants: ";
    PrintConstants(llvm::outs());
    llvm::outs() << "\n";
  }
  switch (p->kind()) {
    case PatternKind::AutoPattern: {
      p->set_static_type(arena_->New<TypeType>());
      return;
    }
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(*p);
      TypeCheckPattern(&binding.type(), std::nullopt, impl_scope);
      Nonnull<const Value*> type =
          InterpPattern(&binding.type(), arena_, trace_);
      if (expected) {
        if (IsConcreteType(type)) {
          ExpectType(p->source_loc(), "name binding", type, *expected);
        } else {
          BindingMap generic_args;
          if (!PatternMatch(type, *expected, binding.type().source_loc(),
                            std::nullopt, generic_args)) {
            FATAL_COMPILATION_ERROR(binding.type().source_loc())
                << "Type pattern '" << *type << "' does not match actual type '"
                << **expected << "'";
          }
          type = *expected;
        }
      }
      ExpectIsConcreteType(binding.source_loc(), type);
      binding.set_static_type(type);
      SetValue(&binding, InterpPattern(&binding, arena_, trace_));
      return;
    }
    case PatternKind::GenericBinding: {
      auto& binding = cast<GenericBinding>(*p);
      TypeCheckExp(&binding.type(), impl_scope);
      Nonnull<const Value*> type = InterpExp(&binding.type(), arena_, trace_);
      if (expected) {
        FATAL_COMPILATION_ERROR(binding.type().source_loc())
            << "Generic binding may not occur in pattern with expected type: "
            << binding;
      }
      binding.set_static_type(type);
      Nonnull<const Value*> val = InterpPattern(&binding, arena_, trace_);
      binding.set_compile_time_value(val);
      Nonnull<ImplBinding*> impl_binding = arena_->New<ImplBinding>(
          binding.source_loc(), &binding, &binding.static_type());
      binding.set_impl_binding(impl_binding);
      SetValue(&binding, val);
      impl_scope.Add(impl_binding->interface(),
                     *impl_binding->type_var()->compile_time_value(),
                     impl_binding);
      return;
    }
    case PatternKind::TuplePattern: {
      auto& tuple = cast<TuplePattern>(*p);
      std::vector<Nonnull<const Value*>> field_types;
      if (expected && (*expected)->kind() != Value::Kind::TupleValue) {
        FATAL_COMPILATION_ERROR(p->source_loc()) << "didn't expect a tuple";
      }
      if (expected && tuple.fields().size() !=
                          cast<TupleValue>(**expected).elements().size()) {
        FATAL_COMPILATION_ERROR(tuple.source_loc())
            << "tuples of different length";
      }
      for (size_t i = 0; i < tuple.fields().size(); ++i) {
        Nonnull<Pattern*> field = tuple.fields()[i];
        std::optional<Nonnull<const Value*>> expected_field_type;
        if (expected) {
          expected_field_type = cast<TupleValue>(**expected).elements()[i];
        }
        TypeCheckPattern(field, expected_field_type, impl_scope);
        if (trace_)
          llvm::outs() << "finished checking tuple pattern field " << *field
                       << "\n";
        field_types.push_back(&field->static_type());
      }
      tuple.set_static_type(arena_->New<TupleValue>(std::move(field_types)));
      SetValue(&tuple, InterpPattern(&tuple, arena_, trace_));
      return;
    }
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(*p);
      TypeCheckExp(&alternative.choice_type(), impl_scope);
      if (alternative.choice_type().static_type().kind() !=
          Value::Kind::TypeOfChoiceType) {
        FATAL_COMPILATION_ERROR(alternative.source_loc())
            << "alternative pattern does not name a choice type.";
      }
      if (expected) {
        ExpectExactType(alternative.source_loc(), "alternative pattern",
                        *expected, &alternative.choice_type().static_type());
      }
      const ChoiceType& choice_type =
          cast<TypeOfChoiceType>(alternative.choice_type().static_type())
              .choice_type();
      std::optional<Nonnull<const Value*>> parameter_types =
          cast<ChoiceType>(choice_type)
              .FindAlternative(alternative.alternative_name());
      if (parameter_types == std::nullopt) {
        FATAL_COMPILATION_ERROR(alternative.source_loc())
            << "'" << alternative.alternative_name()
            << "' is not an alternative of " << choice_type;
      }
      TypeCheckPattern(&alternative.arguments(), *parameter_types, impl_scope);
      alternative.set_static_type(&choice_type);
      SetValue(&alternative, InterpPattern(&alternative, arena_, trace_));
      return;
    }
    case PatternKind::ExpressionPattern: {
      auto& expression = cast<ExpressionPattern>(*p).expression();
      TypeCheckExp(&expression, impl_scope);
      p->set_static_type(&expression.static_type());
      SetValue(p, InterpPattern(p, arena_, trace_));
      return;
    }
  }
}

void TypeChecker::TypeCheckStmt(Nonnull<Statement*> s, ImplScope& impl_scope) {
  if (trace_) {
    llvm::outs() << "checking statement " << *s << "\n";
  }
  switch (s->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*s);
      TypeCheckExp(&match.expression(), impl_scope);
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        TypeCheckPattern(&clause.pattern(), &match.expression().static_type(),
                         impl_scope);
        TypeCheckStmt(&clause.statement(), impl_scope);
      }
      return;
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(*s);
      TypeCheckExp(&while_stmt.condition(), impl_scope);
      ExpectType(s->source_loc(), "condition of `while`",
                 arena_->New<BoolType>(),
                 &while_stmt.condition().static_type());
      TypeCheckStmt(&while_stmt.body(), impl_scope);
      return;
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return;
    case StatementKind::Block: {
      auto& block = cast<Block>(*s);
      for (auto* block_statement : block.statements()) {
        TypeCheckStmt(block_statement, impl_scope);
      }
      return;
    }
    case StatementKind::VariableDefinition: {
      auto& var = cast<VariableDefinition>(*s);
      TypeCheckExp(&var.init(), impl_scope);
      const Value& rhs_ty = var.init().static_type();
      TypeCheckPattern(&var.pattern(), &rhs_ty, impl_scope);
      return;
    }
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(*s);
      TypeCheckExp(&assign.rhs(), impl_scope);
      TypeCheckExp(&assign.lhs(), impl_scope);
      ExpectType(s->source_loc(), "assign", &assign.lhs().static_type(),
                 &assign.rhs().static_type());
      if (assign.lhs().value_category() != ValueCategory::Var) {
        FATAL_COMPILATION_ERROR(assign.source_loc())
            << "Cannot assign to rvalue '" << assign.lhs() << "'";
      }
      return;
    }
    case StatementKind::ExpressionStatement: {
      TypeCheckExp(&cast<ExpressionStatement>(*s).expression(), impl_scope);
      return;
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*s);
      TypeCheckExp(&if_stmt.condition(), impl_scope);
      ExpectType(s->source_loc(), "condition of `if`", arena_->New<BoolType>(),
                 &if_stmt.condition().static_type());
      TypeCheckStmt(&if_stmt.then_block(), impl_scope);
      if (if_stmt.else_block()) {
        TypeCheckStmt(*if_stmt.else_block(), impl_scope);
      }
      return;
    }
    case StatementKind::Return: {
      auto& ret = cast<Return>(*s);
      TypeCheckExp(&ret.expression(), impl_scope);
      ReturnTerm& return_term = ret.function().return_term();
      if (return_term.is_auto()) {
        return_term.set_static_type(&ret.expression().static_type());
      } else {
        ExpectType(s->source_loc(), "return", &return_term.static_type(),
                   &ret.expression().static_type());
      }
      return;
    }
    case StatementKind::Continuation: {
      auto& cont = cast<Continuation>(*s);
      TypeCheckStmt(&cont.body(), impl_scope);
      cont.set_static_type(arena_->New<ContinuationType>());
      return;
    }
    case StatementKind::Run: {
      auto& run = cast<Run>(*s);
      TypeCheckExp(&run.argument(), impl_scope);
      ExpectType(s->source_loc(), "argument of `run`",
                 arena_->New<ContinuationType>(),
                 &run.argument().static_type());
      return;
    }
    case StatementKind::Await: {
      // nothing to do here
      return;
    }
  }  // switch
}

// Returns true if we can statically verify that `match` is exhaustive, meaning
// that one of its clauses will be executed for any possible operand value.
//
// TODO: the current rule is an extremely simplistic placeholder, with
// many false negatives.
static auto IsExhaustive(const Match& match) -> bool {
  for (const Match::Clause& clause : match.clauses()) {
    // A pattern consisting of a single variable binding is guaranteed to match.
    if (clause.pattern().kind() == PatternKind::BindingPattern) {
      return true;
    }
  }
  return false;
}

void TypeChecker::ExpectReturnOnAllPaths(
    std::optional<Nonnull<Statement*>> opt_stmt, SourceLocation source_loc) {
  if (!opt_stmt) {
    FATAL_COMPILATION_ERROR(source_loc)
        << "control-flow reaches end of function that provides a `->` return "
           "type without reaching a return statement";
  }
  Nonnull<Statement*> stmt = *opt_stmt;
  switch (stmt->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*stmt);
      if (!IsExhaustive(match)) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "non-exhaustive match may allow control-flow to reach the end "
               "of a function that provides a `->` return type";
      }
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        ExpectReturnOnAllPaths(&clause.statement(), stmt->source_loc());
      }
      return;
    }
    case StatementKind::Block: {
      auto& block = cast<Block>(*stmt);
      if (block.statements().empty()) {
        FATAL_COMPILATION_ERROR(stmt->source_loc())
            << "control-flow reaches end of function that provides a `->` "
               "return type without reaching a return statement";
      }
      ExpectReturnOnAllPaths(block.statements()[block.statements().size() - 1],
                             block.source_loc());
      return;
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*stmt);
      ExpectReturnOnAllPaths(&if_stmt.then_block(), stmt->source_loc());
      ExpectReturnOnAllPaths(if_stmt.else_block(), stmt->source_loc());
      return;
    }
    case StatementKind::Return:
      return;
    case StatementKind::Continuation:
    case StatementKind::Run:
    case StatementKind::Await:
      return;
    case StatementKind::Assign:
    case StatementKind::ExpressionStatement:
    case StatementKind::While:
    case StatementKind::Break:
    case StatementKind::Continue:
    case StatementKind::VariableDefinition:
      FATAL_COMPILATION_ERROR(stmt->source_loc())
          << "control-flow reaches end of function that provides a `->` "
             "return type without reaching a return statement";
  }
}

// TODO: Add checking to function definitions to ensure that
//   all deduced type parameters will be deduced.
void TypeChecker::DeclareFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                             ImplScope& impl_scope) {
  if (trace_) {
    llvm::outs() << "** declaring function " << f->name() << "\n";
  }
  // Bring the deduced parameters into scope
  for (Nonnull<GenericBinding*> deduced : f->deduced_parameters()) {
    TypeCheckExp(&deduced->type(), impl_scope);
    // SetConstantValue(deduced, arena_->New<VariableType>(deduced));
    deduced->set_compile_time_value(arena_->New<VariableType>(deduced));
    deduced->set_static_type(InterpExp(&deduced->type(), arena_, trace_));
  }
  // Create the impl_bindings
  std::vector<Nonnull<const ImplBinding*>> impl_bindings;
  for (Nonnull<GenericBinding*> deduced : f->deduced_parameters()) {
    Nonnull<ImplBinding*> impl_binding = arena_->New<ImplBinding>(
        deduced->source_loc(), deduced, &deduced->static_type());
    deduced->set_impl_binding(impl_binding);
    impl_binding->set_static_type(&deduced->static_type());
    impl_bindings.push_back(impl_binding);
  }
  // Bring the impl bindings into scope
  ImplScope function_scope;
  function_scope.AddParent(&impl_scope);
  for (Nonnull<const ImplBinding*> impl_binding : impl_bindings) {
    CHECK(impl_binding->type_var()->compile_time_value().has_value());
    function_scope.Add(impl_binding->interface(),
                       *impl_binding->type_var()->compile_time_value(),
                       impl_binding);
  }
  // Type check the receiver pattern
  if (f->is_method()) {
    TypeCheckPattern(&f->me_pattern(), std::nullopt, function_scope);
  }
  // Type check the parameter pattern
  TypeCheckPattern(&f->param_pattern(), std::nullopt, function_scope);

  // Evaluate the return type, if we can do so without examining the body.
  if (std::optional<Nonnull<Expression*>> return_expression =
          f->return_term().type_expression();
      return_expression.has_value()) {
    // We ignore the return value because return type expressions can't bring
    // new types into scope.
    TypeCheckExp(*return_expression, function_scope);
    // Should we be doing SetConstantValue instead? -Jeremy
    // And shouldn't the type of this be Type?
    f->return_term().set_static_type(
        InterpExp(*return_expression, arena_, trace_));
  } else if (f->return_term().is_omitted()) {
    f->return_term().set_static_type(TupleValue::Empty());
  } else {
    // We have to type-check the body in order to determine the return type.
    if (!f->body().has_value()) {
      FATAL_COMPILATION_ERROR(f->return_term().source_loc())
          << "Function declaration has deduced return type but no body";
    }
    TypeCheckStmt(*f->body(), function_scope);
    if (!f->return_term().is_omitted()) {
      ExpectReturnOnAllPaths(f->body(), f->source_loc());
    }
  }

  ExpectIsConcreteType(f->source_loc(), &f->return_term().static_type());
  f->set_static_type(arena_->New<FunctionType>(
      f->deduced_parameters(), &f->param_pattern().static_type(),
      &f->return_term().static_type(), impl_bindings));
  SetConstantValue(f, arena_->New<FunctionValue>(f));

  if (f->name() == "Main") {
    if (!f->return_term().type_expression().has_value()) {
      FATAL_COMPILATION_ERROR(f->return_term().source_loc())
          << "`Main` must have an explicit return type";
    }
    ExpectExactType(f->return_term().source_loc(), "return type of `Main`",
                    arena_->New<IntType>(), &f->return_term().static_type());
    // TODO: Check that main doesn't have any parameters.
  }

  if (trace_) {
    llvm::outs() << "** finished declaring function " << f->name()
                 << " of type " << f->static_type() << "\n";
  }
  return;
}

void TypeChecker::TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                               ImplScope& impl_scope) {
  if (trace_) {
    llvm::outs() << "** checking function " << f->name() << "\n";
  }
  // if f->return_term().is_auto(), the function body was already
  // type checked in DeclareFunctionDeclaration
  if (f->body().has_value() && !f->return_term().is_auto()) {
    // Bring the impl's into scope
    ImplScope function_scope;
    function_scope.AddParent(&impl_scope);
    for (Nonnull<const ImplBinding*> impl_binding :
         cast<FunctionType>(f->static_type()).impl_bindings()) {
      CHECK(impl_binding->type_var()->compile_time_value().has_value());
      function_scope.Add(impl_binding->interface(),
                         *impl_binding->type_var()->compile_time_value(),
                         impl_binding);
    }
    if (trace_)
      llvm::outs() << function_scope;
    TypeCheckStmt(*f->body(), function_scope);
    if (!f->return_term().is_omitted()) {
      ExpectReturnOnAllPaths(f->body(), f->source_loc());
    }
  }
  if (trace_) {
    llvm::outs() << "** finished checking function " << f->name() << "\n";
  }
  return;
}

void TypeChecker::DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                          ImplScope& enclosing_scope) {
  if (trace_) {
    llvm::outs() << "** declaring class " << class_decl->name() << "\n";
  }
  if (class_decl->type_params().has_value()) {
    ImplScope class_scope;
    class_scope.AddParent(&enclosing_scope);
    TypeCheckPattern(*class_decl->type_params(), std::nullopt, class_scope);
    if (trace_)
      llvm::outs() << class_scope;

    BindingMap type_args;
    Nonnull<NominalClassType*> class_type =
        arena_->New<NominalClassType>(class_decl, type_args);
    SetConstantValue(class_decl, class_type);
    class_decl->set_static_type(arena_->New<TypeOfClassType>(class_type));

    for (Nonnull<Declaration*> m : class_decl->members()) {
      DeclareDeclaration(m, class_scope);
    }

    // TODO: when/how to bring impls in generic class into scope?
  } else {
    // The declarations of the members may refer to the class, so we
    // must set the constant value of the class and its static type
    // before we start processing the members.
    BindingMap type_args;
    Nonnull<NominalClassType*> class_type =
        arena_->New<NominalClassType>(class_decl, type_args);
    SetConstantValue(class_decl, class_type);
    class_decl->set_static_type(arena_->New<TypeOfClassType>(class_type));

    for (Nonnull<Declaration*> m : class_decl->members()) {
      DeclareDeclaration(m, enclosing_scope);
    }
  }
  if (trace_) {
    llvm::outs() << "** finished declaring class " << class_decl->name()
                 << "\n";
  }
}

void TypeChecker::TypeCheckClassDeclaration(
    Nonnull<ClassDeclaration*> class_decl, ImplScope& impl_scope) {
  if (trace_) {
    llvm::outs() << "** checking class " << class_decl->name() << "\n";
  }
  ImplScope class_scope;
  class_scope.AddParent(&impl_scope);
  if (class_decl->type_params().has_value()) {
    PatternImpls(*class_decl->type_params(), class_scope);
  }
  if (trace_)
    llvm::outs() << class_scope;
  for (Nonnull<Declaration*> m : class_decl->members()) {
    TypeCheckDeclaration(m, class_scope);
  }
  if (trace_) {
    llvm::outs() << "** finished checking class " << class_decl->name() << "\n";
  }
}

void TypeChecker::DeclareInterfaceDeclaration(
    Nonnull<InterfaceDeclaration*> iface_decl, ImplScope& enclosing_scope) {
  Nonnull<InterfaceType*> iface_type = arena_->New<InterfaceType>(iface_decl);
  SetConstantValue(iface_decl, iface_type);
  iface_decl->set_static_type(arena_->New<TypeOfInterfaceType>(iface_type));

  // Process the Self parameter.
  TypeCheckExp(&iface_decl->self()->type(), enclosing_scope);
  iface_decl->self()->set_static_type(
      arena_->New<VariableType>(iface_decl->self()));
  // SetConstantValue(iface_decl->self(), &iface_decl->self()->static_type());
  iface_decl->self()->set_compile_time_value(
      &iface_decl->self()->static_type());

  for (Nonnull<Declaration*> m : iface_decl->members()) {
    DeclareDeclaration(m, enclosing_scope);
  }
}

void TypeChecker::TypeCheckInterfaceDeclaration(
    Nonnull<InterfaceDeclaration*> iface_decl, ImplScope& impl_scope) {
  for (Nonnull<Declaration*> m : iface_decl->members()) {
    TypeCheckDeclaration(m, impl_scope);
  }
}

void TypeChecker::DeclareImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                         ImplScope& enclosing_scope) {
  if (trace_) {
    llvm::outs() << "declaring " << *impl_decl << "\n";
  }
  TypeCheckExp(&impl_decl->interface(), enclosing_scope);
  Nonnull<const Value*> iface_type =
      InterpExp(&impl_decl->interface(), arena_, trace_);
  const auto& iface_decl = cast<InterfaceType>(*iface_type).declaration();
  impl_decl->set_interface_type(iface_type);

  TypeCheckExp(impl_decl->impl_type(), enclosing_scope);
  Nonnull<const Value*> impl_type_value =
      InterpExp(impl_decl->impl_type(), arena_, trace_);
  enclosing_scope.Add(iface_type, impl_type_value, impl_decl);

  for (Nonnull<Declaration*> m : impl_decl->members()) {
    DeclareDeclaration(m, enclosing_scope);
  }
  // Check that the interface is satisfied by the impl members
  for (Nonnull<Declaration*> m : iface_decl.members()) {
    if (std::optional<std::string> mem_name = GetName(*m);
        mem_name.has_value()) {
      if (std::optional<Nonnull<const Declaration*>> mem =
              FindMember(*mem_name, impl_decl->members());
          mem.has_value()) {
        std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>
            self_map;
        self_map[iface_decl.self()] = impl_type_value;
        Nonnull<const Value*> iface_mem_type =
            Substitute(self_map, &m->static_type());
        ExpectType((*mem)->source_loc(), "member of implementation",
                   iface_mem_type, &(*mem)->static_type());
      } else {
        FATAL_COMPILATION_ERROR(impl_decl->source_loc())
            << "implementation missing " << *mem_name;
      }
    }
  }
  impl_decl->set_constant_value(arena_->New<Witness>(impl_decl));
}

void TypeChecker::TypeCheckImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                           ImplScope& impl_scope) {
  if (trace_) {
    llvm::outs() << "checking " << *impl_decl << "\n";
  }
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    TypeCheckDeclaration(m, impl_scope);
  }
  if (trace_) {
    llvm::outs() << "finished checking impl\n";
  }
}

void TypeChecker::DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                           ImplScope& impl_scope) {
  std::vector<NamedValue> alternatives;
  for (Nonnull<AlternativeSignature*> alternative : choice->alternatives()) {
    TypeCheckExp(&alternative->signature(), impl_scope);
    auto signature = InterpExp(&alternative->signature(), arena_, trace_);
    alternatives.push_back({.name = alternative->name(), .value = signature});
  }
  auto ct = arena_->New<ChoiceType>(choice->name(), std::move(alternatives));
  SetConstantValue(choice, ct);
  choice->set_static_type(arena_->New<TypeOfChoiceType>(ct));
}

void TypeChecker::TypeCheckChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                             ImplScope& impl_scope) {
  // Nothing to do here, but perhaps that will change in the future?
}

void TypeChecker::TypeCheck(AST& ast) {
  ImplScope impl_scope;
  for (Nonnull<Declaration*> declaration : ast.declarations) {
    DeclareDeclaration(declaration, impl_scope);
  }
  for (Nonnull<Declaration*> decl : ast.declarations) {
    TypeCheckDeclaration(decl, impl_scope);
  }
  TypeCheckExp(*ast.main_call, impl_scope);
}

void TypeChecker::TypeCheckDeclaration(Nonnull<Declaration*> d,
                                       ImplScope& impl_scope) {
  switch (d->kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      TypeCheckInterfaceDeclaration(&cast<InterfaceDeclaration>(*d),
                                    impl_scope);
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      TypeCheckImplDeclaration(&cast<ImplDeclaration>(*d), impl_scope);
      break;
    }
    case DeclarationKind::FunctionDeclaration:
      TypeCheckFunctionDeclaration(&cast<FunctionDeclaration>(*d), impl_scope);
      return;
    case DeclarationKind::ClassDeclaration:
      TypeCheckClassDeclaration(&cast<ClassDeclaration>(*d), impl_scope);
      return;
    case DeclarationKind::ChoiceDeclaration:
      TypeCheckChoiceDeclaration(&cast<ChoiceDeclaration>(*d), impl_scope);
      return;
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Signals a type error if the initializing expression does not have
      // the declared type of the variable, otherwise returns this
      // declaration with annotated types.
      if (var.has_initializer()) {
        TypeCheckExp(&var.initializer(), impl_scope);
      }
      const auto* binding_type =
          dyn_cast<ExpressionPattern>(&var.binding().type());
      if (binding_type == nullptr) {
        // TODO: consider adding support for `auto`
        FATAL_COMPILATION_ERROR(var.source_loc())
            << "Type of a top-level variable must be an expression.";
      }
      if (var.has_initializer()) {
        ExpectType(var.source_loc(), "initializer of variable",
                   &var.static_type(), &var.initializer().static_type());
      }
      return;
    }
  }
}

void TypeChecker::DeclareDeclaration(Nonnull<Declaration*> d,
                                     ImplScope& impl_scope) {
  switch (d->kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      auto& iface_decl = cast<InterfaceDeclaration>(*d);
      DeclareInterfaceDeclaration(&iface_decl, impl_scope);
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl_decl = cast<ImplDeclaration>(*d);
      DeclareImplDeclaration(&impl_decl, impl_scope);
      break;
    }
    case DeclarationKind::FunctionDeclaration: {
      auto& func_def = cast<FunctionDeclaration>(*d);
      DeclareFunctionDeclaration(&func_def, impl_scope);
      break;
    }

    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(*d);
      DeclareClassDeclaration(&class_decl, impl_scope);
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(*d);
      DeclareChoiceDeclaration(&choice, impl_scope);
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Associate the variable name with it's declared type in the
      // compile-time symbol table.
      Expression& type =
          cast<ExpressionPattern>(var.binding().type()).expression();
      TypeCheckPattern(&var.binding(), std::nullopt, impl_scope);
      Nonnull<const Value*> declared_type = InterpExp(&type, arena_, trace_);
      var.set_static_type(declared_type);
      break;
    }
  }
}

template <typename T>
void TypeChecker::SetConstantValue(Nonnull<T*> value_node,
                                   Nonnull<const Value*> value) {
  std::optional<Nonnull<const Value*>> old_value = value_node->constant_value();
  CHECK(!old_value.has_value());
  value_node->set_constant_value(value);
  CHECK(constants_.insert(value_node).second);
}

void TypeChecker::PrintConstants(llvm::raw_ostream& out) {
  llvm::ListSeparator sep;
  for (const auto& value_node : constants_) {
    out << sep << value_node;
  }
}

}  // namespace Carbon
