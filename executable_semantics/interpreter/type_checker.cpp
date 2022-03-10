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
#include "llvm/Support/Error.h"

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

static auto ExpectExactType(SourceLocation source_loc,
                                   const std::string& context,
                                   Nonnull<const Value*> expected,
                                   Nonnull<const Value*> actual) ->llvm::Error{
  if (!TypeEqual(expected, actual)) {
    return FATAL_COMPILATION_ERROR(source_loc)
           << "type error in " << context << "\n"
           << "expected: " << *expected << "\n"
           << "actual: " << *actual;
  }
  return llvm::Error::success();
}

static auto ExpectPointerType(SourceLocation source_loc,
                                     const std::string& context,
                                     Nonnull<const Value*> actual) ->llvm::Error{
  if (actual->kind() != Value::Kind::PointerType) {
    return FATAL_COMPILATION_ERROR(source_loc)
           << "type error in " << context << "\n"
           << "expected a pointer type\n"
           << "actual: " << *actual;
  }
  return llvm::Error::success();
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

auto TypeChecker::ExpectIsConcreteType(SourceLocation source_loc,
                                       Nonnull<const Value*> value)
    -> llvm::Error {
  if (!IsConcreteType(value)) {
    return FATAL_COMPILATION_ERROR(source_loc)
           << "Expected a type, but got " << *value;
  } else {
    return llvm::Error::success();
  }
}

// Returns true if *source is implicitly convertible to *destination. *source
// and *destination must be concrete types.
static auto IsImplicitlyConvertible(Nonnull<const Value*> source,
                                    Nonnull<const Value*> destination) -> bool;

// Returns true if source_fields and destination_fields contain the same set
// of names, and each value in source_fields is implicitly convertible to
// the corresponding value in destination_fields. All values in both arguments
// must be types.
static auto FieldTypesImplicitlyConvertible(
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

static auto IsImplicitlyConvertible(Nonnull<const Value*> source,
                                    Nonnull<const Value*> destination) -> bool {
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
    default:
      return false;
  }
}

static auto ExpectType(SourceLocation source_loc, const std::string& context,
                       Nonnull<const Value*> expected,
                       Nonnull<const Value*> actual) -> llvm::Error {
  if (!IsImplicitlyConvertible(actual, expected)) {
    return FATAL_COMPILATION_ERROR(source_loc)
           << "type error in " << context << ": "
           << "'" << *actual << "' is not implicitly convertible to '"
           << *expected << "'";
  } else {
    return llvm::Error::success();
  }
}

auto TypeChecker::ArgumentDeduction(SourceLocation source_loc,
                                    BindingMap& deduced,
                                    Nonnull<const Value*> param,
                                    Nonnull<const Value*> arg) -> llvm::Error {
  switch (param->kind()) {
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*param);
      auto [it, success] = deduced.insert({&var_type.binding(), arg});
      if (!success) {
        // TODO: can we allow implicit conversions here?
        RETURN_IF_ERROR(
            ExpectExactType(source_loc, "argument deduction", it->second, arg));
      }
      return llvm::Error::success();
    }
    case Value::Kind::TupleValue: {
      if (arg->kind() != Value::Kind::TupleValue) {
        return FATAL_COMPILATION_ERROR(source_loc)
               << "type error in argument deduction\n"
               << "expected: " << *param << "\n"
               << "actual: " << *arg;
      }
      const auto& param_tup = cast<TupleValue>(*param);
      const auto& arg_tup = cast<TupleValue>(*arg);
      if (param_tup.elements().size() != arg_tup.elements().size()) {
        return FATAL_COMPILATION_ERROR(source_loc)
               << "mismatch in tuple sizes, expected "
               << param_tup.elements().size() << " but got "
               << arg_tup.elements().size();
      }
      for (size_t i = 0; i < param_tup.elements().size(); ++i) {
        RETURN_IF_ERROR(ArgumentDeduction(source_loc, deduced,
                                          param_tup.elements()[i],
                                          arg_tup.elements()[i]));
      }
      return llvm::Error::success();
    }
    case Value::Kind::StructType: {
      if (arg->kind() != Value::Kind::StructType) {
        return FATAL_COMPILATION_ERROR(source_loc)
               << "type error in argument deduction\n"
               << "expected: " << *param << "\n"
               << "actual: " << *arg;
      }
      const auto& param_struct = cast<StructType>(*param);
      const auto& arg_struct = cast<StructType>(*arg);
      if (param_struct.fields().size() != arg_struct.fields().size()) {
        return FATAL_COMPILATION_ERROR(source_loc)
               << "mismatch in struct field counts, expected "
               << param_struct.fields().size() << " but got "
               << arg_struct.fields().size();
      }
      for (size_t i = 0; i < param_struct.fields().size(); ++i) {
        if (param_struct.fields()[i].name != arg_struct.fields()[i].name) {
          return FATAL_COMPILATION_ERROR(source_loc)
                 << "mismatch in field names, " << param_struct.fields()[i].name
                 << " != " << arg_struct.fields()[i].name;
        }
        RETURN_IF_ERROR(ArgumentDeduction(source_loc, deduced,
                                          param_struct.fields()[i].value,
                                          arg_struct.fields()[i].value));
      }
      return llvm::Error::success();
    }
    case Value::Kind::FunctionType: {
      if (arg->kind() != Value::Kind::FunctionType) {
        return FATAL_COMPILATION_ERROR(source_loc)
               << "type error in argument deduction\n"
               << "expected: " << *param << "\n"
               << "actual: " << *arg;
      }
      const auto& param_fn = cast<FunctionType>(*param);
      const auto& arg_fn = cast<FunctionType>(*arg);
      // TODO: handle situation when arg has deduced parameters.
      RETURN_IF_ERROR(ArgumentDeduction(
          source_loc, deduced, &param_fn.parameters(), &arg_fn.parameters()));
      RETURN_IF_ERROR(ArgumentDeduction(
          source_loc, deduced, &param_fn.return_type(), &arg_fn.return_type()));
      return llvm::Error::success();
    }
    case Value::Kind::PointerType: {
      if (arg->kind() != Value::Kind::PointerType) {
        return FATAL_COMPILATION_ERROR(source_loc)
               << "type error in argument deduction\n"
               << "expected: " << *param << "\n"
               << "actual: " << *arg;
      }
      return ArgumentDeduction(source_loc, deduced,
                               &cast<PointerType>(*param).type(),
                               &cast<PointerType>(*arg).type());
    }
    // Nothing to do in the case for `auto`.
    case Value::Kind::AutoType: {
      return llvm::Error::success();
    }
    // For the following cases, we check for type convertability.
    case Value::Kind::ContinuationType:
    case Value::Kind::NominalClassType:
    case Value::Kind::InterfaceType:
    case Value::Kind::ChoiceType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
      return ExpectType(source_loc, "argument deduction", param, arg);
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
      FATAL() << "In ArgumentDeduction: expected type, not value " << *param;
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
    case Value::Kind::AutoType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::NominalClassType:
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

auto TypeChecker::TypeCheckExp(Nonnull<Expression*> e,
                               const ImplScope& impl_scope) -> llvm::Error {
  if (trace_) {
    llvm::outs() << "checking expression " << *e;
    llvm::outs() << "\nconstants: ";
    PrintConstants(llvm::outs());
    llvm::outs() << "\n";
  }
  switch (e->kind()) {
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(*e);
      RETURN_IF_ERROR(TypeCheckExp(&index.aggregate(), impl_scope));
      const Value& aggregate_type = index.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::TupleValue: {
          const auto& tuple_type = cast<TupleValue>(aggregate_type);
          ASSIGN_OR_RETURN(auto offset_value,
                           InterpExp(&index.offset(), arena_, trace_));
          int i = cast<IntValue>(*offset_value).value();
          if (i < 0 || i >= static_cast<int>(tuple_type.elements().size())) {
            return FATAL_COMPILATION_ERROR(e->source_loc())
                   << "index " << i << " is out of range for type "
                   << tuple_type;
          }
          index.set_static_type(tuple_type.elements()[i]);
          index.set_value_category(index.aggregate().value_category());
          return llvm::Error::success();
        }
        default:
          return FATAL_COMPILATION_ERROR(e->source_loc()) << "expected a tuple";
      }
    }
    case ExpressionKind::TupleLiteral: {
      std::vector<Nonnull<const Value*>> arg_types;
      for (auto& arg : cast<TupleLiteral>(*e).fields()) {
        RETURN_IF_ERROR(TypeCheckExp(arg, impl_scope));
        arg_types.push_back(&arg->static_type());
      }
      e->set_static_type(arena_->New<TupleValue>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return llvm::Error::success();
    }
    case ExpressionKind::StructLiteral: {
      std::vector<NamedValue> arg_types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        RETURN_IF_ERROR(TypeCheckExp(&arg.expression(), impl_scope));
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      e->set_static_type(arena_->New<StructType>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return llvm::Error::success();
    }
    case ExpressionKind::StructTypeLiteral: {
      auto& struct_type = cast<StructTypeLiteral>(*e);
      for (auto& arg : struct_type.fields()) {
        RETURN_IF_ERROR(TypeCheckExp(&arg.expression(), impl_scope));
        ASSIGN_OR_RETURN(auto value,
                         InterpExp(&arg.expression(), arena_, trace_));
        RETURN_IF_ERROR(
            ExpectIsConcreteType(arg.expression().source_loc(), value));
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
      return llvm::Error::success();
    }
    case ExpressionKind::FieldAccessExpression: {
      auto& access = cast<FieldAccessExpression>(*e);
      RETURN_IF_ERROR(TypeCheckExp(&access.aggregate(), impl_scope));
      const Value& aggregate_type = access.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(aggregate_type);
          for (const auto& [field_name, field_type] : struct_type.fields()) {
            if (access.field() == field_name) {
              access.set_static_type(field_type);
              access.set_value_category(access.aggregate().value_category());
              return llvm::Error::success();
            }
          }
          return FATAL_COMPILATION_ERROR(access.source_loc())
                 << "struct " << struct_type << " does not have a field named "
                 << access.field();
        }
        case Value::Kind::NominalClassType: {
          const auto& t_class = cast<NominalClassType>(aggregate_type);
          if (std::optional<Nonnull<const Declaration*>> member =
                  FindMember(access.field(), t_class.declaration().members());
              member.has_value()) {
            access.set_static_type(&(*member)->static_type());
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
            return llvm::Error::success();
          } else {
            return FATAL_COMPILATION_ERROR(e->source_loc())
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
            return FATAL_COMPILATION_ERROR(e->source_loc())
                   << "choice " << choice.name()
                   << " does not have a field named " << access.field();
          }
          access.set_static_type(arena_->New<FunctionType>(
              std::vector<Nonnull<const GenericBinding*>>(), *parameter_types,
              &aggregate_type, std::vector<Nonnull<const ImplBinding*>>()));
          access.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
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
                access.set_static_type(&(*member)->static_type());
                access.set_value_category(ValueCategory::Let);
                return llvm::Error::success();
              }
              default:
                break;
            }
            return FATAL_COMPILATION_ERROR(access.source_loc())
                   << access.field() << " is not a class function";
          } else {
            return FATAL_COMPILATION_ERROR(access.source_loc())
                   << class_type << " does not have a class function named "
                   << access.field();
          }
        }
        case Value::Kind::VariableType: {
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
                access.set_impl(*var_type.binding().impl_binding());
                return llvm::Error::success();
              } else {
                return FATAL_COMPILATION_ERROR(e->source_loc())
                       << "field access, " << access.field() << " not in "
                       << iface_decl.name();
              }
              break;
            }
            default:
              break;
          }
          return FATAL_COMPILATION_ERROR(e->source_loc())
                 << "field access, unexpected " << aggregate_type << " in "
                 << *e;
          break;
        }
        default:
          return FATAL_COMPILATION_ERROR(e->source_loc())
                 << "field access, unexpected " << aggregate_type << " in "
                 << *e;
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
          return FATAL_COMPILATION_ERROR(ident.source_loc())
                 << "Function calls itself, but has a deduced return type";
        }
      }
      ident.set_static_type(&ident.value_node().static_type());
      ident.set_value_category(ident.value_node().value_category());
      return llvm::Error::success();
    }
    case ExpressionKind::IntLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<IntType>());
      return llvm::Error::success();
    case ExpressionKind::BoolLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<BoolType>());
      return llvm::Error::success();
    case ExpressionKind::PrimitiveOperatorExpression: {
      auto& op = cast<PrimitiveOperatorExpression>(*e);
      std::vector<Nonnull<const Value*>> ts;
      for (Nonnull<Expression*> argument : op.arguments()) {
        RETURN_IF_ERROR(TypeCheckExp(argument, impl_scope));
        ts.push_back(&argument->static_type());
      }
      switch (op.op()) {
        case Operator::Neg:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "negation",
                                          arena_->New<IntType>(), ts[0]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::Add:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "addition(1)",
                                          arena_->New<IntType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "addition(2)",
                                          arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::Sub:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "subtraction(1)",
                                          arena_->New<IntType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "subtraction(2)",
                                          arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::Mul:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "multiplication(1)",
                                          arena_->New<IntType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "multiplication(2)",
                                          arena_->New<IntType>(), ts[1]));
          op.set_static_type(arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::And:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "&&(1)",
                                          arena_->New<BoolType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "&&(2)",
                                          arena_->New<BoolType>(), ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::Or:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "||(1)",
                                          arena_->New<BoolType>(), ts[0]));
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "||(2)",
                                          arena_->New<BoolType>(), ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::Not:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "!",
                                          arena_->New<BoolType>(), ts[0]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::Eq:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "==", ts[0], ts[1]));
          op.set_static_type(arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::Deref:
          RETURN_IF_ERROR(ExpectPointerType(e->source_loc(), "*", ts[0]));
          op.set_static_type(&cast<PointerType>(*ts[0]).type());
          op.set_value_category(ValueCategory::Var);
          return llvm::Error::success();
        case Operator::Ptr:
          RETURN_IF_ERROR(ExpectExactType(e->source_loc(), "*",
                                          arena_->New<TypeType>(), ts[0]));
          op.set_static_type(arena_->New<TypeType>());
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        case Operator::AddressOf:
          if (op.arguments()[0]->value_category() != ValueCategory::Var) {
            return FATAL_COMPILATION_ERROR(op.arguments()[0]->source_loc())
                   << "Argument to " << ToString(op.op())
                   << " should be an lvalue.";
          }
          op.set_static_type(arena_->New<PointerType>(ts[0]));
          op.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
      }
      break;
    }
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(*e);
      RETURN_IF_ERROR(TypeCheckExp(&call.function(), impl_scope));
      switch (call.function().static_type().kind()) {
        case Value::Kind::FunctionType: {
          const auto& fun_t = cast<FunctionType>(call.function().static_type());
          RETURN_IF_ERROR(TypeCheckExp(&call.argument(), impl_scope));
          Nonnull<const Value*> parameters = &fun_t.parameters();
          Nonnull<const Value*> return_type = &fun_t.return_type();
          if (!fun_t.deduced().empty()) {
            BindingMap deduced_args;
            RETURN_IF_ERROR(ArgumentDeduction(e->source_loc(), deduced_args,
                                              parameters,
                                              &call.argument().static_type()));
            for (Nonnull<const GenericBinding*> deduced_param :
                 fun_t.deduced()) {
              // TODO: change the following to a CHECK once the real checking
              // has been added to the type checking of function signatures.
              if (auto it = deduced_args.find(deduced_param);
                  it == deduced_args.end()) {
                return FATAL_COMPILATION_ERROR(e->source_loc())
                       << "could not deduce type argument for type parameter "
                       << deduced_param->name();
              }
            }
            parameters = Substitute(deduced_args, parameters);
            return_type = Substitute(deduced_args, return_type);
            // Find impls for all the impl bindings of the function
            std::map<Nonnull<const ImplBinding*>, ValueNodeView> impls;
            for (Nonnull<const ImplBinding*> impl_binding :
                 fun_t.impl_bindings()) {
              switch (impl_binding->interface()->kind()) {
                case Value::Kind::InterfaceType: {
                  ASSIGN_OR_RETURN(
                      ValueNodeView impl,
                      impl_scope.Resolve(impl_binding->interface(),
                                         deduced_args[impl_binding->type_var()],
                                         e->source_loc()));
                  impls.emplace(impl_binding, impl);
                  break;
                }
                case Value::Kind::TypeType:
                  break;
                default:
                  return FATAL_COMPILATION_ERROR(e->source_loc())
                         << "unexpected type of deduced parameter "
                         << *impl_binding->interface();
              }
            }
            call.set_impls(impls);
          } else {
            RETURN_IF_ERROR(ExpectType(e->source_loc(), "call", parameters,
                                       &call.argument().static_type()));
          }
          call.set_static_type(return_type);
          call.set_value_category(ValueCategory::Let);
          return llvm::Error::success();
        }
        default: {
          return FATAL_COMPILATION_ERROR(e->source_loc())
                 << "in call, expected a function\n"
                 << *e;
        }
      }
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fn = cast<FunctionTypeLiteral>(*e);
      ASSIGN_OR_RETURN(Nonnull<const Value*> param_type,
                       InterpExp(&fn.parameter(), arena_, trace_));
      RETURN_IF_ERROR(
          ExpectIsConcreteType(fn.parameter().source_loc(), param_type));
      ASSIGN_OR_RETURN(Nonnull<const Value*> ret_type,
                       InterpExp(&fn.return_type(), arena_, trace_));
      RETURN_IF_ERROR(
          ExpectIsConcreteType(fn.return_type().source_loc(), ret_type));
      fn.set_static_type(arena_->New<TypeType>());
      fn.set_value_category(ValueCategory::Let);
      return llvm::Error::success();
    }
    case ExpressionKind::StringLiteral:
      e->set_static_type(arena_->New<StringType>());
      e->set_value_category(ValueCategory::Let);
      return llvm::Error::success();
    case ExpressionKind::IntrinsicExpression: {
      auto& intrinsic_exp = cast<IntrinsicExpression>(*e);
      RETURN_IF_ERROR(TypeCheckExp(&intrinsic_exp.args(), impl_scope));
      switch (cast<IntrinsicExpression>(*e).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          if (intrinsic_exp.args().fields().size() != 1) {
            return FATAL_COMPILATION_ERROR(e->source_loc())
                   << "__intrinsic_print takes 1 argument";
          }
          RETURN_IF_ERROR(
              ExpectType(e->source_loc(), "__intrinsic_print argument",
                         arena_->New<StringType>(),
                         &intrinsic_exp.args().fields()[0]->static_type()));
          e->set_static_type(TupleValue::Empty());
          e->set_value_category(ValueCategory::Let);
          return llvm::Error::success();
      }
    }
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
      e->set_value_category(ValueCategory::Let);
      e->set_static_type(arena_->New<TypeType>());
      return llvm::Error::success();
    case ExpressionKind::IfExpression: {
      auto& if_expr = cast<IfExpression>(*e);
      RETURN_IF_ERROR(TypeCheckExp(if_expr.condition(), impl_scope));
      RETURN_IF_ERROR(ExpectType(if_expr.source_loc(), "condition of `if`",
                                 arena_->New<BoolType>(),
                                 &if_expr.condition()->static_type()));

      // TODO: Compute the common type and convert both operands to it.
      RETURN_IF_ERROR(TypeCheckExp(if_expr.then_expression(), impl_scope));
      RETURN_IF_ERROR(TypeCheckExp(if_expr.else_expression(), impl_scope));
      RETURN_IF_ERROR(
          ExpectExactType(e->source_loc(), "expression of `if` expression",
                          &if_expr.then_expression()->static_type(),
                          &if_expr.else_expression()->static_type()));
      e->set_static_type(&if_expr.then_expression()->static_type());
      e->set_value_category(ValueCategory::Let);
      return llvm::Error::success();
    }
    case ExpressionKind::UnimplementedExpression:
      FATAL() << "Unimplemented: " << *e;
  }
}

auto TypeChecker::TypeCheckPattern(
    Nonnull<Pattern*> p, std::optional<Nonnull<const Value*>> expected,
    const ImplScope& impl_scope) -> llvm::Error {
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
      return llvm::Error::success();
    }
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(*p);
      RETURN_IF_ERROR(
          TypeCheckPattern(&binding.type(), std::nullopt, impl_scope));
      ASSIGN_OR_RETURN(Nonnull<const Value*> type,
                       InterpPattern(&binding.type(), arena_, trace_));
      if (expected) {
        if (IsConcreteType(type)) {
          RETURN_IF_ERROR(
              ExpectType(p->source_loc(), "name binding", type, *expected));
        } else {
          ASSIGN_OR_RETURN(
              const bool matches,
              PatternMatch(type, *expected, binding.type().source_loc(),
                           std::nullopt));
          if (!matches) {
            return FATAL_COMPILATION_ERROR(binding.type().source_loc())
                   << "Type pattern '" << *type
                   << "' does not match actual type '" << **expected << "'";
          }
          type = *expected;
        }
      }
      RETURN_IF_ERROR(ExpectIsConcreteType(binding.source_loc(), type));
      binding.set_static_type(type);
      ASSIGN_OR_RETURN(Nonnull<const Value*> binding_value,
                       InterpPattern(&binding, arena_, trace_));
      SetValue(&binding, binding_value);
      return llvm::Error::success();
    }
    case PatternKind::TuplePattern: {
      auto& tuple = cast<TuplePattern>(*p);
      std::vector<Nonnull<const Value*>> field_types;
      if (expected && (*expected)->kind() != Value::Kind::TupleValue) {
        return FATAL_COMPILATION_ERROR(p->source_loc())
               << "didn't expect a tuple";
      }
      if (expected && tuple.fields().size() !=
                          cast<TupleValue>(**expected).elements().size()) {
        return FATAL_COMPILATION_ERROR(tuple.source_loc())
               << "tuples of different length";
      }
      for (size_t i = 0; i < tuple.fields().size(); ++i) {
        Nonnull<Pattern*> field = tuple.fields()[i];
        std::optional<Nonnull<const Value*>> expected_field_type;
        if (expected) {
          expected_field_type = cast<TupleValue>(**expected).elements()[i];
        }
        RETURN_IF_ERROR(
            TypeCheckPattern(field, expected_field_type, impl_scope));
        field_types.push_back(&field->static_type());
      }
      tuple.set_static_type(arena_->New<TupleValue>(std::move(field_types)));
      ASSIGN_OR_RETURN(Nonnull<const Value*> tuple_value,
                       InterpPattern(&tuple, arena_, trace_));
      SetValue(&tuple, tuple_value);
      return llvm::Error::success();
    }
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(*p);
      RETURN_IF_ERROR(TypeCheckExp(&alternative.choice_type(), impl_scope));
      if (alternative.choice_type().static_type().kind() !=
          Value::Kind::TypeOfChoiceType) {
        return FATAL_COMPILATION_ERROR(alternative.source_loc())
               << "alternative pattern does not name a choice type.";
      }
      if (expected) {
        RETURN_IF_ERROR(ExpectExactType(
            alternative.source_loc(), "alternative pattern", *expected,
            &alternative.choice_type().static_type()));
      }
      const ChoiceType& choice_type =
          cast<TypeOfChoiceType>(alternative.choice_type().static_type())
              .choice_type();
      std::optional<Nonnull<const Value*>> parameter_types =
          cast<ChoiceType>(choice_type)
              .FindAlternative(alternative.alternative_name());
      if (parameter_types == std::nullopt) {
        return FATAL_COMPILATION_ERROR(alternative.source_loc())
               << "'" << alternative.alternative_name()
               << "' is not an alternative of " << choice_type;
      }
      RETURN_IF_ERROR(TypeCheckPattern(&alternative.arguments(),
                                       *parameter_types, impl_scope));
      alternative.set_static_type(&choice_type);
      ASSIGN_OR_RETURN(Nonnull<const Value*> alternative_value,
                       InterpPattern(&alternative, arena_, trace_));
      SetValue(&alternative, alternative_value);
      return llvm::Error::success();
    }
    case PatternKind::ExpressionPattern: {
      auto& expression = cast<ExpressionPattern>(*p).expression();
      RETURN_IF_ERROR(TypeCheckExp(&expression, impl_scope));
      p->set_static_type(&expression.static_type());
      ASSIGN_OR_RETURN(Nonnull<const Value*> expr_value,
                       InterpPattern(p, arena_, trace_));
      SetValue(p, expr_value);
      return llvm::Error::success();
    }
  }
}

auto TypeChecker::TypeCheckStmt(Nonnull<Statement*> s,
                                const ImplScope& impl_scope) -> llvm::Error {
  if (trace_) {
    llvm::outs() << "checking statement " << *s << "\n";
  }
  switch (s->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&match.expression(), impl_scope));
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        RETURN_IF_ERROR(TypeCheckPattern(
            &clause.pattern(), &match.expression().static_type(), impl_scope));
        RETURN_IF_ERROR(TypeCheckStmt(&clause.statement(), impl_scope));
      }
      return llvm::Error::success();
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&while_stmt.condition(), impl_scope));
      RETURN_IF_ERROR(ExpectType(s->source_loc(), "condition of `while`",
                                 arena_->New<BoolType>(),
                                 &while_stmt.condition().static_type()));
      RETURN_IF_ERROR(TypeCheckStmt(&while_stmt.body(), impl_scope));
      return llvm::Error::success();
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return llvm::Error::success();
    case StatementKind::Block: {
      auto& block = cast<Block>(*s);
      for (auto* block_statement : block.statements()) {
        RETURN_IF_ERROR(TypeCheckStmt(block_statement, impl_scope));
      }
      return llvm::Error::success();
    }
    case StatementKind::VariableDefinition: {
      auto& var = cast<VariableDefinition>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&var.init(), impl_scope));
      const Value& rhs_ty = var.init().static_type();
      RETURN_IF_ERROR(TypeCheckPattern(&var.pattern(), &rhs_ty, impl_scope));
      return llvm::Error::success();
    }
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&assign.rhs(), impl_scope));
      RETURN_IF_ERROR(TypeCheckExp(&assign.lhs(), impl_scope));
      RETURN_IF_ERROR(ExpectType(s->source_loc(), "assign",
                                 &assign.lhs().static_type(),
                                 &assign.rhs().static_type()));
      if (assign.lhs().value_category() != ValueCategory::Var) {
        return FATAL_COMPILATION_ERROR(assign.source_loc())
               << "Cannot assign to rvalue '" << assign.lhs() << "'";
      }
      return llvm::Error::success();
    }
    case StatementKind::ExpressionStatement: {
      RETURN_IF_ERROR(TypeCheckExp(&cast<ExpressionStatement>(*s).expression(),
                                   impl_scope));
      return llvm::Error::success();
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&if_stmt.condition(), impl_scope));
      RETURN_IF_ERROR(ExpectType(s->source_loc(), "condition of `if`",
                                 arena_->New<BoolType>(),
                                 &if_stmt.condition().static_type()));
      RETURN_IF_ERROR(TypeCheckStmt(&if_stmt.then_block(), impl_scope));
      if (if_stmt.else_block()) {
        RETURN_IF_ERROR(TypeCheckStmt(*if_stmt.else_block(), impl_scope));
      }
      return llvm::Error::success();
    }
    case StatementKind::Return: {
      auto& ret = cast<Return>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&ret.expression(), impl_scope));
      ReturnTerm& return_term = ret.function().return_term();
      if (return_term.is_auto()) {
        return_term.set_static_type(&ret.expression().static_type());
      } else {
        RETURN_IF_ERROR(ExpectType(s->source_loc(), "return",
                                   &return_term.static_type(),
                                   &ret.expression().static_type()));
      }
      return llvm::Error::success();
    }
    case StatementKind::Continuation: {
      auto& cont = cast<Continuation>(*s);
      RETURN_IF_ERROR(TypeCheckStmt(&cont.body(), impl_scope));
      cont.set_static_type(arena_->New<ContinuationType>());
      return llvm::Error::success();
    }
    case StatementKind::Run: {
      auto& run = cast<Run>(*s);
      RETURN_IF_ERROR(TypeCheckExp(&run.argument(), impl_scope));
      RETURN_IF_ERROR(ExpectType(s->source_loc(), "argument of `run`",
                                 arena_->New<ContinuationType>(),
                                 &run.argument().static_type()));
      return llvm::Error::success();
    }
    case StatementKind::Await: {
      // nothing to do here
      return llvm::Error::success();
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

auto TypeChecker::ExpectReturnOnAllPaths(
    std::optional<Nonnull<Statement*>> opt_stmt, SourceLocation source_loc)
    -> llvm::Error {
  if (!opt_stmt) {
    return FATAL_COMPILATION_ERROR(source_loc)
           << "control-flow reaches end of function that provides a `->` "
              "return "
              "type without reaching a return statement";
  }
  Nonnull<Statement*> stmt = *opt_stmt;
  switch (stmt->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*stmt);
      if (!IsExhaustive(match)) {
        return FATAL_COMPILATION_ERROR(source_loc)
               << "non-exhaustive match may allow control-flow to reach the "
                  "end "
                  "of a function that provides a `->` return type";
      }
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        RETURN_IF_ERROR(
            ExpectReturnOnAllPaths(&clause.statement(), stmt->source_loc()));
      }
      return llvm::Error::success();
    }
    case StatementKind::Block: {
      auto& block = cast<Block>(*stmt);
      if (block.statements().empty()) {
        return FATAL_COMPILATION_ERROR(stmt->source_loc())
               << "control-flow reaches end of function that provides a `->` "
                  "return type without reaching a return statement";
      }
      RETURN_IF_ERROR(ExpectReturnOnAllPaths(
          block.statements()[block.statements().size() - 1],
          block.source_loc()));
      return llvm::Error::success();
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*stmt);
      RETURN_IF_ERROR(
          ExpectReturnOnAllPaths(&if_stmt.then_block(), stmt->source_loc()));
      RETURN_IF_ERROR(
          ExpectReturnOnAllPaths(if_stmt.else_block(), stmt->source_loc()));
      return llvm::Error::success();
    }
    case StatementKind::Return:
      return llvm::Error::success();
    case StatementKind::Continuation:
    case StatementKind::Run:
    case StatementKind::Await:
      return llvm::Error::success();
    case StatementKind::Assign:
    case StatementKind::ExpressionStatement:
    case StatementKind::While:
    case StatementKind::Break:
    case StatementKind::Continue:
    case StatementKind::VariableDefinition:
      return FATAL_COMPILATION_ERROR(stmt->source_loc())
             << "control-flow reaches end of function that provides a `->` "
                "return type without reaching a return statement";
  }
}

// TODO: Add checking to function definitions to ensure that
//   all deduced type parameters will be deduced.
auto TypeChecker::DeclareFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                             const ImplScope& impl_scope)
    -> llvm::Error {
  if (trace_) {
    llvm::outs() << "** declaring function " << f->name() << "\n";
  }
  // Bring the deduced parameters into scope
  for (Nonnull<GenericBinding*> deduced : f->deduced_parameters()) {
    RETURN_IF_ERROR(TypeCheckExp(&deduced->type(), impl_scope));
    SetConstantValue(deduced, arena_->New<VariableType>(deduced));
    ASSIGN_OR_RETURN(Nonnull<const Value*> deduced_type,
                     InterpExp(&deduced->type(), arena_, trace_));
    deduced->set_static_type(deduced_type);
  }
  // Type check the receiver pattern
  if (f->is_method()) {
    RETURN_IF_ERROR(
        TypeCheckPattern(&f->me_pattern(), std::nullopt, impl_scope));
  }
  // Type check the parameter pattern
  RETURN_IF_ERROR(
      TypeCheckPattern(&f->param_pattern(), std::nullopt, impl_scope));

  // Create the impl_bindings
  std::vector<Nonnull<const ImplBinding*>> impl_bindings;
  for (Nonnull<GenericBinding*> deduced : f->deduced_parameters()) {
    Nonnull<ImplBinding*> impl_binding = arena_->New<ImplBinding>(
        deduced->source_loc(), deduced, &deduced->static_type());
    deduced->set_impl_binding(impl_binding);
    impl_binding->set_static_type(&deduced->static_type());
    impl_bindings.push_back(impl_binding);
  }

  // Evaluate the return type, if we can do so without examining the body.
  if (std::optional<Nonnull<Expression*>> return_expression =
          f->return_term().type_expression();
      return_expression.has_value()) {
    // We ignore the return value because return type expressions can't bring
    // new types into scope.
    RETURN_IF_ERROR(TypeCheckExp(*return_expression, impl_scope));
    // Should we be doing SetConstantValue instead? -Jeremy
    // And shouldn't the type of this be Type?
    ASSIGN_OR_RETURN(Nonnull<const Value*> ret_type,
                     InterpExp(*return_expression, arena_, trace_));
    f->return_term().set_static_type(ret_type);
  } else if (f->return_term().is_omitted()) {
    f->return_term().set_static_type(TupleValue::Empty());
  } else {
    // We have to type-check the body in order to determine the return type.
    if (!f->body().has_value()) {
      return FATAL_COMPILATION_ERROR(f->return_term().source_loc())
             << "Function declaration has deduced return type but no body";
    }
    // Bring the impl bindings into scope
    ImplScope function_scope;
    function_scope.AddParent(&impl_scope);
    for (Nonnull<const ImplBinding*> impl_binding : impl_bindings) {
      function_scope.Add(impl_binding->interface(),
                         *impl_binding->type_var()->constant_value(),
                         impl_binding);
    }
    RETURN_IF_ERROR(TypeCheckStmt(*f->body(), impl_scope));
    if (!f->return_term().is_omitted()) {
      RETURN_IF_ERROR(ExpectReturnOnAllPaths(f->body(), f->source_loc()));
    }
  }

  RETURN_IF_ERROR(
      ExpectIsConcreteType(f->source_loc(), &f->return_term().static_type()));
  f->set_static_type(arena_->New<FunctionType>(
      f->deduced_parameters(), &f->param_pattern().static_type(),
      &f->return_term().static_type(), impl_bindings));
  SetConstantValue(f, arena_->New<FunctionValue>(f));

  if (f->name() == "Main") {
    if (!f->return_term().type_expression().has_value()) {
      return FATAL_COMPILATION_ERROR(f->return_term().source_loc())
             << "`Main` must have an explicit return type";
    }
    RETURN_IF_ERROR(ExpectExactType(
        f->return_term().source_loc(), "return type of `Main`",
        arena_->New<IntType>(), &f->return_term().static_type()));
    // TODO: Check that main doesn't have any parameters.
  }

  if (trace_) {
    llvm::outs() << "** finished declaring function " << f->name() << "\n";
  }
  return llvm::Error::success();
}

auto TypeChecker::TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                               const ImplScope& impl_scope)
    -> llvm::Error {
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
      function_scope.Add(impl_binding->interface(),
                         *impl_binding->type_var()->constant_value(),
                         impl_binding);
    }
    RETURN_IF_ERROR(TypeCheckStmt(*f->body(), function_scope));
    if (!f->return_term().is_omitted()) {
      RETURN_IF_ERROR(ExpectReturnOnAllPaths(f->body(), f->source_loc()));
    }
  }
  if (trace_) {
    llvm::outs() << "** finished checking function " << f->name() << "\n";
  }
  return llvm::Error::success();
}

auto TypeChecker::DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                          ImplScope& enclosing_scope)
    -> llvm::Error {
  // The declarations of the members may refer to the class, so we
  // must set the constant value of the class and its static type
  // before we start processing the members.
  Nonnull<NominalClassType*> class_type =
      arena_->New<NominalClassType>(class_decl);
  SetConstantValue(class_decl, class_type);
  class_decl->set_static_type(arena_->New<TypeOfClassType>(class_type));

  for (Nonnull<Declaration*> m : class_decl->members()) {
    RETURN_IF_ERROR(DeclareDeclaration(m, enclosing_scope));
  }
  return llvm::Error::success();
}

auto TypeChecker::TypeCheckClassDeclaration(
    Nonnull<ClassDeclaration*> class_decl, const ImplScope& impl_scope)
    -> llvm::Error {
  for (Nonnull<Declaration*> m : class_decl->members()) {
    RETURN_IF_ERROR(TypeCheckDeclaration(m, impl_scope));
  }
  return llvm::Error::success();
}

auto TypeChecker::DeclareInterfaceDeclaration(
    Nonnull<InterfaceDeclaration*> iface_decl, ImplScope& enclosing_scope)
    -> llvm::Error {
  Nonnull<InterfaceType*> iface_type = arena_->New<InterfaceType>(iface_decl);
  SetConstantValue(iface_decl, iface_type);
  iface_decl->set_static_type(arena_->New<TypeOfInterfaceType>(iface_type));

  // Process the Self parameter.
  RETURN_IF_ERROR(TypeCheckExp(&iface_decl->self()->type(), enclosing_scope));
  iface_decl->self()->set_static_type(
      arena_->New<VariableType>(iface_decl->self()));
  SetConstantValue(iface_decl->self(), &iface_decl->self()->static_type());

  for (Nonnull<Declaration*> m : iface_decl->members()) {
    RETURN_IF_ERROR(DeclareDeclaration(m, enclosing_scope));
  }
  return llvm::Error::success();
}

auto TypeChecker::TypeCheckInterfaceDeclaration(
    Nonnull<InterfaceDeclaration*> iface_decl, const ImplScope& impl_scope)
    -> llvm::Error {
  for (Nonnull<Declaration*> m : iface_decl->members()) {
    RETURN_IF_ERROR(TypeCheckDeclaration(m, impl_scope));
  }
  return llvm::Error::success();
}

auto TypeChecker::DeclareImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                         ImplScope& enclosing_scope)
    -> llvm::Error {
  if (trace_) {
    llvm::outs() << "declaring " << *impl_decl << "\n";
  }
  RETURN_IF_ERROR(TypeCheckExp(&impl_decl->interface(), enclosing_scope));
  ASSIGN_OR_RETURN(Nonnull<const Value*> iface_type,
                   InterpExp(&impl_decl->interface(), arena_, trace_));
  const auto& iface_decl = cast<InterfaceType>(*iface_type).declaration();
  impl_decl->set_interface_type(iface_type);

  RETURN_IF_ERROR(TypeCheckExp(impl_decl->impl_type(), enclosing_scope));
  ASSIGN_OR_RETURN(Nonnull<const Value*> impl_type_value,
                   InterpExp(impl_decl->impl_type(), arena_, trace_));
  enclosing_scope.Add(iface_type, impl_type_value, impl_decl);

  for (Nonnull<Declaration*> m : impl_decl->members()) {
    RETURN_IF_ERROR(DeclareDeclaration(m, enclosing_scope));
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
        RETURN_IF_ERROR(ExpectType((*mem)->source_loc(),
                                   "member of implementation", iface_mem_type,
                                   &(*mem)->static_type()));
      } else {
        return FATAL_COMPILATION_ERROR(impl_decl->source_loc())
               << "implementation missing " << *mem_name;
      }
    }
  }
  impl_decl->set_constant_value(arena_->New<Witness>(impl_decl));
  return llvm::Error::success();
}

auto TypeChecker::TypeCheckImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                           const ImplScope& impl_scope)
    -> llvm::Error {
  if (trace_) {
    llvm::outs() << "checking " << *impl_decl << "\n";
  }
  for (Nonnull<Declaration*> m : impl_decl->members()) {
    RETURN_IF_ERROR(TypeCheckDeclaration(m, impl_scope));
  }
  if (trace_) {
    llvm::outs() << "finished checking impl\n";
  }
  return llvm::Error::success();
}

auto TypeChecker::DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                           const ImplScope& impl_scope)
    -> llvm::Error {
  std::vector<NamedValue> alternatives;
  for (Nonnull<AlternativeSignature*> alternative : choice->alternatives()) {
    RETURN_IF_ERROR(TypeCheckExp(&alternative->signature(), impl_scope));
    ASSIGN_OR_RETURN(auto signature,
                     InterpExp(&alternative->signature(), arena_, trace_));
    alternatives.push_back({.name = alternative->name(), .value = signature});
  }
  auto ct = arena_->New<ChoiceType>(choice->name(), std::move(alternatives));
  SetConstantValue(choice, ct);
  choice->set_static_type(arena_->New<TypeOfChoiceType>(ct));
  return llvm::Error::success();
}

auto TypeChecker::TypeCheckChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                             const ImplScope& impl_scope)
    -> llvm::Error {
  // Nothing to do here, but perhaps that will change in the future?
  return llvm::Error::success();
}

auto TypeChecker::TypeCheck(AST& ast) -> llvm::Error {
  ImplScope impl_scope;
  for (Nonnull<Declaration*> declaration : ast.declarations) {
    RETURN_IF_ERROR(DeclareDeclaration(declaration, impl_scope));
  }
  for (Nonnull<Declaration*> decl : ast.declarations) {
    RETURN_IF_ERROR(TypeCheckDeclaration(decl, impl_scope));
  }
  RETURN_IF_ERROR(TypeCheckExp(*ast.main_call, impl_scope));
  return llvm::Error::success();
}

auto TypeChecker::TypeCheckDeclaration(Nonnull<Declaration*> d,
                                       const ImplScope& impl_scope)
    -> llvm::Error {
  switch (d->kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      RETURN_IF_ERROR(TypeCheckInterfaceDeclaration(
          &cast<InterfaceDeclaration>(*d), impl_scope));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      RETURN_IF_ERROR(
          TypeCheckImplDeclaration(&cast<ImplDeclaration>(*d), impl_scope));
      break;
    }
    case DeclarationKind::FunctionDeclaration:
      RETURN_IF_ERROR(TypeCheckFunctionDeclaration(
          &cast<FunctionDeclaration>(*d), impl_scope));
      return llvm::Error::success();
    case DeclarationKind::ClassDeclaration:
      RETURN_IF_ERROR(
          TypeCheckClassDeclaration(&cast<ClassDeclaration>(*d), impl_scope));
      return llvm::Error::success();
    case DeclarationKind::ChoiceDeclaration:
      RETURN_IF_ERROR(
          TypeCheckChoiceDeclaration(&cast<ChoiceDeclaration>(*d), impl_scope));
      return llvm::Error::success();
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Signals a type error if the initializing expression does not have
      // the declared type of the variable, otherwise returns this
      // declaration with annotated types.
      if (var.has_initializer()) {
        RETURN_IF_ERROR(TypeCheckExp(&var.initializer(), impl_scope));
      }
      const auto* binding_type =
          dyn_cast<ExpressionPattern>(&var.binding().type());
      if (binding_type == nullptr) {
        // TODO: consider adding support for `auto`
        return FATAL_COMPILATION_ERROR(var.source_loc())
               << "Type of a top-level variable must be an expression.";
      }
      if (var.has_initializer()) {
        RETURN_IF_ERROR(ExpectType(var.source_loc(), "initializer of variable",
                                   &var.static_type(),
                                   &var.initializer().static_type()));
      }
      return llvm::Error::success();
    }
  }
  return llvm::Error::success();
}

auto TypeChecker::DeclareDeclaration(Nonnull<Declaration*> d,
                                     ImplScope& impl_scope) -> llvm::Error {
  switch (d->kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      auto& iface_decl = cast<InterfaceDeclaration>(*d);
      RETURN_IF_ERROR(DeclareInterfaceDeclaration(&iface_decl, impl_scope));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl_decl = cast<ImplDeclaration>(*d);
      RETURN_IF_ERROR(DeclareImplDeclaration(&impl_decl, impl_scope));
      break;
    }
    case DeclarationKind::FunctionDeclaration: {
      auto& func_def = cast<FunctionDeclaration>(*d);
      RETURN_IF_ERROR(DeclareFunctionDeclaration(&func_def, impl_scope));
      break;
    }

    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(*d);
      RETURN_IF_ERROR(DeclareClassDeclaration(&class_decl, impl_scope));
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(*d);
      RETURN_IF_ERROR(DeclareChoiceDeclaration(&choice, impl_scope));
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Associate the variable name with it's declared type in the
      // compile-time symbol table.
      if (!llvm::isa<ExpressionPattern>(var.binding().type())) {
        return FATAL_COMPILATION_ERROR(var.binding().type().source_loc())
               << "Expected expression for variable type";
      }
      Expression& type =
          cast<ExpressionPattern>(var.binding().type()).expression();
      RETURN_IF_ERROR(
          TypeCheckPattern(&var.binding(), std::nullopt, impl_scope));
      ASSIGN_OR_RETURN(Nonnull<const Value*> declared_type,
                       InterpExp(&type, arena_, trace_));
      var.set_static_type(declared_type);
      break;
    }
  }
  return llvm::Error::success();
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
