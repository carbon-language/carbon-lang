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
#include "executable_semantics/interpreter/interpreter.h"
#include "executable_semantics/interpreter/value.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {

// Sets the static type of `*object`. Can be called multiple times on
// the same node, so long as the types are the same on each call.
// T must have static_type, has_static_type, and set_static_type methods.
template <typename T>
static void SetStaticType(Nonnull<T*> object, Nonnull<const Value*> type) {
  if (object->has_static_type()) {
    CHECK(TypeEqual(&object->static_type(), type));
  } else {
    object->set_static_type(type);
  }
}

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
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
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

static void ExpectType(SourceLocation source_loc, const std::string& context,
                       Nonnull<const Value*> expected,
                       Nonnull<const Value*> actual) {
  if (!IsImplicitlyConvertible(actual, expected)) {
    FATAL_COMPILATION_ERROR(source_loc)
        << "type error in " << context << ": "
        << "'" << *actual << "' is not implicitly convertible to '" << *expected
        << "'";
  }
}

void TypeChecker::ArgumentDeduction(
    SourceLocation source_loc,
    std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>& deduced,
    Nonnull<const Value*> param, Nonnull<const Value*> arg) {
  switch (param->kind()) {
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*param);
      auto [it, success] = deduced.insert({&var_type.binding(), arg});
      if (!success) {
        // TODO: can we allow implicit conversions here?
        ExpectExactType(source_loc, "argument deduction", it->second, arg);
      }
      return;
    }
    case Value::Kind::TupleValue: {
      if (arg->kind() != Value::Kind::TupleValue) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "type error in argument deduction\n"
            << "expected: " << *param << "\n"
            << "actual: " << *arg;
      }
      const auto& param_tup = cast<TupleValue>(*param);
      const auto& arg_tup = cast<TupleValue>(*arg);
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
      if (arg->kind() != Value::Kind::StructType) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "type error in argument deduction\n"
            << "expected: " << *param << "\n"
            << "actual: " << *arg;
      }
      const auto& param_struct = cast<StructType>(*param);
      const auto& arg_struct = cast<StructType>(*arg);
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
      if (arg->kind() != Value::Kind::FunctionType) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "type error in argument deduction\n"
            << "expected: " << *param << "\n"
            << "actual: " << *arg;
      }
      const auto& param_fn = cast<FunctionType>(*param);
      const auto& arg_fn = cast<FunctionType>(*arg);
      // TODO: handle situation when arg has deduced parameters.
      ArgumentDeduction(source_loc, deduced, &param_fn.parameters(),
                        &arg_fn.parameters());
      ArgumentDeduction(source_loc, deduced, &param_fn.return_type(),
                        &arg_fn.return_type());
      return;
    }
    case Value::Kind::PointerType: {
      if (arg->kind() != Value::Kind::PointerType) {
        FATAL_COMPILATION_ERROR(source_loc)
            << "type error in argument deduction\n"
            << "expected: " << *param << "\n"
            << "actual: " << *arg;
      }
      ArgumentDeduction(source_loc, deduced, &cast<PointerType>(*param).type(),
                        &cast<PointerType>(*arg).type());
      return;
    }
    // Nothing to do in the case for `auto`.
    case Value::Kind::AutoType: {
      return;
    }
    // For the following cases, we check for type convertability.
    case Value::Kind::ContinuationType:
    case Value::Kind::NominalClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfChoiceType:
      ExpectType(source_loc, "argument deduction", param, arg);
      return;
    // The rest of these cases should never happen.
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
          std::vector<Nonnull<const GenericBinding*>>(), param, ret);
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
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::StringType:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfChoiceType:
      return type;
    // The rest of these cases should never happen.
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

void TypeChecker::TypeCheckExp(Nonnull<Expression*> e) {
  if (trace_) {
    llvm::outs() << "checking expression " << *e << "\nconstants: ";
    PrintConstants(llvm::outs());
    llvm::outs() << "\n";
  }
  switch (e->kind()) {
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(*e);
      TypeCheckExp(&index.aggregate());
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
          SetStaticType(&index, tuple_type.elements()[i]);
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
        TypeCheckExp(arg);
        arg_types.push_back(&arg->static_type());
      }
      SetStaticType(e, arena_->New<TupleValue>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return;
    }
    case ExpressionKind::StructLiteral: {
      std::vector<NamedValue> arg_types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        TypeCheckExp(&arg.expression());
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      SetStaticType(e, arena_->New<StructType>(std::move(arg_types)));
      e->set_value_category(ValueCategory::Let);
      return;
    }
    case ExpressionKind::StructTypeLiteral: {
      auto& struct_type = cast<StructTypeLiteral>(*e);
      for (auto& arg : struct_type.fields()) {
        TypeCheckExp(&arg.expression());
        ExpectIsConcreteType(arg.expression().source_loc(),
                             InterpExp(&arg.expression(), arena_, trace_));
      }
      if (struct_type.fields().empty()) {
        // `{}` is the type of `{}`, just as `()` is the type of `()`.
        // This applies only if there are no fields, because (unlike with
        // tuples) non-empty struct types are syntactically disjoint
        // from non-empty struct values.
        SetStaticType(&struct_type, arena_->New<StructType>());
      } else {
        SetStaticType(&struct_type, arena_->New<TypeType>());
      }
      e->set_value_category(ValueCategory::Let);
      return;
    }
    case ExpressionKind::FieldAccessExpression: {
      auto& access = cast<FieldAccessExpression>(*e);
      TypeCheckExp(&access.aggregate());
      const Value& aggregate_type = access.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(aggregate_type);
          for (const auto& [field_name, field_type] : struct_type.fields()) {
            if (access.field() == field_name) {
              SetStaticType(&access, field_type);
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
                  t_class.FindMember(access.field());
              member.has_value()) {
            SetStaticType(&access, &(*member)->static_type());
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
          SetStaticType(&access,
                        arena_->New<FunctionType>(
                            std::vector<Nonnull<const GenericBinding*>>(),
                            *parameter_types, &aggregate_type));
          access.set_value_category(ValueCategory::Let);
          return;
        }
        case Value::Kind::TypeOfClassType: {
          const NominalClassType& class_type =
              cast<TypeOfClassType>(aggregate_type).class_type();
          if (std::optional<Nonnull<const Declaration*>> member =
                  class_type.FindMember(access.field());
              member.has_value()) {
            switch ((*member)->kind()) {
              case DeclarationKind::FunctionDeclaration: {
                const auto& func = cast<FunctionDeclaration>(*member);
                if (func->is_method()) {
                  break;
                }
                SetStaticType(&access, &(*member)->static_type());
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
        default:
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "field access, unexpected " << aggregate_type << " in " << *e;
      }
    }
    case ExpressionKind::IdentifierExpression: {
      auto& ident = cast<IdentifierExpression>(*e);
      if (ident.named_entity().base().kind() ==
          AstNodeKind::FunctionDeclaration) {
        const auto& function =
            cast<FunctionDeclaration>(ident.named_entity().base());
        if (!function.has_static_type()) {
          CHECK(function.return_term().is_auto());
          FATAL_COMPILATION_ERROR(ident.source_loc())
              << "Function calls itself, but has a deduced return type";
        }
      }
      SetStaticType(&ident, &ident.named_entity().static_type());
      ident.set_value_category(ident.named_entity().value_category());
      return;
    }
    case ExpressionKind::IntLiteral:
      e->set_value_category(ValueCategory::Let);
      SetStaticType(e, arena_->New<IntType>());
      return;
    case ExpressionKind::BoolLiteral:
      e->set_value_category(ValueCategory::Let);
      SetStaticType(e, arena_->New<BoolType>());
      return;
    case ExpressionKind::PrimitiveOperatorExpression: {
      auto& op = cast<PrimitiveOperatorExpression>(*e);
      std::vector<Nonnull<const Value*>> ts;
      for (Nonnull<Expression*> argument : op.arguments()) {
        TypeCheckExp(argument);
        ts.push_back(&argument->static_type());
      }
      switch (op.op()) {
        case Operator::Neg:
          ExpectExactType(e->source_loc(), "negation", arena_->New<IntType>(),
                          ts[0]);
          SetStaticType(&op, arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Add:
          ExpectExactType(e->source_loc(), "addition(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "addition(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Sub:
          ExpectExactType(e->source_loc(), "subtraction(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "subtraction(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Mul:
          ExpectExactType(e->source_loc(), "multiplication(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "multiplication(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::And:
          ExpectExactType(e->source_loc(), "&&(1)", arena_->New<BoolType>(),
                          ts[0]);
          ExpectExactType(e->source_loc(), "&&(2)", arena_->New<BoolType>(),
                          ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Or:
          ExpectExactType(e->source_loc(), "||(1)", arena_->New<BoolType>(),
                          ts[0]);
          ExpectExactType(e->source_loc(), "||(2)", arena_->New<BoolType>(),
                          ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Not:
          ExpectExactType(e->source_loc(), "!", arena_->New<BoolType>(), ts[0]);
          SetStaticType(&op, arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Eq:
          ExpectExactType(e->source_loc(), "==", ts[0], ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::Deref:
          ExpectPointerType(e->source_loc(), "*", ts[0]);
          SetStaticType(&op, &cast<PointerType>(*ts[0]).type());
          op.set_value_category(ValueCategory::Var);
          return;
        case Operator::Ptr:
          ExpectExactType(e->source_loc(), "*", arena_->New<TypeType>(), ts[0]);
          SetStaticType(&op, arena_->New<TypeType>());
          op.set_value_category(ValueCategory::Let);
          return;
        case Operator::AddressOf:
          if (op.arguments()[0]->value_category() != ValueCategory::Var) {
            FATAL_COMPILATION_ERROR(op.arguments()[0]->source_loc())
                << "Argument to " << ToString(op.op())
                << " should be an lvalue.";
          }
          SetStaticType(&op, arena_->New<PointerType>(ts[0]));
          op.set_value_category(ValueCategory::Let);
          return;
      }
      break;
    }
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(*e);
      TypeCheckExp(&call.function());
      switch (call.function().static_type().kind()) {
        case Value::Kind::FunctionType: {
          const auto& fun_t = cast<FunctionType>(call.function().static_type());
          TypeCheckExp(&call.argument());
          Nonnull<const Value*> parameters = &fun_t.parameters();
          Nonnull<const Value*> return_type = &fun_t.return_type();
          if (!fun_t.deduced().empty()) {
            std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>
                deduced_args;
            ArgumentDeduction(e->source_loc(), deduced_args, parameters,
                              &call.argument().static_type());
            for (Nonnull<const GenericBinding*> deduced_param :
                 fun_t.deduced()) {
              // TODO: change the following to a CHECK once the real checking
              // has been added to the type checking of function signatures.
              if (auto it = deduced_args.find(deduced_param);
                  it == deduced_args.end()) {
                FATAL_COMPILATION_ERROR(e->source_loc())
                    << "could not deduce type argument for type parameter "
                    << deduced_param->name();
              }
            }
            parameters = Substitute(deduced_args, parameters);
            return_type = Substitute(deduced_args, return_type);
          } else {
            ExpectType(e->source_loc(), "call", parameters,
                       &call.argument().static_type());
          }
          SetStaticType(&call, return_type);
          call.set_value_category(ValueCategory::Let);
          return;
        }
        default: {
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "in call, expected a function\n"
              << *e;
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
      SetStaticType(&fn, arena_->New<TypeType>());
      fn.set_value_category(ValueCategory::Let);
      return;
    }
    case ExpressionKind::StringLiteral:
      SetStaticType(e, arena_->New<StringType>());
      e->set_value_category(ValueCategory::Let);
      return;
    case ExpressionKind::IntrinsicExpression: {
      auto& intrinsic_exp = cast<IntrinsicExpression>(*e);
      TypeCheckExp(&intrinsic_exp.args());
      switch (cast<IntrinsicExpression>(*e).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          if (intrinsic_exp.args().fields().size() != 1) {
            FATAL_COMPILATION_ERROR(e->source_loc())
                << "__intrinsic_print takes 1 argument";
          }
          ExpectType(e->source_loc(), "__intrinsic_print argument",
                     arena_->New<StringType>(),
                     &intrinsic_exp.args().fields()[0]->static_type());
          SetStaticType(e, TupleValue::Empty());
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
      SetStaticType(e, arena_->New<TypeType>());
      return;
    case ExpressionKind::UnimplementedExpression:
      FATAL() << "Unimplemented: " << *e;
  }
}

void TypeChecker::TypeCheckPattern(
    Nonnull<Pattern*> p, std::optional<Nonnull<const Value*>> expected,
    ValueCategory enclosing_value_category) {
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
      SetStaticType(p, arena_->New<TypeType>());
      return;
    }
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(*p);
      TypeCheckPattern(&binding.type(), std::nullopt, enclosing_value_category);
      Nonnull<const Value*> type =
          InterpPattern(&binding.type(), arena_, trace_);
      if (expected) {
        if (IsConcreteType(type)) {
          ExpectType(p->source_loc(), "name binding", type, *expected);
        } else {
          if (!PatternMatch(type, *expected, binding.type().source_loc(),
                            std::nullopt)) {
            FATAL_COMPILATION_ERROR(binding.type().source_loc())
                << "Type pattern '" << *type << "' does not match actual type '"
                << **expected << "'";
          }
          type = *expected;
        }
      }
      ExpectIsConcreteType(binding.source_loc(), type);
      SetStaticType(&binding, type);
      SetValue(&binding, InterpPattern(&binding, arena_, trace_));

      if (!binding.has_value_category()) {
        binding.set_value_category(enclosing_value_category);
      }
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
        TypeCheckPattern(field, expected_field_type, enclosing_value_category);
        field_types.push_back(&field->static_type());
      }
      SetStaticType(&tuple, arena_->New<TupleValue>(std::move(field_types)));
      SetValue(&tuple, InterpPattern(&tuple, arena_, trace_));
      return;
    }
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(*p);
      TypeCheckExp(&alternative.choice_type());
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
      TypeCheckPattern(&alternative.arguments(), *parameter_types,
                       enclosing_value_category);
      SetStaticType(&alternative, &choice_type);
      SetValue(&alternative, InterpPattern(&alternative, arena_, trace_));
      return;
    }
    case PatternKind::ExpressionPattern: {
      auto& expression = cast<ExpressionPattern>(*p).expression();
      TypeCheckExp(&expression);
      SetStaticType(p, &expression.static_type());
      SetValue(p, InterpPattern(p, arena_, trace_));
      return;
    }
  }
}

void TypeChecker::TypeCheckStmt(Nonnull<Statement*> s) {
  switch (s->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*s);
      TypeCheckExp(&match.expression());
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        TypeCheckPattern(&clause.pattern(), &match.expression().static_type(),
                         ValueCategory::Let);
        TypeCheckStmt(&clause.statement());
      }
      return;
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(*s);
      TypeCheckExp(&while_stmt.condition());
      ExpectType(s->source_loc(), "condition of `while`",
                 arena_->New<BoolType>(),
                 &while_stmt.condition().static_type());
      TypeCheckStmt(&while_stmt.body());
      return;
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return;
    case StatementKind::Block: {
      auto& block = cast<Block>(*s);
      for (auto* block_statement : block.statements()) {
        TypeCheckStmt(block_statement);
      }
      return;
    }
    case StatementKind::VariableDefinition: {
      auto& var = cast<VariableDefinition>(*s);
      TypeCheckExp(&var.init());
      const Value& rhs_ty = var.init().static_type();
      TypeCheckPattern(&var.pattern(), &rhs_ty, var.value_category());
      return;
    }
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(*s);
      TypeCheckExp(&assign.rhs());
      TypeCheckExp(&assign.lhs());
      ExpectType(s->source_loc(), "assign", &assign.lhs().static_type(),
                 &assign.rhs().static_type());
      if (assign.lhs().value_category() != ValueCategory::Var) {
        FATAL_COMPILATION_ERROR(assign.source_loc())
            << "Cannot assign to rvalue '" << assign.lhs() << "'";
      }
      return;
    }
    case StatementKind::ExpressionStatement: {
      TypeCheckExp(&cast<ExpressionStatement>(*s).expression());
      return;
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*s);
      TypeCheckExp(&if_stmt.condition());
      ExpectType(s->source_loc(), "condition of `if`", arena_->New<BoolType>(),
                 &if_stmt.condition().static_type());
      TypeCheckStmt(&if_stmt.then_block());
      if (if_stmt.else_block()) {
        TypeCheckStmt(*if_stmt.else_block());
      }
      return;
    }
    case StatementKind::Return: {
      auto& ret = cast<Return>(*s);
      TypeCheckExp(&ret.expression());
      ReturnTerm& return_term = ret.function().return_term();
      if (return_term.is_auto()) {
        SetStaticType(&return_term, &ret.expression().static_type());
      } else {
        ExpectType(s->source_loc(), "return", &return_term.static_type(),
                   &ret.expression().static_type());
      }
      return;
    }
    case StatementKind::Continuation: {
      auto& cont = cast<Continuation>(*s);
      TypeCheckStmt(&cont.body());
      SetStaticType(&cont, arena_->New<ContinuationType>());
      return;
    }
    case StatementKind::Run: {
      auto& run = cast<Run>(*s);
      TypeCheckExp(&run.argument());
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
void TypeChecker::TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                               bool check_body) {
  // Bring the deduced parameters into scope
  for (Nonnull<GenericBinding*> deduced : f->deduced_parameters()) {
    TypeCheckExp(&deduced->type());
    // auto t = interpreter_.InterpExp(values, deduced.type);
    SetStaticType(deduced, arena_->New<VariableType>(deduced));
    SetConstantValue(deduced, &deduced->static_type());
  }
  if (f->is_method()) {
    // Type check the receiver patter
    TypeCheckPattern(&f->me_pattern(), std::nullopt, ValueCategory::Let);
  }

  // Type check the parameter pattern
  TypeCheckPattern(&f->param_pattern(), std::nullopt, ValueCategory::Let);

  // Evaluate the return type, if we can do so without examining the body.
  if (std::optional<Nonnull<Expression*>> return_expression =
          f->return_term().type_expression();
      return_expression.has_value()) {
    // We ignore the return value because return type expressions can't bring
    // new types into scope.
    TypeCheckExp(*return_expression);
    SetStaticType(&f->return_term(),
                  InterpExp(*return_expression, arena_, trace_));
  } else if (f->return_term().is_omitted()) {
    SetStaticType(&f->return_term(), TupleValue::Empty());
  } else {
    // We have to type-check the body in order to determine the return type.
    check_body = true;
    if (!f->body().has_value()) {
      FATAL_COMPILATION_ERROR(f->return_term().source_loc())
          << "Function declaration has deduced return type but no body";
    }
  }

  if (f->body().has_value() && check_body) {
    TypeCheckStmt(*f->body());
    if (!f->return_term().is_omitted()) {
      ExpectReturnOnAllPaths(f->body(), f->source_loc());
    }
  }

  ExpectIsConcreteType(f->source_loc(), &f->return_term().static_type());
  SetStaticType(f, arena_->New<FunctionType>(f->deduced_parameters(),
                                             &f->param_pattern().static_type(),
                                             &f->return_term().static_type()));
  if (f->name() == "Main") {
    if (!f->return_term().type_expression().has_value()) {
      FATAL_COMPILATION_ERROR(f->return_term().source_loc())
          << "`Main` must have an explicit return type";
    }
    ExpectExactType(f->return_term().source_loc(), "return type of `Main`",
                    arena_->New<IntType>(), &f->return_term().static_type());
    // TODO: Check that main doesn't have any parameters.
  }
  SetConstantValue(f, arena_->New<FunctionValue>(f));
  return;
}

void TypeChecker::TypeCheckClassDeclaration(
    Nonnull<ClassDeclaration*> class_decl) {
  // The declarations of the members may refer to the class, so we
  // must set the constant value of the class and its static type
  // before we start processing the members.
  Nonnull<NominalClassType*> class_type =
      arena_->New<NominalClassType>(class_decl);
  SetConstantValue(class_decl, class_type);
  SetStaticType(class_decl, arena_->New<TypeOfClassType>(class_type));

  // First pass: process the field, class function, and method
  // declarations but not the bodies of class functions or method
  // declarations.
  for (Nonnull<Declaration*> m : class_decl->members()) {
    DeclareDeclaration(m);
  }

  // Second pass: type check the bodies of the class functions and
  // methods.
  for (Nonnull<Declaration*> m : class_decl->members()) {
    TypeCheckDeclaration(m);
  }
}

void TypeChecker::TypeCheckChoiceDeclaration(
    Nonnull<ChoiceDeclaration*> choice) {
  std::vector<NamedValue> alternatives;
  for (Nonnull<AlternativeSignature*> alternative : choice->alternatives()) {
    TypeCheckExp(&alternative->signature());
    auto signature = InterpExp(&alternative->signature(), arena_, trace_);
    alternatives.push_back({.name = alternative->name(), .value = signature});
  }
  auto ct = arena_->New<ChoiceType>(choice->name(), std::move(alternatives));
  SetConstantValue(choice, ct);
  SetStaticType(choice, arena_->New<TypeOfChoiceType>(ct));
}

void TypeChecker::TypeCheck(AST& ast) {
  for (Nonnull<Declaration*> declaration : ast.declarations) {
    DeclareDeclaration(declaration);
  }
  for (Nonnull<Declaration*> decl : ast.declarations) {
    TypeCheckDeclaration(decl);
  }
  TypeCheckExp(*ast.main_call);
}

void TypeChecker::TypeCheckDeclaration(Nonnull<Declaration*> d) {
  switch (d->kind()) {
    case DeclarationKind::FunctionDeclaration:
      TypeCheckFunctionDeclaration(&cast<FunctionDeclaration>(*d),
                                   /*check_body=*/true);
      return;
    case DeclarationKind::ClassDeclaration:
      TypeCheckClassDeclaration(&cast<ClassDeclaration>(*d));
      return;
    case DeclarationKind::ChoiceDeclaration:
      TypeCheckChoiceDeclaration(&cast<ChoiceDeclaration>(*d));
      return;
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Signals a type error if the initializing expression does not have
      // the declared type of the variable, otherwise returns this
      // declaration with annotated types.
      if (var.has_initializer()) {
        TypeCheckExp(&var.initializer());
      }
      const auto* binding_type =
          dyn_cast<ExpressionPattern>(&var.binding().type());
      if (binding_type == nullptr) {
        // TODO: consider adding support for `auto`
        FATAL_COMPILATION_ERROR(var.source_loc())
            << "Type of a top-level variable must be an expression.";
      }
      Nonnull<const Value*> declared_type =
          InterpExp(&binding_type->expression(), arena_, trace_);
      SetStaticType(&var, declared_type);
      if (var.has_initializer()) {
        ExpectType(var.source_loc(), "initializer of variable", declared_type,
                   &var.initializer().static_type());
      }
      return;
    }
  }
}

void TypeChecker::DeclareDeclaration(Nonnull<Declaration*> d) {
  switch (d->kind()) {
    case DeclarationKind::FunctionDeclaration: {
      auto& func_def = cast<FunctionDeclaration>(*d);
      TypeCheckFunctionDeclaration(&func_def, /*check_body=*/false);
      break;
    }

    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(*d);
      TypeCheckClassDeclaration(&class_decl);
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(*d);
      TypeCheckChoiceDeclaration(&choice);
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Associate the variable name with it's declared type in the
      // compile-time symbol table.
      Expression& type =
          cast<ExpressionPattern>(var.binding().type()).expression();
      TypeCheckPattern(&var.binding(), std::nullopt, var.value_category());
      Nonnull<const Value*> declared_type = InterpExp(&type, arena_, trace_);
      SetStaticType(&var, declared_type);
      break;
    }
  }
}

template <typename T>
void TypeChecker::SetConstantValue(Nonnull<T*> named_entity,
                                   Nonnull<const Value*> value) {
  std::optional<Nonnull<const Value*>> old_value =
      named_entity->constant_value();
  if (old_value.has_value()) {
    CHECK(ValueEqual(*old_value, value));
  } else {
    named_entity->set_constant_value(value);
    CHECK(constants_.insert(named_entity).second);
  }
}

void TypeChecker::PrintConstants(llvm::raw_ostream& out) {
  llvm::ListSeparator sep;
  for (const auto& named_entity : constants_) {
    out << sep << named_entity.name() << ": "
        << **named_entity.constant_value();
  }
}

}  // namespace Carbon
