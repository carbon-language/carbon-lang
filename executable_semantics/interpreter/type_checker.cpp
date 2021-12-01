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

void TypeChecker::PrintTypeEnv(TypeEnv types, llvm::raw_ostream& out) {
  llvm::ListSeparator sep;
  for (const auto& [name, type] : types) {
    out << sep << name << ": " << *type;
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
              cast<NominalClassType>(*destination).fields());
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

auto TypeChecker::ArgumentDeduction(SourceLocation source_loc, TypeEnv deduced,
                                    Nonnull<const Value*> param,
                                    Nonnull<const Value*> arg) -> TypeEnv {
  switch (param->kind()) {
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*param);
      std::optional<Nonnull<const Value*>> d = deduced.Get(var_type.name());
      if (!d) {
        deduced.Set(var_type.name(), arg);
      } else {
        // TODO: can we allow implicit conversions here?
        ExpectExactType(source_loc, "argument deduction", *d, arg);
      }
      return deduced;
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
        deduced =
            ArgumentDeduction(source_loc, deduced, param_tup.elements()[i],
                              arg_tup.elements()[i]);
      }
      return deduced;
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
        deduced = ArgumentDeduction(source_loc, deduced,
                                    param_struct.fields()[i].value,
                                    arg_struct.fields()[i].value);
      }
      return deduced;
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
      deduced = ArgumentDeduction(source_loc, deduced, &param_fn.parameters(),
                                  &arg_fn.parameters());
      deduced = ArgumentDeduction(source_loc, deduced, &param_fn.return_type(),
                                  &arg_fn.return_type());
      return deduced;
    }
    case Value::Kind::PointerType: {
      if (arg->kind() != Value::Kind::PointerType) {
        FATAL_COMPILATION_ERROR(source_loc)
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
      return deduced;
    }
    // For the following cases, we check for type convertability.
    case Value::Kind::ContinuationType:
    case Value::Kind::NominalClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::StringType:
      ExpectType(source_loc, "argument deduction", param, arg);
      return deduced;
    // The rest of these cases should never happen.
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
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

auto TypeChecker::Substitute(TypeEnv dict, Nonnull<const Value*> type)
    -> Nonnull<const Value*> {
  switch (type->kind()) {
    case Value::Kind::VariableType: {
      std::optional<Nonnull<const Value*>> t =
          dict.Get(cast<VariableType>(*type).name());
      if (!t) {
        return type;
      } else {
        return *t;
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
      return type;
    // The rest of these cases should never happen.
    case Value::Kind::IntValue:
    case Value::Kind::BoolValue:
    case Value::Kind::FunctionValue:
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

auto TypeChecker::TypeCheckExp(Nonnull<Expression*> e, TypeEnv types,
                               Env values) -> TCResult {
  if (trace_) {
    llvm::outs() << "checking expression " << *e << "\ntypes: ";
    PrintTypeEnv(types, llvm::outs());
    llvm::outs() << "\nvalues: ";
    interpreter_.PrintEnv(values, llvm::outs());
    llvm::outs() << "\n";
  }
  switch (e->kind()) {
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(*e);
      auto res = TypeCheckExp(&index.aggregate(), types, values);
      const Value& aggregate_type = index.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::TupleValue: {
          const auto& tuple_type = cast<TupleValue>(aggregate_type);
          int i =
              cast<IntValue>(*interpreter_.InterpExp(values, &index.offset()))
                  .value();
          if (i < 0 || i >= static_cast<int>(tuple_type.elements().size())) {
            FATAL_COMPILATION_ERROR(e->source_loc())
                << "index " << i << " is out of range for type " << tuple_type;
          }
          SetStaticType(&index, tuple_type.elements()[i]);
          index.set_value_category(index.aggregate().value_category());
          return TCResult(res.types);
        }
        default:
          FATAL_COMPILATION_ERROR(e->source_loc()) << "expected a tuple";
      }
    }
    case ExpressionKind::TupleLiteral: {
      std::vector<Nonnull<const Value*>> arg_types;
      auto new_types = types;
      for (auto& arg : cast<TupleLiteral>(*e).fields()) {
        auto arg_res = TypeCheckExp(arg, new_types, values);
        new_types = arg_res.types;
        arg_types.push_back(&arg->static_type());
      }
      SetStaticType(e, arena_->New<TupleValue>(std::move(arg_types)));
      e->set_value_category(Expression::ValueCategory::Let);
      return TCResult(new_types);
    }
    case ExpressionKind::StructLiteral: {
      std::vector<FieldInitializer> new_args;
      std::vector<NamedValue> arg_types;
      auto new_types = types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        auto arg_res = TypeCheckExp(&arg.expression(), new_types, values);
        new_types = arg_res.types;
        new_args.push_back(FieldInitializer(arg.name(), &arg.expression()));
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      SetStaticType(e, arena_->New<StructType>(std::move(arg_types)));
      e->set_value_category(Expression::ValueCategory::Let);
      return TCResult(new_types);
    }
    case ExpressionKind::StructTypeLiteral: {
      auto& struct_type = cast<StructTypeLiteral>(*e);
      std::vector<FieldInitializer> new_args;
      auto new_types = types;
      for (auto& arg : struct_type.fields()) {
        auto arg_res = TypeCheckExp(&arg.expression(), new_types, values);
        new_types = arg_res.types;
        ExpectIsConcreteType(arg.expression().source_loc(),
                             interpreter_.InterpExp(values, &arg.expression()));
        new_args.push_back(FieldInitializer(arg.name(), &arg.expression()));
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
      e->set_value_category(Expression::ValueCategory::Let);
      return TCResult(new_types);
    }
    case ExpressionKind::FieldAccessExpression: {
      auto& access = cast<FieldAccessExpression>(*e);
      auto res = TypeCheckExp(&access.aggregate(), types, values);
      const Value& aggregate_type = access.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(aggregate_type);
          for (const auto& [field_name, field_type] : struct_type.fields()) {
            if (access.field() == field_name) {
              SetStaticType(&access, field_type);
              access.set_value_category(access.aggregate().value_category());
              return TCResult(res.types);
            }
          }
          FATAL_COMPILATION_ERROR(access.source_loc())
              << "struct " << struct_type << " does not have a field named "
              << access.field();
        }
        case Value::Kind::NominalClassType: {
          const auto& t_class = cast<NominalClassType>(aggregate_type);
          // Search for a field
          for (auto& field : t_class.fields()) {
            if (access.field() == field.name) {
              SetStaticType(&access, field.value);
              access.set_value_category(access.aggregate().value_category());
              return TCResult(res.types);
            }
          }
          // Search for a method
          for (auto& method : t_class.methods()) {
            if (access.field() == method.name) {
              SetStaticType(&access, method.value);
              access.set_value_category(Expression::ValueCategory::Let);
              return TCResult(res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "class " << t_class.name() << " does not have a field named "
              << access.field();
        }
        case Value::Kind::ChoiceType: {
          const auto& choice = cast<ChoiceType>(aggregate_type);
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
          access.set_value_category(Expression::ValueCategory::Let);
          return TCResult(res.types);
        }
        default:
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "field access, expected a struct\n"
              << *e;
      }
    }
    case ExpressionKind::IdentifierExpression: {
      auto& ident = cast<IdentifierExpression>(*e);
      CHECK(ident.has_named_entity()) << "Identifier '" << *e << "' at "
                                      << e->source_loc() << " was not resolved";
      if (ident.named_entity().kind() == NamedEntityKind::FunctionDeclaration) {
        const auto& function = cast<FunctionDeclaration>(ident.named_entity());
        if (!function.has_static_type()) {
          CHECK(function.return_term().is_auto());
          FATAL_COMPILATION_ERROR(ident.source_loc())
              << "Function calls itself, but has a deduced return type";
        }
      }
      SetStaticType(&ident, &ident.named_entity().static_type());
      // TODO: this should depend on what entity this name resolves to, but
      //   we don't have access to that information yet.
      ident.set_value_category(Expression::ValueCategory::Var);
      return TCResult(types);
    }
    case ExpressionKind::IntLiteral:
      e->set_value_category(Expression::ValueCategory::Let);
      SetStaticType(e, arena_->New<IntType>());
      return TCResult(types);
    case ExpressionKind::BoolLiteral:
      e->set_value_category(Expression::ValueCategory::Let);
      SetStaticType(e, arena_->New<BoolType>());
      return TCResult(types);
    case ExpressionKind::PrimitiveOperatorExpression: {
      auto& op = cast<PrimitiveOperatorExpression>(*e);
      std::vector<Nonnull<Expression*>> es;
      std::vector<Nonnull<const Value*>> ts;
      auto new_types = types;
      for (Nonnull<Expression*> argument : op.arguments()) {
        auto res = TypeCheckExp(argument, types, values);
        new_types = res.types;
        es.push_back(argument);
        ts.push_back(&argument->static_type());
      }
      switch (op.op()) {
        case Operator::Neg:
          ExpectExactType(e->source_loc(), "negation", arena_->New<IntType>(),
                          ts[0]);
          SetStaticType(&op, arena_->New<IntType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
        case Operator::Add:
          ExpectExactType(e->source_loc(), "addition(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "addition(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
        case Operator::Sub:
          ExpectExactType(e->source_loc(), "subtraction(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "subtraction(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
        case Operator::Mul:
          ExpectExactType(e->source_loc(), "multiplication(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "multiplication(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
        case Operator::And:
          ExpectExactType(e->source_loc(), "&&(1)", arena_->New<BoolType>(),
                          ts[0]);
          ExpectExactType(e->source_loc(), "&&(2)", arena_->New<BoolType>(),
                          ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
        case Operator::Or:
          ExpectExactType(e->source_loc(), "||(1)", arena_->New<BoolType>(),
                          ts[0]);
          ExpectExactType(e->source_loc(), "||(2)", arena_->New<BoolType>(),
                          ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
        case Operator::Not:
          ExpectExactType(e->source_loc(), "!", arena_->New<BoolType>(), ts[0]);
          SetStaticType(&op, arena_->New<BoolType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
        case Operator::Eq:
          ExpectExactType(e->source_loc(), "==", ts[0], ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
        case Operator::Deref:
          ExpectPointerType(e->source_loc(), "*", ts[0]);
          SetStaticType(&op, &cast<PointerType>(*ts[0]).type());
          op.set_value_category(Expression::ValueCategory::Var);
          return TCResult(new_types);
        case Operator::Ptr:
          ExpectExactType(e->source_loc(), "*", arena_->New<TypeType>(), ts[0]);
          SetStaticType(&op, arena_->New<TypeType>());
          op.set_value_category(Expression::ValueCategory::Let);
          return TCResult(new_types);
      }
      break;
    }
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(*e);
      auto fun_res = TypeCheckExp(&call.function(), types, values);
      switch (call.function().static_type().kind()) {
        case Value::Kind::FunctionType: {
          const auto& fun_t = cast<FunctionType>(call.function().static_type());
          auto arg_res = TypeCheckExp(&call.argument(), fun_res.types, values);
          Nonnull<const Value*> parameters = &fun_t.parameters();
          Nonnull<const Value*> return_type = &fun_t.return_type();
          if (!fun_t.deduced().empty()) {
            auto deduced_args =
                ArgumentDeduction(e->source_loc(), TypeEnv(arena_), parameters,
                                  &call.argument().static_type());
            for (Nonnull<const GenericBinding*> deduced_param :
                 fun_t.deduced()) {
              // TODO: change the following to a CHECK once the real checking
              // has been added to the type checking of function signatures.
              if (!deduced_args.Get(deduced_param->name())) {
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
          call.set_value_category(Expression::ValueCategory::Let);
          return TCResult(arg_res.types);
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
                           interpreter_.InterpExp(values, &fn.parameter()));
      ExpectIsConcreteType(fn.return_type().source_loc(),
                           interpreter_.InterpExp(values, &fn.return_type()));
      SetStaticType(&fn, arena_->New<TypeType>());
      fn.set_value_category(Expression::ValueCategory::Let);
      return TCResult(types);
    }
    case ExpressionKind::StringLiteral:
      SetStaticType(e, arena_->New<StringType>());
      e->set_value_category(Expression::ValueCategory::Let);
      return TCResult(types);
    case ExpressionKind::IntrinsicExpression: {
      auto& intrinsic_exp = cast<IntrinsicExpression>(*e);
      TCResult arg_res = TypeCheckExp(&intrinsic_exp.args(), types, values);
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
          e->set_value_category(Expression::ValueCategory::Let);
          return TCResult(arg_res.types);
      }
    }
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
      e->set_value_category(Expression::ValueCategory::Let);
      SetStaticType(e, arena_->New<TypeType>());
      return TCResult(types);
    case ExpressionKind::UnimplementedExpression:
      FATAL() << "Unimplemented: " << *e;
  }
}

auto TypeChecker::TypeCheckPattern(
    Nonnull<Pattern*> p, TypeEnv types, Env values,
    std::optional<Nonnull<const Value*>> expected) -> TCResult {
  if (trace_) {
    llvm::outs() << "checking pattern " << *p;
    if (expected) {
      llvm::outs() << ", expecting " << **expected;
    }
    llvm::outs() << "\ntypes: ";
    PrintTypeEnv(types, llvm::outs());
    llvm::outs() << "\nvalues: ";
    interpreter_.PrintEnv(values, llvm::outs());
    llvm::outs() << "\n";
  }
  switch (p->kind()) {
    case PatternKind::AutoPattern: {
      SetStaticType(p, arena_->New<TypeType>());
      return TCResult(types);
    }
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(*p);
      TypeCheckPattern(&binding.type(), types, values, std::nullopt);
      Nonnull<const Value*> type =
          interpreter_.InterpPattern(values, &binding.type());
      if (expected) {
        if (IsConcreteType(type)) {
          ExpectType(p->source_loc(), "name binding", type, *expected);
        } else {
          std::optional<Env> values = interpreter_.PatternMatch(
              type, *expected, binding.type().source_loc());
          if (values == std::nullopt) {
            FATAL_COMPILATION_ERROR(binding.type().source_loc())
                << "Type pattern '" << *type << "' does not match actual type '"
                << **expected << "'";
          }
          CHECK(values->begin() == values->end())
              << "Name bindings within type patterns are unsupported";
          type = *expected;
        }
      }
      ExpectIsConcreteType(binding.source_loc(), type);
      if (binding.name().has_value()) {
        types.Set(*binding.name(), type);
      }
      SetStaticType(&binding, type);
      SetValue(&binding, interpreter_.InterpPattern(values, &binding));
      return TCResult(types);
    }
    case PatternKind::TuplePattern: {
      auto& tuple = cast<TuplePattern>(*p);
      std::vector<Nonnull<const Value*>> field_types;
      auto new_types = types;
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
        auto field_result =
            TypeCheckPattern(field, new_types, values, expected_field_type);
        new_types = field_result.types;
        field_types.push_back(&field->static_type());
      }
      SetStaticType(&tuple, arena_->New<TupleValue>(std::move(field_types)));
      SetValue(&tuple, interpreter_.InterpPattern(values, &tuple));
      return TCResult(new_types);
    }
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(*p);
      Nonnull<const Value*> choice_type =
          interpreter_.InterpExp(values, &alternative.choice_type());
      if (choice_type->kind() != Value::Kind::ChoiceType) {
        FATAL_COMPILATION_ERROR(alternative.source_loc())
            << "alternative pattern does not name a choice type.";
      }
      if (expected) {
        ExpectExactType(alternative.source_loc(), "alternative pattern",
                        *expected, choice_type);
      }
      std::optional<Nonnull<const Value*>> parameter_types =
          cast<ChoiceType>(*choice_type)
              .FindAlternative(alternative.alternative_name());
      if (parameter_types == std::nullopt) {
        FATAL_COMPILATION_ERROR(alternative.source_loc())
            << "'" << alternative.alternative_name()
            << "' is not an alternative of " << *choice_type;
      }
      TCResult arg_results = TypeCheckPattern(&alternative.arguments(), types,
                                              values, *parameter_types);
      SetStaticType(&alternative, choice_type);
      SetValue(&alternative, interpreter_.InterpPattern(values, &alternative));
      return TCResult(arg_results.types);
    }
    case PatternKind::ExpressionPattern: {
      auto& expression = cast<ExpressionPattern>(*p).expression();
      TCResult result = TypeCheckExp(&expression, types, values);
      SetStaticType(p, &expression.static_type());
      SetValue(p, interpreter_.InterpPattern(values, p));
      return TCResult(result.types);
    }
  }
}

auto TypeChecker::TypeCheckCase(Nonnull<const Value*> expected,
                                Nonnull<Pattern*> pat, Nonnull<Statement*> body,
                                TypeEnv types, Env values) -> Match::Clause {
  auto pat_res = TypeCheckPattern(pat, types, values, expected);
  TypeCheckStmt(body, pat_res.types, values);
  return Match::Clause(pat, body);
}

auto TypeChecker::TypeCheckStmt(Nonnull<Statement*> s, TypeEnv types,
                                Env values) -> TCResult {
  switch (s->kind()) {
    case StatementKind::Match: {
      auto& match = cast<Match>(*s);
      TypeCheckExp(&match.expression(), types, values);
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        new_clauses.push_back(
            TypeCheckCase(&match.expression().static_type(), &clause.pattern(),
                          &clause.statement(), types, values));
      }
      return TCResult(types);
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(*s);
      TypeCheckExp(&while_stmt.condition(), types, values);
      ExpectType(s->source_loc(), "condition of `while`",
                 arena_->New<BoolType>(),
                 &while_stmt.condition().static_type());
      TypeCheckStmt(&while_stmt.body(), types, values);
      return TCResult(types);
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      return TCResult(types);
    case StatementKind::Block: {
      auto& block = cast<Block>(*s);
      for (auto* block_statement : block.statements()) {
        auto result = TypeCheckStmt(block_statement, types, values);
        types = result.types;
      }
      return TCResult(types);
    }
    case StatementKind::VariableDefinition: {
      auto& var = cast<VariableDefinition>(*s);
      TypeCheckExp(&var.init(), types, values);
      const Value& rhs_ty = var.init().static_type();
      auto lhs_res = TypeCheckPattern(&var.pattern(), types, values, &rhs_ty);
      return TCResult(lhs_res.types);
    }
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(*s);
      TypeCheckExp(&assign.rhs(), types, values);
      auto lhs_res = TypeCheckExp(&assign.lhs(), types, values);
      ExpectType(s->source_loc(), "assign", &assign.lhs().static_type(),
                 &assign.rhs().static_type());
      if (assign.lhs().value_category() != Expression::ValueCategory::Var) {
        FATAL_COMPILATION_ERROR(assign.source_loc())
            << "Cannot assign to rvalue '" << assign.lhs() << "'";
      }
      return TCResult(lhs_res.types);
    }
    case StatementKind::ExpressionStatement: {
      TypeCheckExp(&cast<ExpressionStatement>(*s).expression(), types, values);
      return TCResult(types);
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(*s);
      TypeCheckExp(&if_stmt.condition(), types, values);
      ExpectType(s->source_loc(), "condition of `if`", arena_->New<BoolType>(),
                 &if_stmt.condition().static_type());
      TypeCheckStmt(&if_stmt.then_block(), types, values);
      if (if_stmt.else_block()) {
        TypeCheckStmt(*if_stmt.else_block(), types, values);
      }
      return TCResult(types);
    }
    case StatementKind::Return: {
      auto& ret = cast<Return>(*s);
      TypeCheckExp(&ret.expression(), types, values);
      ReturnTerm& return_term = ret.function().return_term();
      if (return_term.is_auto()) {
        SetStaticType(&return_term, &ret.expression().static_type());
      } else {
        ExpectType(s->source_loc(), "return", &return_term.static_type(),
                   &ret.expression().static_type());
      }
      return TCResult(types);
    }
    case StatementKind::Continuation: {
      auto& cont = cast<Continuation>(*s);
      TypeCheckStmt(&cont.body(), types, values);
      SetStaticType(&cont, arena_->New<ContinuationType>());
      types.Set(cont.continuation_variable(), &cont.static_type());
      return TCResult(types);
    }
    case StatementKind::Run: {
      auto& run = cast<Run>(*s);
      TypeCheckExp(&run.argument(), types, values);
      ExpectType(s->source_loc(), "argument of `run`",
                 arena_->New<ContinuationType>(),
                 &run.argument().static_type());
      return TCResult(types);
    }
    case StatementKind::Await: {
      // nothing to do here
      return TCResult(types);
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
auto TypeChecker::TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                               TypeEnv types, Env values,
                                               bool check_body) -> TCResult {
  // Bring the deduced parameters into scope
  for (Nonnull<GenericBinding*> deduced : f->deduced_parameters()) {
    TypeCheckExp(&deduced->type(), types, values);
    // auto t = interpreter_.InterpExp(values, deduced.type);
    SetStaticType(deduced, arena_->New<VariableType>(deduced->name()));
    types.Set(deduced->name(), &deduced->static_type());
    AllocationId a = interpreter_.AllocateValue(*types.Get(deduced->name()));
    values.Set(deduced->name(), a);
  }
  // Type check the parameter pattern
  auto param_res =
      TypeCheckPattern(&f->param_pattern(), types, values, std::nullopt);

  // Evaluate the return type, if we can do so without examining the body.
  if (std::optional<Nonnull<Expression*>> return_expression =
          f->return_term().type_expression();
      return_expression.has_value()) {
    // We ignore the return value because return type expressions can't bring
    // new types into scope.
    TypeCheckExp(*return_expression, param_res.types, values);
    SetStaticType(&f->return_term(),
                  interpreter_.InterpExp(values, *return_expression));
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
    TypeCheckStmt(*f->body(), param_res.types, values);
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
  return TCResult(types);
}

auto TypeChecker::TypeOfClassDecl(ClassDeclaration& class_decl,
                                  TypeEnv /*types*/, Env ct_top)
    -> Nonnull<const Value*> {
  std::vector<NamedValue> fields;
  std::vector<NamedValue> methods;
  for (Nonnull<const Member*> m : class_decl.members()) {
    switch (m->kind()) {
      case MemberKind::FieldMember: {
        const BindingPattern& binding = cast<FieldMember>(*m).binding();
        if (!binding.name().has_value()) {
          FATAL_COMPILATION_ERROR(binding.source_loc())
              << "Struct members must have names";
        }
        const auto* binding_type = dyn_cast<ExpressionPattern>(&binding.type());
        if (binding_type == nullptr) {
          FATAL_COMPILATION_ERROR(binding.source_loc())
              << "Struct members must have explicit types";
        }
        auto type = interpreter_.InterpExp(ct_top, &binding_type->expression());
        fields.push_back({.name = *binding.name(), .value = type});
        break;
      }
    }
  }
  SetStaticType(&class_decl,
                arena_->New<NominalClassType>(
                    class_decl.name(), std::move(fields), std::move(methods)));
  return &class_decl.static_type();
}

static auto GetName(const Declaration& d) -> const std::string& {
  switch (d.kind()) {
    case DeclarationKind::FunctionDeclaration:
      return cast<FunctionDeclaration>(d).name();
    case DeclarationKind::ClassDeclaration:
      return cast<ClassDeclaration>(d).name();
    case DeclarationKind::ChoiceDeclaration:
      return cast<ChoiceDeclaration>(d).name();
    case DeclarationKind::VariableDeclaration: {
      const BindingPattern& binding = cast<VariableDeclaration>(d).binding();
      if (!binding.name().has_value()) {
        FATAL_COMPILATION_ERROR(binding.source_loc())
            << "Top-level variable declarations must have names";
      }
      return *binding.name();
    }
  }
}

void TypeChecker::TypeCheck(AST& ast) {
  TypeCheckContext p = TopLevel(&ast.declarations);
  TypeEnv top = p.types;
  Env ct_top = p.values;
  for (const auto decl : ast.declarations) {
    TypeCheckDeclaration(decl, top, ct_top);
  }
  if (ast.main_call.has_value()) {
    TypeCheckExp(*ast.main_call, p.types, p.values);
  }
}

void TypeChecker::TypeCheckDeclaration(Nonnull<Declaration*> d,
                                       const TypeEnv& types,
                                       const Env& values) {
  switch (d->kind()) {
    case DeclarationKind::FunctionDeclaration:
      TypeCheckFunctionDeclaration(&cast<FunctionDeclaration>(*d), types,
                                   values, /*check_body=*/true);
      return;
    case DeclarationKind::ClassDeclaration:
      // TODO
      return;

    case DeclarationKind::ChoiceDeclaration:
      // TODO
      return;

    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Signals a type error if the initializing expression does not have
      // the declared type of the variable, otherwise returns this
      // declaration with annotated types.
      TypeCheckExp(&var.initializer(), types, values);
      const auto* binding_type =
          dyn_cast<ExpressionPattern>(&var.binding().type());
      if (binding_type == nullptr) {
        // TODO: consider adding support for `auto`
        FATAL_COMPILATION_ERROR(var.source_loc())
            << "Type of a top-level variable must be an expression.";
      }
      Nonnull<const Value*> declared_type =
          interpreter_.InterpExp(values, &binding_type->expression());
      SetStaticType(&var, declared_type);
      ExpectType(var.source_loc(), "initializer of variable", declared_type,
                 &var.initializer().static_type());
      return;
    }
  }
}

void TypeChecker::TopLevel(Nonnull<Declaration*> d, TypeCheckContext* tops) {
  switch (d->kind()) {
    case DeclarationKind::FunctionDeclaration: {
      auto& func_def = cast<FunctionDeclaration>(*d);
      TypeCheckFunctionDeclaration(&func_def, tops->types, tops->values,
                                   /*check_body=*/false);
      tops->types.Set(func_def.name(), &func_def.static_type());
      interpreter_.InitEnv(*d, &tops->values);
      break;
    }

    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(*d);
      auto st = TypeOfClassDecl(class_decl, tops->types, tops->values);
      AllocationId a = interpreter_.AllocateValue(st);
      tops->values.Set(class_decl.name(), a);  // Is this obsolete?
      tops->types.Set(class_decl.name(), st);
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(*d);
      std::vector<NamedValue> alts;
      for (Nonnull<const AlternativeSignature*> alternative :
           choice.alternatives()) {
        auto t =
            interpreter_.InterpExp(tops->values, &alternative->signature());
        alts.push_back({.name = alternative->name(), .value = t});
      }
      auto ct = arena_->New<ChoiceType>(choice.name(), std::move(alts));
      SetStaticType(&choice, ct);
      AllocationId a = interpreter_.AllocateValue(ct);
      tops->values.Set(choice.name(), a);  // Is this obsolete?
      tops->types.Set(choice.name(), ct);
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Associate the variable name with it's declared type in the
      // compile-time symbol table.
      Expression& type =
          cast<ExpressionPattern>(var.binding().type()).expression();
      tops->types = TypeCheckPattern(&var.binding(), tops->types, tops->values,
                                     std::nullopt)
                        .types;
      Nonnull<const Value*> declared_type =
          interpreter_.InterpExp(tops->values, &type);
      tops->types.Set(*var.binding().name(), declared_type);
      SetStaticType(&var, declared_type);
      break;
    }
  }
}

auto TypeChecker::TopLevel(std::vector<Nonnull<Declaration*>>* fs)
    -> TypeCheckContext {
  TypeCheckContext tops(arena_);
  bool found_main = false;

  for (auto const& d : *fs) {
    if (GetName(*d) == "Main") {
      found_main = true;
    }
    TopLevel(d, &tops);
  }

  if (found_main == false) {
    FATAL_COMPILATION_ERROR_NO_LINE()
        << "program must contain a function named `Main`";
  }
  return tops;
}

}  // namespace Carbon
