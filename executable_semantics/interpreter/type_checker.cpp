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

// Sets the static type of `expression`. Can be called multiple times on
// the same node, so long as the types are the same on each call.
static void SetStaticType(Nonnull<Expression*> expression,
                          Nonnull<const Value*> type) {
  if (expression->has_static_type()) {
    CHECK(TypeEqual(&expression->static_type(), type));
  } else {
    expression->set_static_type(type);
  }
}

// Sets the static type of `pattern`. Can be called multiple times on
// the same node, so long as the types are the same on each call.
static void SetStaticType(Nonnull<Pattern*> pattern,
                          Nonnull<const Value*> type) {
  if (pattern->has_static_type()) {
    CHECK(TypeEqual(&pattern->static_type(), type));
  } else {
    pattern->set_static_type(type);
  }
}

// Sets the static type of `definition`. Can be called multiple times on
// the same node, so long as the types are the same on each call.
static void SetStaticType(Nonnull<Declaration*> definition,
                          Nonnull<const Value*> type) {
  if (definition->has_static_type()) {
    CHECK(TypeEqual(&definition->static_type(), type));
  } else {
    definition->set_static_type(type);
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

TypeChecker::ReturnTypeContext::ReturnTypeContext(
    Nonnull<const Value*> orig_return_type, bool is_omitted)
    : is_auto_(isa<AutoType>(orig_return_type)),
      deduced_return_type_(is_auto_ ? std::nullopt
                                    : std::optional(orig_return_type)),
      is_omitted_(is_omitted) {}

void PrintTypeEnv(TypeEnv types, llvm::raw_ostream& out) {
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
    case Value::Kind::PointerValue:
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
    const VarValues& source_fields, const VarValues& destination_fields) {
  if (source_fields.size() != destination_fields.size()) {
    return false;
  }
  for (const auto& [field_name, source_field_type] : source_fields) {
    std::optional<Nonnull<const Value*>> destination_field_type =
        FindInVarValues(field_name, destination_fields);
    if (!destination_field_type.has_value() ||
        !IsImplicitlyConvertible(source_field_type, *destination_field_type)) {
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

// Perform type argument deduction, matching the parameter type `param`
// against the argument type `arg`. Whenever there is an VariableType
// in the parameter type, it is deduced to be the corresponding type
// inside the argument type.
// The `deduced` parameter is an accumulator, that is, it holds the
// results so-far.
static auto ArgumentDeduction(SourceLocation source_loc, TypeEnv deduced,
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
        if (param_struct.fields()[i].first != arg_struct.fields()[i].first) {
          FATAL_COMPILATION_ERROR(source_loc)
              << "mismatch in field names, " << param_struct.fields()[i].first
              << " != " << arg_struct.fields()[i].first;
        }
        deduced = ArgumentDeduction(source_loc, deduced,
                                    param_struct.fields()[i].second,
                                    arg_struct.fields()[i].second);
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
    case Value::Kind::PointerValue:
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
      VarValues fields;
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
      return arena_->New<FunctionType>(std::vector<GenericBinding>(), param,
                                       ret);
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
    case Value::Kind::PointerValue:
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
    case Expression::Kind::IndexExpression: {
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
          return TCResult(res.types);
        }
        default:
          FATAL_COMPILATION_ERROR(e->source_loc()) << "expected a tuple";
      }
    }
    case Expression::Kind::TupleLiteral: {
      std::vector<Nonnull<const Value*>> arg_types;
      auto new_types = types;
      for (auto& arg : cast<TupleLiteral>(*e).fields()) {
        auto arg_res = TypeCheckExp(arg, new_types, values);
        new_types = arg_res.types;
        arg_types.push_back(&arg->static_type());
      }
      SetStaticType(e, arena_->New<TupleValue>(std::move(arg_types)));
      return TCResult(new_types);
    }
    case Expression::Kind::StructLiteral: {
      std::vector<FieldInitializer> new_args;
      VarValues arg_types;
      auto new_types = types;
      for (auto& arg : cast<StructLiteral>(*e).fields()) {
        auto arg_res = TypeCheckExp(&arg.expression(), new_types, values);
        new_types = arg_res.types;
        new_args.push_back(FieldInitializer(arg.name(), &arg.expression()));
        arg_types.push_back({arg.name(), &arg.expression().static_type()});
      }
      SetStaticType(e, arena_->New<StructType>(std::move(arg_types)));
      return TCResult(new_types);
    }
    case Expression::Kind::StructTypeLiteral: {
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
      return TCResult(new_types);
    }
    case Expression::Kind::FieldAccessExpression: {
      auto& access = cast<FieldAccessExpression>(*e);
      auto res = TypeCheckExp(&access.aggregate(), types, values);
      const Value& aggregate_type = access.aggregate().static_type();
      switch (aggregate_type.kind()) {
        case Value::Kind::StructType: {
          const auto& struct_type = cast<StructType>(aggregate_type);
          for (const auto& [field_name, field_type] : struct_type.fields()) {
            if (access.field() == field_name) {
              SetStaticType(&access, field_type);
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
            if (access.field() == field.first) {
              SetStaticType(&access, field.second);
              return TCResult(res.types);
            }
          }
          // Search for a method
          for (auto& method : t_class.methods()) {
            if (access.field() == method.first) {
              SetStaticType(&access, method.second);
              return TCResult(res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "class " << t_class.name() << " does not have a field named "
              << access.field();
        }
        case Value::Kind::ChoiceType: {
          const auto& choice = cast<ChoiceType>(aggregate_type);
          for (const auto& vt : choice.alternatives()) {
            if (access.field() == vt.first) {
              SetStaticType(&access, arena_->New<FunctionType>(
                                         std::vector<GenericBinding>(),
                                         vt.second, &aggregate_type));
              return TCResult(res.types);
            }
          }
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "choice " << choice.name() << " does not have a field named "
              << access.field();
        }
        default:
          FATAL_COMPILATION_ERROR(e->source_loc())
              << "field access, expected a struct\n"
              << *e;
      }
    }
    case Expression::Kind::IdentifierExpression: {
      auto& ident = cast<IdentifierExpression>(*e);
      std::optional<Nonnull<const Value*>> type = types.Get(ident.name());
      if (type) {
        SetStaticType(&ident, *type);
        return TCResult(types);
      } else {
        FATAL_COMPILATION_ERROR(e->source_loc())
            << "could not find `" << ident.name() << "`";
      }
    }
    case Expression::Kind::IntLiteral:
      SetStaticType(e, arena_->New<IntType>());
      return TCResult(types);
    case Expression::Kind::BoolLiteral:
      SetStaticType(e, arena_->New<BoolType>());
      return TCResult(types);
    case Expression::Kind::PrimitiveOperatorExpression: {
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
          return TCResult(new_types);
        case Operator::Add:
          ExpectExactType(e->source_loc(), "addition(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "addition(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          return TCResult(new_types);
        case Operator::Sub:
          ExpectExactType(e->source_loc(), "subtraction(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "subtraction(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          return TCResult(new_types);
        case Operator::Mul:
          ExpectExactType(e->source_loc(), "multiplication(1)",
                          arena_->New<IntType>(), ts[0]);
          ExpectExactType(e->source_loc(), "multiplication(2)",
                          arena_->New<IntType>(), ts[1]);
          SetStaticType(&op, arena_->New<IntType>());
          return TCResult(new_types);
        case Operator::And:
          ExpectExactType(e->source_loc(), "&&(1)", arena_->New<BoolType>(),
                          ts[0]);
          ExpectExactType(e->source_loc(), "&&(2)", arena_->New<BoolType>(),
                          ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          return TCResult(new_types);
        case Operator::Or:
          ExpectExactType(e->source_loc(), "||(1)", arena_->New<BoolType>(),
                          ts[0]);
          ExpectExactType(e->source_loc(), "||(2)", arena_->New<BoolType>(),
                          ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          return TCResult(new_types);
        case Operator::Not:
          ExpectExactType(e->source_loc(), "!", arena_->New<BoolType>(), ts[0]);
          SetStaticType(&op, arena_->New<BoolType>());
          return TCResult(new_types);
        case Operator::Eq:
          ExpectExactType(e->source_loc(), "==", ts[0], ts[1]);
          SetStaticType(&op, arena_->New<BoolType>());
          return TCResult(new_types);
        case Operator::Deref:
          ExpectPointerType(e->source_loc(), "*", ts[0]);
          SetStaticType(&op, &cast<PointerType>(*ts[0]).type());
          return TCResult(new_types);
        case Operator::Ptr:
          ExpectExactType(e->source_loc(), "*", arena_->New<TypeType>(), ts[0]);
          SetStaticType(&op, arena_->New<TypeType>());
          return TCResult(new_types);
      }
      break;
    }
    case Expression::Kind::CallExpression: {
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
            for (auto& deduced_param : fun_t.deduced()) {
              // TODO: change the following to a CHECK once the real checking
              // has been added to the type checking of function signatures.
              if (!deduced_args.Get(deduced_param.name)) {
                FATAL_COMPILATION_ERROR(e->source_loc())
                    << "could not deduce type argument for type parameter "
                    << deduced_param.name;
              }
            }
            parameters = Substitute(deduced_args, parameters);
            return_type = Substitute(deduced_args, return_type);
          } else {
            ExpectType(e->source_loc(), "call", parameters,
                       &call.argument().static_type());
          }
          SetStaticType(&call, return_type);
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
    case Expression::Kind::FunctionTypeLiteral: {
      auto& fn = cast<FunctionTypeLiteral>(*e);
      ExpectIsConcreteType(fn.parameter().source_loc(),
                           interpreter_.InterpExp(values, &fn.parameter()));
      ExpectIsConcreteType(fn.return_type().source_loc(),
                           interpreter_.InterpExp(values, &fn.return_type()));
      SetStaticType(&fn, arena_->New<TypeType>());
      return TCResult(types);
    }
    case Expression::Kind::StringLiteral:
      SetStaticType(e, arena_->New<StringType>());
      return TCResult(types);
    case Expression::Kind::IntrinsicExpression:
      switch (cast<IntrinsicExpression>(*e).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          SetStaticType(e, TupleValue::Empty());
          return TCResult(types);
      }
    case Expression::Kind::IntTypeLiteral:
    case Expression::Kind::BoolTypeLiteral:
    case Expression::Kind::StringTypeLiteral:
    case Expression::Kind::TypeTypeLiteral:
    case Expression::Kind::ContinuationTypeLiteral:
      SetStaticType(e, arena_->New<TypeType>());
      return TCResult(types);
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
    case Pattern::Kind::AutoPattern: {
      SetStaticType(p, arena_->New<TypeType>());
      return TCResult(types);
    }
    case Pattern::Kind::BindingPattern: {
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
    case Pattern::Kind::TuplePattern: {
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
    case Pattern::Kind::AlternativePattern: {
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
          FindInVarValues(alternative.alternative_name(),
                          cast<ChoiceType>(*choice_type).alternatives());
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
    case Pattern::Kind::ExpressionPattern: {
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
                                TypeEnv types, Env values,
                                Nonnull<ReturnTypeContext*> return_type_context)
    -> Match::Clause {
  auto pat_res = TypeCheckPattern(pat, types, values, expected);
  TypeCheckStmt(body, pat_res.types, values, return_type_context);
  return Match::Clause(pat, body);
}

auto TypeChecker::TypeCheckStmt(Nonnull<Statement*> s, TypeEnv types,
                                Env values,
                                Nonnull<ReturnTypeContext*> return_type_context)
    -> TCResult {
  switch (s->kind()) {
    case Statement::Kind::Match: {
      auto& match = cast<Match>(*s);
      TypeCheckExp(&match.expression(), types, values);
      std::vector<Match::Clause> new_clauses;
      for (auto& clause : match.clauses()) {
        new_clauses.push_back(TypeCheckCase(
            &match.expression().static_type(), &clause.pattern(),
            &clause.statement(), types, values, return_type_context));
      }
      return TCResult(types);
    }
    case Statement::Kind::While: {
      auto& while_stmt = cast<While>(*s);
      TypeCheckExp(&while_stmt.condition(), types, values);
      ExpectType(s->source_loc(), "condition of `while`",
                 arena_->New<BoolType>(),
                 &while_stmt.condition().static_type());
      TypeCheckStmt(&while_stmt.body(), types, values, return_type_context);
      return TCResult(types);
    }
    case Statement::Kind::Break:
    case Statement::Kind::Continue:
      return TCResult(types);
    case Statement::Kind::Block: {
      auto& block = cast<Block>(*s);
      if (block.sequence()) {
        TypeCheckStmt(*block.sequence(), types, values, return_type_context);
        return TCResult(types);
      } else {
        return TCResult(types);
      }
    }
    case Statement::Kind::VariableDefinition: {
      auto& var = cast<VariableDefinition>(*s);
      TypeCheckExp(&var.init(), types, values);
      const Value& rhs_ty = var.init().static_type();
      auto lhs_res = TypeCheckPattern(&var.pattern(), types, values, &rhs_ty);
      return TCResult(lhs_res.types);
    }
    case Statement::Kind::Sequence: {
      auto& seq = cast<Sequence>(*s);
      auto stmt_res =
          TypeCheckStmt(&seq.statement(), types, values, return_type_context);
      auto checked_types = stmt_res.types;
      if (seq.next()) {
        auto next_res = TypeCheckStmt(*seq.next(), checked_types, values,
                                      return_type_context);
        checked_types = next_res.types;
      }
      return TCResult(checked_types);
    }
    case Statement::Kind::Assign: {
      auto& assign = cast<Assign>(*s);
      TypeCheckExp(&assign.rhs(), types, values);
      auto lhs_res = TypeCheckExp(&assign.lhs(), types, values);
      ExpectType(s->source_loc(), "assign", &assign.lhs().static_type(),
                 &assign.rhs().static_type());
      return TCResult(lhs_res.types);
    }
    case Statement::Kind::ExpressionStatement: {
      TypeCheckExp(&cast<ExpressionStatement>(*s).expression(), types, values);
      return TCResult(types);
    }
    case Statement::Kind::If: {
      auto& if_stmt = cast<If>(*s);
      TypeCheckExp(&if_stmt.condition(), types, values);
      ExpectType(s->source_loc(), "condition of `if`", arena_->New<BoolType>(),
                 &if_stmt.condition().static_type());
      TypeCheckStmt(&if_stmt.then_block(), types, values, return_type_context);
      if (if_stmt.else_block()) {
        TypeCheckStmt(*if_stmt.else_block(), types, values,
                      return_type_context);
      }
      return TCResult(types);
    }
    case Statement::Kind::Return: {
      auto& ret = cast<Return>(*s);
      TypeCheckExp(&ret.expression(), types, values);
      if (return_type_context->is_auto()) {
        if (return_type_context->deduced_return_type()) {
          // Only one return is allowed when the return type is `auto`.
          FATAL_COMPILATION_ERROR(s->source_loc())
              << "Only one return is allowed in a function with an `auto` "
                 "return type.";
        } else {
          // Infer the auto return from the first `return` statement.
          return_type_context->set_deduced_return_type(
              &ret.expression().static_type());
        }
      } else {
        ExpectType(s->source_loc(), "return",
                   *return_type_context->deduced_return_type(),
                   &ret.expression().static_type());
      }
      if (ret.is_omitted_expression() != return_type_context->is_omitted()) {
        FATAL_COMPILATION_ERROR(s->source_loc())
            << *s << " should"
            << (return_type_context->is_omitted() ? " not" : "")
            << " provide a return value, to match the function's signature.";
      }
      return TCResult(types);
    }
    case Statement::Kind::Continuation: {
      auto& cont = cast<Continuation>(*s);
      TypeCheckStmt(&cont.body(), types, values, return_type_context);
      types.Set(cont.continuation_variable(), arena_->New<ContinuationType>());
      return TCResult(types);
    }
    case Statement::Kind::Run: {
      auto& run = cast<Run>(*s);
      TypeCheckExp(&run.argument(), types, values);
      ExpectType(s->source_loc(), "argument of `run`",
                 arena_->New<ContinuationType>(),
                 &run.argument().static_type());
      return TCResult(types);
    }
    case Statement::Kind::Await: {
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
    if (clause.pattern().kind() == Pattern::Kind::BindingPattern) {
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
    case Statement::Kind::Match: {
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
    case Statement::Kind::Block:
      ExpectReturnOnAllPaths(cast<Block>(*stmt).sequence(), stmt->source_loc());
      return;
    case Statement::Kind::If: {
      auto& if_stmt = cast<If>(*stmt);
      ExpectReturnOnAllPaths(&if_stmt.then_block(), stmt->source_loc());
      ExpectReturnOnAllPaths(if_stmt.else_block(), stmt->source_loc());
      return;
    }
    case Statement::Kind::Return:
      return;
    case Statement::Kind::Sequence: {
      auto& seq = cast<Sequence>(*stmt);
      if (seq.next()) {
        ExpectReturnOnAllPaths(seq.next(), stmt->source_loc());
      } else {
        ExpectReturnOnAllPaths(&seq.statement(), stmt->source_loc());
      }
      return;
    }
    case Statement::Kind::Continuation:
    case Statement::Kind::Run:
    case Statement::Kind::Await:
      return;
    case Statement::Kind::Assign:
    case Statement::Kind::ExpressionStatement:
    case Statement::Kind::While:
    case Statement::Kind::Break:
    case Statement::Kind::Continue:
    case Statement::Kind::VariableDefinition:
      FATAL_COMPILATION_ERROR(stmt->source_loc())
          << "control-flow reaches end of function that provides a `->` "
             "return type without reaching a return statement";
  }
}

// TODO: factor common parts of TypeCheckFunDef and TypeOfFunDef into
// a function.
// TODO: Add checking to function definitions to ensure that
//   all deduced type parameters will be deduced.
auto TypeChecker::TypeCheckFunDef(FunctionDeclaration* f, TypeEnv types,
                                  Env values) -> TCResult {
  // Bring the deduced parameters into scope
  for (const auto& deduced : f->deduced_parameters()) {
    // auto t = interpreter_.InterpExp(values, deduced.type);
    types.Set(deduced.name, arena_->New<VariableType>(deduced.name));
    Address a = interpreter_.AllocateValue(*types.Get(deduced.name));
    values.Set(deduced.name, a);
  }
  // Type check the parameter pattern
  auto param_res =
      TypeCheckPattern(&f->param_pattern(), types, values, std::nullopt);
  // Evaluate the return type expression
  auto return_type = interpreter_.InterpPattern(values, &f->return_type());
  if (f->name() == "main") {
    ExpectExactType(f->source_loc(), "return type of `main`",
                    arena_->New<IntType>(), return_type);
    // TODO: Check that main doesn't have any parameters.
  }
  std::optional<Nonnull<Statement*>> body_stmt;
  if (f->body()) {
    ReturnTypeContext return_type_context(return_type,
                                          f->is_omitted_return_type());
    TypeCheckStmt(*f->body(), param_res.types, values, &return_type_context);
    body_stmt = *f->body();
    // Save the return type in case it changed.
    if (return_type_context.deduced_return_type().has_value()) {
      return_type = *return_type_context.deduced_return_type();
    }
  }
  if (!f->is_omitted_return_type()) {
    ExpectReturnOnAllPaths(body_stmt, f->source_loc());
  }
  ExpectIsConcreteType(f->return_type().source_loc(), return_type);
  SetStaticType(f, arena_->New<FunctionType>(f->deduced_parameters(),
                                             &f->param_pattern().static_type(),
                                             return_type));
  return TCResult(types);
}

auto TypeChecker::TypeOfFunDef(TypeEnv types, Env values,
                               FunctionDeclaration* fun_def)
    -> Nonnull<const Value*> {
  // Bring the deduced parameters into scope
  for (const auto& deduced : fun_def->deduced_parameters()) {
    // auto t = interpreter_.InterpExp(values, deduced.type);
    types.Set(deduced.name, arena_->New<VariableType>(deduced.name));
    Address a = interpreter_.AllocateValue(*types.Get(deduced.name));
    values.Set(deduced.name, a);
  }
  // Type check the parameter pattern
  TypeCheckPattern(&fun_def->param_pattern(), types, values, std::nullopt);
  // Evaluate the return type expression
  auto ret = interpreter_.InterpPattern(values, &fun_def->return_type());
  if (ret->kind() == Value::Kind::AutoType) {
    // FIXME do this unconditionally?
    TypeCheckFunDef(fun_def, types, values);
    return &fun_def->static_type();
  }
  return arena_->New<FunctionType>(fun_def->deduced_parameters(),
                                   &fun_def->param_pattern().static_type(),
                                   ret);
}

auto TypeChecker::TypeOfClassDef(const ClassDefinition* sd, TypeEnv /*types*/,
                                 Env ct_top) -> Nonnull<const Value*> {
  VarValues fields;
  VarValues methods;
  for (Nonnull<const Member*> m : sd->members()) {
    switch (m->kind()) {
      case Member::Kind::FieldMember: {
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
        fields.push_back(std::make_pair(*binding.name(), type));
        break;
      }
    }
  }
  return arena_->New<NominalClassType>(sd->name(), std::move(fields),
                                       std::move(methods));
}

static auto GetName(const Declaration& d) -> const std::string& {
  switch (d.kind()) {
    case Declaration::Kind::FunctionDeclaration:
      return cast<FunctionDeclaration>(d).name();
    case Declaration::Kind::ClassDeclaration:
      return cast<ClassDeclaration>(d).definition().name();
    case Declaration::Kind::ChoiceDeclaration:
      return cast<ChoiceDeclaration>(d).name();
    case Declaration::Kind::VariableDeclaration: {
      const BindingPattern& binding = cast<VariableDeclaration>(d).binding();
      if (!binding.name().has_value()) {
        FATAL_COMPILATION_ERROR(binding.source_loc())
            << "Top-level variable declarations must have names";
      }
      return *binding.name();
    }
  }
}

void TypeChecker::TypeCheck(Nonnull<Declaration*> d, const TypeEnv& types,
                            const Env& values) {
  switch (d->kind()) {
    case Declaration::Kind::FunctionDeclaration:
      TypeCheckFunDef(&cast<FunctionDeclaration>(*d), types, values);
      return;
    case Declaration::Kind::ClassDeclaration:
      // TODO
      return;

    case Declaration::Kind::ChoiceDeclaration:
      // TODO
      return;

    case Declaration::Kind::VariableDeclaration: {
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
    case Declaration::Kind::FunctionDeclaration: {
      auto& func_def = cast<FunctionDeclaration>(*d);
      auto t = TypeOfFunDef(tops->types, tops->values, &func_def);
      tops->types.Set(func_def.name(), t);
      interpreter_.InitEnv(*d, &tops->values);
      break;
    }

    case Declaration::Kind::ClassDeclaration: {
      const auto& class_def = cast<ClassDeclaration>(*d).definition();
      auto st = TypeOfClassDef(&class_def, tops->types, tops->values);
      Address a = interpreter_.AllocateValue(st);
      tops->values.Set(class_def.name(), a);  // Is this obsolete?
      tops->types.Set(class_def.name(), st);
      break;
    }

    case Declaration::Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(*d);
      VarValues alts;
      for (const auto& alternative : choice.alternatives()) {
        auto t = interpreter_.InterpExp(tops->values, &alternative.signature());
        alts.push_back(std::make_pair(alternative.name(), t));
      }
      auto ct = arena_->New<ChoiceType>(choice.name(), std::move(alts));
      Address a = interpreter_.AllocateValue(ct);
      tops->values.Set(choice.name(), a);  // Is this obsolete?
      tops->types.Set(choice.name(), ct);
      break;
    }

    case Declaration::Kind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(*d);
      // Associate the variable name with it's declared type in the
      // compile-time symbol table.
      Expression& type =
          cast<ExpressionPattern>(var.binding().type()).expression();
      Nonnull<const Value*> declared_type =
          interpreter_.InterpExp(tops->values, &type);
      tops->types.Set(*var.binding().name(), declared_type);
      break;
    }
  }
}

auto TypeChecker::TopLevel(std::vector<Nonnull<Declaration*>>* fs)
    -> TypeCheckContext {
  TypeCheckContext tops(arena_);
  bool found_main = false;

  for (auto const& d : *fs) {
    if (GetName(*d) == "main") {
      found_main = true;
    }
    TopLevel(d, &tops);
  }

  if (found_main == false) {
    FATAL_COMPILATION_ERROR_NO_LINE()
        << "program must contain a function named `main`";
  }
  return tops;
}

}  // namespace Carbon
