// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/interpreter.h"

#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "common/check.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
#include "executable_semantics/common/tracing_flag.h"
#include "executable_semantics/interpreter/action.h"
#include "executable_semantics/interpreter/stack.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;

namespace Carbon {

//
// Auxiliary Functions
//

void Interpreter::PrintEnv(Env values, llvm::raw_ostream& out) {
  llvm::ListSeparator sep;
  for (const auto& [name, address] : values) {
    out << sep << name << ": ";
    heap.PrintAddress(address, out);
  }
}

//
// State Operations
//

auto Interpreter::CurrentScope() -> Scope& {
  for (Nonnull<Action*> action : todo) {
    if (action->scope().has_value()) {
      return *action->scope();
    }
  }
  FATAL() << "No current scope";
}

auto Interpreter::CurrentEnv() -> Env { return CurrentScope().values; }

// Returns the given name from the environment, printing an error if not found.
auto Interpreter::GetFromEnv(SourceLocation source_loc, const std::string& name)
    -> Address {
  std::optional<Address> pointer = CurrentEnv().Get(name);
  if (!pointer) {
    FATAL_RUNTIME_ERROR(source_loc) << "could not find `" << name << "`";
  }
  return *pointer;
}

void Interpreter::PrintState(llvm::raw_ostream& out) {
  out << "{\nstack: ";
  llvm::ListSeparator sep(" :: ");
  for (Nonnull<const Action*> action : todo) {
    out << sep << *action;
  }
  out << "\nheap: " << heap;
  if (!todo.IsEmpty()) {
    out << "\nvalues: ";
    PrintEnv(CurrentEnv(), out);
  }
  out << "\n}\n";
}

auto Interpreter::EvalPrim(Operator op,
                           const std::vector<Nonnull<const Value*>>& args,
                           SourceLocation source_loc) -> Nonnull<const Value*> {
  switch (op) {
    case Operator::Neg:
      return arena->New<IntValue>(-cast<IntValue>(*args[0]).Val());
    case Operator::Add:
      return arena->New<IntValue>(cast<IntValue>(*args[0]).Val() +
                                  cast<IntValue>(*args[1]).Val());
    case Operator::Sub:
      return arena->New<IntValue>(cast<IntValue>(*args[0]).Val() -
                                  cast<IntValue>(*args[1]).Val());
    case Operator::Mul:
      return arena->New<IntValue>(cast<IntValue>(*args[0]).Val() *
                                  cast<IntValue>(*args[1]).Val());
    case Operator::Not:
      return arena->New<BoolValue>(!cast<BoolValue>(*args[0]).Val());
    case Operator::And:
      return arena->New<BoolValue>(cast<BoolValue>(*args[0]).Val() &&
                                   cast<BoolValue>(*args[1]).Val());
    case Operator::Or:
      return arena->New<BoolValue>(cast<BoolValue>(*args[0]).Val() ||
                                   cast<BoolValue>(*args[1]).Val());
    case Operator::Eq:
      return arena->New<BoolValue>(ValueEqual(args[0], args[1], source_loc));
    case Operator::Ptr:
      return arena->New<PointerType>(args[0]);
    case Operator::Deref:
      FATAL() << "dereference not implemented yet";
  }
}

void Interpreter::InitEnv(const Declaration& d, Env* env) {
  switch (d.kind()) {
    case Declaration::Kind::FunctionDeclaration: {
      const FunctionDefinition& func_def =
          cast<FunctionDeclaration>(d).definition();
      Env new_env = *env;
      // Bring the deduced parameters into scope.
      for (const auto& deduced : func_def.deduced_parameters()) {
        Address a = heap.AllocateValue(arena->New<VariableType>(deduced.name));
        new_env.Set(deduced.name, a);
      }
      auto pt = InterpPattern(new_env, &func_def.param_pattern());
      auto f = arena->New<FunctionValue>(func_def.name(), pt, func_def.body());
      Address a = heap.AllocateValue(f);
      env->Set(func_def.name(), a);
      break;
    }

    case Declaration::Kind::ClassDeclaration: {
      const ClassDefinition& class_def = cast<ClassDeclaration>(d).definition();
      VarValues fields;
      VarValues methods;
      for (Nonnull<const Member*> m : class_def.members()) {
        switch (m->kind()) {
          case Member::Kind::FieldMember: {
            Nonnull<const BindingPattern*> binding =
                cast<FieldMember>(*m).Binding();
            Nonnull<const Expression*> type_expression =
                cast<ExpressionPattern>(*binding->Type()).Expression();
            auto type = InterpExp(Env(arena), type_expression);
            fields.push_back(make_pair(*binding->Name(), type));
            break;
          }
        }
      }
      auto st = arena->New<NominalClassType>(
          class_def.name(), std::move(fields), std::move(methods));
      auto a = heap.AllocateValue(st);
      env->Set(class_def.name(), a);
      break;
    }

    case Declaration::Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(d);
      VarValues alts;
      for (const auto& alternative : choice.alternatives()) {
        auto t = InterpExp(Env(arena), &alternative.signature());
        alts.push_back(make_pair(alternative.name(), t));
      }
      auto ct = arena->New<ChoiceType>(choice.name(), std::move(alts));
      auto a = heap.AllocateValue(ct);
      env->Set(choice.name(), a);
      break;
    }

    case Declaration::Kind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(d);
      // Adds an entry in `globals` mapping the variable's name to the
      // result of evaluating the initializer.
      auto v = InterpExp(*env, &var.initializer());
      Address a = heap.AllocateValue(v);
      env->Set(*var.binding().Name(), a);
      break;
    }
  }
}

void Interpreter::InitGlobals(llvm::ArrayRef<Nonnull<Declaration*>> fs) {
  for (const auto d : fs) {
    InitEnv(*d, &globals);
  }
}

void Interpreter::DeallocateScope(Scope& scope) {
  CHECK(!scope.deallocated);
  for (const auto& l : scope.locals) {
    std::optional<Address> a = scope.values.Get(l);
    CHECK(a);
    heap.Deallocate(*a);
  }
  scope.deallocated = true;
}

auto Interpreter::CreateTuple(Nonnull<Action*> act,
                              Nonnull<const Expression*> exp)
    -> Nonnull<const Value*> {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  const auto& tup_lit = cast<TupleLiteral>(*exp);
  CHECK(act->results().size() == tup_lit.fields().size());
  std::vector<TupleElement> elements;
  for (size_t i = 0; i < act->results().size(); ++i) {
    elements.push_back(
        {.name = tup_lit.fields()[i].name(), .value = act->results()[i]});
  }

  return arena->New<TupleValue>(std::move(elements));
}

auto Interpreter::CreateStruct(const std::vector<FieldInitializer>& fields,
                               const std::vector<Nonnull<const Value*>>& values)
    -> Nonnull<const Value*> {
  CHECK(fields.size() == values.size());
  std::vector<TupleElement> elements;
  for (size_t i = 0; i < fields.size(); ++i) {
    elements.push_back({.name = fields[i].name(), .value = values[i]});
  }

  return arena->New<StructValue>(std::move(elements));
}

auto Interpreter::PatternMatch(Nonnull<const Value*> p, Nonnull<const Value*> v,
                               SourceLocation source_loc)
    -> std::optional<Env> {
  switch (p->kind()) {
    case Value::Kind::BindingPlaceholderValue: {
      const auto& placeholder = cast<BindingPlaceholderValue>(*p);
      Env values(arena);
      if (placeholder.Name().has_value()) {
        Address a = heap.AllocateValue(CopyVal(arena, v, source_loc));
        values.Set(*placeholder.Name(), a);
      }
      return values;
    }
    case Value::Kind::TupleValue:
      switch (v->kind()) {
        case Value::Kind::TupleValue: {
          const auto& p_tup = cast<TupleValue>(*p);
          const auto& v_tup = cast<TupleValue>(*v);
          if (p_tup.Elements().size() != v_tup.Elements().size()) {
            FATAL_PROGRAM_ERROR(source_loc)
                << "arity mismatch in tuple pattern match:\n  pattern: "
                << p_tup << "\n  value: " << v_tup;
          }
          Env values(arena);
          for (size_t i = 0; i < p_tup.Elements().size(); ++i) {
            if (p_tup.Elements()[i].name != v_tup.Elements()[i].name) {
              FATAL_PROGRAM_ERROR(source_loc)
                  << "Tuple field name '" << v_tup.Elements()[i].name
                  << "' does not match pattern field name '"
                  << p_tup.Elements()[i].name << "'";
            }
            std::optional<Env> matches =
                PatternMatch(p_tup.Elements()[i].value,
                             v_tup.Elements()[i].value, source_loc);
            if (!matches) {
              return std::nullopt;
            }
            for (const auto& [name, value] : *matches) {
              values.Set(name, value);
            }
          }  // for
          return values;
        }
        default:
          FATAL() << "expected a tuple value in pattern, not " << *v;
      }
    case Value::Kind::StructValue: {
      const auto& p_struct = cast<StructValue>(*p);
      const auto& v_struct = cast<StructValue>(*v);
      CHECK(p_struct.elements().size() == v_struct.elements().size());
      Env values(arena);
      for (size_t i = 0; i < p_struct.elements().size(); ++i) {
        CHECK(p_struct.elements()[i].name == v_struct.elements()[i].name);
        std::optional<Env> matches =
            PatternMatch(p_struct.elements()[i].value,
                         v_struct.elements()[i].value, source_loc);
        if (!matches) {
          return std::nullopt;
        }
        for (const auto& [name, value] : *matches) {
          values.Set(name, value);
        }
      }
      return values;
    }
    case Value::Kind::AlternativeValue:
      switch (v->kind()) {
        case Value::Kind::AlternativeValue: {
          const auto& p_alt = cast<AlternativeValue>(*p);
          const auto& v_alt = cast<AlternativeValue>(*v);
          if (p_alt.ChoiceName() != v_alt.ChoiceName() ||
              p_alt.AltName() != v_alt.AltName()) {
            return std::nullopt;
          }
          return PatternMatch(p_alt.Argument(), v_alt.Argument(), source_loc);
        }
        default:
          FATAL() << "expected a choice alternative in pattern, not " << *v;
      }
    case Value::Kind::FunctionType:
      switch (v->kind()) {
        case Value::Kind::FunctionType: {
          const auto& p_fn = cast<FunctionType>(*p);
          const auto& v_fn = cast<FunctionType>(*v);
          std::optional<Env> param_matches =
              PatternMatch(p_fn.Param(), v_fn.Param(), source_loc);
          if (!param_matches) {
            return std::nullopt;
          }
          std::optional<Env> ret_matches =
              PatternMatch(p_fn.Ret(), v_fn.Ret(), source_loc);
          if (!ret_matches) {
            return std::nullopt;
          }
          Env values = *param_matches;
          for (const auto& [name, value] : *ret_matches) {
            values.Set(name, value);
          }
          return values;
        }
        default:
          return std::nullopt;
      }
    case Value::Kind::AutoType:
      // `auto` matches any type, without binding any new names. We rely
      // on the typechecker to ensure that `v` is a type.
      return Env(arena);
    default:
      if (ValueEqual(p, v, source_loc)) {
        return Env(arena);
      } else {
        return std::nullopt;
      }
  }
}

void Interpreter::PatternAssignment(Nonnull<const Value*> pat,
                                    Nonnull<const Value*> val,
                                    SourceLocation source_loc) {
  switch (pat->kind()) {
    case Value::Kind::PointerValue:
      heap.Write(cast<PointerValue>(*pat).Val(),
                 CopyVal(arena, val, source_loc), source_loc);
      break;
    case Value::Kind::TupleValue: {
      switch (val->kind()) {
        case Value::Kind::TupleValue: {
          const auto& pat_tup = cast<TupleValue>(*pat);
          const auto& val_tup = cast<TupleValue>(*val);
          if (pat_tup.Elements().size() != val_tup.Elements().size()) {
            FATAL_RUNTIME_ERROR(source_loc)
                << "arity mismatch in tuple pattern assignment:\n  pattern: "
                << pat_tup << "\n  value: " << val_tup;
          }
          for (const TupleElement& pattern_element : pat_tup.Elements()) {
            std::optional<Nonnull<const Value*>> value_field =
                val_tup.FindField(pattern_element.name);
            if (!value_field) {
              FATAL_RUNTIME_ERROR(source_loc)
                  << "field " << pattern_element.name << "not in " << *val;
            }
            PatternAssignment(pattern_element.value, *value_field, source_loc);
          }
          break;
        }
        default:
          FATAL() << "expected a tuple value on right-hand-side, not " << *val;
      }
      break;
    }
    case Value::Kind::AlternativeValue: {
      switch (val->kind()) {
        case Value::Kind::AlternativeValue: {
          const auto& pat_alt = cast<AlternativeValue>(*pat);
          const auto& val_alt = cast<AlternativeValue>(*val);
          CHECK(val_alt.ChoiceName() == pat_alt.ChoiceName() &&
                val_alt.AltName() == pat_alt.AltName())
              << "internal error in pattern assignment";
          PatternAssignment(pat_alt.Argument(), val_alt.Argument(), source_loc);
          break;
        }
        default:
          FATAL() << "expected an alternative in left-hand-side, not " << *val;
      }
      break;
    }
    default:
      CHECK(ValueEqual(pat, val, source_loc))
          << "internal error in pattern assignment";
  }
}

auto Interpreter::StepLvalue() -> Transition {
  Nonnull<Action*> act = todo.Top();
  Nonnull<const Expression*> exp = cast<LValAction>(*act).Exp();
  if (tracing_output) {
    llvm::outs() << "--- step lvalue " << *exp << " (" << exp->source_loc()
                 << ") --->\n";
  }
  switch (exp->kind()) {
    case Expression::Kind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp->source_loc(),
                                   cast<IdentifierExpression>(*exp).Name());
      Nonnull<const Value*> v = arena->New<PointerValue>(pointer);
      return Done{v};
    }
    case Expression::Kind::FieldAccessExpression: {
      if (act->pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        return Spawn{arena->New<LValAction>(
            cast<FieldAccessExpression>(*exp).Aggregate())};
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->results()[0]).Val();
        Address field = aggregate.SubobjectAddress(
            cast<FieldAccessExpression>(*exp).Field());
        return Done{arena->New<PointerValue>(field)};
      }
    }
    case Expression::Kind::IndexExpression: {
      if (act->pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{
            arena->New<LValAction>(cast<IndexExpression>(*exp).Aggregate())};

      } else if (act->pos() == 1) {
        return Spawn{
            arena->New<ExpressionAction>(cast<IndexExpression>(*exp).Offset())};
      } else {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->results()[0]).Val();
        std::string f =
            std::to_string(cast<IntValue>(*act->results()[1]).Val());
        Address field = aggregate.SubobjectAddress(f);
        return Done{arena->New<PointerValue>(field)};
      }
    }
    case Expression::Kind::TupleLiteral: {
      if (act->pos() <
          static_cast<int>(cast<TupleLiteral>(*exp).fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        Nonnull<const Expression*> elt =
            cast<TupleLiteral>(*exp).fields()[act->pos()].expression();
        return Spawn{arena->New<LValAction>(elt)};
      } else {
        return Done{CreateTuple(act, exp)};
      }
    }
    case Expression::Kind::StructLiteral:
    case Expression::Kind::StructTypeLiteral:
    case Expression::Kind::IntLiteral:
    case Expression::Kind::BoolLiteral:
    case Expression::Kind::CallExpression:
    case Expression::Kind::PrimitiveOperatorExpression:
    case Expression::Kind::IntTypeLiteral:
    case Expression::Kind::BoolTypeLiteral:
    case Expression::Kind::TypeTypeLiteral:
    case Expression::Kind::FunctionTypeLiteral:
    case Expression::Kind::ContinuationTypeLiteral:
    case Expression::Kind::StringLiteral:
    case Expression::Kind::StringTypeLiteral:
    case Expression::Kind::IntrinsicExpression:
      FATAL_RUNTIME_ERROR_NO_LINE()
          << "Can't treat expression as lvalue: " << *exp;
  }
}

auto Interpreter::StepExp() -> Transition {
  Nonnull<Action*> act = todo.Top();
  Nonnull<const Expression*> exp = cast<ExpressionAction>(*act).Exp();
  if (tracing_output) {
    llvm::outs() << "--- step exp " << *exp << " (" << exp->source_loc()
                 << ") --->\n";
  }
  switch (exp->kind()) {
    case Expression::Kind::IndexExpression: {
      if (act->pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            cast<IndexExpression>(*exp).Aggregate())};
      } else if (act->pos() == 1) {
        return Spawn{
            arena->New<ExpressionAction>(cast<IndexExpression>(*exp).Offset())};
      } else {
        //    { { v :: [][i] :: C, E, F} :: S, H}
        // -> { { v_i :: C, E, F} : S, H}
        auto* tuple = dyn_cast<TupleValue>(act->results()[0]);
        if (tuple == nullptr) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "expected a tuple in field access, not " << *act->results()[0];
        }
        std::string f =
            std::to_string(cast<IntValue>(*act->results()[1]).Val());
        std::optional<Nonnull<const Value*>> field = tuple->FindField(f);
        if (!field) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "field " << f << " not in " << *tuple;
        }
        return Done{*field};
      }
    }
    case Expression::Kind::TupleLiteral: {
      if (act->pos() <
          static_cast<int>(cast<TupleLiteral>(*exp).fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        Nonnull<const Expression*> elt =
            cast<TupleLiteral>(*exp).fields()[act->pos()].expression();
        return Spawn{arena->New<ExpressionAction>(elt)};
      } else {
        return Done{CreateTuple(act, exp)};
      }
    }
    case Expression::Kind::StructLiteral: {
      const auto& literal = cast<StructLiteral>(*exp);
      if (act->pos() < static_cast<int>(literal.fields().size())) {
        Nonnull<const Expression*> elt =
            literal.fields()[act->pos()].expression();
        return Spawn{arena->New<ExpressionAction>(elt)};
      } else {
        return Done{CreateStruct(literal.fields(), act->results())};
      }
    }
    case Expression::Kind::StructTypeLiteral: {
      const auto& struct_type = cast<StructTypeLiteral>(*exp);
      if (act->pos() < static_cast<int>(struct_type.fields().size())) {
        return Spawn{arena->New<ExpressionAction>(
            struct_type.fields()[act->pos()].expression())};
      } else {
        VarValues fields;
        for (size_t i = 0; i < struct_type.fields().size(); ++i) {
          fields.push_back({struct_type.fields()[i].name(), act->results()[i]});
        }
        return Done{arena->New<StructType>(std::move(fields))};
      }
    }
    case Expression::Kind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(*exp);
      if (act->pos() == 0) {
        //    { { e.f :: C, E, F} :: S, H}
        // -> { { e :: [].f :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(access.Aggregate())};
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        return Done{act->results()[0]->GetField(
            arena, FieldPath(access.Field()), exp->source_loc())};
      }
    }
    case Expression::Kind::IdentifierExpression: {
      CHECK(act->pos() == 0);
      const auto& ident = cast<IdentifierExpression>(*exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp->source_loc(), ident.Name());
      return Done{heap.Read(pointer, exp->source_loc())};
    }
    case Expression::Kind::IntLiteral:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena->New<IntValue>(cast<IntLiteral>(*exp).Val())};
    case Expression::Kind::BoolLiteral:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena->New<BoolValue>(cast<BoolLiteral>(*exp).Val())};
    case Expression::Kind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(*exp);
      if (act->pos() != static_cast<int>(op.Arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        Nonnull<const Expression*> arg = op.Arguments()[act->pos()];
        return Spawn{arena->New<ExpressionAction>(arg)};
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        return Done{EvalPrim(op.Op(), act->results(), exp->source_loc())};
      }
    }
    case Expression::Kind::CallExpression:
      if (act->pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            cast<CallExpression>(*exp).Function())};
      } else if (act->pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            cast<CallExpression>(*exp).Argument())};
      } else if (act->pos() == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        switch (act->results()[0]->kind()) {
          case Value::Kind::NominalClassType: {
            Nonnull<const Value*> arg =
                CopyVal(arena, act->results()[1], exp->source_loc());
            return Done{arena->New<NominalClassValue>(act->results()[0], arg)};
          }
          case Value::Kind::AlternativeConstructorValue: {
            const auto& alt =
                cast<AlternativeConstructorValue>(*act->results()[0]);
            Nonnull<const Value*> arg =
                CopyVal(arena, act->results()[1], exp->source_loc());
            return Done{arena->New<AlternativeValue>(alt.AltName(),
                                                     alt.ChoiceName(), arg)};
          }
          case Value::Kind::FunctionValue:
            return CallFunction{
                // TODO: Think about a cleaner way to cast between Ptr types.
                // (multiple TODOs)
                .function = Nonnull<const FunctionValue*>(
                    cast<FunctionValue>(act->results()[0])),
                .args = act->results()[1],
                .source_loc = exp->source_loc()};
          default:
            FATAL_RUNTIME_ERROR(exp->source_loc())
                << "in call, expected a function, not " << *act->results()[0];
        }
      } else if (act->pos() == 3) {
        if (act->results().size() < 3) {
          // Control fell through without explicit return.
          return Done{TupleValue::Empty()};
        } else {
          return Done{act->results()[2]};
        }
      } else {
        FATAL() << "in handle_value with Call pos " << act->pos();
      }
    case Expression::Kind::IntrinsicExpression:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      switch (cast<IntrinsicExpression>(*exp).Intrinsic()) {
        case IntrinsicExpression::IntrinsicKind::Print:
          Address pointer = GetFromEnv(exp->source_loc(), "format_str");
          Nonnull<const Value*> pointee = heap.Read(pointer, exp->source_loc());
          CHECK(pointee->kind() == Value::Kind::StringValue);
          // TODO: This could eventually use something like llvm::formatv.
          llvm::outs() << cast<StringValue>(*pointee).Val();
          return Done{TupleValue::Empty()};
      }

    case Expression::Kind::IntTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena->New<IntType>()};
    }
    case Expression::Kind::BoolTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena->New<BoolType>()};
    }
    case Expression::Kind::TypeTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena->New<TypeType>()};
    }
    case Expression::Kind::FunctionTypeLiteral: {
      if (act->pos() == 0) {
        return Spawn{arena->New<ExpressionAction>(
            cast<FunctionTypeLiteral>(*exp).Parameter())};
      } else if (act->pos() == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            cast<FunctionTypeLiteral>(*exp).ReturnType())};
      } else {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        return Done{arena->New<FunctionType>(std::vector<GenericBinding>(),
                                             act->results()[0],
                                             act->results()[1])};
      }
    }
    case Expression::Kind::ContinuationTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena->New<ContinuationType>()};
    }
    case Expression::Kind::StringLiteral:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena->New<StringValue>(cast<StringLiteral>(*exp).Val())};
    case Expression::Kind::StringTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena->New<StringType>()};
    }
  }  // switch (exp->kind)
}

auto Interpreter::StepPattern() -> Transition {
  Nonnull<Action*> act = todo.Top();
  Nonnull<const Pattern*> pattern = cast<PatternAction>(*act).Pat();
  if (tracing_output) {
    llvm::outs() << "--- step pattern " << *pattern << " ("
                 << pattern->source_loc() << ") --->\n";
  }
  switch (pattern->kind()) {
    case Pattern::Kind::AutoPattern: {
      CHECK(act->pos() == 0);
      return Done{arena->New<AutoType>()};
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*pattern);
      if (act->pos() == 0) {
        return Spawn{arena->New<PatternAction>(binding.Type())};
      } else {
        return Done{arena->New<BindingPlaceholderValue>(binding.Name(),
                                                        act->results()[0])};
      }
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*pattern);
      if (act->pos() < static_cast<int>(tuple.Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        Nonnull<const Pattern*> elt = tuple.Fields()[act->pos()].pattern;
        return Spawn{arena->New<PatternAction>(elt)};
      } else {
        std::vector<TupleElement> elements;
        for (size_t i = 0; i < tuple.Fields().size(); ++i) {
          elements.push_back(
              {.name = tuple.Fields()[i].name, .value = act->results()[i]});
        }
        return Done{arena->New<TupleValue>(std::move(elements))};
      }
    }
    case Pattern::Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*pattern);
      if (act->pos() == 0) {
        return Spawn{arena->New<ExpressionAction>(alternative.ChoiceType())};
      } else if (act->pos() == 1) {
        return Spawn{arena->New<PatternAction>(alternative.Arguments())};
      } else {
        CHECK(act->pos() == 2);
        const auto& choice_type = cast<ChoiceType>(*act->results()[0]);
        return Done{arena->New<AlternativeValue>(alternative.AlternativeName(),
                                                 choice_type.Name(),
                                                 act->results()[1])};
      }
    }
    case Pattern::Kind::ExpressionPattern:
      return Delegate{arena->New<ExpressionAction>(
          cast<ExpressionPattern>(*pattern).Expression())};
  }
}

static auto IsRunAction(Nonnull<Action*> action) -> bool {
  const auto* statement = dyn_cast<StatementAction>(action);
  return statement != nullptr && llvm::isa<Run>(*statement->Stmt());
}

auto Interpreter::StepStmt() -> Transition {
  Nonnull<Action*> act = todo.Top();
  Nonnull<const Statement*> stmt = cast<StatementAction>(*act).Stmt();
  if (tracing_output) {
    llvm::outs() << "--- step stmt ";
    stmt->PrintDepth(1, llvm::outs());
    llvm::outs() << " (" << stmt->source_loc() << ") --->\n";
  }
  switch (stmt->kind()) {
    case Statement::Kind::Match: {
      const auto& match_stmt = cast<Match>(*stmt);
      if (act->pos() == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        act->StartScope(Scope(CurrentEnv()));
        return Spawn{arena->New<ExpressionAction>(&match_stmt.expression())};
      } else {
        // Regarding act->pos():
        // * odd: start interpreting the pattern of a clause
        // * even: finished interpreting the pattern, now try to match
        //
        // Regarding act->results():
        // * 0: the value that we're matching
        // * 1: the pattern for clause 0
        // * 2: the pattern for clause 1
        // * ...
        auto clause_num = (act->pos() - 1) / 2;
        if (clause_num >= static_cast<int>(match_stmt.clauses().size())) {
          return Done{};
        }
        auto c = match_stmt.clauses()[clause_num];

        if (act->pos() % 2 == 1) {
          // start interpreting the pattern of the clause
          //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
          // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
          return Spawn{arena->New<PatternAction>(&c.pattern())};
        } else {  // try to match
          auto v = act->results()[0];
          auto pat = act->results()[clause_num + 1];
          std::optional<Env> matches = PatternMatch(pat, v, stmt->source_loc());
          if (matches) {  // we have a match, start the body
            // Ensure we don't process any more clauses.
            act->set_pos(2 * match_stmt.clauses().size() + 1);

            for (const auto& [name, value] : *matches) {
              act->scope()->values.Set(name, value);
              act->scope()->locals.push_back(name);
            }
            return Spawn{arena->New<StatementAction>(&c.statement())};
          } else {
            return RunAgain{};
          }
        }
      }
    }
    case Statement::Kind::While:
      if (act->pos() % 2 == 0) {
        //    { { (while (e) s) :: C, E, F} :: S, H}
        // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
        act->Clear();
        return Spawn{arena->New<ExpressionAction>(cast<While>(*stmt).Cond())};
      } else if (cast<BoolValue>(*act->results().back()).Val()) {
        //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
        return Spawn{arena->New<StatementAction>(cast<While>(*stmt).Body())};
      } else {
        //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { C, E, F } :: S, H}
        return Done{};
      }
    case Statement::Kind::Break: {
      CHECK(act->pos() == 0);
      //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { C, E', F} :: S, H}
      return UnwindPast{cast<Break>(*stmt).loop()};
    }
    case Statement::Kind::Continue: {
      CHECK(act->pos() == 0);
      //    { { continue; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { (while (e) s) :: C, E', F} :: S, H}
      return UnwindTo{cast<Continue>(*stmt).loop()};
    }
    case Statement::Kind::Block: {
      if (act->pos() == 0) {
        const Block& block = cast<Block>(*stmt);
        if (block.Stmt()) {
          act->StartScope(Scope(CurrentEnv()));
          return Spawn{arena->New<StatementAction>(*block.Stmt())};
        } else {
          return Done{};
        }
      } else {
        return Done{};
      }
    }
    case Statement::Kind::VariableDefinition:
      if (act->pos() == 0) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            cast<VariableDefinition>(*stmt).Init())};
      } else if (act->pos() == 1) {
        return Spawn{
            arena->New<PatternAction>(cast<VariableDefinition>(*stmt).Pat())};
      } else {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        Nonnull<const Value*> v = act->results()[0];
        Nonnull<const Value*> p = act->results()[1];

        std::optional<Env> matches = PatternMatch(p, v, stmt->source_loc());
        CHECK(matches)
            << stmt->source_loc()
            << ": internal error in variable definition, match failed";
        for (const auto& [name, value] : *matches) {
          Scope& current_scope = CurrentScope();
          current_scope.values.Set(name, value);
          current_scope.locals.push_back(name);
        }
        return Done{};
      }
    case Statement::Kind::ExpressionStatement:
      if (act->pos() == 0) {
        //    { {e :: C, E, F} :: S, H}
        // -> { {e :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            cast<ExpressionStatement>(*stmt).Exp())};
      } else {
        return Done{};
      }
    case Statement::Kind::Assign:
      if (act->pos() == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        return Spawn{arena->New<LValAction>(cast<Assign>(*stmt).Lhs())};
      } else if (act->pos() == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(cast<Assign>(*stmt).Rhs())};
      } else {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        auto pat = act->results()[0];
        auto val = act->results()[1];
        PatternAssignment(pat, val, stmt->source_loc());
        return Done{};
      }
    case Statement::Kind::If:
      if (act->pos() == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(cast<If>(*stmt).Cond())};
      } else if (cast<BoolValue>(*act->results()[0]).Val()) {
        //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { then_stmt :: C, E, F } :: S, H}
        return Delegate{
            arena->New<StatementAction>(cast<If>(*stmt).ThenStmt())};
      } else if (cast<If>(*stmt).ElseStmt()) {
        //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { else_stmt :: C, E, F } :: S, H}
        return Delegate{
            arena->New<StatementAction>(*cast<If>(*stmt).ElseStmt())};
      } else {
        return Done{};
      }
    case Statement::Kind::Return:
      if (act->pos() == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(cast<Return>(*stmt).Exp())};
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        Nonnull<const Value*> ret_val =
            CopyVal(arena, act->results()[0], stmt->source_loc());
        return UnwindPast{*cast<Return>(*stmt).function()->body(), ret_val};
      }
    case Statement::Kind::Sequence: {
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      const Sequence& seq = cast<Sequence>(*stmt);
      if (act->pos() == 0) {
        return Spawn{arena->New<StatementAction>(seq.Stmt())};
      } else {
        if (seq.Next()) {
          return Delegate{
              arena->New<StatementAction>(*cast<Sequence>(*stmt).Next())};
        } else {
          return Done{};
        }
      }
    }
    case Statement::Kind::Continuation: {
      CHECK(act->pos() == 0);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      auto continuation_stack = arena->New<std::vector<Nonnull<Action*>>>();
      continuation_stack->push_back(
          arena->New<StatementAction>(cast<Continuation>(*stmt).Body()));
      continuation_stack->push_back(
          arena->New<ScopeAction>(Scope(CurrentEnv())));
      Address continuation_address =
          heap.AllocateValue(arena->New<ContinuationValue>(continuation_stack));
      // Bind the continuation object to the continuation variable
      CurrentScope().values.Set(
          cast<Continuation>(*stmt).ContinuationVariable(),
          continuation_address);
      return Done{};
    }
    case Statement::Kind::Run: {
      auto& run = cast<Run>(*stmt);
      if (act->pos() == 0) {
        // Evaluate the argument of the run statement.
        return Spawn{arena->New<ExpressionAction>(run.Argument())};
      } else if (act->pos() == 1) {
        // Push the continuation onto the current stack.
        std::vector<Nonnull<Action*>>& continuation_vector =
            *cast<const ContinuationValue>(*act->results()[0]).Stack();
        while (!continuation_vector.empty()) {
          todo.Push(continuation_vector.back());
          continuation_vector.pop_back();
        }
        act->set_pos(2);
        return ManualTransition{};
      } else {
        return Done{};
      }
    }
    case Statement::Kind::Await:
      CHECK(act->pos() == 0);
      // Pause the current continuation
      todo.Pop();
      std::vector<Nonnull<Action*>> paused;
      while (!IsRunAction(todo.Top())) {
        paused.push_back(todo.Pop());
      }
      const auto& continuation =
          cast<const ContinuationValue>(*todo.Top()->results()[0]);
      CHECK(continuation.Stack()->empty());
      // Update the continuation with the paused stack.
      *continuation.Stack() = std::move(paused);
      return ManualTransition{};
  }
}

class Interpreter::DoTransition {
 public:
  // Does not take ownership of interpreter.
  DoTransition(Interpreter* interpreter) : interpreter(interpreter) {}

  void operator()(const Done& done) {
    Nonnull<Action*> act = interpreter->todo.Pop();
    if (act->scope().has_value()) {
      interpreter->DeallocateScope(*act->scope());
    }
    switch (act->kind()) {
      case Action::Kind::ExpressionAction:
      case Action::Kind::LValAction:
      case Action::Kind::PatternAction:
        CHECK(done.result.has_value());
        interpreter->todo.Top()->AddResult(*done.result);
        break;
      case Action::Kind::StatementAction:
        CHECK(!done.result.has_value());
        break;
      case Action::Kind::ScopeAction:
        if (done.result.has_value()) {
          interpreter->todo.Top()->AddResult(*done.result);
        }
        break;
    }
  }

  void operator()(const Spawn& spawn) {
    Nonnull<Action*> action = interpreter->todo.Top();
    action->set_pos(action->pos() + 1);
    interpreter->todo.Push(spawn.child);
  }

  void operator()(const Delegate& delegate) {
    Nonnull<Action*> act = interpreter->todo.Pop();
    if (act->scope().has_value()) {
      delegate.delegate->StartScope(*act->scope());
    }
    interpreter->todo.Push(delegate.delegate);
  }

  void operator()(const RunAgain&) {
    Nonnull<Action*> action = interpreter->todo.Top();
    action->set_pos(action->pos() + 1);
  }

  void operator()(const UnwindTo& unwind_to) {
    while (interpreter->todo.Top()->ast_node() != unwind_to.ast_node) {
      if (interpreter->todo.Top()->scope().has_value()) {
        interpreter->DeallocateScope(*interpreter->todo.Top()->scope());
      }
      interpreter->todo.Pop();
    }
  }

  void operator()(const UnwindPast& unwind_past) {
    Nonnull<Action*> action;
    do {
      action = interpreter->todo.Pop();
      if (action->scope().has_value()) {
        interpreter->DeallocateScope(*action->scope());
      }
    } while (action->ast_node() != unwind_past.ast_node);
    if (unwind_past.result.has_value()) {
      interpreter->todo.Top()->AddResult(*unwind_past.result);
    }
  }

  void operator()(const CallFunction& call) {
    Nonnull<Action*> action = interpreter->todo.Top();
    action->set_pos(action->pos() + 1);
    std::optional<Env> matches = interpreter->PatternMatch(
        call.function->Param(), call.args, call.source_loc);
    CHECK(matches.has_value())
        << "internal error in call_function, pattern match failed";
    // Create the new frame and push it on the stack
    Scope new_scope(interpreter->globals);
    for (const auto& [name, value] : *matches) {
      new_scope.values.Set(name, value);
      new_scope.locals.push_back(name);
    }
    interpreter->todo.Push(
        interpreter->arena->New<ScopeAction>(std::move(new_scope)));
    CHECK(call.function->Body()) << "Calling a function that's missing a body";
    interpreter->todo.Push(
        interpreter->arena->New<StatementAction>(*call.function->Body()));
  }

  void operator()(const ManualTransition&) {}

 private:
  Nonnull<Interpreter*> interpreter;
};

// State transition.
void Interpreter::Step() {
  Nonnull<Action*> act = todo.Top();
  switch (act->kind()) {
    case Action::Kind::LValAction:
      std::visit(DoTransition(this), StepLvalue());
      break;
    case Action::Kind::ExpressionAction:
      std::visit(DoTransition(this), StepExp());
      break;
    case Action::Kind::PatternAction:
      std::visit(DoTransition(this), StepPattern());
      break;
    case Action::Kind::StatementAction:
      std::visit(DoTransition(this), StepStmt());
      break;
    case Action::Kind::ScopeAction:
      if (act->results().empty()) {
        std::visit(DoTransition(this), Transition{Done{}});
      } else {
        CHECK(act->results().size() == 1);
        std::visit(DoTransition(this), Transition{Done{act->results()[0]}});
      }
  }  // switch
}

// Runs `action` in a scope consisting of `values`, and returns the result.
// `action` must produce a result. In other words, it must not be a
// StatementAction or ScopeAction.
//
// TODO: consider whether to use this->tracing_output rather than a separate
// trace_steps parameter.
auto Interpreter::ExecuteAction(Nonnull<Action*> action, Env values,
                                bool trace_steps) -> Nonnull<const Value*> {
  todo = {};
  todo.Push(arena->New<ScopeAction>(Scope(values)));
  todo.Push(action);

  while (todo.Count() > 1) {
    Step();
    if (trace_steps) {
      PrintState(llvm::outs());
    }
  }
  CHECK(todo.Top()->results().size() == 1);
  return todo.Top()->results()[0];
}

auto Interpreter::InterpProgram(llvm::ArrayRef<Nonnull<Declaration*>> fs,
                                Nonnull<const Expression*> call_main) -> int {
  // Check that the interpreter is in a clean state.
  CHECK(globals.IsEmpty());
  CHECK(todo.IsEmpty());

  if (tracing_output) {
    llvm::outs() << "********** initializing globals **********\n";
  }
  InitGlobals(fs);

  if (tracing_output) {
    llvm::outs() << "********** calling main function **********\n";
    PrintState(llvm::outs());
  }

  return cast<IntValue>(*ExecuteAction(arena->New<ExpressionAction>(call_main),
                                       globals, tracing_output))
      .Val();
}

auto Interpreter::InterpExp(Env values, Nonnull<const Expression*> e)
    -> Nonnull<const Value*> {
  return ExecuteAction(arena->New<ExpressionAction>(e), values,
                       /*trace_steps=*/false);
}

auto Interpreter::InterpPattern(Env values, Nonnull<const Pattern*> p)
    -> Nonnull<const Value*> {
  return ExecuteAction(arena->New<PatternAction>(p), values,
                       /*trace_steps=*/false);
}

}  // namespace Carbon
