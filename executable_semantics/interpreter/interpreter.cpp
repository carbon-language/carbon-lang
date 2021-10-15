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
#include "executable_semantics/interpreter/frame.h"
#include "executable_semantics/interpreter/stack.h"
#include "llvm/ADT/ScopeExit.h"
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

auto Interpreter::CurrentEnv() -> Env {
  Nonnull<Frame*> frame = stack.Top();
  return frame->scopes.Top()->values;
}

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
  for (const auto& frame : stack) {
    out << sep << *frame;
  }
  out << "\nheap: " << heap;
  if (!stack.IsEmpty() && !stack.Top()->scopes.IsEmpty()) {
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
      return arena->New<IntValue>(-cast<IntValue>(*args[0]).value());
    case Operator::Add:
      return arena->New<IntValue>(cast<IntValue>(*args[0]).value() +
                                  cast<IntValue>(*args[1]).value());
    case Operator::Sub:
      return arena->New<IntValue>(cast<IntValue>(*args[0]).value() -
                                  cast<IntValue>(*args[1]).value());
    case Operator::Mul:
      return arena->New<IntValue>(cast<IntValue>(*args[0]).value() *
                                  cast<IntValue>(*args[1]).value());
    case Operator::Not:
      return arena->New<BoolValue>(!cast<BoolValue>(*args[0]).value());
    case Operator::And:
      return arena->New<BoolValue>(cast<BoolValue>(*args[0]).value() &&
                                   cast<BoolValue>(*args[1]).value());
    case Operator::Or:
      return arena->New<BoolValue>(cast<BoolValue>(*args[0]).value() ||
                                   cast<BoolValue>(*args[1]).value());
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

void Interpreter::DeallocateScope(Nonnull<Scope*> scope) {
  for (const auto& l : scope->locals) {
    std::optional<Address> a = scope->values.Get(l);
    CHECK(a);
    heap.Deallocate(*a);
  }
}

void Interpreter::DeallocateLocals(Nonnull<Frame*> frame) {
  while (!frame->scopes.IsEmpty()) {
    DeallocateScope(frame->scopes.Top());
    frame->scopes.Pop();
  }
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
      if (placeholder.name().has_value()) {
        Address a = heap.AllocateValue(CopyVal(arena, v, source_loc));
        values.Set(*placeholder.name(), a);
      }
      return values;
    }
    case Value::Kind::TupleValue:
      switch (v->kind()) {
        case Value::Kind::TupleValue: {
          const auto& p_tup = cast<TupleValue>(*p);
          const auto& v_tup = cast<TupleValue>(*v);
          if (p_tup.elements().size() != v_tup.elements().size()) {
            FATAL_PROGRAM_ERROR(source_loc)
                << "arity mismatch in tuple pattern match:\n  pattern: "
                << p_tup << "\n  value: " << v_tup;
          }
          Env values(arena);
          for (size_t i = 0; i < p_tup.elements().size(); ++i) {
            if (p_tup.elements()[i].name != v_tup.elements()[i].name) {
              FATAL_PROGRAM_ERROR(source_loc)
                  << "Tuple field name '" << v_tup.elements()[i].name
                  << "' does not match pattern field name '"
                  << p_tup.elements()[i].name << "'";
            }
            std::optional<Env> matches =
                PatternMatch(p_tup.elements()[i].value,
                             v_tup.elements()[i].value, source_loc);
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
          if (p_alt.choice_name() != v_alt.choice_name() ||
              p_alt.alt_name() != v_alt.alt_name()) {
            return std::nullopt;
          }
          return PatternMatch(&p_alt.argument(), &v_alt.argument(), source_loc);
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
              PatternMatch(&p_fn.parameters(), &v_fn.parameters(), source_loc);
          if (!param_matches) {
            return std::nullopt;
          }
          std::optional<Env> ret_matches = PatternMatch(
              &p_fn.return_type(), &v_fn.return_type(), source_loc);
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
      heap.Write(cast<PointerValue>(*pat).value(),
                 CopyVal(arena, val, source_loc), source_loc);
      break;
    case Value::Kind::TupleValue: {
      switch (val->kind()) {
        case Value::Kind::TupleValue: {
          const auto& pat_tup = cast<TupleValue>(*pat);
          const auto& val_tup = cast<TupleValue>(*val);
          if (pat_tup.elements().size() != val_tup.elements().size()) {
            FATAL_RUNTIME_ERROR(source_loc)
                << "arity mismatch in tuple pattern assignment:\n  pattern: "
                << pat_tup << "\n  value: " << val_tup;
          }
          for (const TupleElement& pattern_element : pat_tup.elements()) {
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
          CHECK(val_alt.choice_name() == pat_alt.choice_name() &&
                val_alt.alt_name() == pat_alt.alt_name())
              << "internal error in pattern assignment";
          PatternAssignment(&pat_alt.argument(), &val_alt.argument(),
                            source_loc);
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
  Nonnull<Action*> act = stack.Top()->todo.Top();
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
                                   cast<IdentifierExpression>(*exp).name());
      Nonnull<const Value*> v = arena->New<PointerValue>(pointer);
      return Done{v};
    }
    case Expression::Kind::FieldAccessExpression: {
      if (act->pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        return Spawn{arena->New<LValAction>(
            &cast<FieldAccessExpression>(*exp).aggregate())};
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->results()[0]).value();
        Address field = aggregate.SubobjectAddress(
            cast<FieldAccessExpression>(*exp).field());
        return Done{arena->New<PointerValue>(field)};
      }
    }
    case Expression::Kind::IndexExpression: {
      if (act->pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{
            arena->New<LValAction>(&cast<IndexExpression>(*exp).aggregate())};

      } else if (act->pos() == 1) {
        return Spawn{arena->New<ExpressionAction>(
            &cast<IndexExpression>(*exp).offset())};
      } else {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->results()[0]).value();
        std::string f =
            std::to_string(cast<IntValue>(*act->results()[1]).value());
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
        return Spawn{arena->New<LValAction>(
            &cast<TupleLiteral>(*exp).fields()[act->pos()].expression())};
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
  Nonnull<Action*> act = stack.Top()->todo.Top();
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
            &cast<IndexExpression>(*exp).aggregate())};
      } else if (act->pos() == 1) {
        return Spawn{arena->New<ExpressionAction>(
            &cast<IndexExpression>(*exp).offset())};
      } else {
        //    { { v :: [][i] :: C, E, F} :: S, H}
        // -> { { v_i :: C, E, F} : S, H}
        auto* tuple = dyn_cast<TupleValue>(act->results()[0]);
        if (tuple == nullptr) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "expected a tuple in field access, not " << *act->results()[0];
        }
        std::string f =
            std::to_string(cast<IntValue>(*act->results()[1]).value());
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
        return Spawn{arena->New<ExpressionAction>(
            &cast<TupleLiteral>(*exp).fields()[act->pos()].expression())};
      } else {
        return Done{CreateTuple(act, exp)};
      }
    }
    case Expression::Kind::StructLiteral: {
      const auto& literal = cast<StructLiteral>(*exp);
      if (act->pos() < static_cast<int>(literal.fields().size())) {
        return Spawn{arena->New<ExpressionAction>(
            &literal.fields()[act->pos()].expression())};
      } else {
        return Done{CreateStruct(literal.fields(), act->results())};
      }
    }
    case Expression::Kind::StructTypeLiteral: {
      const auto& struct_type = cast<StructTypeLiteral>(*exp);
      if (act->pos() < static_cast<int>(struct_type.fields().size())) {
        return Spawn{arena->New<ExpressionAction>(
            &struct_type.fields()[act->pos()].expression())};
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
        return Spawn{arena->New<ExpressionAction>(&access.aggregate())};
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        return Done{act->results()[0]->GetField(
            arena, FieldPath(access.field()), exp->source_loc())};
      }
    }
    case Expression::Kind::IdentifierExpression: {
      CHECK(act->pos() == 0);
      const auto& ident = cast<IdentifierExpression>(*exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp->source_loc(), ident.name());
      return Done{heap.Read(pointer, exp->source_loc())};
    }
    case Expression::Kind::IntLiteral:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena->New<IntValue>(cast<IntLiteral>(*exp).value())};
    case Expression::Kind::BoolLiteral:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena->New<BoolValue>(cast<BoolLiteral>(*exp).value())};
    case Expression::Kind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(*exp);
      if (act->pos() != static_cast<int>(op.arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        Nonnull<const Expression*> arg = op.arguments()[act->pos()];
        return Spawn{arena->New<ExpressionAction>(arg)};
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        return Done{EvalPrim(op.op(), act->results(), exp->source_loc())};
      }
    }
    case Expression::Kind::CallExpression:
      if (act->pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            &cast<CallExpression>(*exp).function())};
      } else if (act->pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            &cast<CallExpression>(*exp).argument())};
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
            return Done{arena->New<AlternativeValue>(alt.alt_name(),
                                                     alt.choice_name(), arg)};
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
      } else {
        FATAL() << "in handle_value with Call pos " << act->pos();
      }
    case Expression::Kind::IntrinsicExpression:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      switch (cast<IntrinsicExpression>(*exp).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          Address pointer = GetFromEnv(exp->source_loc(), "format_str");
          Nonnull<const Value*> pointee = heap.Read(pointer, exp->source_loc());
          CHECK(pointee->kind() == Value::Kind::StringValue);
          // TODO: This could eventually use something like llvm::formatv.
          llvm::outs() << cast<StringValue>(*pointee).value();
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
            &cast<FunctionTypeLiteral>(*exp).parameter())};
      } else if (act->pos() == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        return Spawn{arena->New<ExpressionAction>(
            &cast<FunctionTypeLiteral>(*exp).return_type())};
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
      return Done{arena->New<StringValue>(cast<StringLiteral>(*exp).value())};
    case Expression::Kind::StringTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena->New<StringType>()};
    }
  }  // switch (exp->kind)
}

auto Interpreter::StepPattern() -> Transition {
  Nonnull<Action*> act = stack.Top()->todo.Top();
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
        return Spawn{
            arena->New<PatternAction>(tuple.Fields()[act->pos()].pattern)};
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
                                                 choice_type.name(),
                                                 act->results()[1])};
      }
    }
    case Pattern::Kind::ExpressionPattern:
      return Delegate{arena->New<ExpressionAction>(
          cast<ExpressionPattern>(*pattern).Expression())};
  }
}

static auto IsWhileAct(Nonnull<Action*> act) -> bool {
  switch (act->kind()) {
    case Action::Kind::StatementAction:
      switch (cast<StatementAction>(*act).Stmt()->kind()) {
        case Statement::Kind::While:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

static auto HasLocalScope(Nonnull<Action*> act) -> bool {
  switch (act->kind()) {
    case Action::Kind::StatementAction:
      switch (cast<StatementAction>(*act).Stmt()->kind()) {
        case Statement::Kind::Block:
        case Statement::Kind::Match:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

auto Interpreter::StepStmt() -> Transition {
  Nonnull<Frame*> frame = stack.Top();
  Nonnull<Action*> act = frame->todo.Top();
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
        frame->scopes.Push(arena->New<Scope>(CurrentEnv()));
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
          DeallocateScope(frame->scopes.Top());
          frame->scopes.Pop();
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
              frame->scopes.Top()->values.Set(name, value);
              frame->scopes.Top()->locals.push_back(name);
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
      } else if (cast<BoolValue>(*act->results().back()).value()) {
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
      auto it =
          std::find_if(frame->todo.begin(), frame->todo.end(), &IsWhileAct);
      if (it == frame->todo.end()) {
        FATAL_RUNTIME_ERROR(stmt->source_loc())
            << "`break` not inside `while` statement";
      }
      ++it;
      return UnwindTo{*it};
    }
    case Statement::Kind::Continue: {
      CHECK(act->pos() == 0);
      //    { { continue; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { (while (e) s) :: C, E', F} :: S, H}
      auto it =
          std::find_if(frame->todo.begin(), frame->todo.end(), &IsWhileAct);
      if (it == frame->todo.end()) {
        FATAL_RUNTIME_ERROR(stmt->source_loc())
            << "`continue` not inside `while` statement";
      }
      return UnwindTo{*it};
    }
    case Statement::Kind::Block: {
      if (act->pos() == 0) {
        const auto& block = cast<Block>(*stmt);
        if (block.Stmt()) {
          frame->scopes.Push(arena->New<Scope>(CurrentEnv()));
          return Spawn{arena->New<StatementAction>(*block.Stmt())};
        } else {
          return Done{};
        }
      } else {
        Nonnull<Scope*> scope = frame->scopes.Top();
        DeallocateScope(scope);
        frame->scopes.Pop(1);
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
          frame->scopes.Top()->values.Set(name, value);
          frame->scopes.Top()->locals.push_back(name);
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
      } else if (cast<BoolValue>(*act->results()[0]).value()) {
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
        return UnwindFunctionCall{ret_val};
      }
    case Statement::Kind::Sequence: {
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      const auto& seq = cast<Sequence>(*stmt);
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
      auto scopes = Stack<Nonnull<Scope*>>(arena->New<Scope>(CurrentEnv()));
      Stack<Nonnull<Action*>> todo;
      todo.Push(arena->New<StatementAction>(
          arena->New<Return>(arena, stmt->source_loc())));
      todo.Push(arena->New<StatementAction>(cast<Continuation>(*stmt).Body()));
      auto continuation_stack = arena->New<std::vector<Nonnull<Frame*>>>();
      auto continuation_frame =
          arena->New<Frame>("__continuation", scopes, todo);
      continuation_stack->push_back(continuation_frame);
      Address continuation_address =
          heap.AllocateValue(arena->New<ContinuationValue>(continuation_stack));
      // Store the continuation's address in the frame.
      continuation_frame->continuation = continuation_address;
      // Bind the continuation object to the continuation variable
      frame->scopes.Top()->values.Set(
          cast<Continuation>(*stmt).ContinuationVariable(),
          continuation_address);
      // Pop the continuation statement.
      frame->todo.Pop();
      return ManualTransition{};
    }
    case Statement::Kind::Run:
      if (act->pos() == 0) {
        // Evaluate the argument of the run statement.
        return Spawn{arena->New<ExpressionAction>(cast<Run>(*stmt).Argument())};
      } else {
        frame->todo.Pop(1);
        // Push an expression statement action to ignore the result
        // value from the continuation.
        auto ignore_result =
            arena->New<StatementAction>(arena->New<ExpressionStatement>(
                stmt->source_loc(),
                arena->New<TupleLiteral>(stmt->source_loc())));
        frame->todo.Push(ignore_result);
        // Push the continuation onto the current stack.
        std::vector<Nonnull<Frame*>>& continuation_vector =
            cast<ContinuationValue>(*act->results()[0]).stack();
        while (!continuation_vector.empty()) {
          stack.Push(continuation_vector.back());
          continuation_vector.pop_back();
        }
        return ManualTransition{};
      }
    case Statement::Kind::Await:
      CHECK(act->pos() == 0);
      // Pause the current continuation
      frame->todo.Pop();
      std::vector<Nonnull<Frame*>> paused;
      do {
        paused.push_back(stack.Pop());
      } while (paused.back()->continuation == std::nullopt);
      // Update the continuation with the paused stack.
      const auto& continuation = cast<ContinuationValue>(
          *heap.Read(*paused.back()->continuation, stmt->source_loc()));
      CHECK(continuation.stack().empty());
      continuation.stack() = std::move(paused);
      return ManualTransition{};
  }
}

class Interpreter::DoTransition {
 public:
  // Does not take ownership of interpreter.
  explicit DoTransition(Interpreter* interpreter) : interpreter(interpreter) {}

  void operator()(const Done& done) {
    Nonnull<Frame*> frame = interpreter->stack.Top();
    if (frame->todo.Top()->kind() != Action::Kind::StatementAction) {
      CHECK(done.result);
      frame->todo.Pop();
      if (frame->todo.IsEmpty()) {
        interpreter->program_value = *done.result;
      } else {
        frame->todo.Top()->AddResult(*done.result);
      }
    } else {
      CHECK(!done.result);
      frame->todo.Pop();
    }
  }

  void operator()(const Spawn& spawn) {
    Nonnull<Frame*> frame = interpreter->stack.Top();
    Nonnull<Action*> action = frame->todo.Top();
    action->set_pos(action->pos() + 1);
    frame->todo.Push(spawn.child);
  }

  void operator()(const Delegate& delegate) {
    Nonnull<Frame*> frame = interpreter->stack.Top();
    frame->todo.Pop();
    frame->todo.Push(delegate.delegate);
  }

  void operator()(const RunAgain&) {
    Nonnull<Action*> action = interpreter->stack.Top()->todo.Top();
    action->set_pos(action->pos() + 1);
  }

  void operator()(const UnwindTo& unwind_to) {
    Nonnull<Frame*> frame = interpreter->stack.Top();
    while (frame->todo.Top() != unwind_to.new_top) {
      if (HasLocalScope(frame->todo.Top())) {
        interpreter->DeallocateScope(frame->scopes.Top());
        frame->scopes.Pop();
      }
      frame->todo.Pop();
    }
  }

  void operator()(const UnwindFunctionCall& unwind) {
    interpreter->DeallocateLocals(interpreter->stack.Top());
    interpreter->stack.Pop();
    if (interpreter->stack.Top()->todo.IsEmpty()) {
      interpreter->program_value = unwind.return_val;
    } else {
      interpreter->stack.Top()->todo.Top()->AddResult(unwind.return_val);
    }
  }

  void operator()(const CallFunction& call) {
    interpreter->stack.Top()->todo.Pop();
    std::optional<Env> matches = interpreter->PatternMatch(
        &call.function->parameters(), call.args, call.source_loc);
    CHECK(matches.has_value())
        << "internal error in call_function, pattern match failed";
    // Create the new frame and push it on the stack
    Env values = interpreter->globals;
    std::vector<std::string> params;
    for (const auto& [name, value] : *matches) {
      values.Set(name, value);
      params.push_back(name);
    }
    auto scopes =
        Stack<Nonnull<Scope*>>(interpreter->arena->New<Scope>(values, params));
    CHECK(call.function->body()) << "Calling a function that's missing a body";
    auto todo = Stack<Nonnull<Action*>>(
        interpreter->arena->New<StatementAction>(*call.function->body()));
    auto frame =
        interpreter->arena->New<Frame>(call.function->name(), scopes, todo);
    interpreter->stack.Push(frame);
  }

  void operator()(const ManualTransition&) {}

 private:
  Nonnull<Interpreter*> interpreter;
};

// State transition.
void Interpreter::Step() {
  Nonnull<Frame*> frame = stack.Top();
  if (frame->todo.IsEmpty()) {
    std::visit(DoTransition(this),
               Transition{UnwindFunctionCall{TupleValue::Empty()}});
    return;
  }

  Nonnull<Action*> act = frame->todo.Top();
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
  }  // switch
}

auto Interpreter::InterpProgram(llvm::ArrayRef<Nonnull<Declaration*>> fs,
                                Nonnull<const Expression*> call_main) -> int {
  // Check that the interpreter is in a clean state.
  CHECK(globals.IsEmpty());
  CHECK(stack.IsEmpty());
  CHECK(program_value == std::nullopt);

  if (tracing_output) {
    llvm::outs() << "********** initializing globals **********\n";
  }
  InitGlobals(fs);

  auto todo = Stack<Nonnull<Action*>>(arena->New<ExpressionAction>(call_main));
  auto scopes = Stack<Nonnull<Scope*>>(arena->New<Scope>(globals));
  stack = Stack<Nonnull<Frame*>>(arena->New<Frame>("top", scopes, todo));

  if (tracing_output) {
    llvm::outs() << "********** calling main function **********\n";
    PrintState(llvm::outs());
  }

  while (stack.Count() > 1 || !stack.Top()->todo.IsEmpty()) {
    Step();
    if (tracing_output) {
      PrintState(llvm::outs());
    }
  }
  return cast<IntValue>(**program_value).value();
}

auto Interpreter::InterpExp(Env values, Nonnull<const Expression*> e)
    -> Nonnull<const Value*> {
  CHECK(program_value == std::nullopt);
  auto program_value_guard =
      llvm::make_scope_exit([&] { program_value = std::nullopt; });
  auto todo = Stack<Nonnull<Action*>>(arena->New<ExpressionAction>(e));
  auto scopes = Stack<Nonnull<Scope*>>(arena->New<Scope>(values));
  stack = Stack<Nonnull<Frame*>>(arena->New<Frame>("InterpExp", scopes, todo));

  while (stack.Count() > 1 || !stack.Top()->todo.IsEmpty()) {
    Step();
  }
  CHECK(program_value != std::nullopt);
  return *program_value;
}

auto Interpreter::InterpPattern(Env values, Nonnull<const Pattern*> p)
    -> Nonnull<const Value*> {
  CHECK(program_value == std::nullopt);
  auto program_value_guard =
      llvm::make_scope_exit([&] { program_value = std::nullopt; });
  auto todo = Stack<Nonnull<Action*>>(arena->New<PatternAction>(p));
  auto scopes = Stack<Nonnull<Scope*>>(arena->New<Scope>(values));
  stack =
      Stack<Nonnull<Frame*>>(arena->New<Frame>("InterpPattern", scopes, todo));

  while (stack.Count() > 1 || !stack.Top()->todo.IsEmpty()) {
    Step();
  }
  CHECK(program_value != std::nullopt);
  return *program_value;
}

}  // namespace Carbon
