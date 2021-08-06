// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/interpreter.h"

#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <utility>
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

State* state = nullptr;

auto PatternMatch(const Value* pat, const Value* val, Env,
                  std::list<std::string>*, int) -> std::optional<Env>;
void Step();
//
// Auxiliary Functions
//

void PrintEnv(Env values, llvm::raw_ostream& out) {
  llvm::ListSeparator sep;
  for (const auto& [name, address] : values) {
    out << sep << name << ": ";
    state->heap.PrintAddress(address, out);
  }
}

//
// State Operations
//

void PrintStack(const Stack<Frame*>& ls, llvm::raw_ostream& out) {
  llvm::ListSeparator sep(" :: ");
  for (const auto& frame : ls) {
    out << sep << *frame;
  }
}

auto CurrentEnv(State* state) -> Env {
  Frame* frame = state->stack.Top();
  return frame->scopes.Top()->values;
}

void PrintState(llvm::raw_ostream& out) {
  out << "{\nstack: ";
  PrintStack(state->stack, out);
  out << "\nheap: " << state->heap;
  if (!state->stack.IsEmpty() && !state->stack.Top()->scopes.IsEmpty()) {
    out << "\nvalues: ";
    PrintEnv(CurrentEnv(state), out);
  }
  out << "\n}\n";
}

auto EvalPrim(Operator op, const std::vector<const Value*>& args, int line_num)
    -> const Value* {
  switch (op) {
    case Operator::Neg:
      return global_arena->New<IntValue>(-cast<IntValue>(*args[0]).Val());
    case Operator::Add:
      return global_arena->New<IntValue>(cast<IntValue>(*args[0]).Val() +
                                         cast<IntValue>(*args[1]).Val());
    case Operator::Sub:
      return global_arena->New<IntValue>(cast<IntValue>(*args[0]).Val() -
                                         cast<IntValue>(*args[1]).Val());
    case Operator::Mul:
      return global_arena->New<IntValue>(cast<IntValue>(*args[0]).Val() *
                                         cast<IntValue>(*args[1]).Val());
    case Operator::Not:
      return global_arena->New<BoolValue>(!cast<BoolValue>(*args[0]).Val());
    case Operator::And:
      return global_arena->New<BoolValue>(cast<BoolValue>(*args[0]).Val() &&
                                          cast<BoolValue>(*args[1]).Val());
    case Operator::Or:
      return global_arena->New<BoolValue>(cast<BoolValue>(*args[0]).Val() ||
                                          cast<BoolValue>(*args[1]).Val());
    case Operator::Eq:
      return global_arena->New<BoolValue>(
          ValueEqual(args[0], args[1], line_num));
    case Operator::Ptr:
      return global_arena->New<PointerType>(args[0]);
    case Operator::Deref:
      FATAL() << "dereference not implemented yet";
  }
}

// Globally-defined entities, such as functions, structs, choices.
static Env globals;

void InitEnv(const Declaration& d, Env* env) {
  switch (d.Tag()) {
    case Declaration::Kind::FunctionDeclaration: {
      const FunctionDefinition& func_def =
          cast<FunctionDeclaration>(d).Definition();
      Env new_env = *env;
      // Bring the deduced parameters into scope.
      for (const auto& deduced : func_def.deduced_parameters) {
        Address a = state->heap.AllocateValue(
            global_arena->New<VariableType>(deduced.name));
        new_env.Set(deduced.name, a);
      }
      auto pt = InterpPattern(new_env, func_def.param_pattern);
      auto f =
          global_arena->New<FunctionValue>(func_def.name, pt, func_def.body);
      Address a = state->heap.AllocateValue(f);
      env->Set(func_def.name, a);
      break;
    }

    case Declaration::Kind::StructDeclaration: {
      const StructDefinition& struct_def =
          cast<StructDeclaration>(d).Definition();
      VarValues fields;
      VarValues methods;
      for (const Member* m : struct_def.members) {
        switch (m->tag()) {
          case MemberKind::FieldMember: {
            const BindingPattern* binding = m->GetFieldMember().binding;
            const Expression* type_expression =
                cast<ExpressionPattern>(binding->Type())->Expression();
            auto type = InterpExp(Env(), type_expression);
            fields.push_back(make_pair(*binding->Name(), type));
            break;
          }
        }
      }
      auto st = global_arena->New<StructType>(
          struct_def.name, std::move(fields), std::move(methods));
      auto a = state->heap.AllocateValue(st);
      env->Set(struct_def.name, a);
      break;
    }

    case Declaration::Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(d);
      VarValues alts;
      for (const auto& [name, signature] : choice.Alternatives()) {
        auto t = InterpExp(Env(), signature);
        alts.push_back(make_pair(name, t));
      }
      auto ct = global_arena->New<ChoiceType>(choice.Name(), std::move(alts));
      auto a = state->heap.AllocateValue(ct);
      env->Set(choice.Name(), a);
      break;
    }

    case Declaration::Kind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(d);
      // Adds an entry in `globals` mapping the variable's name to the
      // result of evaluating the initializer.
      auto v = InterpExp(*env, var.Initializer());
      Address a = state->heap.AllocateValue(v);
      env->Set(*var.Binding()->Name(), a);
      break;
    }
  }
}

static void InitGlobals(const std::list<const Declaration*>& fs) {
  for (const auto* d : fs) {
    InitEnv(*d, &globals);
  }
}

//    { S, H} -> { { C, E, F} :: S, H}
// where C is the body of the function,
//       E is the environment (functions + parameters + locals)
//       F is the function
void CallFunction(int line_num, std::vector<const Value*> operas,
                  State* state) {
  switch (operas[0]->Tag()) {
    case Value::Kind::FunctionValue: {
      const auto& fn = cast<FunctionValue>(*operas[0]);
      // Bind arguments to parameters
      std::list<std::string> params;
      std::optional<Env> matches =
          PatternMatch(fn.Param(), operas[1], globals, &params, line_num);
      CHECK(matches) << "internal error in call_function, pattern match failed";
      // Create the new frame and push it on the stack
      auto* scope = global_arena->New<Scope>(*matches, params);
      auto* frame = global_arena->New<Frame>(
          fn.Name(), Stack(scope),
          Stack<Action*>(global_arena->New<StatementAction>(fn.Body())));
      state->stack.Push(frame);
      break;
    }
    case Value::Kind::StructType: {
      const Value* arg = CopyVal(operas[1], line_num);
      const Value* sv = global_arena->New<StructValue>(operas[0], arg);
      Frame* frame = state->stack.Top();
      frame->todo.Push(global_arena->New<ValAction>(sv));
      break;
    }
    case Value::Kind::AlternativeConstructorValue: {
      const auto& alt = cast<AlternativeConstructorValue>(*operas[0]);
      const Value* arg = CopyVal(operas[1], line_num);
      const Value* av = global_arena->New<AlternativeValue>(
          alt.AltName(), alt.ChoiceName(), arg);
      Frame* frame = state->stack.Top();
      frame->todo.Push(global_arena->New<ValAction>(av));
      break;
    }
    default:
      FATAL_RUNTIME_ERROR(line_num)
          << "in call, expected a function, not " << *operas[0];
  }
}

void DeallocateScope(int line_num, Scope* scope) {
  for (const auto& l : scope->locals) {
    std::optional<Address> a = scope->values.Get(l);
    CHECK(a);
    state->heap.Deallocate(*a);
  }
}

void DeallocateLocals(int line_num, Frame* frame) {
  while (!frame->scopes.IsEmpty()) {
    DeallocateScope(line_num, frame->scopes.Top());
    frame->scopes.Pop();
  }
}

void CreateTuple(Frame* frame, Action* act, const Expression* exp) {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  const auto& tup_lit = exp->GetTupleLiteral();
  CHECK(act->Results().size() == tup_lit.fields.size());
  std::vector<TupleElement> elements;
  for (size_t i = 0; i < act->Results().size(); ++i) {
    elements.push_back(
        {.name = tup_lit.fields[i].name, .value = act->Results()[i]});
  }

  const Value* tv = global_arena->New<TupleValue>(std::move(elements));
  frame->todo.Pop(1);
  frame->todo.Push(global_arena->New<ValAction>(tv));
}

// Returns an updated environment that includes the bindings of
//    pattern variables to their matched values, if matching succeeds.
//
// The names of the pattern variables are added to the vars parameter.
// Returns nullopt if the value doesn't match the pattern.
auto PatternMatch(const Value* p, const Value* v, Env values,
                  std::list<std::string>* vars, int line_num)
    -> std::optional<Env> {
  switch (p->Tag()) {
    case Value::Kind::BindingPlaceholderValue: {
      const auto& placeholder = cast<BindingPlaceholderValue>(*p);
      if (placeholder.Name().has_value()) {
        Address a = state->heap.AllocateValue(CopyVal(v, line_num));
        vars->push_back(*placeholder.Name());
        values.Set(*placeholder.Name(), a);
      }
      return values;
    }
    case Value::Kind::TupleValue:
      switch (v->Tag()) {
        case Value::Kind::TupleValue: {
          const auto& p_tup = cast<TupleValue>(*p);
          const auto& v_tup = cast<TupleValue>(*v);
          if (p_tup.Elements().size() != v_tup.Elements().size()) {
            FATAL_RUNTIME_ERROR(line_num)
                << "arity mismatch in tuple pattern match:\n  pattern: "
                << p_tup << "\n  value: " << v_tup;
          }
          for (const TupleElement& pattern_element : p_tup.Elements()) {
            const Value* value_field = v_tup.FindField(pattern_element.name);
            if (value_field == nullptr) {
              FATAL_RUNTIME_ERROR(line_num)
                  << "field " << pattern_element.name << "not in " << *v;
            }
            std::optional<Env> matches = PatternMatch(
                pattern_element.value, value_field, values, vars, line_num);
            if (!matches) {
              return std::nullopt;
            }
            values = *matches;
          }  // for
          return values;
        }
        default:
          FATAL() << "expected a tuple value in pattern, not " << *v;
      }
    case Value::Kind::AlternativeValue:
      switch (v->Tag()) {
        case Value::Kind::AlternativeValue: {
          const auto& p_alt = cast<AlternativeValue>(*p);
          const auto& v_alt = cast<AlternativeValue>(*v);
          if (p_alt.ChoiceName() != v_alt.ChoiceName() ||
              p_alt.AltName() != v_alt.AltName()) {
            return std::nullopt;
          }
          std::optional<Env> matches = PatternMatch(
              p_alt.Argument(), v_alt.Argument(), values, vars, line_num);
          if (!matches) {
            return std::nullopt;
          }
          return *matches;
        }
        default:
          FATAL() << "expected a choice alternative in pattern, not " << *v;
      }
    case Value::Kind::FunctionType:
      switch (v->Tag()) {
        case Value::Kind::FunctionType: {
          const auto& p_fn = cast<FunctionType>(*p);
          const auto& v_fn = cast<FunctionType>(*v);
          std::optional<Env> matches =
              PatternMatch(p_fn.Param(), v_fn.Param(), values, vars, line_num);
          if (!matches) {
            return std::nullopt;
          }
          return PatternMatch(p_fn.Ret(), v_fn.Ret(), *matches, vars, line_num);
        }
        default:
          return std::nullopt;
      }
    default:
      if (ValueEqual(p, v, line_num)) {
        return values;
      } else {
        return std::nullopt;
      }
  }
}

void PatternAssignment(const Value* pat, const Value* val, int line_num) {
  switch (pat->Tag()) {
    case Value::Kind::PointerValue:
      state->heap.Write(cast<PointerValue>(*pat).Val(), CopyVal(val, line_num),
                        line_num);
      break;
    case Value::Kind::TupleValue: {
      switch (val->Tag()) {
        case Value::Kind::TupleValue: {
          const auto& pat_tup = cast<TupleValue>(*pat);
          const auto& val_tup = cast<TupleValue>(*val);
          if (pat_tup.Elements().size() != val_tup.Elements().size()) {
            FATAL_RUNTIME_ERROR(line_num)
                << "arity mismatch in tuple pattern assignment:\n  pattern: "
                << pat_tup << "\n  value: " << val_tup;
          }
          for (const TupleElement& pattern_element : pat_tup.Elements()) {
            const Value* value_field = val_tup.FindField(pattern_element.name);
            if (value_field == nullptr) {
              FATAL_RUNTIME_ERROR(line_num)
                  << "field " << pattern_element.name << "not in " << *val;
            }
            PatternAssignment(pattern_element.value, value_field, line_num);
          }
          break;
        }
        default:
          FATAL() << "expected a tuple value on right-hand-side, not " << *val;
      }
      break;
    }
    case Value::Kind::AlternativeValue: {
      switch (val->Tag()) {
        case Value::Kind::AlternativeValue: {
          const auto& pat_alt = cast<AlternativeValue>(*pat);
          const auto& val_alt = cast<AlternativeValue>(*val);
          CHECK(val_alt.ChoiceName() == pat_alt.ChoiceName() &&
                val_alt.AltName() == pat_alt.AltName())
              << "internal error in pattern assignment";
          PatternAssignment(pat_alt.Argument(), val_alt.Argument(), line_num);
          break;
        }
        default:
          FATAL() << "expected an alternative in left-hand-side, not " << *val;
      }
      break;
    }
    default:
      CHECK(ValueEqual(pat, val, line_num))
          << "internal error in pattern assignment";
  }
}

// State transitions for lvalues.

void StepLvalue() {
  Frame* frame = state->stack.Top();
  Action* act = frame->todo.Top();
  const Expression* exp = cast<LValAction>(*act).Exp();
  if (tracing_output) {
    llvm::outs() << "--- step lvalue " << *exp << " --->\n";
  }
  switch (exp->tag()) {
    case ExpressionKind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      std::optional<Address> pointer =
          CurrentEnv(state).Get(exp->GetIdentifierExpression().name);
      if (!pointer) {
        FATAL_RUNTIME_ERROR(exp->line_num)
            << "could not find `" << exp->GetIdentifierExpression().name << "`";
      }
      const Value* v = global_arena->New<PointerValue>(*pointer);
      frame->todo.Pop();
      frame->todo.Push(global_arena->New<ValAction>(v));
      break;
    }
    case ExpressionKind::FieldAccessExpression: {
      if (act->Pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<LValAction>(
            exp->GetFieldAccessExpression().aggregate));
        act->IncrementPos();
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->Results()[0]).Val();
        Address field =
            aggregate.SubobjectAddress(exp->GetFieldAccessExpression().field);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->New<ValAction>(
            global_arena->New<PointerValue>(field)));
      }
      break;
    }
    case ExpressionKind::IndexExpression: {
      if (act->Pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->New<LValAction>(exp->GetIndexExpression().aggregate));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        frame->todo.Push(global_arena->New<ExpressionAction>(
            exp->GetIndexExpression().offset));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->Results()[0]).Val();
        std::string f =
            std::to_string(cast<IntValue>(*act->Results()[1]).Val());
        Address field = aggregate.SubobjectAddress(f);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->New<ValAction>(
            global_arena->New<PointerValue>(field)));
      }
      break;
    }
    case ExpressionKind::TupleLiteral: {
      if (act->Pos() == 0) {
        //    { {(f1=e1,...) :: C, E, F} :: S, H}
        // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
        const Expression* e1 = exp->GetTupleLiteral().fields[0].expression;
        frame->todo.Push(global_arena->New<LValAction>(e1));
        act->IncrementPos();
      } else if (act->Pos() !=
                 static_cast<int>(exp->GetTupleLiteral().fields.size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Expression* elt =
            exp->GetTupleLiteral().fields[act->Pos()].expression;
        frame->todo.Push(global_arena->New<LValAction>(elt));
        act->IncrementPos();
      } else {
        CreateTuple(frame, act, exp);
      }
      break;
    }
    case ExpressionKind::IntLiteral:
    case ExpressionKind::BoolLiteral:
    case ExpressionKind::CallExpression:
    case ExpressionKind::PrimitiveOperatorExpression:
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::FunctionTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral: {
      FATAL_RUNTIME_ERROR_NO_LINE()
          << "Can't treat expression as lvalue: " << *exp;
    }
  }
}

// State transitions for expressions.

void StepExp() {
  Frame* frame = state->stack.Top();
  Action* act = frame->todo.Top();
  const Expression* exp = cast<ExpressionAction>(*act).Exp();
  if (tracing_output) {
    llvm::outs() << "--- step exp " << *exp << " --->\n";
  }
  switch (exp->tag()) {
    case ExpressionKind::IndexExpression: {
      if (act->Pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<ExpressionAction>(
            exp->GetIndexExpression().aggregate));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        frame->todo.Push(global_arena->New<ExpressionAction>(
            exp->GetIndexExpression().offset));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        auto tuple = act->Results()[0];
        switch (tuple->Tag()) {
          case Value::Kind::TupleValue: {
            //    { { v :: [][i] :: C, E, F} :: S, H}
            // -> { { v_i :: C, E, F} : S, H}
            std::string f =
                std::to_string(cast<IntValue>(*act->Results()[1]).Val());
            const Value* field = cast<TupleValue>(*tuple).FindField(f);
            if (field == nullptr) {
              FATAL_RUNTIME_ERROR_NO_LINE()
                  << "field " << f << " not in " << *tuple;
            }
            frame->todo.Pop(1);
            frame->todo.Push(global_arena->New<ValAction>(field));
            break;
          }
          default:
            FATAL_RUNTIME_ERROR_NO_LINE()
                << "expected a tuple in field access, not " << *tuple;
        }
      }
      break;
    }
    case ExpressionKind::TupleLiteral: {
      if (act->Pos() == 0) {
        if (exp->GetTupleLiteral().fields.size() > 0) {
          //    { {(f1=e1,...) :: C, E, F} :: S, H}
          // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
          const Expression* e1 = exp->GetTupleLiteral().fields[0].expression;
          frame->todo.Push(global_arena->New<ExpressionAction>(e1));
          act->IncrementPos();
        } else {
          CreateTuple(frame, act, exp);
        }
      } else if (act->Pos() !=
                 static_cast<int>(exp->GetTupleLiteral().fields.size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Expression* elt =
            exp->GetTupleLiteral().fields[act->Pos()].expression;
        frame->todo.Push(global_arena->New<ExpressionAction>(elt));
        act->IncrementPos();
      } else {
        CreateTuple(frame, act, exp);
      }
      break;
    }
    case ExpressionKind::FieldAccessExpression: {
      if (act->Pos() == 0) {
        //    { { e.f :: C, E, F} :: S, H}
        // -> { { e :: [].f :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<ExpressionAction>(
            exp->GetFieldAccessExpression().aggregate));
        act->IncrementPos();
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        const Value* element = act->Results()[0]->GetField(
            FieldPath(exp->GetFieldAccessExpression().field), exp->line_num);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->New<ValAction>(element));
      }
      break;
    }
    case ExpressionKind::IdentifierExpression: {
      CHECK(act->Pos() == 0);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      std::optional<Address> pointer =
          CurrentEnv(state).Get(exp->GetIdentifierExpression().name);
      if (!pointer) {
        FATAL_RUNTIME_ERROR(exp->line_num)
            << "could not find `" << exp->GetIdentifierExpression().name << "`";
      }
      const Value* pointee = state->heap.Read(*pointer, exp->line_num);
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ValAction>(pointee));
      break;
    }
    case ExpressionKind::IntLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ValAction>(
          global_arena->New<IntValue>(exp->GetIntLiteral())));
      break;
    case ExpressionKind::BoolLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ValAction>(
          global_arena->New<BoolValue>(exp->GetBoolLiteral())));
      break;
    case ExpressionKind::PrimitiveOperatorExpression:
      if (act->Pos() !=
          static_cast<int>(
              exp->GetPrimitiveOperatorExpression().arguments.size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        const Expression* arg =
            exp->GetPrimitiveOperatorExpression().arguments[act->Pos()];
        frame->todo.Push(global_arena->New<ExpressionAction>(arg));
        act->IncrementPos();
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        const Value* v = EvalPrim(exp->GetPrimitiveOperatorExpression().op,
                                  act->Results(), exp->line_num);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->New<ValAction>(v));
      }
      break;
    case ExpressionKind::CallExpression:
      if (act->Pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<ExpressionAction>(
            exp->GetCallExpression().function));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<ExpressionAction>(
            exp->GetCallExpression().argument));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        frame->todo.Pop(1);
        CallFunction(exp->line_num, act->Results(), state);
      } else {
        FATAL() << "in handle_value with Call pos " << act->Pos();
      }
      break;
    case ExpressionKind::IntTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->New<IntType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ValAction>(v));
      break;
    }
    case ExpressionKind::BoolTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->New<BoolType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ValAction>(v));
      break;
    }
    case ExpressionKind::TypeTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->New<TypeType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ValAction>(v));
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      if (act->Pos() == 0) {
        frame->todo.Push(global_arena->New<ExpressionAction>(
            exp->GetFunctionTypeLiteral().parameter));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<ExpressionAction>(
            exp->GetFunctionTypeLiteral().return_type));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        const Value* v = global_arena->New<FunctionType>(
            std::vector<GenericBinding>(), act->Results()[0],
            act->Results()[1]);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->New<ValAction>(v));
      }
      break;
    }
    case ExpressionKind::ContinuationTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->New<ContinuationType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ValAction>(v));
      break;
    }
  }  // switch (exp->tag)
}

void StepPattern() {
  Frame* frame = state->stack.Top();
  Action* act = frame->todo.Top();
  const Pattern* pattern = cast<PatternAction>(*act).Pat();
  if (tracing_output) {
    llvm::outs() << "--- step pattern " << *pattern << " --->\n";
  }
  switch (pattern->Tag()) {
    case Pattern::Kind::AutoPattern: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->New<AutoType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ValAction>(v));
      break;
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*pattern);
      if (act->Pos() == 0) {
        frame->todo.Push(global_arena->New<PatternAction>(binding.Type()));
        act->IncrementPos();
      } else {
        auto v = global_arena->New<BindingPlaceholderValue>(binding.Name(),
                                                            act->Results()[0]);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->New<ValAction>(v));
      }
      break;
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*pattern);
      if (act->Pos() == 0) {
        if (tuple.Fields().empty()) {
          frame->todo.Pop(1);
          frame->todo.Push(global_arena->New<ValAction>(&TupleValue::Empty()));
        } else {
          const Pattern* p1 = tuple.Fields()[0].pattern;
          frame->todo.Push(global_arena->New<PatternAction>(p1));
          act->IncrementPos();
        }
      } else if (act->Pos() != static_cast<int>(tuple.Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Pattern* elt = tuple.Fields()[act->Pos()].pattern;
        frame->todo.Push(global_arena->New<PatternAction>(elt));
        act->IncrementPos();
      } else {
        std::vector<TupleElement> elements;
        for (size_t i = 0; i < tuple.Fields().size(); ++i) {
          elements.push_back(
              {.name = tuple.Fields()[i].name, .value = act->Results()[i]});
        }
        const Value* tuple_value =
            global_arena->New<TupleValue>(std::move(elements));
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->New<ValAction>(tuple_value));
      }
      break;
    }
    case Pattern::Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*pattern);
      if (act->Pos() == 0) {
        frame->todo.Push(
            global_arena->New<ExpressionAction>(alternative.ChoiceType()));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        frame->todo.Push(
            global_arena->New<PatternAction>(alternative.Arguments()));
        act->IncrementPos();
      } else {
        CHECK(act->Pos() == 2);
        const auto& choice_type = cast<ChoiceType>(*act->Results()[0]);
        frame->todo.Pop(1);
        frame->todo.Push(
            global_arena->New<ValAction>(global_arena->New<AlternativeValue>(
                alternative.AlternativeName(), choice_type.Name(),
                act->Results()[1])));
      }
      break;
    }
    case Pattern::Kind::ExpressionPattern:
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->New<ExpressionAction>(
          cast<ExpressionPattern>(pattern)->Expression()));
      break;
  }
}

auto IsWhileAct(Action* act) -> bool {
  switch (act->Tag()) {
    case Action::Kind::StatementAction:
      switch (cast<StatementAction>(*act).Stmt()->tag()) {
        case StatementKind::While:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

auto IsBlockAct(Action* act) -> bool {
  switch (act->Tag()) {
    case Action::Kind::StatementAction:
      switch (cast<StatementAction>(*act).Stmt()->tag()) {
        case StatementKind::Block:
          return true;
        default:
          return false;
      }
    default:
      return false;
  }
}

// State transitions for statements.

void StepStmt() {
  Frame* frame = state->stack.Top();
  Action* act = frame->todo.Top();
  const Statement* stmt = cast<StatementAction>(*act).Stmt();
  CHECK(stmt != nullptr) << "null statement!";
  if (tracing_output) {
    llvm::outs() << "--- step stmt ";
    stmt->PrintDepth(1, llvm::outs());
    llvm::outs() << " --->\n";
  }
  switch (stmt->tag()) {
    case StatementKind::Match:
      if (act->Pos() == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->New<ExpressionAction>(stmt->GetMatch().exp));
        act->IncrementPos();
      } else {
        // Regarding act->Pos():
        // * odd: start interpreting the pattern of a clause
        // * even: finished interpreting the pattern, now try to match
        //
        // Regarding act->Results():
        // * 0: the value that we're matching
        // * 1: the pattern for clause 0
        // * 2: the pattern for clause 1
        // * ...
        auto clause_num = (act->Pos() - 1) / 2;
        if (clause_num >= static_cast<int>(stmt->GetMatch().clauses->size())) {
          frame->todo.Pop(1);
          break;
        }
        auto c = stmt->GetMatch().clauses->begin();
        std::advance(c, clause_num);

        if (act->Pos() % 2 == 1) {
          // start interpreting the pattern of the clause
          //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
          // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
          frame->todo.Push(global_arena->New<PatternAction>(c->first));
          act->IncrementPos();
        } else {  // try to match
          auto v = act->Results()[0];
          auto pat = act->Results()[clause_num + 1];
          auto values = CurrentEnv(state);
          std::list<std::string> vars;
          std::optional<Env> matches =
              PatternMatch(pat, v, values, &vars, stmt->line_num);
          if (matches) {  // we have a match, start the body
            auto* new_scope = global_arena->New<Scope>(*matches, vars);
            frame->scopes.Push(new_scope);
            const Statement* body_block =
                Statement::MakeBlock(stmt->line_num, c->second);
            Action* body_act = global_arena->New<StatementAction>(body_block);
            body_act->IncrementPos();
            frame->todo.Pop(1);
            frame->todo.Push(body_act);
            frame->todo.Push(global_arena->New<StatementAction>(c->second));
          } else {
            // this case did not match, moving on
            act->IncrementPos();
            clause_num = (act->Pos() - 1) / 2;
            if (clause_num ==
                static_cast<int>(stmt->GetMatch().clauses->size())) {
              frame->todo.Pop(1);
            }
          }
        }
      }
      break;
    case StatementKind::While:
      if (act->Pos() == 0) {
        //    { { (while (e) s) :: C, E, F} :: S, H}
        // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->New<ExpressionAction>(stmt->GetWhile().cond));
        act->IncrementPos();
      } else if (cast<BoolValue>(*act->Results()[0]).Val()) {
        //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
        frame->todo.Top()->Clear();
        frame->todo.Push(
            global_arena->New<StatementAction>(stmt->GetWhile().body));
      } else {
        //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { C, E, F } :: S, H}
        frame->todo.Top()->Clear();
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::Break:
      CHECK(act->Pos() == 0);
      //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { C, E', F} :: S, H}
      frame->todo.Pop(1);
      while (!frame->todo.IsEmpty() && !IsWhileAct(frame->todo.Top())) {
        if (IsBlockAct(frame->todo.Top())) {
          DeallocateScope(stmt->line_num, frame->scopes.Top());
          frame->scopes.Pop(1);
        }
        frame->todo.Pop(1);
      }
      frame->todo.Pop(1);
      break;
    case StatementKind::Continue:
      CHECK(act->Pos() == 0);
      //    { { continue; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { (while (e) s) :: C, E', F} :: S, H}
      frame->todo.Pop(1);
      while (!frame->todo.IsEmpty() && !IsWhileAct(frame->todo.Top())) {
        if (IsBlockAct(frame->todo.Top())) {
          DeallocateScope(stmt->line_num, frame->scopes.Top());
          frame->scopes.Pop(1);
        }
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::Block: {
      if (act->Pos() == 0) {
        if (stmt->GetBlock().stmt) {
          auto* scope = global_arena->New<Scope>(CurrentEnv(state),
                                                 std::list<std::string>());
          frame->scopes.Push(scope);
          frame->todo.Push(
              global_arena->New<StatementAction>(stmt->GetBlock().stmt));
          act->IncrementPos();
          act->IncrementPos();
        } else {
          frame->todo.Pop();
        }
      } else {
        Scope* scope = frame->scopes.Top();
        DeallocateScope(stmt->line_num, scope);
        frame->scopes.Pop(1);
        frame->todo.Pop(1);
      }
      break;
    }
    case StatementKind::VariableDefinition:
      if (act->Pos() == 0) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<ExpressionAction>(
            stmt->GetVariableDefinition().init));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        frame->todo.Push(global_arena->New<PatternAction>(
            stmt->GetVariableDefinition().pat));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        const Value* v = act->Results()[0];
        const Value* p = act->Results()[1];

        std::optional<Env> matches =
            PatternMatch(p, v, frame->scopes.Top()->values,
                         &frame->scopes.Top()->locals, stmt->line_num);
        CHECK(matches)
            << stmt->line_num
            << ": internal error in variable definition, match failed";
        frame->scopes.Top()->values = *matches;
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::ExpressionStatement:
      if (act->Pos() == 0) {
        //    { {e :: C, E, F} :: S, H}
        // -> { {e :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<ExpressionAction>(
            stmt->GetExpressionStatement().exp));
        act->IncrementPos();
      } else {
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::Assign:
      if (act->Pos() == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->New<LValAction>(stmt->GetAssign().lhs));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->New<ExpressionAction>(stmt->GetAssign().rhs));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        auto pat = act->Results()[0];
        auto val = act->Results()[1];
        PatternAssignment(pat, val, stmt->line_num);
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::If:
      if (act->Pos() == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->New<ExpressionAction>(stmt->GetIf().cond));
        act->IncrementPos();
      } else if (cast<BoolValue>(*act->Results()[0]).Val()) {
        //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { then_stmt :: C, E, F } :: S, H}
        frame->todo.Pop(1);
        frame->todo.Push(
            global_arena->New<StatementAction>(stmt->GetIf().then_stmt));
      } else if (stmt->GetIf().else_stmt) {
        //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { else_stmt :: C, E, F } :: S, H}
        frame->todo.Pop(1);
        frame->todo.Push(
            global_arena->New<StatementAction>(stmt->GetIf().else_stmt));
      } else {
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::Return:
      if (act->Pos() == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->New<ExpressionAction>(stmt->GetReturn().exp));
        act->IncrementPos();
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        const Value* ret_val = CopyVal(act->Results()[0], stmt->line_num);
        DeallocateLocals(stmt->line_num, frame);
        state->stack.Pop(1);
        frame = state->stack.Top();
        frame->todo.Push(global_arena->New<ValAction>(ret_val));
      }
      break;
    case StatementKind::Sequence:
      CHECK(act->Pos() == 0);
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      if (stmt->GetSequence().next) {
        frame->todo.Push(
            global_arena->New<StatementAction>(stmt->GetSequence().next));
      }
      frame->todo.Push(
          global_arena->New<StatementAction>(stmt->GetSequence().stmt));
      break;
    case StatementKind::Continuation: {
      CHECK(act->Pos() == 0);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      Scope* scope =
          global_arena->New<Scope>(CurrentEnv(state), std::list<std::string>());
      Stack<Scope*> scopes;
      scopes.Push(scope);
      Stack<Action*> todo;
      todo.Push(global_arena->New<StatementAction>(
          Statement::MakeReturn(stmt->line_num, nullptr,
                                /*is_omitted_exp=*/true)));
      todo.Push(
          global_arena->New<StatementAction>(stmt->GetContinuation().body));
      Frame* continuation_frame =
          global_arena->New<Frame>("__continuation", scopes, todo);
      Address continuation_address =
          state->heap.AllocateValue(global_arena->New<ContinuationValue>(
              std::vector<Frame*>({continuation_frame})));
      // Store the continuation's address in the frame.
      continuation_frame->continuation = continuation_address;
      // Bind the continuation object to the continuation variable
      frame->scopes.Top()->values.Set(
          stmt->GetContinuation().continuation_variable, continuation_address);
      // Pop the continuation statement.
      frame->todo.Pop();
      break;
    }
    case StatementKind::Run:
      if (act->Pos() == 0) {
        // Evaluate the argument of the run statement.
        frame->todo.Push(
            global_arena->New<ExpressionAction>(stmt->GetRun().argument));
        act->IncrementPos();
      } else {
        frame->todo.Pop(1);
        // Push an expression statement action to ignore the result
        // value from the continuation.
        Action* ignore_result = global_arena->New<StatementAction>(
            Statement::MakeExpressionStatement(
                stmt->line_num,
                Expression::MakeTupleLiteral(stmt->line_num, {})));
        frame->todo.Push(ignore_result);
        // Push the continuation onto the current stack.
        const std::vector<Frame*>& continuation_vector =
            cast<ContinuationValue>(*act->Results()[0]).Stack();
        for (auto frame_iter = continuation_vector.rbegin();
             frame_iter != continuation_vector.rend(); ++frame_iter) {
          state->stack.Push(*frame_iter);
        }
      }
      break;
    case StatementKind::Await:
      CHECK(act->Pos() == 0);
      // Pause the current continuation
      frame->todo.Pop();
      std::vector<Frame*> paused;
      do {
        paused.push_back(state->stack.Pop());
      } while (paused.back()->continuation == std::nullopt);
      // Update the continuation with the paused stack.
      state->heap.Write(*paused.back()->continuation,
                        global_arena->New<ContinuationValue>(paused),
                        stmt->line_num);
      break;
  }
}

// State transition.
void Step() {
  Frame* frame = state->stack.Top();
  if (frame->todo.IsEmpty()) {
    FATAL_RUNTIME_ERROR_NO_LINE()
        << "fell off end of function " << frame->name << " without `return`";
  }

  Action* act = frame->todo.Top();
  switch (act->Tag()) {
    case Action::Kind::ValAction: {
      const ValAction& val_act = cast<ValAction>(*frame->todo.Pop());
      Action* act = frame->todo.Top();
      act->AddResult(val_act.Val());
      break;
    }
    case Action::Kind::LValAction:
      StepLvalue();
      break;
    case Action::Kind::ExpressionAction:
      StepExp();
      break;
    case Action::Kind::PatternAction:
      StepPattern();
      break;
    case Action::Kind::StatementAction:
      StepStmt();
      break;
  }  // switch
}

// Interpret the whole porogram.
auto InterpProgram(const std::list<const Declaration*>& fs) -> int {
  state = global_arena->New<State>();  // Runtime state.
  if (tracing_output) {
    llvm::outs() << "********** initializing globals **********\n";
  }
  InitGlobals(fs);

  const Expression* arg = Expression::MakeTupleLiteral(0, {});
  const Expression* call_main = Expression::MakeCallExpression(
      0, Expression::MakeIdentifierExpression(0, "main"), arg);
  auto todo = Stack<Action*>(global_arena->New<ExpressionAction>(call_main));
  auto* scope = global_arena->New<Scope>(globals, std::list<std::string>());
  auto* frame = global_arena->New<Frame>("top", Stack(scope), todo);
  state->stack = Stack(frame);

  if (tracing_output) {
    llvm::outs() << "********** calling main function **********\n";
    PrintState(llvm::outs());
  }

  while (state->stack.Count() > 1 || state->stack.Top()->todo.Count() > 1 ||
         state->stack.Top()->todo.Top()->Tag() != Action::Kind::ValAction) {
    Step();
    if (tracing_output) {
      PrintState(llvm::outs());
    }
  }
  const Value* v = cast<ValAction>(*state->stack.Top()->todo.Top()).Val();
  return cast<IntValue>(*v).Val();
}

// Interpret an expression at compile-time.
auto InterpExp(Env values, const Expression* e) -> const Value* {
  auto todo = Stack<Action*>(global_arena->New<ExpressionAction>(e));
  auto* scope = global_arena->New<Scope>(values, std::list<std::string>());
  auto* frame = global_arena->New<Frame>("InterpExp", Stack(scope), todo);
  state->stack = Stack(frame);

  while (state->stack.Count() > 1 || state->stack.Top()->todo.Count() > 1 ||
         state->stack.Top()->todo.Top()->Tag() != Action::Kind::ValAction) {
    Step();
  }
  return cast<ValAction>(*state->stack.Top()->todo.Top()).Val();
}

// Interpret a pattern at compile-time.
auto InterpPattern(Env values, const Pattern* p) -> const Value* {
  auto todo = Stack<Action*>(global_arena->New<PatternAction>(p));
  auto* scope = global_arena->New<Scope>(values, std::list<std::string>());
  auto* frame = global_arena->New<Frame>("InterpPattern", Stack(scope), todo);
  state->stack = Stack(frame);

  while (state->stack.Count() > 1 || state->stack.Top()->todo.Count() > 1 ||
         state->stack.Top()->todo.Top()->Tag() != Action::Kind::ValAction) {
    Step();
  }
  return cast<ValAction>(*state->stack.Top()->todo.Top()).Val();
}

}  // namespace Carbon
