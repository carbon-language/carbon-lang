// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/interpreter.h"

#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "common/check.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/function_definition.h"
#include "executable_semantics/interpreter/action.h"
#include "executable_semantics/interpreter/frame.h"
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/tracing_flag.h"

namespace Carbon {

State* state = nullptr;

auto PatternMatch(const Value* pat, const Value* val, Env,
                  std::list<std::string>*, int) -> std::optional<Env>;
auto Step() -> void;
//
// Auxiliary Functions
//

void PrintEnv(Env values, std::ostream& out) {
  for (const auto& [name, address] : values) {
    out << name << ": ";
    state->heap.PrintAddress(address, out);
    out << ", ";
  }
}

//
// State Operations
//

void PrintStack(Stack<Frame*> ls, std::ostream& out) {
  if (!ls.IsEmpty()) {
    PrintFrame(ls.Pop(), out);
    if (!ls.IsEmpty()) {
      out << " :: ";
      PrintStack(ls, out);
    }
  }
}

auto CurrentEnv(State* state) -> Env {
  Frame* frame = state->stack.Top();
  return frame->scopes.Top()->values;
}

void PrintState(std::ostream& out) {
  out << "{" << std::endl;
  out << "stack: ";
  PrintStack(state->stack, out);
  out << std::endl << "heap: ";
  state->heap.PrintHeap(out);
  if (!state->stack.IsEmpty() && !state->stack.Top()->scopes.IsEmpty()) {
    out << std::endl << "values: ";
    PrintEnv(CurrentEnv(state), out);
  }
  out << std::endl << "}" << std::endl;
}

auto EvalPrim(Operator op, const std::vector<const Value*>& args, int line_num)
    -> const Value* {
  switch (op) {
    case Operator::Neg:
      return Value::MakeIntValue(-args[0]->GetIntValue());
    case Operator::Add:
      return Value::MakeIntValue(args[0]->GetIntValue() +
                                 args[1]->GetIntValue());
    case Operator::Sub:
      return Value::MakeIntValue(args[0]->GetIntValue() -
                                 args[1]->GetIntValue());
    case Operator::Mul:
      return Value::MakeIntValue(args[0]->GetIntValue() *
                                 args[1]->GetIntValue());
    case Operator::Not:
      return Value::MakeBoolValue(!args[0]->GetBoolValue());
    case Operator::And:
      return Value::MakeBoolValue(args[0]->GetBoolValue() &&
                                  args[1]->GetBoolValue());
    case Operator::Or:
      return Value::MakeBoolValue(args[0]->GetBoolValue() ||
                                  args[1]->GetBoolValue());
    case Operator::Eq:
      return Value::MakeBoolValue(ValueEqual(args[0], args[1], line_num));
    case Operator::Ptr:
      return Value::MakePointerType(args[0]);
    case Operator::Deref:
      std::cerr << line_num << ": dereference not implemented yet\n";
      exit(-1);
  }
}

// Globally-defined entities, such as functions, structs, choices.
static Env globals;

void InitEnv(const Declaration& d, Env* env) {
  switch (d.tag()) {
    case DeclarationKind::FunctionDeclaration: {
      const FunctionDefinition& func_def =
          d.GetFunctionDeclaration().definition;
      auto pt = InterpExp(*env, func_def.param_pattern);
      auto f = Value::MakeFunctionValue(func_def.name, pt, func_def.body);
      Address a = state->heap.AllocateValue(f);
      env->Set(func_def.name, a);
      break;
    }

    case DeclarationKind::StructDeclaration: {
      const StructDefinition& struct_def = d.GetStructDeclaration().definition;
      VarValues fields;
      VarValues methods;
      for (const Member* m : struct_def.members) {
        switch (m->tag()) {
          case MemberKind::FieldMember: {
            const auto& field = m->GetFieldMember();
            auto t = InterpExp(Env(), field.type);
            fields.push_back(make_pair(field.name, t));
            break;
          }
        }
      }
      auto st = Value::MakeStructType(struct_def.name, std::move(fields),
                                      std::move(methods));
      auto a = state->heap.AllocateValue(st);
      env->Set(struct_def.name, a);
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = d.GetChoiceDeclaration();
      VarValues alts;
      for (const auto& [name, signature] : choice.alternatives) {
        auto t = InterpExp(Env(), signature);
        alts.push_back(make_pair(name, t));
      }
      auto ct = Value::MakeChoiceType(choice.name, std::move(alts));
      auto a = state->heap.AllocateValue(ct);
      env->Set(choice.name, a);
      break;
    }

    case DeclarationKind::VariableDeclaration: {
      const auto& var = d.GetVariableDeclaration();
      // Adds an entry in `globals` mapping the variable's name to the
      // result of evaluating the initializer.
      auto v = InterpExp(*env, var.initializer);
      Address a = state->heap.AllocateValue(v);
      env->Set(var.name, a);
      break;
    }
  }
}

static void InitGlobals(std::list<Declaration>* fs) {
  for (auto const& d : *fs) {
    InitEnv(d, &globals);
  }
}

//    { S, H} -> { { C, E, F} :: S, H}
// where C is the body of the function,
//       E is the environment (functions + parameters + locals)
//       F is the function
void CallFunction(int line_num, std::vector<const Value*> operas,
                  State* state) {
  switch (operas[0]->tag()) {
    case ValKind::FunctionValue: {
      // Bind arguments to parameters
      std::list<std::string> params;
      std::optional<Env> matches =
          PatternMatch(operas[0]->GetFunctionValue().param, operas[1], globals,
                       &params, line_num);
      if (!matches) {
        std::cerr << "internal error in call_function, pattern match failed"
                  << std::endl;
        exit(-1);
      }
      // Create the new frame and push it on the stack
      auto* scope = new Scope(*matches, params);
      auto* frame = new Frame(operas[0]->GetFunctionValue().name, Stack(scope),
                              Stack(Action::MakeStatementAction(
                                  operas[0]->GetFunctionValue().body)));
      state->stack.Push(frame);
      break;
    }
    case ValKind::StructType: {
      const Value* arg = CopyVal(operas[1], line_num);
      const Value* sv = Value::MakeStructValue(operas[0], arg);
      Frame* frame = state->stack.Top();
      frame->todo.Push(Action::MakeValAction(sv));
      break;
    }
    case ValKind::AlternativeConstructorValue: {
      const Value* arg = CopyVal(operas[1], line_num);
      const Value* av = Value::MakeAlternativeValue(
          operas[0]->GetAlternativeConstructorValue().alt_name,
          operas[0]->GetAlternativeConstructorValue().choice_name, arg);
      Frame* frame = state->stack.Top();
      frame->todo.Push(Action::MakeValAction(av));
      break;
    }
    default:
      std::cerr << line_num << ": in call, expected a function, not ";
      PrintValue(operas[0], std::cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

void DeallocateScope(int line_num, Scope* scope) {
  for (const auto& l : scope->locals) {
    std::optional<Address> a = scope->values.Get(l);
    if (!a) {
      std::cerr << "internal error in DeallocateScope" << std::endl;
      exit(-1);
    }
    state->heap.Deallocate(*a);
  }
}

void DeallocateLocals(int line_num, Frame* frame) {
  for (auto scope : frame->scopes) {
    DeallocateScope(line_num, scope);
  }
}

void CreateTuple(Frame* frame, Action* act, const Expression* exp) {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  std::vector<TupleElement> elements;
  auto f = exp->GetTupleLiteral().fields.begin();

  for (auto i = act->results.begin(); i != act->results.end(); ++i, ++f) {
    elements.push_back({.name = f->name, .value = *i});
  }
  const Value* tv = Value::MakeTupleValue(std::move(elements));
  frame->todo.Pop(1);
  frame->todo.Push(Action::MakeValAction(tv));
}

// Returns an updated environment that includes the bindings of
//    pattern variables to their matched values, if matching succeeds.
//
// The names of the pattern variables are added to the vars parameter.
// Returns nullopt if the value doesn't match the pattern.
auto PatternMatch(const Value* p, const Value* v, Env values,
                  std::list<std::string>* vars, int line_num)
    -> std::optional<Env> {
  switch (p->tag()) {
    case ValKind::BindingPlaceholderValue: {
      const BindingPlaceholderValue& placeholder =
          p->GetBindingPlaceholderValue();
      if (placeholder.name.has_value()) {
        Address a = state->heap.AllocateValue(CopyVal(v, line_num));
        vars->push_back(*placeholder.name);
        values.Set(*placeholder.name, a);
      }
      return values;
    }
    case ValKind::TupleValue:
      switch (v->tag()) {
        case ValKind::TupleValue: {
          if (p->GetTupleValue().elements.size() !=
              v->GetTupleValue().elements.size()) {
            std::cerr << "runtime error: arity mismatch in tuple pattern match"
                      << std::endl;
            exit(-1);
          }
          for (const TupleElement& pattern_element :
               p->GetTupleValue().elements) {
            const Value* value_field =
                v->GetTupleValue().FindField(pattern_element.name);
            if (value_field == nullptr) {
              std::cerr << "runtime error: field " << pattern_element.name
                        << "not in ";
              PrintValue(v, std::cerr);
              std::cerr << std::endl;
              exit(-1);
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
          std::cerr
              << "internal error, expected a tuple value in pattern, not ";
          PrintValue(v, std::cerr);
          std::cerr << std::endl;
          exit(-1);
      }
    case ValKind::AlternativeValue:
      switch (v->tag()) {
        case ValKind::AlternativeValue: {
          if (p->GetAlternativeValue().choice_name !=
                  v->GetAlternativeValue().choice_name ||
              p->GetAlternativeValue().alt_name !=
                  v->GetAlternativeValue().alt_name) {
            return std::nullopt;
          }
          std::optional<Env> matches = PatternMatch(
              p->GetAlternativeValue().argument,
              v->GetAlternativeValue().argument, values, vars, line_num);
          if (!matches) {
            return std::nullopt;
          }
          return *matches;
        }
        default:
          std::cerr
              << "internal error, expected a choice alternative in pattern, "
                 "not ";
          PrintValue(v, std::cerr);
          std::cerr << std::endl;
          exit(-1);
      }
    case ValKind::FunctionType:
      switch (v->tag()) {
        case ValKind::FunctionType: {
          std::optional<Env> matches =
              PatternMatch(p->GetFunctionType().param,
                           v->GetFunctionType().param, values, vars, line_num);
          if (!matches) {
            return std::nullopt;
          }
          return PatternMatch(p->GetFunctionType().ret,
                              v->GetFunctionType().ret, *matches, vars,
                              line_num);
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
  switch (pat->tag()) {
    case ValKind::PointerValue:
      state->heap.Write(pat->GetPointerValue(), CopyVal(val, line_num),
                        line_num);
      break;
    case ValKind::TupleValue: {
      switch (val->tag()) {
        case ValKind::TupleValue: {
          if (pat->GetTupleValue().elements.size() !=
              val->GetTupleValue().elements.size()) {
            std::cerr << "runtime error: arity mismatch in tuple pattern match"
                      << std::endl;
            exit(-1);
          }
          for (const TupleElement& pattern_element :
               pat->GetTupleValue().elements) {
            const Value* value_field =
                val->GetTupleValue().FindField(pattern_element.name);
            if (value_field == nullptr) {
              std::cerr << "runtime error: field " << pattern_element.name
                        << "not in ";
              PrintValue(val, std::cerr);
              std::cerr << std::endl;
              exit(-1);
            }
            PatternAssignment(pattern_element.value, value_field, line_num);
          }
          break;
        }
        default:
          std::cerr
              << "internal error, expected a tuple value on right-hand-side, "
                 "not ";
          PrintValue(val, std::cerr);
          std::cerr << std::endl;
          exit(-1);
      }
      break;
    }
    case ValKind::AlternativeValue: {
      switch (val->tag()) {
        case ValKind::AlternativeValue: {
          if (pat->GetAlternativeValue().choice_name !=
                  val->GetAlternativeValue().choice_name ||
              pat->GetAlternativeValue().alt_name !=
                  val->GetAlternativeValue().alt_name) {
            std::cerr << "internal error in pattern assignment" << std::endl;
            exit(-1);
          }
          PatternAssignment(pat->GetAlternativeValue().argument,
                            val->GetAlternativeValue().argument, line_num);
          break;
        }
        default:
          std::cerr
              << "internal error, expected an alternative in left-hand-side, "
                 "not ";
          PrintValue(val, std::cerr);
          std::cerr << std::endl;
          exit(-1);
      }
      break;
    }
    default:
      if (!ValueEqual(pat, val, line_num)) {
        std::cerr << "internal error in pattern assignment" << std::endl;
        exit(-1);
      }
  }
}

// State transitions for lvalues.

void StepLvalue() {
  Frame* frame = state->stack.Top();
  Action* act = frame->todo.Top();
  const Expression* exp = act->GetLValAction().exp;
  if (tracing_output) {
    std::cout << "--- step lvalue ";
    PrintExp(exp);
    std::cout << " --->" << std::endl;
  }
  switch (exp->tag()) {
    case ExpressionKind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      std::optional<Address> pointer =
          CurrentEnv(state).Get(exp->GetIdentifierExpression().name);
      if (!pointer) {
        std::cerr << exp->line_num << ": could not find `"
                  << exp->GetIdentifierExpression().name << "`" << std::endl;
        exit(-1);
      }
      const Value* v = Value::MakePointerValue(*pointer);
      frame->todo.Pop();
      frame->todo.Push(Action::MakeValAction(v));
      break;
    }
    case ExpressionKind::FieldAccessExpression: {
      if (act->pos == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        frame->todo.Push(
            Action::MakeLValAction(exp->GetFieldAccessExpression().aggregate));
        act->pos++;
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = act->results[0]->GetPointerValue();
        Address field =
            aggregate.SubobjectAddress(exp->GetFieldAccessExpression().field);
        frame->todo.Pop(1);
        frame->todo.Push(Action::MakeValAction(Value::MakePointerValue(field)));
      }
      break;
    }
    case ExpressionKind::IndexExpression: {
      if (act->pos == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        frame->todo.Push(
            Action::MakeLValAction(exp->GetIndexExpression().aggregate));
        act->pos++;
      } else if (act->pos == 1) {
        frame->todo.Push(
            Action::MakeExpressionAction(exp->GetIndexExpression().offset));
        act->pos++;
      } else if (act->pos == 2) {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = act->results[0]->GetPointerValue();
        std::string f = std::to_string(act->results[1]->GetIntValue());
        Address field = aggregate.SubobjectAddress(f);
        frame->todo.Pop(1);
        frame->todo.Push(Action::MakeValAction(Value::MakePointerValue(field)));
      }
      break;
    }
    case ExpressionKind::TupleLiteral: {
      if (act->pos == 0) {
        //    { {(f1=e1,...) :: C, E, F} :: S, H}
        // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
        const Expression* e1 = exp->GetTupleLiteral().fields[0].expression;
        frame->todo.Push(Action::MakeLValAction(e1));
        act->pos++;
      } else if (act->pos !=
                 static_cast<int>(exp->GetTupleLiteral().fields.size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Expression* elt =
            exp->GetTupleLiteral().fields[act->pos].expression;
        frame->todo.Push(Action::MakeLValAction(elt));
        act->pos++;
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
    case ExpressionKind::AutoTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
    case ExpressionKind::BindingExpression: {
      std::cerr << "Can't treat expression as lvalue: ";
      PrintExp(exp);
      std::cerr << std::endl;
      exit(-1);
    }
  }
}

// State transitions for expressions.

void StepExp() {
  Frame* frame = state->stack.Top();
  Action* act = frame->todo.Top();
  const Expression* exp = act->GetExpressionAction().exp;
  if (tracing_output) {
    std::cout << "--- step exp ";
    PrintExp(exp);
    std::cout << " --->" << std::endl;
  }
  switch (exp->tag()) {
    case ExpressionKind::BindingExpression: {
      if (act->pos == 0) {
        frame->todo.Push(
            Action::MakeExpressionAction(exp->GetBindingExpression().type));
        act->pos++;
      } else {
        auto v = Value::MakeBindingPlaceholderValue(
            exp->GetBindingExpression().name, act->results[0]);
        frame->todo.Pop(1);
        frame->todo.Push(Action::MakeValAction(v));
      }
      break;
    }
    case ExpressionKind::IndexExpression: {
      if (act->pos == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        frame->todo.Push(
            Action::MakeExpressionAction(exp->GetIndexExpression().aggregate));
        act->pos++;
      } else if (act->pos == 1) {
        frame->todo.Push(
            Action::MakeExpressionAction(exp->GetIndexExpression().offset));
        act->pos++;
      } else if (act->pos == 2) {
        auto tuple = act->results[0];
        switch (tuple->tag()) {
          case ValKind::TupleValue: {
            //    { { v :: [][i] :: C, E, F} :: S, H}
            // -> { { v_i :: C, E, F} : S, H}
            std::string f = std::to_string(act->results[1]->GetIntValue());
            const Value* field = tuple->GetTupleValue().FindField(f);
            if (field == nullptr) {
              std::cerr << "runtime error, field " << f << " not in ";
              PrintValue(tuple, std::cerr);
              std::cerr << std::endl;
              exit(-1);
            }
            frame->todo.Pop(1);
            frame->todo.Push(Action::MakeValAction(field));
            break;
          }
          default:
            std::cerr
                << "runtime type error, expected a tuple in field access, "
                   "not ";
            PrintValue(tuple, std::cerr);
            exit(-1);
        }
      }
      break;
    }
    case ExpressionKind::TupleLiteral: {
      if (act->pos == 0) {
        if (exp->GetTupleLiteral().fields.size() > 0) {
          //    { {(f1=e1,...) :: C, E, F} :: S, H}
          // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
          const Expression* e1 = exp->GetTupleLiteral().fields[0].expression;
          frame->todo.Push(Action::MakeExpressionAction(e1));
          act->pos++;
        } else {
          CreateTuple(frame, act, exp);
        }
      } else if (act->pos !=
                 static_cast<int>(exp->GetTupleLiteral().fields.size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Expression* elt =
            exp->GetTupleLiteral().fields[act->pos].expression;
        frame->todo.Push(Action::MakeExpressionAction(elt));
        act->pos++;
      } else {
        CreateTuple(frame, act, exp);
      }
      break;
    }
    case ExpressionKind::FieldAccessExpression: {
      if (act->pos == 0) {
        //    { { e.f :: C, E, F} :: S, H}
        // -> { { e :: [].f :: C, E, F} :: S, H}
        frame->todo.Push(Action::MakeExpressionAction(
            exp->GetFieldAccessExpression().aggregate));
        act->pos++;
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        const Value* element = act->results[0]->GetField(
            FieldPath(exp->GetFieldAccessExpression().field), exp->line_num);
        frame->todo.Pop(1);
        frame->todo.Push(Action::MakeValAction(element));
      }
      break;
    }
    case ExpressionKind::IdentifierExpression: {
      CHECK(act->pos == 0);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      std::optional<Address> pointer =
          CurrentEnv(state).Get(exp->GetIdentifierExpression().name);
      if (!pointer) {
        std::cerr << exp->line_num << ": could not find `"
                  << exp->GetIdentifierExpression().name << "`" << std::endl;
        exit(-1);
      }
      const Value* pointee = state->heap.Read(*pointer, exp->line_num);
      frame->todo.Pop(1);
      frame->todo.Push(Action::MakeValAction(pointee));
      break;
    }
    case ExpressionKind::IntLiteral:
      CHECK(act->pos == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(
          Action::MakeValAction(Value::MakeIntValue(exp->GetIntLiteral())));
      break;
    case ExpressionKind::BoolLiteral:
      CHECK(act->pos == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(
          Action::MakeValAction(Value::MakeBoolValue(exp->GetBoolLiteral())));
      break;
    case ExpressionKind::PrimitiveOperatorExpression:
      if (act->pos !=
          static_cast<int>(
              exp->GetPrimitiveOperatorExpression().arguments.size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        const Expression* arg =
            exp->GetPrimitiveOperatorExpression().arguments[act->pos];
        frame->todo.Push(Action::MakeExpressionAction(arg));
        act->pos++;
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        const Value* v = EvalPrim(exp->GetPrimitiveOperatorExpression().op,
                                  act->results, exp->line_num);
        frame->todo.Pop(1);
        frame->todo.Push(Action::MakeValAction(v));
      }
      break;
    case ExpressionKind::CallExpression:
      if (act->pos == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        frame->todo.Push(
            Action::MakeExpressionAction(exp->GetCallExpression().function));
        act->pos++;
      } else if (act->pos == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        frame->todo.Push(
            Action::MakeExpressionAction(exp->GetCallExpression().argument));
        act->pos++;
      } else if (act->pos == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        frame->todo.Pop(1);
        CallFunction(exp->line_num, act->results, state);
      } else {
        std::cerr << "internal error in handle_value with Call" << std::endl;
        exit(-1);
      }
      break;
    case ExpressionKind::IntTypeLiteral: {
      CHECK(act->pos == 0);
      const Value* v = Value::MakeIntType();
      frame->todo.Pop(1);
      frame->todo.Push(Action::MakeValAction(v));
      break;
    }
    case ExpressionKind::BoolTypeLiteral: {
      CHECK(act->pos == 0);
      const Value* v = Value::MakeBoolType();
      frame->todo.Pop(1);
      frame->todo.Push(Action::MakeValAction(v));
      break;
    }
    case ExpressionKind::AutoTypeLiteral: {
      CHECK(act->pos == 0);
      const Value* v = Value::MakeAutoType();
      frame->todo.Pop(1);
      frame->todo.Push(Action::MakeValAction(v));
      break;
    }
    case ExpressionKind::TypeTypeLiteral: {
      CHECK(act->pos == 0);
      const Value* v = Value::MakeTypeType();
      frame->todo.Pop(1);
      frame->todo.Push(Action::MakeValAction(v));
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      if (act->pos == 0) {
        frame->todo.Push(Action::MakeExpressionAction(
            exp->GetFunctionTypeLiteral().parameter));
        act->pos++;
      } else if (act->pos == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        frame->todo.Push(Action::MakeExpressionAction(
            exp->GetFunctionTypeLiteral().return_type));
        act->pos++;
      } else if (act->pos == 2) {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        const Value* v =
            Value::MakeFunctionType(act->results[0], act->results[1]);
        frame->todo.Pop(1);
        frame->todo.Push(Action::MakeValAction(v));
      }
      break;
    }
    case ExpressionKind::ContinuationTypeLiteral: {
      CHECK(act->pos == 0);
      const Value* v = Value::MakeContinuationType();
      frame->todo.Pop(1);
      frame->todo.Push(Action::MakeValAction(v));
      break;
    }
  }  // switch (exp->tag)
}

auto IsWhileAct(Action* act) -> bool {
  switch (act->tag()) {
    case ActionKind::StatementAction:
      switch (act->GetStatementAction().stmt->tag()) {
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
  switch (act->tag()) {
    case ActionKind::StatementAction:
      switch (act->GetStatementAction().stmt->tag()) {
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
  const Statement* stmt = act->GetStatementAction().stmt;
  CHECK(stmt != nullptr && "null statement!");
  if (tracing_output) {
    std::cout << "--- step stmt ";
    PrintStatement(stmt, 1);
    std::cout << " --->" << std::endl;
  }
  switch (stmt->tag()) {
    case StatementKind::Match:
      if (act->pos == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        frame->todo.Push(Action::MakeExpressionAction(stmt->GetMatch().exp));
        act->pos++;
      } else {
        // Regarding act->pos:
        // * odd: start interpreting the pattern of a clause
        // * even: finished interpreting the pattern, now try to match
        //
        // Regarding act->results:
        // * 0: the value that we're matching
        // * 1: the pattern for clause 0
        // * 2: the pattern for clause 1
        // * ...
        auto clause_num = (act->pos - 1) / 2;
        if (clause_num >= static_cast<int>(stmt->GetMatch().clauses->size())) {
          frame->todo.Pop(1);
          break;
        }
        auto c = stmt->GetMatch().clauses->begin();
        std::advance(c, clause_num);

        if (act->pos % 2 == 1) {
          // start interpreting the pattern of the clause
          //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
          // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
          frame->todo.Push(Action::MakeExpressionAction(c->first));
          act->pos++;
        } else {  // try to match
          auto v = act->results[0];
          auto pat = act->results[clause_num + 1];
          auto values = CurrentEnv(state);
          std::list<std::string> vars;
          std::optional<Env> matches =
              PatternMatch(pat, v, values, &vars, stmt->line_num);
          if (matches) {  // we have a match, start the body
            auto* new_scope = new Scope(*matches, vars);
            frame->scopes.Push(new_scope);
            const Statement* body_block =
                Statement::MakeBlock(stmt->line_num, c->second);
            Action* body_act = Action::MakeStatementAction(body_block);
            body_act->pos = 1;
            frame->todo.Pop(1);
            frame->todo.Push(body_act);
            frame->todo.Push(Action::MakeStatementAction(c->second));
          } else {
            // this case did not match, moving on
            act->pos++;
            clause_num = (act->pos - 1) / 2;
            if (clause_num ==
                static_cast<int>(stmt->GetMatch().clauses->size())) {
              frame->todo.Pop(2);
            }
          }
        }
      }
      break;
    case StatementKind::While:
      if (act->pos == 0) {
        //    { { (while (e) s) :: C, E, F} :: S, H}
        // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
        frame->todo.Push(Action::MakeExpressionAction(stmt->GetWhile().cond));
        act->pos++;
      } else if (act->results[0]->GetBoolValue()) {
        //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
        frame->todo.Top()->pos = 0;
        frame->todo.Top()->results.clear();
        frame->todo.Push(Action::MakeStatementAction(stmt->GetWhile().body));
      } else {
        //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { C, E, F } :: S, H}
        frame->todo.Top()->pos = 0;
        frame->todo.Top()->results.clear();
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::Break:
      CHECK(act->pos == 0);
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
      CHECK(act->pos == 0);
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
      if (act->pos == 0) {
        if (stmt->GetBlock().stmt) {
          auto* scope = new Scope(CurrentEnv(state), {});
          frame->scopes.Push(scope);
          frame->todo.Push(Action::MakeStatementAction(stmt->GetBlock().stmt));
          act->pos++;
          act->pos++;
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
      if (act->pos == 0) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        frame->todo.Push(
            Action::MakeExpressionAction(stmt->GetVariableDefinition().init));
        act->pos++;
      } else if (act->pos == 1) {
        frame->todo.Push(
            Action::MakeExpressionAction(stmt->GetVariableDefinition().pat));
        act->pos++;
      } else if (act->pos == 2) {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        const Value* v = act->results[0];
        const Value* p = act->results[1];

        std::optional<Env> matches =
            PatternMatch(p, v, frame->scopes.Top()->values,
                         &frame->scopes.Top()->locals, stmt->line_num);
        if (!matches) {
          std::cerr << stmt->line_num
                    << ": internal error in variable definition, match failed"
                    << std::endl;
          exit(-1);
        }
        frame->scopes.Top()->values = *matches;
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::ExpressionStatement:
      if (act->pos == 0) {
        //    { {e :: C, E, F} :: S, H}
        // -> { {e :: C, E, F} :: S, H}
        frame->todo.Push(
            Action::MakeExpressionAction(stmt->GetExpressionStatement().exp));
        act->pos++;
      } else {
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::Assign:
      if (act->pos == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        frame->todo.Push(Action::MakeLValAction(stmt->GetAssign().lhs));
        act->pos++;
      } else if (act->pos == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        frame->todo.Push(Action::MakeExpressionAction(stmt->GetAssign().rhs));
        act->pos++;
      } else if (act->pos == 2) {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        auto pat = act->results[0];
        auto val = act->results[1];
        PatternAssignment(pat, val, stmt->line_num);
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::If:
      if (act->pos == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        frame->todo.Push(Action::MakeExpressionAction(stmt->GetIf().cond));
        act->pos++;
      } else if (act->results[0]->GetBoolValue()) {
        //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { then_stmt :: C, E, F } :: S, H}
        frame->todo.Pop(1);
        frame->todo.Push(Action::MakeStatementAction(stmt->GetIf().then_stmt));
      } else if (stmt->GetIf().else_stmt) {
        //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { else_stmt :: C, E, F } :: S, H}
        frame->todo.Pop(1);
        frame->todo.Push(Action::MakeStatementAction(stmt->GetIf().else_stmt));
      } else {
        frame->todo.Pop(1);
      }
      break;
    case StatementKind::Return:
      if (act->pos == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        frame->todo.Push(Action::MakeExpressionAction(stmt->GetReturn().exp));
        act->pos++;
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        const Value* ret_val = CopyVal(act->results[0], stmt->line_num);
        DeallocateLocals(stmt->line_num, frame);
        state->stack.Pop(1);
        frame = state->stack.Top();
        frame->todo.Push(Action::MakeValAction(ret_val));
      }
      break;
    case StatementKind::Sequence:
      CHECK(act->pos == 0);
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      if (stmt->GetSequence().next) {
        frame->todo.Push(Action::MakeStatementAction(stmt->GetSequence().next));
      }
      frame->todo.Push(Action::MakeStatementAction(stmt->GetSequence().stmt));
      break;
    case StatementKind::Continuation: {
      CHECK(act->pos == 0);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      Scope* scope = new Scope(CurrentEnv(state), std::list<std::string>());
      Stack<Scope*> scopes;
      scopes.Push(scope);
      Stack<Action*> todo;
      todo.Push(Action::MakeStatementAction(Statement::MakeReturn(
          stmt->line_num, Expression::MakeTupleLiteral(stmt->line_num, {}))));
      todo.Push(Action::MakeStatementAction(stmt->GetContinuation().body));
      Frame* continuation_frame = new Frame("__continuation", scopes, todo);
      Address continuation_address = state->heap.AllocateValue(
          Value::MakeContinuationValue({continuation_frame}));
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
      if (act->pos == 0) {
        // Evaluate the argument of the run statement.
        frame->todo.Push(Action::MakeExpressionAction(stmt->GetRun().argument));
        act->pos++;
      } else {
        frame->todo.Pop(1);
        // Push an expression statement action to ignore the result
        // value from the continuation.
        Action* ignore_result =
            Action::MakeStatementAction(Statement::MakeExpressionStatement(
                stmt->line_num,
                Expression::MakeTupleLiteral(stmt->line_num, {})));
        ignore_result->pos = 0;
        frame->todo.Push(ignore_result);
        // Push the continuation onto the current stack.
        const std::vector<Frame*>& continuation_vector =
            act->results[0]->GetContinuationValue().stack;
        for (auto frame_iter = continuation_vector.rbegin();
             frame_iter != continuation_vector.rend(); ++frame_iter) {
          state->stack.Push(*frame_iter);
        }
      }
      break;
    case StatementKind::Await:
      CHECK(act->pos == 0);
      // Pause the current continuation
      frame->todo.Pop();
      std::vector<Frame*> paused;
      do {
        paused.push_back(state->stack.Pop());
      } while (paused.back()->continuation == std::nullopt);
      // Update the continuation with the paused stack.
      state->heap.Write(*paused.back()->continuation,
                        Value::MakeContinuationValue(paused), stmt->line_num);
      break;
  }
}

// State transition.
void Step() {
  Frame* frame = state->stack.Top();
  if (frame->todo.IsEmpty()) {
    std::cerr << "runtime error: fell off end of function " << frame->name
              << " without `return`" << std::endl;
    exit(-1);
  }

  Action* act = frame->todo.Top();
  switch (act->tag()) {
    case ActionKind::ValAction: {
      Action* val_act = frame->todo.Pop();
      Action* act = frame->todo.Top();
      act->results.push_back(val_act->GetValAction().val);
      break;
    }
    case ActionKind::LValAction:
      StepLvalue();
      break;
    case ActionKind::ExpressionAction:
      StepExp();
      break;
    case ActionKind::StatementAction:
      StepStmt();
      break;
  }  // switch
}

// Interpret the whole porogram.
auto InterpProgram(std::list<Declaration>* fs) -> int {
  state = new State();  // Runtime state.
  if (tracing_output) {
    std::cout << "********** initializing globals **********" << std::endl;
  }
  InitGlobals(fs);

  const Expression* arg = Expression::MakeTupleLiteral(0, {});
  const Expression* call_main = Expression::MakeCallExpression(
      0, Expression::MakeIdentifierExpression(0, "main"), arg);
  auto todo = Stack(Action::MakeExpressionAction(call_main));
  auto* scope = new Scope(globals, std::list<std::string>());
  auto* frame = new Frame("top", Stack(scope), todo);
  state->stack = Stack(frame);

  if (tracing_output) {
    std::cout << "********** calling main function **********" << std::endl;
    PrintState(std::cout);
  }

  while (state->stack.CountExceeds(1) ||
         state->stack.Top()->todo.CountExceeds(1) ||
         state->stack.Top()->todo.Top()->tag() != ActionKind::ValAction) {
    Step();
    if (tracing_output) {
      PrintState(std::cout);
    }
  }
  const Value* v = state->stack.Top()->todo.Top()->GetValAction().val;
  return v->GetIntValue();
}

// Interpret an expression at compile-time.
auto InterpExp(Env values, const Expression* e) -> const Value* {
  auto todo = Stack(Action::MakeExpressionAction(e));
  auto* scope = new Scope(values, std::list<std::string>());
  auto* frame = new Frame("InterpExp", Stack(scope), todo);
  state->stack = Stack(frame);

  while (state->stack.CountExceeds(1) ||
         state->stack.Top()->todo.CountExceeds(1) ||
         state->stack.Top()->todo.Top()->tag() != ActionKind::ValAction) {
    Step();
  }
  const Value* v = state->stack.Top()->todo.Top()->GetValAction().val;
  return v;
}

}  // namespace Carbon
