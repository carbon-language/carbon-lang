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

// Returns the given name from the environment, printing an error if not found.
static auto GetFromEnv(int line_num, const std::string& name) -> Address {
  std::optional<Address> pointer = CurrentEnv(state).Get(name);
  if (!pointer) {
    FATAL_RUNTIME_ERROR(line_num) << "could not find `" << name << "`";
  }
  return *pointer;
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
      return global_arena->RawNew<IntValue>(-cast<IntValue>(*args[0]).Val());
    case Operator::Add:
      return global_arena->RawNew<IntValue>(cast<IntValue>(*args[0]).Val() +
                                            cast<IntValue>(*args[1]).Val());
    case Operator::Sub:
      return global_arena->RawNew<IntValue>(cast<IntValue>(*args[0]).Val() -
                                            cast<IntValue>(*args[1]).Val());
    case Operator::Mul:
      return global_arena->RawNew<IntValue>(cast<IntValue>(*args[0]).Val() *
                                            cast<IntValue>(*args[1]).Val());
    case Operator::Not:
      return global_arena->RawNew<BoolValue>(!cast<BoolValue>(*args[0]).Val());
    case Operator::And:
      return global_arena->RawNew<BoolValue>(cast<BoolValue>(*args[0]).Val() &&
                                             cast<BoolValue>(*args[1]).Val());
    case Operator::Or:
      return global_arena->RawNew<BoolValue>(cast<BoolValue>(*args[0]).Val() ||
                                             cast<BoolValue>(*args[1]).Val());
    case Operator::Eq:
      return global_arena->RawNew<BoolValue>(
          ValueEqual(args[0], args[1], line_num));
    case Operator::Ptr:
      return global_arena->RawNew<PointerType>(args[0]);
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
            global_arena->RawNew<VariableType>(deduced.name));
        new_env.Set(deduced.name, a);
      }
      auto pt = InterpPattern(new_env, func_def.param_pattern);
      auto f =
          global_arena->RawNew<FunctionValue>(func_def.name, pt, func_def.body);
      Address a = state->heap.AllocateValue(f);
      env->Set(func_def.name, a);
      break;
    }

    case Declaration::Kind::ClassDeclaration: {
      const ClassDefinition& class_def = cast<ClassDeclaration>(d).Definition();
      VarValues fields;
      VarValues methods;
      for (const Member* m : class_def.members) {
        switch (m->Tag()) {
          case Member::Kind::FieldMember: {
            const BindingPattern* binding = cast<FieldMember>(*m).Binding();
            const Expression* type_expression =
                cast<ExpressionPattern>(binding->Type())->Expression();
            auto type = InterpExp(Env(), type_expression);
            fields.push_back(make_pair(*binding->Name(), type));
            break;
          }
        }
      }
      auto st = global_arena->RawNew<ClassType>(
          class_def.name, std::move(fields), std::move(methods));
      auto a = state->heap.AllocateValue(st);
      env->Set(class_def.name, a);
      break;
    }

    case Declaration::Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(d);
      VarValues alts;
      for (const auto& [name, signature] : choice.Alternatives()) {
        auto t = InterpExp(Env(), signature);
        alts.push_back(make_pair(name, t));
      }
      auto ct =
          global_arena->RawNew<ChoiceType>(choice.Name(), std::move(alts));
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
      std::optional<Env> matches =
          PatternMatch(fn.Param(), operas[1], line_num);
      CHECK(matches) << "internal error in call_function, pattern match failed";
      // Create the new frame and push it on the stack
      Env values = globals;
      std::list<std::string> params;
      for (const auto& [name, value] : *matches) {
        values.Set(name, value);
        params.push_back(name);
      }
      auto* scope = global_arena->RawNew<Scope>(values, params);
      auto* frame = global_arena->RawNew<Frame>(
          fn.Name(), Stack(scope),
          Stack<Action*>(global_arena->RawNew<StatementAction>(fn.Body())));
      state->stack.Push(frame);
      break;
    }
    case Value::Kind::ClassType: {
      const Value* arg = CopyVal(operas[1], line_num);
      const Value* sv = global_arena->RawNew<StructValue>(operas[0], arg);
      Frame* frame = state->stack.Top();
      frame->todo.Push(global_arena->RawNew<ValAction>(sv));
      break;
    }
    case Value::Kind::AlternativeConstructorValue: {
      const auto& alt = cast<AlternativeConstructorValue>(*operas[0]);
      const Value* arg = CopyVal(operas[1], line_num);
      const Value* av = global_arena->RawNew<AlternativeValue>(
          alt.AltName(), alt.ChoiceName(), arg);
      Frame* frame = state->stack.Top();
      frame->todo.Push(global_arena->RawNew<ValAction>(av));
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
  const auto& tup_lit = cast<TupleLiteral>(*exp);
  CHECK(act->Results().size() == tup_lit.Fields().size());
  std::vector<TupleElement> elements;
  for (size_t i = 0; i < act->Results().size(); ++i) {
    elements.push_back(
        {.name = tup_lit.Fields()[i].name, .value = act->Results()[i]});
  }

  const Value* tv = global_arena->RawNew<TupleValue>(std::move(elements));
  frame->todo.Pop(1);
  frame->todo.Push(global_arena->RawNew<ValAction>(tv));
}

auto PatternMatch(const Value* p, const Value* v, int line_num)
    -> std::optional<Env> {
  switch (p->Tag()) {
    case Value::Kind::BindingPlaceholderValue: {
      const auto& placeholder = cast<BindingPlaceholderValue>(*p);
      Env values;
      if (placeholder.Name().has_value()) {
        Address a = state->heap.AllocateValue(CopyVal(v, line_num));
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
            FATAL_PROGRAM_ERROR(line_num)
                << "arity mismatch in tuple pattern match:\n  pattern: "
                << p_tup << "\n  value: " << v_tup;
          }
          Env values;
          for (size_t i = 0; i < p_tup.Elements().size(); ++i) {
            if (p_tup.Elements()[i].name != v_tup.Elements()[i].name) {
              FATAL_PROGRAM_ERROR(line_num)
                  << "Tuple field name '" << v_tup.Elements()[i].name
                  << "' does not match pattern field name '"
                  << p_tup.Elements()[i].name << "'";
            }
            std::optional<Env> matches = PatternMatch(
                p_tup.Elements()[i].value, v_tup.Elements()[i].value, line_num);
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
    case Value::Kind::AlternativeValue:
      switch (v->Tag()) {
        case Value::Kind::AlternativeValue: {
          const auto& p_alt = cast<AlternativeValue>(*p);
          const auto& v_alt = cast<AlternativeValue>(*v);
          if (p_alt.ChoiceName() != v_alt.ChoiceName() ||
              p_alt.AltName() != v_alt.AltName()) {
            return std::nullopt;
          }
          return PatternMatch(p_alt.Argument(), v_alt.Argument(), line_num);
        }
        default:
          FATAL() << "expected a choice alternative in pattern, not " << *v;
      }
    case Value::Kind::FunctionType:
      switch (v->Tag()) {
        case Value::Kind::FunctionType: {
          const auto& p_fn = cast<FunctionType>(*p);
          const auto& v_fn = cast<FunctionType>(*v);
          std::optional<Env> param_matches =
              PatternMatch(p_fn.Param(), v_fn.Param(), line_num);
          if (!param_matches) {
            return std::nullopt;
          }
          std::optional<Env> ret_matches =
              PatternMatch(p_fn.Ret(), v_fn.Ret(), line_num);
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
      return Env();
    default:
      if (ValueEqual(p, v, line_num)) {
        return Env();
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
  switch (exp->Tag()) {
    case Expression::Kind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp->LineNumber(),
                                   cast<IdentifierExpression>(*exp).Name());
      const Value* v = global_arena->RawNew<PointerValue>(pointer);
      frame->todo.Pop();
      frame->todo.Push(global_arena->RawNew<ValAction>(v));
      break;
    }
    case Expression::Kind::FieldAccessExpression: {
      if (act->Pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->RawNew<LValAction>(
            cast<FieldAccessExpression>(*exp).Aggregate()));
        act->IncrementPos();
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->Results()[0]).Val();
        Address field = aggregate.SubobjectAddress(
            cast<FieldAccessExpression>(*exp).Field());
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->RawNew<ValAction>(
            global_arena->RawNew<PointerValue>(field)));
      }
      break;
    }
    case Expression::Kind::IndexExpression: {
      if (act->Pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->RawNew<LValAction>(
            cast<IndexExpression>(*exp).Aggregate()));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<IndexExpression>(*exp).Offset()));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->Results()[0]).Val();
        std::string f =
            std::to_string(cast<IntValue>(*act->Results()[1]).Val());
        Address field = aggregate.SubobjectAddress(f);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->RawNew<ValAction>(
            global_arena->RawNew<PointerValue>(field)));
      }
      break;
    }
    case Expression::Kind::TupleLiteral: {
      if (act->Pos() == 0) {
        //    { {(f1=e1,...) :: C, E, F} :: S, H}
        // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
        const Expression* e1 = cast<TupleLiteral>(*exp).Fields()[0].expression;
        frame->todo.Push(global_arena->RawNew<LValAction>(e1));
        act->IncrementPos();
      } else if (act->Pos() !=
                 static_cast<int>(cast<TupleLiteral>(*exp).Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Expression* elt =
            cast<TupleLiteral>(*exp).Fields()[act->Pos()].expression;
        frame->todo.Push(global_arena->RawNew<LValAction>(elt));
        act->IncrementPos();
      } else {
        CreateTuple(frame, act, exp);
      }
      break;
    }
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

// State transitions for expressions.

void StepExp() {
  Frame* frame = state->stack.Top();
  Action* act = frame->todo.Top();
  const Expression* exp = cast<ExpressionAction>(*act).Exp();
  if (tracing_output) {
    llvm::outs() << "--- step exp " << *exp << " --->\n";
  }
  switch (exp->Tag()) {
    case Expression::Kind::IndexExpression: {
      if (act->Pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<IndexExpression>(*exp).Aggregate()));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<IndexExpression>(*exp).Offset()));
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
            frame->todo.Push(global_arena->RawNew<ValAction>(field));
            break;
          }
          default:
            FATAL_RUNTIME_ERROR_NO_LINE()
                << "expected a tuple in field access, not " << *tuple;
        }
      }
      break;
    }
    case Expression::Kind::TupleLiteral: {
      if (act->Pos() == 0) {
        if (cast<TupleLiteral>(*exp).Fields().size() > 0) {
          //    { {(f1=e1,...) :: C, E, F} :: S, H}
          // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
          const Expression* e1 =
              cast<TupleLiteral>(*exp).Fields()[0].expression;
          frame->todo.Push(global_arena->RawNew<ExpressionAction>(e1));
          act->IncrementPos();
        } else {
          CreateTuple(frame, act, exp);
        }
      } else if (act->Pos() !=
                 static_cast<int>(cast<TupleLiteral>(*exp).Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Expression* elt =
            cast<TupleLiteral>(*exp).Fields()[act->Pos()].expression;
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(elt));
        act->IncrementPos();
      } else {
        CreateTuple(frame, act, exp);
      }
      break;
    }
    case Expression::Kind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(*exp);
      if (act->Pos() == 0) {
        //    { { e.f :: C, E, F} :: S, H}
        // -> { { e :: [].f :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->RawNew<ExpressionAction>(access.Aggregate()));
        act->IncrementPos();
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        const Value* element = act->Results()[0]->GetField(
            FieldPath(access.Field()), exp->LineNumber());
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->RawNew<ValAction>(element));
      }
      break;
    }
    case Expression::Kind::IdentifierExpression: {
      CHECK(act->Pos() == 0);
      const auto& ident = cast<IdentifierExpression>(*exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp->LineNumber(), ident.Name());
      const Value* pointee = state->heap.Read(pointer, exp->LineNumber());
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(pointee));
      break;
    }
    case Expression::Kind::IntLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(
          global_arena->RawNew<IntValue>(cast<IntLiteral>(*exp).Val())));
      break;
    case Expression::Kind::BoolLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(
          global_arena->RawNew<BoolValue>(cast<BoolLiteral>(*exp).Val())));
      break;
    case Expression::Kind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(*exp);
      if (act->Pos() != static_cast<int>(op.Arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        const Expression* arg = op.Arguments()[act->Pos()];
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(arg));
        act->IncrementPos();
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        const Value* v = EvalPrim(op.Op(), act->Results(), exp->LineNumber());
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->RawNew<ValAction>(v));
      }
      break;
    }
    case Expression::Kind::CallExpression:
      if (act->Pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<CallExpression>(*exp).Function()));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<CallExpression>(*exp).Argument()));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        frame->todo.Pop(1);
        CallFunction(exp->LineNumber(), act->Results(), state);
      } else {
        FATAL() << "in handle_value with Call pos " << act->Pos();
      }
      break;
    case Expression::Kind::IntrinsicExpression:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      switch (cast<IntrinsicExpression>(*exp).Intrinsic()) {
        case IntrinsicExpression::IntrinsicKind::Print:
          Address pointer = GetFromEnv(exp->LineNumber(), "format_str");
          const Value* pointee = state->heap.Read(pointer, exp->LineNumber());
          CHECK(pointee->Tag() == Value::Kind::StringValue);
          // TODO: This could eventually use something like llvm::formatv.
          llvm::outs() << cast<StringValue>(*pointee).Val();
          frame->todo.Push(
              global_arena->RawNew<ValAction>(&TupleValue::Empty()));
          break;
      }
      break;

    case Expression::Kind::IntTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->RawNew<IntType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(v));
      break;
    }
    case Expression::Kind::BoolTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->RawNew<BoolType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(v));
      break;
    }
    case Expression::Kind::TypeTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->RawNew<TypeType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(v));
      break;
    }
    case Expression::Kind::FunctionTypeLiteral: {
      if (act->Pos() == 0) {
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<FunctionTypeLiteral>(*exp).Parameter()));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<FunctionTypeLiteral>(*exp).ReturnType()));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        const Value* v = global_arena->RawNew<FunctionType>(
            std::vector<GenericBinding>(), act->Results()[0],
            act->Results()[1]);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->RawNew<ValAction>(v));
      }
      break;
    }
    case Expression::Kind::ContinuationTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->RawNew<ContinuationType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(v));
      break;
    }
    case Expression::Kind::StringLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(
          global_arena->RawNew<StringValue>(cast<StringLiteral>(*exp).Val())));
      break;
    case Expression::Kind::StringTypeLiteral: {
      CHECK(act->Pos() == 0);
      const Value* v = global_arena->RawNew<StringType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(v));
      break;
    }
  }  // switch (exp->Tag)
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
      const Value* v = global_arena->RawNew<AutoType>();
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ValAction>(v));
      break;
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*pattern);
      if (act->Pos() == 0) {
        frame->todo.Push(global_arena->RawNew<PatternAction>(binding.Type()));
        act->IncrementPos();
      } else {
        auto v = global_arena->RawNew<BindingPlaceholderValue>(
            binding.Name(), act->Results()[0]);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->RawNew<ValAction>(v));
      }
      break;
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*pattern);
      if (act->Pos() == 0) {
        if (tuple.Fields().empty()) {
          frame->todo.Pop(1);
          frame->todo.Push(
              global_arena->RawNew<ValAction>(&TupleValue::Empty()));
        } else {
          const Pattern* p1 = tuple.Fields()[0].pattern;
          frame->todo.Push(global_arena->RawNew<PatternAction>(p1));
          act->IncrementPos();
        }
      } else if (act->Pos() != static_cast<int>(tuple.Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Pattern* elt = tuple.Fields()[act->Pos()].pattern;
        frame->todo.Push(global_arena->RawNew<PatternAction>(elt));
        act->IncrementPos();
      } else {
        std::vector<TupleElement> elements;
        for (size_t i = 0; i < tuple.Fields().size(); ++i) {
          elements.push_back(
              {.name = tuple.Fields()[i].name, .value = act->Results()[i]});
        }
        const Value* tuple_value =
            global_arena->RawNew<TupleValue>(std::move(elements));
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->RawNew<ValAction>(tuple_value));
      }
      break;
    }
    case Pattern::Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*pattern);
      if (act->Pos() == 0) {
        frame->todo.Push(
            global_arena->RawNew<ExpressionAction>(alternative.ChoiceType()));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        frame->todo.Push(
            global_arena->RawNew<PatternAction>(alternative.Arguments()));
        act->IncrementPos();
      } else {
        CHECK(act->Pos() == 2);
        const auto& choice_type = cast<ChoiceType>(*act->Results()[0]);
        frame->todo.Pop(1);
        frame->todo.Push(global_arena->RawNew<ValAction>(
            global_arena->RawNew<AlternativeValue>(
                alternative.AlternativeName(), choice_type.Name(),
                act->Results()[1])));
      }
      break;
    }
    case Pattern::Kind::ExpressionPattern:
      frame->todo.Pop(1);
      frame->todo.Push(global_arena->RawNew<ExpressionAction>(
          cast<ExpressionPattern>(pattern)->Expression()));
      break;
  }
}

auto IsWhileAct(Action* act) -> bool {
  switch (act->Tag()) {
    case Action::Kind::StatementAction:
      switch (cast<StatementAction>(*act).Stmt()->Tag()) {
        case Statement::Kind::While:
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
      switch (cast<StatementAction>(*act).Stmt()->Tag()) {
        case Statement::Kind::Block:
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
  switch (stmt->Tag()) {
    case Statement::Kind::Match:
      if (act->Pos() == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->RawNew<ExpressionAction>(cast<Match>(*stmt).Exp()));
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
        if (clause_num >=
            static_cast<int>(cast<Match>(*stmt).Clauses()->size())) {
          frame->todo.Pop(1);
          break;
        }
        auto c = cast<Match>(*stmt).Clauses()->begin();
        std::advance(c, clause_num);

        if (act->Pos() % 2 == 1) {
          // start interpreting the pattern of the clause
          //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
          // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
          frame->todo.Push(global_arena->RawNew<PatternAction>(c->first));
          act->IncrementPos();
        } else {  // try to match
          auto v = act->Results()[0];
          auto pat = act->Results()[clause_num + 1];
          std::optional<Env> matches = PatternMatch(pat, v, stmt->LineNumber());
          if (matches) {  // we have a match, start the body
            Env values = CurrentEnv(state);
            std::list<std::string> vars;
            for (const auto& [name, value] : *matches) {
              values.Set(name, value);
              vars.push_back(name);
            }
            auto* new_scope = global_arena->RawNew<Scope>(values, vars);
            frame->scopes.Push(new_scope);
            const Statement* body_block =
                global_arena->RawNew<Block>(stmt->LineNumber(), c->second);
            Action* body_act =
                global_arena->RawNew<StatementAction>(body_block);
            body_act->IncrementPos();
            frame->todo.Pop(1);
            frame->todo.Push(body_act);
            frame->todo.Push(global_arena->RawNew<StatementAction>(c->second));
          } else {
            // this case did not match, moving on
            act->IncrementPos();
            clause_num = (act->Pos() - 1) / 2;
            if (clause_num ==
                static_cast<int>(cast<Match>(*stmt).Clauses()->size())) {
              frame->todo.Pop(1);
            }
          }
        }
      }
      break;
    case Statement::Kind::While:
      if (act->Pos() == 0) {
        //    { { (while (e) s) :: C, E, F} :: S, H}
        // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->RawNew<ExpressionAction>(cast<While>(*stmt).Cond()));
        act->IncrementPos();
      } else if (cast<BoolValue>(*act->Results()[0]).Val()) {
        //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
        frame->todo.Top()->Clear();
        frame->todo.Push(
            global_arena->RawNew<StatementAction>(cast<While>(*stmt).Body()));
      } else {
        //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { C, E, F } :: S, H}
        frame->todo.Top()->Clear();
        frame->todo.Pop(1);
      }
      break;
    case Statement::Kind::Break:
      CHECK(act->Pos() == 0);
      //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { C, E', F} :: S, H}
      frame->todo.Pop(1);
      while (!frame->todo.IsEmpty() && !IsWhileAct(frame->todo.Top())) {
        if (IsBlockAct(frame->todo.Top())) {
          DeallocateScope(stmt->LineNumber(), frame->scopes.Top());
          frame->scopes.Pop(1);
        }
        frame->todo.Pop(1);
      }
      frame->todo.Pop(1);
      break;
    case Statement::Kind::Continue:
      CHECK(act->Pos() == 0);
      //    { { continue; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { (while (e) s) :: C, E', F} :: S, H}
      frame->todo.Pop(1);
      while (!frame->todo.IsEmpty() && !IsWhileAct(frame->todo.Top())) {
        if (IsBlockAct(frame->todo.Top())) {
          DeallocateScope(stmt->LineNumber(), frame->scopes.Top());
          frame->scopes.Pop(1);
        }
        frame->todo.Pop(1);
      }
      break;
    case Statement::Kind::Block: {
      if (act->Pos() == 0) {
        if (cast<Block>(*stmt).Stmt()) {
          auto* scope = global_arena->RawNew<Scope>(CurrentEnv(state),
                                                    std::list<std::string>());
          frame->scopes.Push(scope);
          frame->todo.Push(
              global_arena->RawNew<StatementAction>(cast<Block>(*stmt).Stmt()));
          act->IncrementPos();
          act->IncrementPos();
        } else {
          frame->todo.Pop();
        }
      } else {
        Scope* scope = frame->scopes.Top();
        DeallocateScope(stmt->LineNumber(), scope);
        frame->scopes.Pop(1);
        frame->todo.Pop(1);
      }
      break;
    }
    case Statement::Kind::VariableDefinition:
      if (act->Pos() == 0) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<VariableDefinition>(*stmt).Init()));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        frame->todo.Push(global_arena->RawNew<PatternAction>(
            cast<VariableDefinition>(*stmt).Pat()));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        const Value* v = act->Results()[0];
        const Value* p = act->Results()[1];

        std::optional<Env> matches = PatternMatch(p, v, stmt->LineNumber());
        CHECK(matches)
            << stmt->LineNumber()
            << ": internal error in variable definition, match failed";
        for (const auto& [name, value] : *matches) {
          frame->scopes.Top()->values.Set(name, value);
          frame->scopes.Top()->locals.push_back(name);
        }
        frame->todo.Pop(1);
      }
      break;
    case Statement::Kind::ExpressionStatement:
      if (act->Pos() == 0) {
        //    { {e :: C, E, F} :: S, H}
        // -> { {e :: C, E, F} :: S, H}
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<ExpressionStatement>(*stmt).Exp()));
        act->IncrementPos();
      } else {
        frame->todo.Pop(1);
      }
      break;
    case Statement::Kind::Assign:
      if (act->Pos() == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->RawNew<LValAction>(cast<Assign>(*stmt).Lhs()));
        act->IncrementPos();
      } else if (act->Pos() == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->RawNew<ExpressionAction>(cast<Assign>(*stmt).Rhs()));
        act->IncrementPos();
      } else if (act->Pos() == 2) {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        auto pat = act->Results()[0];
        auto val = act->Results()[1];
        PatternAssignment(pat, val, stmt->LineNumber());
        frame->todo.Pop(1);
      }
      break;
    case Statement::Kind::If:
      if (act->Pos() == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->RawNew<ExpressionAction>(cast<If>(*stmt).Cond()));
        act->IncrementPos();
      } else if (cast<BoolValue>(*act->Results()[0]).Val()) {
        //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { then_stmt :: C, E, F } :: S, H}
        frame->todo.Pop(1);
        frame->todo.Push(
            global_arena->RawNew<StatementAction>(cast<If>(*stmt).ThenStmt()));
      } else if (cast<If>(*stmt).ElseStmt()) {
        //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { else_stmt :: C, E, F } :: S, H}
        frame->todo.Pop(1);
        frame->todo.Push(
            global_arena->RawNew<StatementAction>(cast<If>(*stmt).ElseStmt()));
      } else {
        frame->todo.Pop(1);
      }
      break;
    case Statement::Kind::Return:
      if (act->Pos() == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        frame->todo.Push(
            global_arena->RawNew<ExpressionAction>(cast<Return>(*stmt).Exp()));
        act->IncrementPos();
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        const Value* ret_val = CopyVal(act->Results()[0], stmt->LineNumber());
        DeallocateLocals(stmt->LineNumber(), frame);
        state->stack.Pop(1);
        frame = state->stack.Top();
        frame->todo.Push(global_arena->RawNew<ValAction>(ret_val));
      }
      break;
    case Statement::Kind::Sequence:
      CHECK(act->Pos() == 0);
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      if (cast<Sequence>(*stmt).Next()) {
        frame->todo.Push(global_arena->RawNew<StatementAction>(
            cast<Sequence>(*stmt).Next()));
      }
      frame->todo.Push(
          global_arena->RawNew<StatementAction>(cast<Sequence>(*stmt).Stmt()));
      break;
    case Statement::Kind::Continuation: {
      CHECK(act->Pos() == 0);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      Scope* scope = global_arena->RawNew<Scope>(CurrentEnv(state),
                                                 std::list<std::string>());
      Stack<Scope*> scopes;
      scopes.Push(scope);
      Stack<Action*> todo;
      todo.Push(global_arena->RawNew<StatementAction>(
          global_arena->RawNew<Return>(stmt->LineNumber(), nullptr,
                                       /*is_omitted_exp=*/true)));
      todo.Push(global_arena->RawNew<StatementAction>(
          cast<Continuation>(*stmt).Body()));
      Frame* continuation_frame =
          global_arena->RawNew<Frame>("__continuation", scopes, todo);
      Address continuation_address =
          state->heap.AllocateValue(global_arena->RawNew<ContinuationValue>(
              std::vector<Frame*>({continuation_frame})));
      // Store the continuation's address in the frame.
      continuation_frame->continuation = continuation_address;
      // Bind the continuation object to the continuation variable
      frame->scopes.Top()->values.Set(
          cast<Continuation>(*stmt).ContinuationVariable(),
          continuation_address);
      // Pop the continuation statement.
      frame->todo.Pop();
      break;
    }
    case Statement::Kind::Run:
      if (act->Pos() == 0) {
        // Evaluate the argument of the run statement.
        frame->todo.Push(global_arena->RawNew<ExpressionAction>(
            cast<Run>(*stmt).Argument()));
        act->IncrementPos();
      } else {
        frame->todo.Pop(1);
        // Push an expression statement action to ignore the result
        // value from the continuation.
        Action* ignore_result = global_arena->RawNew<StatementAction>(
            global_arena->RawNew<ExpressionStatement>(
                stmt->LineNumber(),
                global_arena->RawNew<TupleLiteral>(stmt->LineNumber())));
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
    case Statement::Kind::Await:
      CHECK(act->Pos() == 0);
      // Pause the current continuation
      frame->todo.Pop();
      std::vector<Frame*> paused;
      do {
        paused.push_back(state->stack.Pop());
      } while (paused.back()->continuation == std::nullopt);
      // Update the continuation with the paused stack.
      state->heap.Write(*paused.back()->continuation,
                        global_arena->RawNew<ContinuationValue>(paused),
                        stmt->LineNumber());
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
  state = global_arena->RawNew<State>();  // Runtime state.
  if (tracing_output) {
    llvm::outs() << "********** initializing globals **********\n";
  }
  InitGlobals(fs);

  const Expression* arg = global_arena->RawNew<TupleLiteral>(0);
  const Expression* call_main = global_arena->RawNew<CallExpression>(
      0, global_arena->RawNew<IdentifierExpression>(0, "main"), arg);
  auto todo = Stack<Action*>(global_arena->RawNew<ExpressionAction>(call_main));
  auto* scope = global_arena->RawNew<Scope>(globals, std::list<std::string>());
  auto* frame = global_arena->RawNew<Frame>("top", Stack(scope), todo);
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
  auto todo = Stack<Action*>(global_arena->RawNew<ExpressionAction>(e));
  auto* scope = global_arena->RawNew<Scope>(values, std::list<std::string>());
  auto* frame = global_arena->RawNew<Frame>("InterpExp", Stack(scope), todo);
  state->stack = Stack(frame);

  while (state->stack.Count() > 1 || state->stack.Top()->todo.Count() > 1 ||
         state->stack.Top()->todo.Top()->Tag() != Action::Kind::ValAction) {
    Step();
  }
  return cast<ValAction>(*state->stack.Top()->todo.Top()).Val();
}

// Interpret a pattern at compile-time.
auto InterpPattern(Env values, const Pattern* p) -> const Value* {
  auto todo = Stack<Action*>(global_arena->RawNew<PatternAction>(p));
  auto* scope = global_arena->RawNew<Scope>(values, std::list<std::string>());
  auto* frame =
      global_arena->RawNew<Frame>("InterpPattern", Stack(scope), todo);
  state->stack = Stack(frame);

  while (state->stack.Count() > 1 || state->stack.Top()->todo.Count() > 1 ||
         state->stack.Top()->todo.Top()->Tag() != Action::Kind::ValAction) {
    Step();
  }
  return cast<ValAction>(*state->stack.Top()->todo.Top()).Val();
}

}  // namespace Carbon
