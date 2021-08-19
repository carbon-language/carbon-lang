// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/interpreter.h"

#include <iterator>
#include <list>
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

void PrintStack(const Stack<Ptr<Frame>>& ls, llvm::raw_ostream& out) {
  llvm::ListSeparator sep(" :: ");
  for (const auto& frame : ls) {
    out << sep << *frame;
  }
}

auto CurrentEnv(State* state) -> Env {
  Ptr<Frame> frame = state->stack.Top();
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

static void InitGlobals(const std::list<Ptr<const Declaration>>& fs) {
  for (const auto d : fs) {
    InitEnv(*d, &globals);
  }
}

void DeallocateScope(Ptr<Scope> scope) {
  for (const auto& l : scope->locals) {
    std::optional<Address> a = scope->values.Get(l);
    CHECK(a);
    state->heap.Deallocate(*a);
  }
}

void DeallocateLocals(Ptr<Frame> frame) {
  while (!frame->scopes.IsEmpty()) {
    DeallocateScope(frame->scopes.Top());
    frame->scopes.Pop();
  }
}

const Value* CreateTuple(Ptr<Action> act, const Expression* exp) {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  const auto& tup_lit = cast<TupleLiteral>(*exp);
  CHECK(act->Results().size() == tup_lit.Fields().size());
  std::vector<TupleElement> elements;
  for (size_t i = 0; i < act->Results().size(); ++i) {
    elements.push_back(
        {.name = tup_lit.Fields()[i].name, .value = act->Results()[i]});
  }

  return global_arena->RawNew<TupleValue>(std::move(elements));
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

// State transition functions
//
// The `Step*` family of functions implement state transitions in the
// interpreter by executing a step of the Action at the top of the todo stack,
// and then returning a Transition that specifies how `state.stack` should be
// updated. `Transition` is a variant of several "transition types" representing
// the different kinds of state transition.

// Transition type which indicates that the current Action is now done.
struct Done {
  // The value computed by the Action. Should always be null for Statement
  // Actions, and never null for any other kind of Action.
  const Value* result = nullptr;
};

// Transition type which spawns a new Action on the todo stack above the current
// Action, and increments the current Action's position counter.
struct Spawn {
  Ptr<Action> child;
};

// Transition type which spawns a new Action that replaces the current action
// on the todo stack.
struct Delegate {
  Ptr<Action> delegate;
};

// Transition type which keeps the current Action at the top of the stack,
// and increments its position counter.
struct RunAgain {};

// Transition type which unwinds the `todo` and `scopes` stacks until it
// reaches a specified Action lower in the stack.
struct UnwindTo {
  const Ptr<Action> new_top;
};

// Transition type which unwinds the entire current stack frame, and returns
// a specified value to the caller.
struct UnwindFunctionCall {
  const Value* return_val;
};

// Transition type which removes the current action from the top of the todo
// stack, then creates a new stack frame which calls the specified function
// with the specified arguments.
struct CallFunction {
  const FunctionValue* function;
  const Value* args;
  int line_num;
};

// Transition type which does nothing.
//
// TODO(geoffromer): This is a temporary placeholder during refactoring. All
// uses of this type should be replaced with meaningful transitions.
struct ManualTransition {};

using Transition =
    std::variant<Done, Spawn, Delegate, RunAgain, UnwindTo, UnwindFunctionCall,
                 CallFunction, ManualTransition>;

// State transitions for lvalues.
Transition StepLvalue() {
  Ptr<Action> act = state->stack.Top()->todo.Top();
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
      return Done{v};
    }
    case Expression::Kind::FieldAccessExpression: {
      if (act->Pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        return Spawn{global_arena->New<LValAction>(
            cast<FieldAccessExpression>(*exp).Aggregate())};
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->Results()[0]).Val();
        Address field = aggregate.SubobjectAddress(
            cast<FieldAccessExpression>(*exp).Field());
        return Done{global_arena->RawNew<PointerValue>(field)};
      }
    }
    case Expression::Kind::IndexExpression: {
      if (act->Pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{global_arena->New<LValAction>(
            cast<IndexExpression>(*exp).Aggregate())};

      } else if (act->Pos() == 1) {
        return Spawn{global_arena->New<ExpressionAction>(
            cast<IndexExpression>(*exp).Offset())};
      } else {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->Results()[0]).Val();
        std::string f =
            std::to_string(cast<IntValue>(*act->Results()[1]).Val());
        Address field = aggregate.SubobjectAddress(f);
        return Done{global_arena->RawNew<PointerValue>(field)};
      }
    }
    case Expression::Kind::TupleLiteral: {
      if (act->Pos() == 0) {
        //    { {(f1=e1,...) :: C, E, F} :: S, H}
        // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
        const Expression* e1 = cast<TupleLiteral>(*exp).Fields()[0].expression;
        return Spawn{global_arena->New<LValAction>(e1)};
      } else if (act->Pos() !=
                 static_cast<int>(cast<TupleLiteral>(*exp).Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Expression* elt =
            cast<TupleLiteral>(*exp).Fields()[act->Pos()].expression;
        return Spawn{global_arena->New<LValAction>(elt)};
      } else {
        return Done{CreateTuple(act, exp)};
      }
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
Transition StepExp() {
  Ptr<Action> act = state->stack.Top()->todo.Top();
  const Expression* exp = cast<ExpressionAction>(*act).Exp();
  if (tracing_output) {
    llvm::outs() << "--- step exp " << *exp << " --->\n";
  }
  switch (exp->Tag()) {
    case Expression::Kind::IndexExpression: {
      if (act->Pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{global_arena->New<ExpressionAction>(
            cast<IndexExpression>(*exp).Aggregate())};
      } else if (act->Pos() == 1) {
        return Spawn{global_arena->New<ExpressionAction>(
            cast<IndexExpression>(*exp).Offset())};
      } else {
        //    { { v :: [][i] :: C, E, F} :: S, H}
        // -> { { v_i :: C, E, F} : S, H}
        auto* tuple = dyn_cast<TupleValue>(act->Results()[0]);
        if (tuple == nullptr) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "expected a tuple in field access, not " << *tuple;
        }
        std::string f =
            std::to_string(cast<IntValue>(*act->Results()[1]).Val());
        const Value* field = tuple->FindField(f);
        if (field == nullptr) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "field " << f << " not in " << *tuple;
        }
        return Done{field};
      }
    }
    case Expression::Kind::TupleLiteral: {
      if (act->Pos() == 0) {
        if (cast<TupleLiteral>(*exp).Fields().size() > 0) {
          //    { {(f1=e1,...) :: C, E, F} :: S, H}
          // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
          const Expression* e1 =
              cast<TupleLiteral>(*exp).Fields()[0].expression;
          return Spawn{global_arena->New<ExpressionAction>(e1)};
        } else {
          return Done{CreateTuple(act, exp)};
        }
      } else if (act->Pos() !=
                 static_cast<int>(cast<TupleLiteral>(*exp).Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Expression* elt =
            cast<TupleLiteral>(*exp).Fields()[act->Pos()].expression;
        return Spawn{global_arena->New<ExpressionAction>(elt)};
      } else {
        return Done{CreateTuple(act, exp)};
      }
    }
    case Expression::Kind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(*exp);
      if (act->Pos() == 0) {
        //    { { e.f :: C, E, F} :: S, H}
        // -> { { e :: [].f :: C, E, F} :: S, H}
        return Spawn{global_arena->New<ExpressionAction>(access.Aggregate())};
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        return Done{act->Results()[0]->GetField(FieldPath(access.Field()),
                                                exp->LineNumber())};
      }
    }
    case Expression::Kind::IdentifierExpression: {
      CHECK(act->Pos() == 0);
      const auto& ident = cast<IdentifierExpression>(*exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp->LineNumber(), ident.Name());
      return Done{state->heap.Read(pointer, exp->LineNumber())};
    }
    case Expression::Kind::IntLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{global_arena->RawNew<IntValue>(cast<IntLiteral>(*exp).Val())};
    case Expression::Kind::BoolLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{
          global_arena->RawNew<BoolValue>(cast<BoolLiteral>(*exp).Val())};
    case Expression::Kind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(*exp);
      if (act->Pos() != static_cast<int>(op.Arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        const Expression* arg = op.Arguments()[act->Pos()];
        return Spawn{global_arena->New<ExpressionAction>(arg)};
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        return Done{EvalPrim(op.Op(), act->Results(), exp->LineNumber())};
      }
    }
    case Expression::Kind::CallExpression:
      if (act->Pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        return Spawn{global_arena->New<ExpressionAction>(
            cast<CallExpression>(*exp).Function())};
      } else if (act->Pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        return Spawn{global_arena->New<ExpressionAction>(
            cast<CallExpression>(*exp).Argument())};
      } else if (act->Pos() == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        switch (act->Results()[0]->Tag()) {
          case Value::Kind::ClassType: {
            const Value* arg = CopyVal(act->Results()[1], exp->LineNumber());
            return Done{
                global_arena->RawNew<StructValue>(act->Results()[0], arg)};
          }
          case Value::Kind::AlternativeConstructorValue: {
            const auto& alt =
                cast<AlternativeConstructorValue>(*act->Results()[0]);
            const Value* arg = CopyVal(act->Results()[1], exp->LineNumber());
            return Done{global_arena->RawNew<AlternativeValue>(
                alt.AltName(), alt.ChoiceName(), arg)};
          }
          case Value::Kind::FunctionValue:
            return CallFunction{
                .function = cast<FunctionValue>(act->Results()[0]),
                .args = act->Results()[1],
                .line_num = exp->LineNumber()};
          default:
            FATAL_RUNTIME_ERROR(exp->LineNumber())
                << "in call, expected a function, not " << *act->Results()[0];
        }
      } else {
        FATAL() << "in handle_value with Call pos " << act->Pos();
      }
    case Expression::Kind::IntrinsicExpression:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      switch (cast<IntrinsicExpression>(*exp).Intrinsic()) {
        case IntrinsicExpression::IntrinsicKind::Print:
          Address pointer = GetFromEnv(exp->LineNumber(), "format_str");
          const Value* pointee = state->heap.Read(pointer, exp->LineNumber());
          CHECK(pointee->Tag() == Value::Kind::StringValue);
          // TODO: This could eventually use something like llvm::formatv.
          llvm::outs() << cast<StringValue>(*pointee).Val();
          return Done{&TupleValue::Empty()};
      }

    case Expression::Kind::IntTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{global_arena->RawNew<IntType>()};
    }
    case Expression::Kind::BoolTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{global_arena->RawNew<BoolType>()};
    }
    case Expression::Kind::TypeTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{global_arena->RawNew<TypeType>()};
    }
    case Expression::Kind::FunctionTypeLiteral: {
      if (act->Pos() == 0) {
        return Spawn{global_arena->New<ExpressionAction>(
            cast<FunctionTypeLiteral>(*exp).Parameter())};
      } else if (act->Pos() == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        return Spawn{global_arena->New<ExpressionAction>(
            cast<FunctionTypeLiteral>(*exp).ReturnType())};
      } else {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        return Done{global_arena->RawNew<FunctionType>(
            std::vector<GenericBinding>(), act->Results()[0],
            act->Results()[1])};
      }
    }
    case Expression::Kind::ContinuationTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{global_arena->RawNew<ContinuationType>()};
    }
    case Expression::Kind::StringLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{
          global_arena->RawNew<StringValue>(cast<StringLiteral>(*exp).Val())};
    case Expression::Kind::StringTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{global_arena->RawNew<StringType>()};
    }
  }  // switch (exp->Tag)
}

Transition StepPattern() {
  Ptr<Action> act = state->stack.Top()->todo.Top();
  const Pattern* pattern = cast<PatternAction>(*act).Pat();
  if (tracing_output) {
    llvm::outs() << "--- step pattern " << *pattern << " --->\n";
  }
  switch (pattern->Tag()) {
    case Pattern::Kind::AutoPattern: {
      CHECK(act->Pos() == 0);
      return Done{global_arena->RawNew<AutoType>()};
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*pattern);
      if (act->Pos() == 0) {
        return Spawn{global_arena->New<PatternAction>(binding.Type())};
      } else {
        return Done{global_arena->RawNew<BindingPlaceholderValue>(
            binding.Name(), act->Results()[0])};
      }
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*pattern);
      if (act->Pos() == 0) {
        if (tuple.Fields().empty()) {
          return Done{&TupleValue::Empty()};
        } else {
          const Pattern* p1 = tuple.Fields()[0].pattern;
          return Spawn{(global_arena->New<PatternAction>(p1))};
        }
      } else if (act->Pos() != static_cast<int>(tuple.Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        const Pattern* elt = tuple.Fields()[act->Pos()].pattern;
        return Spawn{global_arena->New<PatternAction>(elt)};
      } else {
        std::vector<TupleElement> elements;
        for (size_t i = 0; i < tuple.Fields().size(); ++i) {
          elements.push_back(
              {.name = tuple.Fields()[i].name, .value = act->Results()[i]});
        }
        return Done{global_arena->RawNew<TupleValue>(std::move(elements))};
      }
    }
    case Pattern::Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*pattern);
      if (act->Pos() == 0) {
        return Spawn{
            global_arena->New<ExpressionAction>(alternative.ChoiceType())};
      } else if (act->Pos() == 1) {
        return Spawn{global_arena->New<PatternAction>(alternative.Arguments())};
      } else {
        CHECK(act->Pos() == 2);
        const auto& choice_type = cast<ChoiceType>(*act->Results()[0]);
        return Done{global_arena->RawNew<AlternativeValue>(
            alternative.AlternativeName(), choice_type.Name(),
            act->Results()[1])};
      }
    }
    case Pattern::Kind::ExpressionPattern:
      return Delegate{global_arena->New<ExpressionAction>(
          cast<ExpressionPattern>(pattern)->Expression())};
  }
}

auto IsWhileAct(Ptr<Action> act) -> bool {
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

auto IsBlockAct(Ptr<Action> act) -> bool {
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
Transition StepStmt() {
  Ptr<Frame> frame = state->stack.Top();
  Ptr<Action> act = frame->todo.Top();
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
        return Spawn{
            global_arena->New<ExpressionAction>(cast<Match>(*stmt).Exp())};
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
          return Done{};
        }
        auto c = cast<Match>(*stmt).Clauses()->begin();
        std::advance(c, clause_num);

        if (act->Pos() % 2 == 1) {
          // start interpreting the pattern of the clause
          //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
          // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
          return Spawn{global_arena->New<PatternAction>(c->first)};
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
            frame->scopes.Push(global_arena->New<Scope>(values, vars));
            const Statement* body_block =
                global_arena->RawNew<Block>(stmt->LineNumber(), c->second);
            auto body_act = global_arena->New<StatementAction>(body_block);
            body_act->IncrementPos();
            frame->todo.Pop(1);
            frame->todo.Push(body_act);
            frame->todo.Push(global_arena->New<StatementAction>(c->second));
            return ManualTransition{};
          } else {
            // this case did not match, moving on
            int next_clause_num = act->Pos() / 2;
            if (next_clause_num ==
                static_cast<int>(cast<Match>(*stmt).Clauses()->size())) {
              return Done{};
            }
            return RunAgain{};
          }
        }
      }
    case Statement::Kind::While:
      if (act->Pos() % 2 == 0) {
        //    { { (while (e) s) :: C, E, F} :: S, H}
        // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
        act->Clear();
        return Spawn{
            global_arena->New<ExpressionAction>(cast<While>(*stmt).Cond())};
      } else if (cast<BoolValue>(*act->Results().back()).Val()) {
        //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
        return Spawn{
            global_arena->New<StatementAction>(cast<While>(*stmt).Body())};
      } else {
        //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { C, E, F } :: S, H}
        return Done{};
      }
    case Statement::Kind::Break: {
      CHECK(act->Pos() == 0);
      //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { C, E', F} :: S, H}
      auto it =
          std::find_if(frame->todo.begin(), frame->todo.end(), &IsWhileAct);
      if (it == frame->todo.end()) {
        FATAL_RUNTIME_ERROR(stmt->LineNumber())
            << "`break` not inside `while` statement";
      }
      ++it;
      return UnwindTo{*it};
    }
    case Statement::Kind::Continue: {
      CHECK(act->Pos() == 0);
      //    { { continue; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { (while (e) s) :: C, E', F} :: S, H}
      auto it =
          std::find_if(frame->todo.begin(), frame->todo.end(), &IsWhileAct);
      if (it == frame->todo.end()) {
        FATAL_RUNTIME_ERROR(stmt->LineNumber())
            << "`continue` not inside `while` statement";
      }
      return UnwindTo{*it};
    }
    case Statement::Kind::Block: {
      if (act->Pos() == 0) {
        const Block& block = cast<Block>(*stmt);
        if (block.Stmt() != nullptr) {
          frame->scopes.Push(global_arena->New<Scope>(CurrentEnv(state)));
          return Spawn{global_arena->New<StatementAction>(block.Stmt())};
        } else {
          return Done{};
        }
      } else {
        Ptr<Scope> scope = frame->scopes.Top();
        DeallocateScope(scope);
        frame->scopes.Pop(1);
        return Done{};
      }
    }
    case Statement::Kind::VariableDefinition:
      if (act->Pos() == 0) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        return Spawn{global_arena->New<ExpressionAction>(
            cast<VariableDefinition>(*stmt).Init())};
      } else if (act->Pos() == 1) {
        return Spawn{global_arena->New<PatternAction>(
            cast<VariableDefinition>(*stmt).Pat())};
      } else {
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
        return Done{};
      }
    case Statement::Kind::ExpressionStatement:
      if (act->Pos() == 0) {
        //    { {e :: C, E, F} :: S, H}
        // -> { {e :: C, E, F} :: S, H}
        return Spawn{global_arena->New<ExpressionAction>(
            cast<ExpressionStatement>(*stmt).Exp())};
      } else {
        return Done{};
      }
    case Statement::Kind::Assign:
      if (act->Pos() == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        return Spawn{global_arena->New<LValAction>(cast<Assign>(*stmt).Lhs())};
      } else if (act->Pos() == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        return Spawn{
            global_arena->New<ExpressionAction>(cast<Assign>(*stmt).Rhs())};
      } else {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        auto pat = act->Results()[0];
        auto val = act->Results()[1];
        PatternAssignment(pat, val, stmt->LineNumber());
        return Done{};
      }
    case Statement::Kind::If:
      if (act->Pos() == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        return Spawn{
            global_arena->New<ExpressionAction>(cast<If>(*stmt).Cond())};
      } else if (cast<BoolValue>(*act->Results()[0]).Val()) {
        //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { then_stmt :: C, E, F } :: S, H}
        return Delegate{
            global_arena->New<StatementAction>(cast<If>(*stmt).ThenStmt())};
      } else if (cast<If>(*stmt).ElseStmt()) {
        //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { else_stmt :: C, E, F } :: S, H}
        return Delegate{
            global_arena->New<StatementAction>(cast<If>(*stmt).ElseStmt())};
      } else {
        return Done{};
      }
    case Statement::Kind::Return:
      if (act->Pos() == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        return Spawn{
            global_arena->New<ExpressionAction>(cast<Return>(*stmt).Exp())};
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        const Value* ret_val = CopyVal(act->Results()[0], stmt->LineNumber());
        return UnwindFunctionCall{ret_val};
      }
    case Statement::Kind::Sequence: {
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      const Sequence& seq = cast<Sequence>(*stmt);
      if (act->Pos() == 0) {
        return Spawn{global_arena->New<StatementAction>(seq.Stmt())};
      } else {
        if (seq.Next() != nullptr) {
          return Delegate{
              global_arena->New<StatementAction>(cast<Sequence>(*stmt).Next())};
        } else {
          return Done{};
        }
      }
    }
    case Statement::Kind::Continuation: {
      CHECK(act->Pos() == 0);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      auto scopes =
          Stack<Ptr<Scope>>(global_arena->New<Scope>(CurrentEnv(state)));
      Stack<Ptr<Action>> todo;
      todo.Push(global_arena->New<StatementAction>(
          global_arena->RawNew<Return>(stmt->LineNumber(), nullptr,
                                       /*is_omitted_exp=*/true)));
      todo.Push(
          global_arena->New<StatementAction>(cast<Continuation>(*stmt).Body()));
      auto continuation_frame =
          global_arena->New<Frame>("__continuation", scopes, todo);
      Address continuation_address =
          state->heap.AllocateValue(global_arena->RawNew<ContinuationValue>(
              std::vector<Ptr<Frame>>({continuation_frame})));
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
      if (act->Pos() == 0) {
        // Evaluate the argument of the run statement.
        return Spawn{
            global_arena->New<ExpressionAction>(cast<Run>(*stmt).Argument())};
      } else {
        frame->todo.Pop(1);
        // Push an expression statement action to ignore the result
        // value from the continuation.
        auto ignore_result = global_arena->New<StatementAction>(
            global_arena->RawNew<ExpressionStatement>(
                stmt->LineNumber(),
                global_arena->RawNew<TupleLiteral>(stmt->LineNumber())));
        frame->todo.Push(ignore_result);
        // Push the continuation onto the current stack.
        const std::vector<Ptr<Frame>>& continuation_vector =
            cast<ContinuationValue>(*act->Results()[0]).Stack();
        for (auto frame_iter = continuation_vector.rbegin();
             frame_iter != continuation_vector.rend(); ++frame_iter) {
          state->stack.Push(*frame_iter);
        }
        return ManualTransition{};
      }
    case Statement::Kind::Await:
      CHECK(act->Pos() == 0);
      // Pause the current continuation
      frame->todo.Pop();
      std::vector<Ptr<Frame>> paused;
      do {
        paused.push_back(state->stack.Pop());
      } while (paused.back()->continuation == std::nullopt);
      // Update the continuation with the paused stack.
      state->heap.Write(*paused.back()->continuation,
                        global_arena->RawNew<ContinuationValue>(paused),
                        stmt->LineNumber());
      return ManualTransition{};
  }
}

// Visitor which implements the behavior associated with each transition type.
struct DoTransition {
  void operator()(const Done& done) {
    Ptr<Frame> frame = state->stack.Top();
    if (frame->todo.Top()->Tag() != Action::Kind::StatementAction) {
      CHECK(done.result != nullptr);
      frame->todo.Pop();
      if (frame->todo.IsEmpty()) {
        state->program_value = done.result;
      } else {
        frame->todo.Top()->AddResult(done.result);
      }
    } else {
      CHECK(done.result == nullptr);
      frame->todo.Pop();
    }
  }

  void operator()(const Spawn& spawn) {
    Ptr<Frame> frame = state->stack.Top();
    frame->todo.Top()->IncrementPos();
    frame->todo.Push(spawn.child);
  }

  void operator()(const Delegate& delegate) {
    Ptr<Frame> frame = state->stack.Top();
    frame->todo.Pop();
    frame->todo.Push(delegate.delegate);
  }

  void operator()(const RunAgain&) {
    state->stack.Top()->todo.Top()->IncrementPos();
  }

  void operator()(const UnwindTo& unwind_to) {
    Ptr<Frame> frame = state->stack.Top();
    // TODO: drop .Get() calls once `Ptr` has comparison operators
    while (frame->todo.Top().Get() != unwind_to.new_top.Get()) {
      if (IsBlockAct(frame->todo.Top())) {
        DeallocateScope(frame->scopes.Top());
        frame->scopes.Pop();
      }
      frame->todo.Pop();
    }
  }

  void operator()(const UnwindFunctionCall& unwind) {
    DeallocateLocals(state->stack.Top());
    state->stack.Pop();
    if (state->stack.Top()->todo.IsEmpty()) {
      state->program_value = unwind.return_val;
    } else {
      state->stack.Top()->todo.Top()->AddResult(unwind.return_val);
    }
  }

  void operator()(const CallFunction& call) {
    state->stack.Top()->todo.Pop();
    std::optional<Env> matches =
        PatternMatch(call.function->Param(), call.args, call.line_num);
    CHECK(matches.has_value())
        << "internal error in call_function, pattern match failed";
    // Create the new frame and push it on the stack
    Env values = globals;
    std::list<std::string> params;
    for (const auto& [name, value] : *matches) {
      values.Set(name, value);
      params.push_back(name);
    }
    auto scopes = Stack<Ptr<Scope>>(global_arena->New<Scope>(values, params));
    auto todo = Stack<Ptr<Action>>(
        global_arena->New<StatementAction>(call.function->Body()));
    auto frame = global_arena->New<Frame>(call.function->Name(), scopes, todo);
    state->stack.Push(frame);
  }

  void operator()(const ManualTransition&) {}
};

// State transition.
void Step() {
  Ptr<Frame> frame = state->stack.Top();
  if (frame->todo.IsEmpty()) {
    FATAL_RUNTIME_ERROR_NO_LINE()
        << "fell off end of function " << frame->name << " without `return`";
  }

  Ptr<Action> act = frame->todo.Top();
  switch (act->Tag()) {
    case Action::Kind::LValAction:
      std::visit(DoTransition(), StepLvalue());
      break;
    case Action::Kind::ExpressionAction:
      std::visit(DoTransition(), StepExp());
      break;
    case Action::Kind::PatternAction:
      std::visit(DoTransition(), StepPattern());
      break;
    case Action::Kind::StatementAction:
      std::visit(DoTransition(), StepStmt());
      break;
  }  // switch
}

// Interpret the whole porogram.
auto InterpProgram(const std::list<Ptr<const Declaration>>& fs) -> int {
  state = global_arena->RawNew<State>();  // Runtime state.
  if (tracing_output) {
    llvm::outs() << "********** initializing globals **********\n";
  }
  InitGlobals(fs);

  const Expression* arg = global_arena->RawNew<TupleLiteral>(0);
  const Expression* call_main = global_arena->RawNew<CallExpression>(
      0, global_arena->RawNew<IdentifierExpression>(0, "main"), arg);
  auto todo =
      Stack<Ptr<Action>>(global_arena->New<ExpressionAction>(call_main));
  auto scopes = Stack<Ptr<Scope>>(global_arena->New<Scope>(globals));
  state->stack =
      Stack<Ptr<Frame>>(global_arena->New<Frame>("top", scopes, todo));

  if (tracing_output) {
    llvm::outs() << "********** calling main function **********\n";
    PrintState(llvm::outs());
  }

  while (state->stack.Count() > 1 || !state->stack.Top()->todo.IsEmpty()) {
    Step();
    if (tracing_output) {
      PrintState(llvm::outs());
    }
  }
  return cast<IntValue>(**state->program_value).Val();
}

// Interpret an expression at compile-time.
auto InterpExp(Env values, const Expression* e) -> const Value* {
  CHECK(state->program_value == std::nullopt);
  auto program_value_guard =
      llvm::make_scope_exit([] { state->program_value = std::nullopt; });
  auto todo = Stack<Ptr<Action>>(global_arena->New<ExpressionAction>(e));
  auto scopes = Stack<Ptr<Scope>>(global_arena->New<Scope>(values));
  state->stack =
      Stack<Ptr<Frame>>(global_arena->New<Frame>("InterpExp", scopes, todo));

  while (state->stack.Count() > 1 || !state->stack.Top()->todo.IsEmpty()) {
    Step();
  }
  CHECK(state->program_value != std::nullopt);
  return *state->program_value;
}

// Interpret a pattern at compile-time.
auto InterpPattern(Env values, const Pattern* p) -> const Value* {
  CHECK(state->program_value == std::nullopt);
  auto program_value_guard =
      llvm::make_scope_exit([] { state->program_value = std::nullopt; });
  auto todo = Stack<Ptr<Action>>(global_arena->New<PatternAction>(p));
  auto scopes = Stack<Ptr<Scope>>(global_arena->New<Scope>(values));
  state->stack = Stack<Ptr<Frame>>(
      global_arena->New<Frame>("InterpPattern", scopes, todo));

  while (state->stack.Count() > 1 || !state->stack.Top()->todo.IsEmpty()) {
    Step();
  }
  CHECK(state->program_value != std::nullopt);
  return *state->program_value;
}

}  // namespace Carbon
