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
#include "executable_semantics/interpreter/stack.h"
#include "executable_semantics/interpreter/typecheck.h"
#include "executable_semantics/tracing_flag.h"

namespace Carbon {

State* state = nullptr;

auto PatternMatch(const Value* pat, const Value* val, Env,
                  std::list<std::string>*, int) -> std::optional<Env>;
void HandleValue();

//
// Auxiliary Functions
//

auto Heap::AllocateValue(const Value* v) -> Address {
  // Putting the following two side effects together in this function
  // ensures that we don't do anything else in between, which is really bad!
  // Consider whether to include a copy of the input v in this function
  // or to leave it up to the caller.
  CHECK(v != nullptr);
  Address a = values_.size();
  values_.push_back(v);
  alive_.push_back(true);
  return a;
}

auto Heap::Read(Address a, int line_num) -> const Value* {
  this->CheckAlive(a, line_num);
  return values_[a];
}

auto Heap::Write(Address a, const Value* v, int line_num) -> void {
  CHECK(v != nullptr);
  this->CheckAlive(a, line_num);
  values_[a] = v;
}

void Heap::CheckAlive(Address address, int line_num) {
  if (!alive_[address]) {
    std::cerr << line_num << ": undefined behavior: access to dead value ";
    PrintValue(values_[address], std::cerr);
    std::cerr << std::endl;
    exit(-1);
  }
}

auto CopyVal(const Value* val, int line_num) -> const Value* {
  switch (val->tag) {
    case ValKind::TupleV: {
      auto* elements = new std::vector<TupleElement>();
      for (const TupleElement& element : *val->GetTuple().elements) {
        const Value* new_element =
            CopyVal(state->heap.Read(element.address, line_num), line_num);
        Address new_address = state->heap.AllocateValue(new_element);
        elements->push_back({.name = element.name, .address = new_address});
      }
      return Value::MakeTupleVal(elements);
    }
    case ValKind::AltV: {
      const Value* arg = CopyVal(
          state->heap.Read(val->GetAlternative().argument, line_num), line_num);
      Address argument_address = state->heap.AllocateValue(arg);
      return Value::MakeAltVal(*val->GetAlternative().alt_name,
                               *val->GetAlternative().choice_name,
                               argument_address);
    }
    case ValKind::StructV: {
      const Value* inits = CopyVal(val->GetStruct().inits, line_num);
      return Value::MakeStructVal(val->GetStruct().type, inits);
    }
    case ValKind::IntV:
      return Value::MakeIntVal(val->GetInteger());
    case ValKind::BoolV:
      return Value::MakeBoolVal(val->GetBoolean());
    case ValKind::FunV:
      return Value::MakeFunVal(*val->GetFunction().name,
                               val->GetFunction().param,
                               val->GetFunction().body);
    case ValKind::PtrV:
      return Value::MakePtrVal(val->GetPointer());
    case ValKind::ContinuationV:
      // Copying a continuation is "shallow".
      return val;
    case ValKind::FunctionTV:
      return Value::MakeFunTypeVal(
          CopyVal(val->GetFunctionType().param, line_num),
          CopyVal(val->GetFunctionType().ret, line_num));

    case ValKind::PointerTV:
      return Value::MakePtrTypeVal(
          CopyVal(val->GetPointerType().type, line_num));
    case ValKind::IntTV:
      return Value::MakeIntTypeVal();
    case ValKind::BoolTV:
      return Value::MakeBoolTypeVal();
    case ValKind::TypeTV:
      return Value::MakeTypeTypeVal();
    case ValKind::VarTV:
      return Value::MakeVarTypeVal(*val->GetVariableType());
    case ValKind::AutoTV:
      return Value::MakeAutoTypeVal();
    case ValKind::ContinuationTV:
      return Value::MakeContinuationTypeVal();
    case ValKind::StructTV:
    case ValKind::ChoiceTV:
    case ValKind::VarPatV:
    case ValKind::AltConsV:
      return val;  // no need to copy these because they are immutable?
      // No, they need to be copied so they don't get killed. -Jeremy
  }
}

void Heap::DeallocateSubObjects(const Value* val) {
  switch (val->tag) {
    case ValKind::AltV:
      Deallocate(val->GetAlternative().argument);
      break;
    case ValKind::StructV:
      DeallocateSubObjects(val->GetStruct().inits);
      break;
    case ValKind::TupleV:
      for (const TupleElement& element : *val->GetTuple().elements) {
        Deallocate(element.address);
      }
      break;
    default:
      break;
  }
}

void Heap::Deallocate(Address address) {
  if (alive_[address]) {
    alive_[address] = false;
    DeallocateSubObjects(values_[address]);
  } else {
    std::cerr << "runtime error, deallocating an already dead value"
              << std::endl;
    exit(-1);
  }
}

void PrintEnv(Env values, std::ostream& out) {
  for (const auto& [name, address] : values) {
    out << name << ": ";
    state->heap.PrintAddress(address, out);
    out << ", ";
  }
}

//
// Frame and State Operations
//

void PrintFrame(Frame* frame, std::ostream& out) {
  out << frame->name;
  out << "{";
  PrintActList(frame->todo, out);
  out << "}";
}

void PrintStack(Stack<Frame*> ls, std::ostream& out) {
  if (!ls.IsEmpty()) {
    PrintFrame(ls.Pop(), out);
    if (!ls.IsEmpty()) {
      out << " :: ";
      PrintStack(ls, out);
    }
  }
}

void Heap::PrintHeap(std::ostream& out) {
  for (Address i = 0; i < values_.size(); ++i) {
    PrintAddress(i, out);
    out << ", ";
  }
}

auto Heap::PrintAddress(Address a, std::ostream& out) -> void {
  if (!alive_[a]) {
    out << "!!";
  }
  PrintValue(values_[a], out);
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

//
// More Auxiliary Functions
//

auto ValToInt(const Value* v, int line_num) -> int {
  switch (v->tag) {
    case ValKind::IntV:
      return v->GetInteger();
    default:
      std::cerr << line_num << ": runtime error: expected an integer"
                << std::endl;
      exit(-1);
  }
}

auto ValToBool(const Value* v, int line_num) -> int {
  switch (v->tag) {
    case ValKind::BoolV:
      return v->GetBoolean();
    default:
      std::cerr << "runtime type error: expected a Boolean" << std::endl;
      exit(-1);
  }
}

auto ValToPtr(const Value* v, int line_num) -> Address {
  switch (v->tag) {
    case ValKind::PtrV:
      return v->GetPointer();
    default:
      std::cerr << "runtime type error: expected a pointer, not ";
      PrintValue(v, std::cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

// Returns *continuation represented as a list of frames.
//
// - Precondition: continuation->tag == ValKind::ContinuationV.
auto ContinuationToVector(const Value* continuation, int sourceLocation)
    -> std::vector<Frame*> {
  if (continuation->tag == ValKind::ContinuationV) {
    return *continuation->GetContinuation().stack;
  } else {
    std::cerr << sourceLocation << ": runtime error: expected an integer"
              << std::endl;
    exit(-1);
  }
}

auto EvalPrim(Operator op, const std::vector<const Value*>& args, int line_num)
    -> const Value* {
  switch (op) {
    case Operator::Neg:
      return Value::MakeIntVal(-ValToInt(args[0], line_num));
    case Operator::Add:
      return Value::MakeIntVal(ValToInt(args[0], line_num) +
                               ValToInt(args[1], line_num));
    case Operator::Sub:
      return Value::MakeIntVal(ValToInt(args[0], line_num) -
                               ValToInt(args[1], line_num));
    case Operator::Mul:
      return Value::MakeIntVal(ValToInt(args[0], line_num) *
                               ValToInt(args[1], line_num));
    case Operator::Not:
      return Value::MakeBoolVal(!ValToBool(args[0], line_num));
    case Operator::And:
      return Value::MakeBoolVal(ValToBool(args[0], line_num) &&
                                ValToBool(args[1], line_num));
    case Operator::Or:
      return Value::MakeBoolVal(ValToBool(args[0], line_num) ||
                                ValToBool(args[1], line_num));
    case Operator::Eq:
      return Value::MakeBoolVal(ValueEqual(args[0], args[1], line_num));
    case Operator::Ptr:
      return Value::MakePtrTypeVal(args[0]);
    case Operator::Deref:
      std::cerr << line_num << ": dereference not implemented yet\n";
      exit(-1);
  }
}

// Globally-defined entities, such as functions, structs, choices.
Env globals;

void InitGlobals(std::list<Declaration>* fs) {
  for (auto const& d : *fs) {
    d.InitGlobals(globals);
  }
}

auto ChoiceDeclaration::InitGlobals(Env& globals) const -> void {
  auto alts = new VarValues();
  for (auto kv : alternatives) {
    auto t = InterpExp(Env(), kv.second);
    alts->push_back(make_pair(kv.first, t));
  }
  auto ct = Value::MakeChoiceTypeVal(name, alts);
  auto a = state->heap.AllocateValue(ct);
  globals.Set(name, a);
}

auto StructDeclaration::InitGlobals(Env& globals) const -> void {
  auto fields = new VarValues();
  auto methods = new VarValues();
  for (auto i = definition.members->begin(); i != definition.members->end();
       ++i) {
    switch ((*i)->tag) {
      case MemberKind::FieldMember: {
        auto t = InterpExp(Env(), (*i)->u.field.type);
        fields->push_back(make_pair(*(*i)->u.field.name, t));
        break;
      }
    }
  }
  auto st = Value::MakeStructTypeVal(*definition.name, fields, methods);
  auto a = state->heap.AllocateValue(st);
  globals.Set(*definition.name, a);
}

auto FunctionDeclaration::InitGlobals(Env& globals) const -> void {
  auto pt = InterpExp(globals, definition->param_pattern);
  auto f = Value::MakeFunVal(definition->name, pt, definition->body);
  Address a = state->heap.AllocateValue(f);
  globals.Set(definition->name, a);
}

// Adds an entry in `globals` mapping the variable's name to the
// result of evaluating the initializer.
auto VariableDeclaration::InitGlobals(Env& globals) const -> void {
  auto v = InterpExp(globals, initializer);
  Address a = state->heap.AllocateValue(v);
  globals.Set(name, a);
}

//    { S, H} -> { { C, E, F} :: S, H}
// where C is the body of the function,
//       E is the environment (functions + parameters + locals)
//       F is the function
void CallFunction(int line_num, std::vector<const Value*> operas,
                  State* state) {
  switch (operas[0]->tag) {
    case ValKind::FunV: {
      // Bind arguments to parameters
      std::list<std::string> params;
      std::optional<Env> matches =
          PatternMatch(operas[0]->GetFunction().param, operas[1], globals,
                       &params, line_num);
      if (!matches) {
        std::cerr << "internal error in call_function, pattern match failed"
                  << std::endl;
        exit(-1);
      }
      // Create the new frame and push it on the stack
      auto* scope = new Scope(*matches, params);
      auto* frame =
          new Frame(*operas[0]->GetFunction().name, Stack(scope),
                    Stack(MakeStmtAct(operas[0]->GetFunction().body)));
      state->stack.Push(frame);
      break;
    }
    case ValKind::StructTV: {
      const Value* arg = CopyVal(operas[1], line_num);
      const Value* sv = Value::MakeStructVal(operas[0], arg);
      Frame* frame = state->stack.Top();
      frame->todo.Push(MakeValAct(sv));
      break;
    }
    case ValKind::AltConsV: {
      const Value* arg = CopyVal(operas[1], line_num);
      const Value* av =
          Value::MakeAltVal(*operas[0]->GetAlternativeConstructor().alt_name,
                            *operas[0]->GetAlternativeConstructor().choice_name,
                            state->heap.AllocateValue(arg));
      Frame* frame = state->stack.Top();
      frame->todo.Push(MakeValAct(av));
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

void CreateTuple(Frame* frame, Action* act, const Expression* /*exp*/) {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  auto elements = new std::vector<TupleElement>();
  auto f = act->u.exp->GetTuple().fields.begin();
  for (auto i = act->results.begin(); i != act->results.end(); ++i, ++f) {
    Address a = state->heap.AllocateValue(*i);  // copy?
    elements->push_back({.name = f->name, .address = a});
  }
  const Value* tv = Value::MakeTupleVal(elements);
  frame->todo.Pop(1);
  frame->todo.Push(MakeValAct(tv));
}

// Returns an updated environment that includes the bindings of
//    pattern variables to their matched values, if matching succeeds.
//
// The names of the pattern variables are added to the vars parameter.
// Returns nullopt if the value doesn't match the pattern.
auto PatternMatch(const Value* p, const Value* v, Env values,
                  std::list<std::string>* vars, int line_num)
    -> std::optional<Env> {
  switch (p->tag) {
    case ValKind::VarPatV: {
      Address a = state->heap.AllocateValue(CopyVal(v, line_num));
      vars->push_back(*p->GetVariablePattern().name);
      values.Set(*p->GetVariablePattern().name, a);
      return values;
    }
    case ValKind::TupleV:
      switch (v->tag) {
        case ValKind::TupleV: {
          if (p->GetTuple().elements->size() !=
              v->GetTuple().elements->size()) {
            std::cerr << "runtime error: arity mismatch in tuple pattern match"
                      << std::endl;
            exit(-1);
          }
          for (const TupleElement& element : *p->GetTuple().elements) {
            auto a = FindTupleField(element.name, v);
            if (a == std::nullopt) {
              std::cerr << "runtime error: field " << element.name << "not in ";
              PrintValue(v, std::cerr);
              std::cerr << std::endl;
              exit(-1);
            }
            std::optional<Env> matches = PatternMatch(
                state->heap.Read(element.address, line_num),
                state->heap.Read(*a, line_num), values, vars, line_num);
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
    case ValKind::AltV:
      switch (v->tag) {
        case ValKind::AltV: {
          if (*p->GetAlternative().choice_name !=
                  *v->GetAlternative().choice_name ||
              *p->GetAlternative().alt_name != *v->GetAlternative().alt_name) {
            return std::nullopt;
          }
          std::optional<Env> matches = PatternMatch(
              state->heap.Read(p->GetAlternative().argument, line_num),
              state->heap.Read(v->GetAlternative().argument, line_num), values,
              vars, line_num);
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
    case ValKind::FunctionTV:
      switch (v->tag) {
        case ValKind::FunctionTV: {
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
  switch (pat->tag) {
    case ValKind::PtrV:
      state->heap.Write(ValToPtr(pat, line_num), CopyVal(val, line_num),
                        line_num);
      break;
    case ValKind::TupleV: {
      switch (val->tag) {
        case ValKind::TupleV: {
          if (pat->GetTuple().elements->size() !=
              val->GetTuple().elements->size()) {
            std::cerr << "runtime error: arity mismatch in tuple pattern match"
                      << std::endl;
            exit(-1);
          }
          for (const TupleElement& element : *pat->GetTuple().elements) {
            auto a = FindTupleField(element.name, val);
            if (a == std::nullopt) {
              std::cerr << "runtime error: field " << element.name << "not in ";
              PrintValue(val, std::cerr);
              std::cerr << std::endl;
              exit(-1);
            }
            PatternAssignment(state->heap.Read(element.address, line_num),
                              state->heap.Read(*a, line_num), line_num);
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
    case ValKind::AltV: {
      switch (val->tag) {
        case ValKind::AltV: {
          if (*pat->GetAlternative().choice_name !=
                  *val->GetAlternative().choice_name ||
              *pat->GetAlternative().alt_name !=
                  *val->GetAlternative().alt_name) {
            std::cerr << "internal error in pattern assignment" << std::endl;
            exit(-1);
          }
          PatternAssignment(
              state->heap.Read(pat->GetAlternative().argument, line_num),
              state->heap.Read(val->GetAlternative().argument, line_num),
              line_num);
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
  const Expression* exp = act->u.exp;
  if (tracing_output) {
    std::cout << "--- step lvalue ";
    PrintExp(exp);
    std::cout << " --->" << std::endl;
  }
  switch (exp->tag()) {
    case ExpressionKind::Variable: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      std::optional<Address> pointer =
          CurrentEnv(state).Get(exp->GetVariable().name);
      if (!pointer) {
        std::cerr << exp->line_num << ": could not find `"
                  << exp->GetVariable().name << "`" << std::endl;
        exit(-1);
      }
      const Value* v = Value::MakePtrVal(*pointer);
      frame->todo.Pop();
      frame->todo.Push(MakeValAct(v));
      break;
    }
    case ExpressionKind::GetField: {
      //    { {e.f :: C, E, F} :: S, H}
      // -> { e :: [].f :: C, E, F} :: S, H}
      frame->todo.Push(
          MakeLvalAct(exp->GetFieldAccess().aggregate.GetPointer()));
      act->pos++;
      break;
    }
    case ExpressionKind::Index: {
      //    { {e[i] :: C, E, F} :: S, H}
      // -> { e :: [][i] :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(exp->GetIndex().aggregate));
      act->pos++;
      break;
    }
    case ExpressionKind::Tuple: {
      //    { {(f1=e1,...) :: C, E, F} :: S, H}
      // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
      const Expression* e1 = exp->GetTuple().fields[0].expression.GetPointer();
      frame->todo.Push(MakeLvalAct(e1));
      act->pos++;
      break;
    }
    case ExpressionKind::Integer:
    case ExpressionKind::Boolean:
    case ExpressionKind::Call:
    case ExpressionKind::PrimitiveOp:
    case ExpressionKind::IntT:
    case ExpressionKind::BoolT:
    case ExpressionKind::TypeT:
    case ExpressionKind::FunctionT:
    case ExpressionKind::AutoT:
    case ExpressionKind::ContinuationT:
    case ExpressionKind::PatternVariable: {
      frame->todo.Pop();
      frame->todo.Push(MakeExpToLvalAct());
      frame->todo.Push(MakeExpAct(exp));
    }
  }
}

// State transitions for expressions.

void StepExp() {
  Frame* frame = state->stack.Top();
  Action* act = frame->todo.Top();
  const Expression* exp = act->u.exp;
  if (tracing_output) {
    std::cout << "--- step exp ";
    PrintExp(exp);
    std::cout << " --->" << std::endl;
  }
  switch (exp->tag()) {
    case ExpressionKind::PatternVariable: {
      frame->todo.Push(MakeExpAct(exp->GetPatternVariable().type));
      act->pos++;
      break;
    }
    case ExpressionKind::Index: {
      //    { { e[i] :: C, E, F} :: S, H}
      // -> { { e :: [][i] :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(exp->GetIndex().aggregate));
      act->pos++;
      break;
    }
    case ExpressionKind::Tuple: {
      if (exp->GetTuple().fields.size() > 0) {
        //    { {(f1=e1,...) :: C, E, F} :: S, H}
        // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
        const Expression* e1 =
            exp->GetTuple().fields[0].expression.GetPointer();
        frame->todo.Push(MakeExpAct(e1));
        act->pos++;
      } else {
        CreateTuple(frame, act, exp);
      }
      break;
    }
    case ExpressionKind::GetField: {
      //    { { e.f :: C, E, F} :: S, H}
      // -> { { e :: [].f :: C, E, F} :: S, H}
      frame->todo.Push(
          MakeLvalAct(exp->GetFieldAccess().aggregate.GetPointer()));
      act->pos++;
      break;
    }
    case ExpressionKind::Variable: {
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      std::optional<Address> pointer =
          CurrentEnv(state).Get(exp->GetVariable().name);
      if (!pointer) {
        std::cerr << exp->line_num << ": could not find `"
                  << exp->GetVariable().name << "`" << std::endl;
        exit(-1);
      }
      const Value* pointee = state->heap.Read(*pointer, exp->line_num);
      frame->todo.Pop(1);
      frame->todo.Push(MakeValAct(pointee));
      break;
    }
    case ExpressionKind::Integer:
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(MakeValAct(Value::MakeIntVal(exp->GetInteger())));
      break;
    case ExpressionKind::Boolean:
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      frame->todo.Push(MakeValAct(Value::MakeBoolVal(exp->GetBoolean())));
      break;
    case ExpressionKind::PrimitiveOp:
      if (exp->GetPrimitiveOperator().arguments->size() > 0) {
        //    { {op(e :: es) :: C, E, F} :: S, H}
        // -> { e :: op([] :: es) :: C, E, F} :: S, H}
        frame->todo.Push(
            MakeExpAct(exp->GetPrimitiveOperator().arguments->front()));
        act->pos++;
      } else {
        //    { {v :: op(]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, ()) :: C, E, F} :: S, H}
        const Value* v = EvalPrim(exp->GetPrimitiveOperator().op, act->results,
                                  exp->line_num);
        frame->todo.Pop(2);
        frame->todo.Push(MakeValAct(v));
      }
      break;
    case ExpressionKind::Call:
      //    { {e1(e2) :: C, E, F} :: S, H}
      // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(exp->GetCall().function));
      act->pos++;
      break;
    case ExpressionKind::IntT: {
      const Value* v = Value::MakeIntTypeVal();
      frame->todo.Pop(1);
      frame->todo.Push(MakeValAct(v));
      break;
    }
    case ExpressionKind::BoolT: {
      const Value* v = Value::MakeBoolTypeVal();
      frame->todo.Pop(1);
      frame->todo.Push(MakeValAct(v));
      break;
    }
    case ExpressionKind::AutoT: {
      const Value* v = Value::MakeAutoTypeVal();
      frame->todo.Pop(1);
      frame->todo.Push(MakeValAct(v));
      break;
    }
    case ExpressionKind::TypeT: {
      const Value* v = Value::MakeTypeTypeVal();
      frame->todo.Pop(1);
      frame->todo.Push(MakeValAct(v));
      break;
    }
    case ExpressionKind::FunctionT: {
      frame->todo.Push(MakeExpAct(exp->GetFunctionType().parameter));
      act->pos++;
      break;
    }
    case ExpressionKind::ContinuationT: {
      const Value* v = Value::MakeContinuationTypeVal();
      frame->todo.Pop(1);
      frame->todo.Push(MakeValAct(v));
      break;
    }
  }  // switch (exp->tag)
}

auto IsWhileAct(Action* act) -> bool {
  switch (act->tag) {
    case ActionKind::StatementAction:
      switch (act->u.stmt->tag) {
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
  switch (act->tag) {
    case ActionKind::StatementAction:
      switch (act->u.stmt->tag) {
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
  const Statement* stmt = act->u.stmt;
  CHECK(stmt != nullptr && "null statement!");
  if (tracing_output) {
    std::cout << "--- step stmt ";
    PrintStatement(stmt, 1);
    std::cout << " --->" << std::endl;
  }
  switch (stmt->tag) {
    case StatementKind::Match:
      //    { { (match (e) ...) :: C, E, F} :: S, H}
      // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(stmt->GetMatch().exp));
      act->pos++;
      break;
    case StatementKind::While:
      //    { { (while (e) s) :: C, E, F} :: S, H}
      // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(stmt->GetWhile().cond));
      act->pos++;
      break;
    case StatementKind::Break:
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
      if (act->pos == -1) {
        if (stmt->GetBlock().stmt) {
          auto* scope = new Scope(CurrentEnv(state), {});
          frame->scopes.Push(scope);
          frame->todo.Push(MakeStmtAct(stmt->GetBlock().stmt));
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
      //    { {(var x = e) :: C, E, F} :: S, H}
      // -> { {e :: (var x = []) :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(stmt->GetVariableDefinition().init));
      act->pos++;
      break;
    case StatementKind::ExpressionStatement:
      //    { {e :: C, E, F} :: S, H}
      // -> { {e :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(stmt->GetExpression()));
      break;
    case StatementKind::Assign:
      //    { {(lv = e) :: C, E, F} :: S, H}
      // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
      frame->todo.Push(MakeLvalAct(stmt->GetAssign().lhs));
      act->pos++;
      break;
    case StatementKind::If:
      //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
      // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(stmt->GetIf().cond));
      act->pos++;
      break;
    case StatementKind::Return:
      //    { {return e :: C, E, F} :: S, H}
      // -> { {e :: return [] :: C, E, F} :: S, H}
      frame->todo.Push(MakeExpAct(stmt->GetReturn()));
      act->pos++;
      break;
    case StatementKind::Sequence:
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      frame->todo.Pop(1);
      if (stmt->GetSequence().next) {
        frame->todo.Push(MakeStmtAct(stmt->GetSequence().next));
      }
      frame->todo.Push(MakeStmtAct(stmt->GetSequence().stmt));
      break;
    case StatementKind::Continuation: {
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      Scope* scope = new Scope(CurrentEnv(state), std::list<std::string>());
      Stack<Scope*> scopes;
      scopes.Push(scope);
      Stack<Action*> todo;
      todo.Push(MakeStmtAct(Statement::MakeReturn(
          stmt->line_num, Expression::MakeTuple(stmt->line_num, {}))));
      todo.Push(MakeStmtAct(stmt->GetContinuation().body));
      Frame* continuation_frame = new Frame("__continuation", scopes, todo);
      Address continuation_address = state->heap.AllocateValue(
          Value::MakeContinuation({continuation_frame}));
      // Store the continuation's address in the frame.
      continuation_frame->continuation = continuation_address;
      // Bind the continuation object to the continuation variable
      frame->scopes.Top()->values.Set(
          *stmt->GetContinuation().continuation_variable, continuation_address);
      // Pop the continuation statement.
      frame->todo.Pop();
      break;
    }
    case StatementKind::Run:
      // Evaluate the argument of the run statement.
      frame->todo.Push(MakeExpAct(stmt->GetRun().argument));
      act->pos++;
      break;
    case StatementKind::Await:
      // Pause the current continuation
      frame->todo.Pop();
      std::vector<Frame*> paused;
      do {
        paused.push_back(state->stack.Pop());
      } while (!paused.back()->IsContinuation());
      // Update the continuation with the paused stack.
      state->heap.Write(paused.back()->continuation,
                        Value::MakeContinuation(paused), stmt->line_num);
      break;
  }
}

auto GetMember(Address a, const std::string& f, int line_num) -> Address {
  const Value* v = state->heap.Read(a, line_num);
  switch (v->tag) {
    case ValKind::StructV: {
      auto a = FindTupleField(f, v->GetStruct().inits);
      if (a == std::nullopt) {
        std::cerr << "runtime error, member " << f << " not in ";
        PrintValue(v, std::cerr);
        std::cerr << std::endl;
        exit(-1);
      }
      return *a;
    }
    case ValKind::TupleV: {
      auto a = FindTupleField(f, v);
      if (a == std::nullopt) {
        std::cerr << "field " << f << " not in ";
        PrintValue(v, std::cerr);
        std::cerr << std::endl;
        exit(-1);
      }
      return *a;
    }
    case ValKind::ChoiceTV: {
      if (FindInVarValues(f, v->GetChoiceType().alternatives) == nullptr) {
        std::cerr << "alternative " << f << " not in ";
        PrintValue(v, std::cerr);
        std::cerr << std::endl;
        exit(-1);
      }
      auto ac = Value::MakeAltCons(f, *v->GetChoiceType().name);
      return state->heap.AllocateValue(ac);
    }
    default:
      std::cerr << "field access not allowed for value ";
      PrintValue(v, std::cerr);
      std::cerr << std::endl;
      exit(-1);
  }
}

void InsertDelete(Action* del, Stack<Action*>& todo) {
  if (!todo.IsEmpty()) {
    switch (todo.Top()->tag) {
      case ActionKind::StatementAction: {
        // This places the delete before the enclosing statement.
        // Not sure if that is OK. Conceptually it should go after
        // but that is tricky for some statements, like 'return'. -Jeremy
        todo.Push(del);
        break;
      }
      case ActionKind::LValAction:
      case ActionKind::ExpressionAction:
      case ActionKind::ValAction:
      case ActionKind::ExpToLValAction:
      case ActionKind::DeleteTmpAction:
        auto top = todo.Pop();
        InsertDelete(del, todo);
        todo.Push(top);
        break;
    }
  } else {
    todo.Push(del);
  }
}

// State transition for handling a value.

void HandleValue() {
  Frame* frame = state->stack.Top();
  Action* val_act = frame->todo.Top();
  Action* act = frame->todo.Popped().Top();
  act->results.push_back(val_act->u.val);
  act->pos++;

  if (tracing_output) {
    std::cout << "--- handle value ";
    PrintValue(val_act->u.val, std::cout);
    std::cout << " with ";
    PrintAct(act, std::cout);
    std::cout << " --->" << std::endl;
  }
  switch (act->tag) {
    case ActionKind::DeleteTmpAction: {
      state->heap.Deallocate(act->u.delete_tmp);
      frame->todo.Pop(2);
      frame->todo.Push(val_act);
      break;
    }
    case ActionKind::ExpToLValAction: {
      Address a = state->heap.AllocateValue(act->results[0]);
      auto del = MakeDeleteAct(a);
      frame->todo.Pop(2);
      InsertDelete(del, frame->todo);
      frame->todo.Push(MakeValAct(Value::MakePtrVal(a)));
      break;
    }
    case ActionKind::LValAction: {
      const Expression* exp = act->u.exp;
      switch (exp->tag()) {
        case ExpressionKind::GetField: {
          //    { v :: [].f :: C, E, F} :: S, H}
          // -> { { &v.f :: C, E, F} :: S, H }
          const Value* str = act->results[0];
          Address a = GetMember(ValToPtr(str, exp->line_num),
                                exp->GetFieldAccess().field, exp->line_num);
          frame->todo.Pop(2);
          frame->todo.Push(MakeValAct(Value::MakePtrVal(a)));
          break;
        }
        case ExpressionKind::Index: {
          if (act->pos == 1) {
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(exp->GetIndex().offset));
          } else if (act->pos == 2) {
            //    { v :: [][i] :: C, E, F} :: S, H}
            // -> { { &v[i] :: C, E, F} :: S, H }
            const Value* tuple = act->results[0];
            std::string f = std::to_string(ToInteger(act->results[1]));
            auto a = FindTupleField(f, tuple);
            if (a == std::nullopt) {
              std::cerr << "runtime error: field " << f << "not in ";
              PrintValue(tuple, std::cerr);
              std::cerr << std::endl;
              exit(-1);
            }
            frame->todo.Pop(2);
            frame->todo.Push(MakeValAct(Value::MakePtrVal(*a)));
          }
          break;
        }
        case ExpressionKind::Tuple: {
          if (act->pos != static_cast<int>(exp->GetTuple().fields.size())) {
            //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
            //    H}
            // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
            // H}
            const Expression* elt =
                exp->GetTuple().fields[act->pos].expression.GetPointer();
            frame->todo.Pop(1);
            frame->todo.Push(MakeLvalAct(elt));
          } else {
            frame->todo.Pop(1);
            CreateTuple(frame, act, exp);
          }
          break;
        }
        default:
          std::cerr << "internal error in handle_value, LValAction"
                    << std::endl;
          exit(-1);
      }
      break;
    }
    case ActionKind::ExpressionAction: {
      const Expression* exp = act->u.exp;
      switch (exp->tag()) {
        case ExpressionKind::PatternVariable: {
          auto v = Value::MakeVarPatVal(exp->GetPatternVariable().name,
                                        act->results[0]);
          frame->todo.Pop(2);
          frame->todo.Push(MakeValAct(v));
          break;
        }
        case ExpressionKind::Tuple: {
          if (act->pos != static_cast<int>(exp->GetTuple().fields.size())) {
            //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
            //    H}
            // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
            // H}
            const Expression* elt =
                exp->GetTuple().fields[act->pos].expression.GetPointer();
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(elt));
          } else {
            frame->todo.Pop(1);
            CreateTuple(frame, act, exp);
          }
          break;
        }
        case ExpressionKind::Index: {
          if (act->pos == 1) {
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(exp->GetIndex().offset));
          } else if (act->pos == 2) {
            auto tuple = act->results[0];
            switch (tuple->tag) {
              case ValKind::TupleV: {
                //    { { v :: [][i] :: C, E, F} :: S, H}
                // -> { { v_i :: C, E, F} : S, H}
                std::string f = std::to_string(ToInteger(act->results[1]));
                auto a = FindTupleField(f, tuple);
                if (a == std::nullopt) {
                  std::cerr << "runtime error, field " << f << " not in ";
                  PrintValue(tuple, std::cerr);
                  std::cerr << std::endl;
                  exit(-1);
                }
                frame->todo.Pop(2);
                const Value* element = state->heap.Read(*a, exp->line_num);
                frame->todo.Push(MakeValAct(element));
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
        case ExpressionKind::GetField: {
          //    { { v :: [].f :: C, E, F} :: S, H}
          // -> { { v_f :: C, E, F} : S, H}
          auto a = GetMember(ValToPtr(act->results[0], exp->line_num),
                             exp->GetFieldAccess().field, exp->line_num);
          const Value* element = state->heap.Read(a, exp->line_num);
          frame->todo.Pop(2);
          frame->todo.Push(MakeValAct(element));
          break;
        }
        case ExpressionKind::PrimitiveOp: {
          if (act->pos !=
              static_cast<int>(exp->GetPrimitiveOperator().arguments->size())) {
            //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
            // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
            const Expression* arg =
                (*exp->GetPrimitiveOperator().arguments)[act->pos];
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(arg));
          } else {
            //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
            // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
            const Value* v = EvalPrim(exp->GetPrimitiveOperator().op,
                                      act->results, exp->line_num);
            frame->todo.Pop(2);
            frame->todo.Push(MakeValAct(v));
          }
          break;
        }
        case ExpressionKind::Call: {
          if (act->pos == 1) {
            //    { { v :: [](e) :: C, E, F} :: S, H}
            // -> { { e :: v([]) :: C, E, F} :: S, H}
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(exp->GetCall().argument));
          } else if (act->pos == 2) {
            //    { { v2 :: v1([]) :: C, E, F} :: S, H}
            // -> { {C',E',F'} :: {C, E, F} :: S, H}
            frame->todo.Pop(2);
            CallFunction(exp->line_num, act->results, state);
          } else {
            std::cerr << "internal error in handle_value with Call"
                      << std::endl;
            exit(-1);
          }
          break;
        }
        case ExpressionKind::FunctionT: {
          if (act->pos == 2) {
            //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
            // -> { fn pt -> rt :: {C, E, F} :: S, H}
            const Value* v =
                Value::MakeFunTypeVal(act->results[0], act->results[1]);
            frame->todo.Pop(2);
            frame->todo.Push(MakeValAct(v));
          } else {
            //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
            // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(exp->GetFunctionType().return_type));
          }
          break;
        }
        case ExpressionKind::Variable:
        case ExpressionKind::Integer:
        case ExpressionKind::Boolean:
        case ExpressionKind::IntT:
        case ExpressionKind::BoolT:
        case ExpressionKind::TypeT:
        case ExpressionKind::AutoT:
        case ExpressionKind::ContinuationT:
          std::cerr << "internal error, bad expression context in handle_value"
                    << std::endl;
          exit(-1);
      }
      break;
    }
    case ActionKind::StatementAction: {
      const Statement* stmt = act->u.stmt;
      switch (stmt->tag) {
        case StatementKind::ExpressionStatement:
          frame->todo.Pop(2);
          break;
        case StatementKind::VariableDefinition: {
          if (act->pos == 1) {
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(stmt->GetVariableDefinition().pat));
          } else if (act->pos == 2) {
            //    { { v :: (x = []) :: C, E, F} :: S, H}
            // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
            const Value* v = act->results[0];
            const Value* p = act->results[1];

            std::optional<Env> matches =
                PatternMatch(p, v, frame->scopes.Top()->values,
                             &frame->scopes.Top()->locals, stmt->line_num);
            if (!matches) {
              std::cerr
                  << stmt->line_num
                  << ": internal error in variable definition, match failed"
                  << std::endl;
              exit(-1);
            }
            frame->scopes.Top()->values = *matches;
            frame->todo.Pop(2);
          }
          break;
        }
        case StatementKind::Assign:
          if (act->pos == 1) {
            //    { { a :: ([] = e) :: C, E, F} :: S, H}
            // -> { { e :: (a = []) :: C, E, F} :: S, H}
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(stmt->GetAssign().rhs));
          } else if (act->pos == 2) {
            //    { { v :: (a = []) :: C, E, F} :: S, H}
            // -> { { C, E, F} :: S, H(a := v)}
            auto pat = act->results[0];
            auto val = act->results[1];
            PatternAssignment(pat, val, stmt->line_num);
            frame->todo.Pop(2);
          }
          break;
        case StatementKind::If:
          if (ValToBool(act->results[0], stmt->line_num)) {
            //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
            //      S, H}
            // -> { { then_stmt :: C, E, F } :: S, H}
            frame->todo.Pop(2);
            frame->todo.Push(MakeStmtAct(stmt->GetIf().then_stmt));
          } else if (stmt->GetIf().else_stmt) {
            //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
            //      S, H}
            // -> { { else_stmt :: C, E, F } :: S, H}
            frame->todo.Pop(2);
            frame->todo.Push(MakeStmtAct(stmt->GetIf().else_stmt));
          } else {
            frame->todo.Pop(2);
          }
          break;
        case StatementKind::While:
          if (ValToBool(act->results[0], stmt->line_num)) {
            //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
            // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
            frame->todo.Pop(1);
            frame->todo.Top()->pos = -1;
            frame->todo.Top()->results.clear();
            frame->todo.Push(MakeStmtAct(stmt->GetWhile().body));
          } else {
            //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
            // -> { { C, E, F } :: S, H}
            frame->todo.Pop(1);
            frame->todo.Top()->pos = -1;
            frame->todo.Top()->results.clear();
            frame->todo.Pop(1);
          }
          break;
        case StatementKind::Match: {
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
          if (clause_num >=
              static_cast<int>(stmt->GetMatch().clauses->size())) {
            frame->todo.Pop(2);
            break;
          }
          auto c = stmt->GetMatch().clauses->begin();
          std::advance(c, clause_num);

          if (act->pos % 2 == 1) {
            // start interpreting the pattern of the clause
            //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
            // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
            frame->todo.Pop(1);
            frame->todo.Push(MakeExpAct(c->first));
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
              Action* body_act = MakeStmtAct(body_block);
              body_act->pos = 0;
              frame->todo.Pop(2);
              frame->todo.Push(body_act);
              frame->todo.Push(MakeStmtAct(c->second));
            } else {
              // this case did not match, moving on
              act->pos++;
              clause_num = (act->pos - 1) / 2;
              if (clause_num <
                  static_cast<int>(stmt->GetMatch().clauses->size())) {
                // interpret the next clause
                c = stmt->GetMatch().clauses->begin();
                std::advance(c, clause_num);
                frame->todo.Pop(1);
                frame->todo.Push(MakeExpAct(c->first));
              } else {  // No more clauses in match
                frame->todo.Pop(2);
              }
            }
          }
          break;
        }
        case StatementKind::Return: {
          //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
          // -> { {v :: C', E', F'} :: S, H}
          const Value* ret_val = CopyVal(val_act->u.val, stmt->line_num);
          DeallocateLocals(stmt->line_num, frame);
          state->stack.Pop(1);
          frame = state->stack.Top();
          frame->todo.Push(MakeValAct(ret_val));
          break;
        }
        case StatementKind::Run: {
          frame->todo.Pop(2);
          // Push an expression statement action to ignore the result
          // value from the continuation.
          Action* ignore_result = MakeStmtAct(Statement::MakeExpStmt(
              stmt->line_num, Expression::MakeTuple(stmt->line_num, {})));
          ignore_result->pos = 0;
          frame->todo.Push(ignore_result);
          // Push the continuation onto the current stack.
          std::vector<Frame*> continuation_vector =
              ContinuationToVector(val_act->u.val, stmt->line_num);
          for (auto frame_iter = continuation_vector.rbegin();
               frame_iter != continuation_vector.rend(); ++frame_iter) {
            state->stack.Push(*frame_iter);
          }
          break;
        }
        case StatementKind::Continuation:
        case StatementKind::Await:
        case StatementKind::Block:
        case StatementKind::Sequence:
        case StatementKind::Break:
        case StatementKind::Continue:
          std::cerr << "internal error in handle_value, unhandled statement ";
          PrintStatement(stmt, 1);
          std::cerr << std::endl;
          exit(-1);
      }  // switch stmt
      break;
    }
    case ActionKind::ValAction:
      std::cerr << "internal error, ValAction in handle_value" << std::endl;
      exit(-1);
  }  // switch act
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
  switch (act->tag) {
    case ActionKind::DeleteTmpAction:
      std::cerr << "internal error in step, did not expect DeleteTmpAction"
                << std::endl;
      break;
    case ActionKind::ExpToLValAction:
      std::cerr << "internal error in step, did not expect ExpToLValAction"
                << std::endl;
      break;
    case ActionKind::ValAction:
      HandleValue();
      break;
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

  const Expression* arg = Expression::MakeTuple(0, {});
  const Expression* call_main =
      Expression::MakeCall(0, Expression::MakeVar(0, "main"), arg);
  auto todo = Stack(MakeExpAct(call_main));
  auto* scope = new Scope(globals, std::list<std::string>());
  auto* frame = new Frame("top", Stack(scope), todo);
  state->stack = Stack(frame);

  if (tracing_output) {
    std::cout << "********** calling main function **********" << std::endl;
    PrintState(std::cout);
  }

  while (state->stack.CountExceeds(1) ||
         state->stack.Top()->todo.CountExceeds(1) ||
         state->stack.Top()->todo.Top()->tag != ActionKind::ValAction) {
    Step();
    if (tracing_output) {
      PrintState(std::cout);
    }
  }
  const Value* v = state->stack.Top()->todo.Top()->u.val;
  return ValToInt(v, 0);
}

// Interpret an expression at compile-time.
auto InterpExp(Env values, const Expression* e) -> const Value* {
  auto todo = Stack(MakeExpAct(e));
  auto* scope = new Scope(values, std::list<std::string>());
  auto* frame = new Frame("InterpExp", Stack(scope), todo);
  state->stack = Stack(frame);

  while (state->stack.CountExceeds(1) ||
         state->stack.Top()->todo.CountExceeds(1) ||
         state->stack.Top()->todo.Top()->tag != ActionKind::ValAction) {
    Step();
  }
  const Value* v = state->stack.Top()->todo.Top()->u.val;
  return v;
}

}  // namespace Carbon
