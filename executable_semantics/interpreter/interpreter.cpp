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
  Ptr<Frame> frame = stack.Top();
  return frame->scopes.Top()->values;
}

// Returns the given name from the environment, printing an error if not found.
auto Interpreter::GetFromEnv(SourceLocation loc, const std::string& name)
    -> Address {
  std::optional<Address> pointer = CurrentEnv().Get(name);
  if (!pointer) {
    FATAL_RUNTIME_ERROR(loc) << "could not find `" << name << "`";
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
                           const std::vector<Ptr<const Value>>& args,
                           SourceLocation loc) -> Ptr<const Value> {
  switch (op) {
    case Operator::Neg:
      return arena.New<IntValue>(-cast<IntValue>(*args[0]).Val());
    case Operator::Add:
      return arena.New<IntValue>(cast<IntValue>(*args[0]).Val() +
                                 cast<IntValue>(*args[1]).Val());
    case Operator::Sub:
      return arena.New<IntValue>(cast<IntValue>(*args[0]).Val() -
                                 cast<IntValue>(*args[1]).Val());
    case Operator::Mul:
      return arena.New<IntValue>(cast<IntValue>(*args[0]).Val() *
                                 cast<IntValue>(*args[1]).Val());
    case Operator::Not:
      return arena.New<BoolValue>(!cast<BoolValue>(*args[0]).Val());
    case Operator::And:
      return arena.New<BoolValue>(cast<BoolValue>(*args[0]).Val() &&
                                  cast<BoolValue>(*args[1]).Val());
    case Operator::Or:
      return arena.New<BoolValue>(cast<BoolValue>(*args[0]).Val() ||
                                  cast<BoolValue>(*args[1]).Val());
    case Operator::Eq:
      return arena.New<BoolValue>(ValueEqual(args[0], args[1], loc));
    case Operator::Ptr:
      return arena.New<PointerType>(args[0]);
    case Operator::Deref:
      FATAL() << "dereference not implemented yet";
  }
}

void Interpreter::InitEnv(const Declaration& d, Env* env) {
  switch (d.Tag()) {
    case Declaration::Kind::FunctionDeclaration: {
      const FunctionDefinition& func_def =
          cast<FunctionDeclaration>(d).Definition();
      Env new_env = *env;
      // Bring the deduced parameters into scope.
      for (const auto& deduced : func_def.deduced_parameters) {
        Address a = heap.AllocateValue(arena.New<VariableType>(deduced.name));
        new_env.Set(deduced.name, a);
      }
      auto pt = InterpPattern(new_env, func_def.param_pattern);
      auto f = arena.New<FunctionValue>(func_def.name, pt, func_def.body);
      Address a = heap.AllocateValue(f);
      env->Set(func_def.name, a);
      break;
    }

    case Declaration::Kind::ClassDeclaration: {
      const ClassDefinition& class_def = cast<ClassDeclaration>(d).Definition();
      VarValues fields;
      VarValues methods;
      for (Ptr<const Member> m : class_def.members) {
        switch (m->Tag()) {
          case Member::Kind::FieldMember: {
            Ptr<const BindingPattern> binding = cast<FieldMember>(*m).Binding();
            Ptr<const Expression> type_expression =
                cast<ExpressionPattern>(*binding->Type()).Expression();
            auto type = InterpExp(Env(&arena), type_expression);
            fields.push_back(make_pair(*binding->Name(), type));
            break;
          }
        }
      }
      auto st = arena.New<ClassType>(class_def.name, std::move(fields),
                                     std::move(methods));
      auto a = heap.AllocateValue(st);
      env->Set(class_def.name, a);
      break;
    }

    case Declaration::Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(d);
      VarValues alts;
      for (const auto& [name, signature] : choice.Alternatives()) {
        auto t = InterpExp(Env(&arena), signature);
        alts.push_back(make_pair(name, t));
      }
      auto ct = arena.New<ChoiceType>(choice.Name(), std::move(alts));
      auto a = heap.AllocateValue(ct);
      env->Set(choice.Name(), a);
      break;
    }

    case Declaration::Kind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(d);
      // Adds an entry in `globals` mapping the variable's name to the
      // result of evaluating the initializer.
      auto v = InterpExp(*env, var.Initializer());
      Address a = heap.AllocateValue(v);
      env->Set(*var.Binding()->Name(), a);
      break;
    }
  }
}

void Interpreter::InitGlobals(const std::list<Ptr<const Declaration>>& fs) {
  for (const auto d : fs) {
    InitEnv(*d, &globals);
  }
}

void Interpreter::DeallocateScope(Ptr<Scope> scope) {
  for (const auto& l : scope->locals) {
    std::optional<Address> a = scope->values.Get(l);
    CHECK(a);
    heap.Deallocate(*a);
  }
}

void Interpreter::DeallocateLocals(Ptr<Frame> frame) {
  while (!frame->scopes.IsEmpty()) {
    DeallocateScope(frame->scopes.Top());
    frame->scopes.Pop();
  }
}

auto Interpreter::CreateTuple(Ptr<Action> act, Ptr<const Expression> exp)
    -> Ptr<const Value> {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  const auto& tup_lit = cast<TupleLiteral>(*exp);
  CHECK(act->Results().size() == tup_lit.Fields().size());
  std::vector<TupleElement> elements;
  for (size_t i = 0; i < act->Results().size(); ++i) {
    elements.push_back(
        {.name = tup_lit.Fields()[i].name, .value = act->Results()[i]});
  }

  return arena.New<TupleValue>(std::move(elements));
}

auto Interpreter::PatternMatch(Ptr<const Value> p, Ptr<const Value> v,
                               SourceLocation loc) -> std::optional<Env> {
  switch (p->Tag()) {
    case Value::Kind::BindingPlaceholderValue: {
      const auto& placeholder = cast<BindingPlaceholderValue>(*p);
      Env values(&arena);
      if (placeholder.Name().has_value()) {
        Address a = heap.AllocateValue(CopyVal(&arena, v, loc));
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
            FATAL_PROGRAM_ERROR(loc)
                << "arity mismatch in tuple pattern match:\n  pattern: "
                << p_tup << "\n  value: " << v_tup;
          }
          Env values(&arena);
          for (size_t i = 0; i < p_tup.Elements().size(); ++i) {
            if (p_tup.Elements()[i].name != v_tup.Elements()[i].name) {
              FATAL_PROGRAM_ERROR(loc)
                  << "Tuple field name '" << v_tup.Elements()[i].name
                  << "' does not match pattern field name '"
                  << p_tup.Elements()[i].name << "'";
            }
            std::optional<Env> matches = PatternMatch(
                p_tup.Elements()[i].value, v_tup.Elements()[i].value, loc);
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
          return PatternMatch(p_alt.Argument(), v_alt.Argument(), loc);
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
              PatternMatch(p_fn.Param(), v_fn.Param(), loc);
          if (!param_matches) {
            return std::nullopt;
          }
          std::optional<Env> ret_matches =
              PatternMatch(p_fn.Ret(), v_fn.Ret(), loc);
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
      return Env(&arena);
    default:
      if (ValueEqual(p, v, loc)) {
        return Env(&arena);
      } else {
        return std::nullopt;
      }
  }
}

void Interpreter::PatternAssignment(Ptr<const Value> pat, Ptr<const Value> val,
                                    SourceLocation loc) {
  switch (pat->Tag()) {
    case Value::Kind::PointerValue:
      heap.Write(cast<PointerValue>(*pat).Val(), CopyVal(&arena, val, loc),
                 loc);
      break;
    case Value::Kind::TupleValue: {
      switch (val->Tag()) {
        case Value::Kind::TupleValue: {
          const auto& pat_tup = cast<TupleValue>(*pat);
          const auto& val_tup = cast<TupleValue>(*val);
          if (pat_tup.Elements().size() != val_tup.Elements().size()) {
            FATAL_RUNTIME_ERROR(loc)
                << "arity mismatch in tuple pattern assignment:\n  pattern: "
                << pat_tup << "\n  value: " << val_tup;
          }
          for (const TupleElement& pattern_element : pat_tup.Elements()) {
            std::optional<Ptr<const Value>> value_field =
                val_tup.FindField(pattern_element.name);
            if (!value_field) {
              FATAL_RUNTIME_ERROR(loc)
                  << "field " << pattern_element.name << "not in " << *val;
            }
            PatternAssignment(pattern_element.value, *value_field, loc);
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
          PatternAssignment(pat_alt.Argument(), val_alt.Argument(), loc);
          break;
        }
        default:
          FATAL() << "expected an alternative in left-hand-side, not " << *val;
      }
      break;
    }
    default:
      CHECK(ValueEqual(pat, val, loc))
          << "internal error in pattern assignment";
  }
}

auto Interpreter::StepLvalue() -> Transition {
  Ptr<Action> act = stack.Top()->todo.Top();
  Ptr<const Expression> exp = cast<LValAction>(*act).Exp();
  if (tracing_output) {
    llvm::outs() << "--- step lvalue " << *exp << " --->\n";
  }
  switch (exp->Tag()) {
    case Expression::Kind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      Address pointer =
          GetFromEnv(exp->SourceLoc(), cast<IdentifierExpression>(*exp).Name());
      Ptr<const Value> v = arena.New<PointerValue>(pointer);
      return Done{v};
    }
    case Expression::Kind::FieldAccessExpression: {
      if (act->Pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        return Spawn{arena.New<LValAction>(
            cast<FieldAccessExpression>(*exp).Aggregate())};
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->Results()[0]).Val();
        Address field = aggregate.SubobjectAddress(
            cast<FieldAccessExpression>(*exp).Field());
        return Done{arena.New<PointerValue>(field)};
      }
    }
    case Expression::Kind::IndexExpression: {
      if (act->Pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{
            arena.New<LValAction>(cast<IndexExpression>(*exp).Aggregate())};

      } else if (act->Pos() == 1) {
        return Spawn{
            arena.New<ExpressionAction>(cast<IndexExpression>(*exp).Offset())};
      } else {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->Results()[0]).Val();
        std::string f =
            std::to_string(cast<IntValue>(*act->Results()[1]).Val());
        Address field = aggregate.SubobjectAddress(f);
        return Done{arena.New<PointerValue>(field)};
      }
    }
    case Expression::Kind::TupleLiteral: {
      if (act->Pos() == 0) {
        //    { {(f1=e1,...) :: C, E, F} :: S, H}
        // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
        Ptr<const Expression> e1 =
            cast<TupleLiteral>(*exp).Fields()[0].expression;
        return Spawn{arena.New<LValAction>(e1)};
      } else if (act->Pos() !=
                 static_cast<int>(cast<TupleLiteral>(*exp).Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        Ptr<const Expression> elt =
            cast<TupleLiteral>(*exp).Fields()[act->Pos()].expression;
        return Spawn{arena.New<LValAction>(elt)};
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

auto Interpreter::StepExp() -> Transition {
  Ptr<Action> act = stack.Top()->todo.Top();
  Ptr<const Expression> exp = cast<ExpressionAction>(*act).Exp();
  if (tracing_output) {
    llvm::outs() << "--- step exp " << *exp << " --->\n";
  }
  switch (exp->Tag()) {
    case Expression::Kind::IndexExpression: {
      if (act->Pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{arena.New<ExpressionAction>(
            cast<IndexExpression>(*exp).Aggregate())};
      } else if (act->Pos() == 1) {
        return Spawn{
            arena.New<ExpressionAction>(cast<IndexExpression>(*exp).Offset())};
      } else {
        //    { { v :: [][i] :: C, E, F} :: S, H}
        // -> { { v_i :: C, E, F} : S, H}
        auto* tuple = dyn_cast<TupleValue>(act->Results()[0].Get());
        if (tuple == nullptr) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "expected a tuple in field access, not " << *act->Results()[0];
        }
        std::string f =
            std::to_string(cast<IntValue>(*act->Results()[1]).Val());
        std::optional<Ptr<const Value>> field = tuple->FindField(f);
        if (!field) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "field " << f << " not in " << *tuple;
        }
        return Done{*field};
      }
    }
    case Expression::Kind::TupleLiteral: {
      if (act->Pos() == 0) {
        if (cast<TupleLiteral>(*exp).Fields().size() > 0) {
          //    { {(f1=e1,...) :: C, E, F} :: S, H}
          // -> { {e1 :: (f1=[],...) :: C, E, F} :: S, H}
          Ptr<const Expression> e1 =
              cast<TupleLiteral>(*exp).Fields()[0].expression;
          return Spawn{arena.New<ExpressionAction>(e1)};
        } else {
          return Done{CreateTuple(act, exp)};
        }
      } else if (act->Pos() !=
                 static_cast<int>(cast<TupleLiteral>(*exp).Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        Ptr<const Expression> elt =
            cast<TupleLiteral>(*exp).Fields()[act->Pos()].expression;
        return Spawn{arena.New<ExpressionAction>(elt)};
      } else {
        return Done{CreateTuple(act, exp)};
      }
    }
    case Expression::Kind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(*exp);
      if (act->Pos() == 0) {
        //    { { e.f :: C, E, F} :: S, H}
        // -> { { e :: [].f :: C, E, F} :: S, H}
        return Spawn{arena.New<ExpressionAction>(access.Aggregate())};
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        return Done{act->Results()[0]->GetField(
            &arena, FieldPath(access.Field()), exp->SourceLoc())};
      }
    }
    case Expression::Kind::IdentifierExpression: {
      CHECK(act->Pos() == 0);
      const auto& ident = cast<IdentifierExpression>(*exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp->SourceLoc(), ident.Name());
      return Done{heap.Read(pointer, exp->SourceLoc())};
    }
    case Expression::Kind::IntLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena.New<IntValue>(cast<IntLiteral>(*exp).Val())};
    case Expression::Kind::BoolLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena.New<BoolValue>(cast<BoolLiteral>(*exp).Val())};
    case Expression::Kind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(*exp);
      if (act->Pos() != static_cast<int>(op.Arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        Ptr<const Expression> arg = op.Arguments()[act->Pos()];
        return Spawn{arena.New<ExpressionAction>(arg)};
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        return Done{EvalPrim(op.Op(), act->Results(), exp->SourceLoc())};
      }
    }
    case Expression::Kind::CallExpression:
      if (act->Pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        return Spawn{
            arena.New<ExpressionAction>(cast<CallExpression>(*exp).Function())};
      } else if (act->Pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        return Spawn{
            arena.New<ExpressionAction>(cast<CallExpression>(*exp).Argument())};
      } else if (act->Pos() == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        switch (act->Results()[0]->Tag()) {
          case Value::Kind::ClassType: {
            Ptr<const Value> arg =
                CopyVal(&arena, act->Results()[1], exp->SourceLoc());
            return Done{arena.New<StructValue>(act->Results()[0], arg)};
          }
          case Value::Kind::AlternativeConstructorValue: {
            const auto& alt =
                cast<AlternativeConstructorValue>(*act->Results()[0]);
            Ptr<const Value> arg =
                CopyVal(&arena, act->Results()[1], exp->SourceLoc());
            return Done{arena.New<AlternativeValue>(alt.AltName(),
                                                    alt.ChoiceName(), arg)};
          }
          case Value::Kind::FunctionValue:
            return CallFunction{
                // TODO: Think about a cleaner way to cast between Ptr types.
                // (multiple TODOs)
                .function = Ptr<const FunctionValue>(
                    cast<FunctionValue>(act->Results()[0].Get())),
                .args = act->Results()[1],
                .loc = exp->SourceLoc()};
          default:
            FATAL_RUNTIME_ERROR(exp->SourceLoc())
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
          Address pointer = GetFromEnv(exp->SourceLoc(), "format_str");
          Ptr<const Value> pointee = heap.Read(pointer, exp->SourceLoc());
          CHECK(pointee->Tag() == Value::Kind::StringValue);
          // TODO: This could eventually use something like llvm::formatv.
          llvm::outs() << cast<StringValue>(*pointee).Val();
          return Done{TupleValue::Empty()};
      }

    case Expression::Kind::IntTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{arena.New<IntType>()};
    }
    case Expression::Kind::BoolTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{arena.New<BoolType>()};
    }
    case Expression::Kind::TypeTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{arena.New<TypeType>()};
    }
    case Expression::Kind::FunctionTypeLiteral: {
      if (act->Pos() == 0) {
        return Spawn{arena.New<ExpressionAction>(
            cast<FunctionTypeLiteral>(*exp).Parameter())};
      } else if (act->Pos() == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        return Spawn{arena.New<ExpressionAction>(
            cast<FunctionTypeLiteral>(*exp).ReturnType())};
      } else {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        return Done{arena.New<FunctionType>(std::vector<GenericBinding>(),
                                            act->Results()[0],
                                            act->Results()[1])};
      }
    }
    case Expression::Kind::ContinuationTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{arena.New<ContinuationType>()};
    }
    case Expression::Kind::StringLiteral:
      CHECK(act->Pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena.New<StringValue>(cast<StringLiteral>(*exp).Val())};
    case Expression::Kind::StringTypeLiteral: {
      CHECK(act->Pos() == 0);
      return Done{arena.New<StringType>()};
    }
  }  // switch (exp->Tag)
}

auto Interpreter::StepPattern() -> Transition {
  Ptr<Action> act = stack.Top()->todo.Top();
  Ptr<const Pattern> pattern = cast<PatternAction>(*act).Pat();
  if (tracing_output) {
    llvm::outs() << "--- step pattern " << *pattern << " --->\n";
  }
  switch (pattern->Tag()) {
    case Pattern::Kind::AutoPattern: {
      CHECK(act->Pos() == 0);
      return Done{arena.New<AutoType>()};
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(*pattern);
      if (act->Pos() == 0) {
        return Spawn{arena.New<PatternAction>(binding.Type())};
      } else {
        return Done{arena.New<BindingPlaceholderValue>(binding.Name(),
                                                       act->Results()[0])};
      }
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(*pattern);
      if (act->Pos() == 0) {
        if (tuple.Fields().empty()) {
          return Done{TupleValue::Empty()};
        } else {
          Ptr<const Pattern> p1 = tuple.Fields()[0].pattern;
          return Spawn{(arena.New<PatternAction>(p1))};
        }
      } else if (act->Pos() != static_cast<int>(tuple.Fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        Ptr<const Pattern> elt = tuple.Fields()[act->Pos()].pattern;
        return Spawn{arena.New<PatternAction>(elt)};
      } else {
        std::vector<TupleElement> elements;
        for (size_t i = 0; i < tuple.Fields().size(); ++i) {
          elements.push_back(
              {.name = tuple.Fields()[i].name, .value = act->Results()[i]});
        }
        return Done{arena.New<TupleValue>(std::move(elements))};
      }
    }
    case Pattern::Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(*pattern);
      if (act->Pos() == 0) {
        return Spawn{arena.New<ExpressionAction>(alternative.ChoiceType())};
      } else if (act->Pos() == 1) {
        return Spawn{arena.New<PatternAction>(alternative.Arguments())};
      } else {
        CHECK(act->Pos() == 2);
        const auto& choice_type = cast<ChoiceType>(*act->Results()[0]);
        return Done{arena.New<AlternativeValue>(alternative.AlternativeName(),
                                                choice_type.Name(),
                                                act->Results()[1])};
      }
    }
    case Pattern::Kind::ExpressionPattern:
      return Delegate{arena.New<ExpressionAction>(
          cast<ExpressionPattern>(*pattern).Expression())};
  }
}

static auto IsWhileAct(Ptr<Action> act) -> bool {
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

static auto HasLocalScope(Ptr<Action> act) -> bool {
  switch (act->Tag()) {
    case Action::Kind::StatementAction:
      switch (cast<StatementAction>(*act).Stmt()->Tag()) {
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
  Ptr<Frame> frame = stack.Top();
  Ptr<Action> act = frame->todo.Top();
  Ptr<const Statement> stmt = cast<StatementAction>(*act).Stmt();
  if (tracing_output) {
    llvm::outs() << "--- step stmt ";
    stmt->PrintDepth(1, llvm::outs());
    llvm::outs() << " --->\n";
  }
  switch (stmt->Tag()) {
    case Statement::Kind::Match: {
      const auto& match_stmt = cast<Match>(*stmt);
      if (act->Pos() == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        frame->scopes.Push(arena.New<Scope>(CurrentEnv()));
        return Spawn{arena.New<ExpressionAction>(match_stmt.Exp())};
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
        if (clause_num >= static_cast<int>(match_stmt.Clauses().size())) {
          DeallocateScope(frame->scopes.Top());
          frame->scopes.Pop();
          return Done{};
        }
        auto c = match_stmt.Clauses().begin();
        std::advance(c, clause_num);

        if (act->Pos() % 2 == 1) {
          // start interpreting the pattern of the clause
          //    { {v :: (match ([]) ...) :: C, E, F} :: S, H}
          // -> { {pi :: (match ([]) ...) :: C, E, F} :: S, H}
          return Spawn{arena.New<PatternAction>(c->first)};
        } else {  // try to match
          auto v = act->Results()[0];
          auto pat = act->Results()[clause_num + 1];
          std::optional<Env> matches = PatternMatch(pat, v, stmt->SourceLoc());
          if (matches) {  // we have a match, start the body
            // Ensure we don't process any more clauses.
            act->SetPos(2 * match_stmt.Clauses().size() + 1);

            for (const auto& [name, value] : *matches) {
              frame->scopes.Top()->values.Set(name, value);
              frame->scopes.Top()->locals.push_back(name);
            }
            return Spawn{arena.New<StatementAction>(c->second)};
          } else {
            return RunAgain{};
          }
        }
      }
    }
    case Statement::Kind::While:
      if (act->Pos() % 2 == 0) {
        //    { { (while (e) s) :: C, E, F} :: S, H}
        // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
        act->Clear();
        return Spawn{arena.New<ExpressionAction>(cast<While>(*stmt).Cond())};
      } else if (cast<BoolValue>(*act->Results().back()).Val()) {
        //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
        // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
        return Spawn{arena.New<StatementAction>(cast<While>(*stmt).Body())};
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
        FATAL_RUNTIME_ERROR(stmt->SourceLoc())
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
        FATAL_RUNTIME_ERROR(stmt->SourceLoc())
            << "`continue` not inside `while` statement";
      }
      return UnwindTo{*it};
    }
    case Statement::Kind::Block: {
      if (act->Pos() == 0) {
        const Block& block = cast<Block>(*stmt);
        if (block.Stmt()) {
          frame->scopes.Push(arena.New<Scope>(CurrentEnv()));
          return Spawn{arena.New<StatementAction>(*block.Stmt())};
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
        return Spawn{arena.New<ExpressionAction>(
            cast<VariableDefinition>(*stmt).Init())};
      } else if (act->Pos() == 1) {
        return Spawn{
            arena.New<PatternAction>(cast<VariableDefinition>(*stmt).Pat())};
      } else {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        Ptr<const Value> v = act->Results()[0];
        Ptr<const Value> p = act->Results()[1];

        std::optional<Env> matches = PatternMatch(p, v, stmt->SourceLoc());
        CHECK(matches)
            << stmt->SourceLoc()
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
        return Spawn{arena.New<ExpressionAction>(
            cast<ExpressionStatement>(*stmt).Exp())};
      } else {
        return Done{};
      }
    case Statement::Kind::Assign:
      if (act->Pos() == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        return Spawn{arena.New<LValAction>(cast<Assign>(*stmt).Lhs())};
      } else if (act->Pos() == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        return Spawn{arena.New<ExpressionAction>(cast<Assign>(*stmt).Rhs())};
      } else {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        auto pat = act->Results()[0];
        auto val = act->Results()[1];
        PatternAssignment(pat, val, stmt->SourceLoc());
        return Done{};
      }
    case Statement::Kind::If:
      if (act->Pos() == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        return Spawn{arena.New<ExpressionAction>(cast<If>(*stmt).Cond())};
      } else if (cast<BoolValue>(*act->Results()[0]).Val()) {
        //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { then_stmt :: C, E, F } :: S, H}
        return Delegate{arena.New<StatementAction>(cast<If>(*stmt).ThenStmt())};
      } else if (cast<If>(*stmt).ElseStmt()) {
        //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
        //      S, H}
        // -> { { else_stmt :: C, E, F } :: S, H}
        return Delegate{
            arena.New<StatementAction>(*cast<If>(*stmt).ElseStmt())};
      } else {
        return Done{};
      }
    case Statement::Kind::Return:
      if (act->Pos() == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        return Spawn{arena.New<ExpressionAction>(cast<Return>(*stmt).Exp())};
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        Ptr<const Value> ret_val =
            CopyVal(&arena, act->Results()[0], stmt->SourceLoc());
        return UnwindFunctionCall{ret_val};
      }
    case Statement::Kind::Sequence: {
      //    { { (s1,s2) :: C, E, F} :: S, H}
      // -> { { s1 :: s2 :: C, E, F} :: S, H}
      const Sequence& seq = cast<Sequence>(*stmt);
      if (act->Pos() == 0) {
        return Spawn{arena.New<StatementAction>(seq.Stmt())};
      } else {
        if (seq.Next()) {
          return Delegate{
              arena.New<StatementAction>(*cast<Sequence>(*stmt).Next())};
        } else {
          return Done{};
        }
      }
    }
    case Statement::Kind::Continuation: {
      CHECK(act->Pos() == 0);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      auto scopes = Stack<Ptr<Scope>>(arena.New<Scope>(CurrentEnv()));
      Stack<Ptr<Action>> todo;
      todo.Push(arena.New<StatementAction>(
          arena.New<Return>(&arena, stmt->SourceLoc())));
      todo.Push(arena.New<StatementAction>(cast<Continuation>(*stmt).Body()));
      auto continuation_frame =
          arena.New<Frame>("__continuation", scopes, todo);
      Address continuation_address =
          heap.AllocateValue(arena.New<ContinuationValue>(
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
        return Spawn{arena.New<ExpressionAction>(cast<Run>(*stmt).Argument())};
      } else {
        frame->todo.Pop(1);
        // Push an expression statement action to ignore the result
        // value from the continuation.
        auto ignore_result =
            arena.New<StatementAction>(arena.New<ExpressionStatement>(
                stmt->SourceLoc(), arena.New<TupleLiteral>(stmt->SourceLoc())));
        frame->todo.Push(ignore_result);
        // Push the continuation onto the current stack.
        const std::vector<Ptr<Frame>>& continuation_vector =
            cast<ContinuationValue>(*act->Results()[0]).Stack();
        for (auto frame_iter = continuation_vector.rbegin();
             frame_iter != continuation_vector.rend(); ++frame_iter) {
          stack.Push(*frame_iter);
        }
        return ManualTransition{};
      }
    case Statement::Kind::Await:
      CHECK(act->Pos() == 0);
      // Pause the current continuation
      frame->todo.Pop();
      std::vector<Ptr<Frame>> paused;
      do {
        paused.push_back(stack.Pop());
      } while (paused.back()->continuation == std::nullopt);
      // Update the continuation with the paused stack.
      heap.Write(*paused.back()->continuation,
                 arena.New<ContinuationValue>(paused), stmt->SourceLoc());
      return ManualTransition{};
  }
}

class Interpreter::DoTransition {
 public:
  // Does not take ownership of interpreter.
  DoTransition(Interpreter* interpreter) : interpreter(interpreter) {}

  void operator()(const Done& done) {
    Ptr<Frame> frame = interpreter->stack.Top();
    if (frame->todo.Top()->Tag() != Action::Kind::StatementAction) {
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
    Ptr<Frame> frame = interpreter->stack.Top();
    Ptr<Action> action = frame->todo.Top();
    action->SetPos(action->Pos() + 1);
    frame->todo.Push(spawn.child);
  }

  void operator()(const Delegate& delegate) {
    Ptr<Frame> frame = interpreter->stack.Top();
    frame->todo.Pop();
    frame->todo.Push(delegate.delegate);
  }

  void operator()(const RunAgain&) {
    Ptr<Action> action = interpreter->stack.Top()->todo.Top();
    action->SetPos(action->Pos() + 1);
  }

  void operator()(const UnwindTo& unwind_to) {
    Ptr<Frame> frame = interpreter->stack.Top();
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
    std::optional<Env> matches =
        interpreter->PatternMatch(call.function->Param(), call.args, call.loc);
    CHECK(matches.has_value())
        << "internal error in call_function, pattern match failed";
    // Create the new frame and push it on the stack
    Env values = interpreter->globals;
    std::list<std::string> params;
    for (const auto& [name, value] : *matches) {
      values.Set(name, value);
      params.push_back(name);
    }
    auto scopes =
        Stack<Ptr<Scope>>(interpreter->arena.New<Scope>(values, params));
    CHECK(call.function->Body()) << "Calling a function that's missing a body";
    auto todo = Stack<Ptr<Action>>(
        interpreter->arena.New<StatementAction>(*call.function->Body()));
    auto frame =
        interpreter->arena.New<Frame>(call.function->Name(), scopes, todo);
    interpreter->stack.Push(frame);
  }

  void operator()(const ManualTransition&) {}

 private:
  Ptr<Interpreter> interpreter;
};

// State transition.
void Interpreter::Step() {
  Ptr<Frame> frame = stack.Top();
  if (frame->todo.IsEmpty()) {
    FATAL_RUNTIME_ERROR_NO_LINE()
        << "fell off end of function " << frame->name << " without `return`";
  }

  Ptr<Action> act = frame->todo.Top();
  switch (act->Tag()) {
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

auto Interpreter::InterpProgram(const std::list<Ptr<const Declaration>>& fs)
    -> int {
  // Check that the interpreter is in a clean state.
  CHECK(globals.IsEmpty());
  CHECK(stack.IsEmpty());
  CHECK(program_value == std::nullopt);

  if (tracing_output) {
    llvm::outs() << "********** initializing globals **********\n";
  }
  InitGlobals(fs);

  SourceLocation loc("<InterpProgram()>", 0);

  Ptr<const Expression> arg = arena.New<TupleLiteral>(loc);
  Ptr<const Expression> call_main = arena.New<CallExpression>(
      loc, arena.New<IdentifierExpression>(loc, "main"), arg);
  auto todo = Stack<Ptr<Action>>(arena.New<ExpressionAction>(call_main));
  auto scopes = Stack<Ptr<Scope>>(arena.New<Scope>(globals));
  stack = Stack<Ptr<Frame>>(arena.New<Frame>("top", scopes, todo));

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
  return cast<IntValue>(**program_value).Val();
}

auto Interpreter::InterpExp(Env values, Ptr<const Expression> e)
    -> Ptr<const Value> {
  CHECK(program_value == std::nullopt);
  auto program_value_guard =
      llvm::make_scope_exit([&] { program_value = std::nullopt; });
  auto todo = Stack<Ptr<Action>>(arena.New<ExpressionAction>(e));
  auto scopes = Stack<Ptr<Scope>>(arena.New<Scope>(values));
  stack = Stack<Ptr<Frame>>(arena.New<Frame>("InterpExp", scopes, todo));

  while (stack.Count() > 1 || !stack.Top()->todo.IsEmpty()) {
    Step();
  }
  CHECK(program_value != std::nullopt);
  return *program_value;
}

auto Interpreter::InterpPattern(Env values, Ptr<const Pattern> p)
    -> Ptr<const Value> {
  CHECK(program_value == std::nullopt);
  auto program_value_guard =
      llvm::make_scope_exit([&] { program_value = std::nullopt; });
  auto todo = Stack<Ptr<Action>>(arena.New<PatternAction>(p));
  auto scopes = Stack<Ptr<Scope>>(arena.New<Scope>(values));
  stack = Stack<Ptr<Frame>>(arena.New<Frame>("InterpPattern", scopes, todo));

  while (stack.Count() > 1 || !stack.Top()->todo.IsEmpty()) {
    Step();
  }
  CHECK(program_value != std::nullopt);
  return *program_value;
}

}  // namespace Carbon
