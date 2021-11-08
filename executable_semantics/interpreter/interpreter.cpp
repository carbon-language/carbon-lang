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
#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/common/arena.h"
#include "executable_semantics/common/error.h"
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
  for (const auto& [name, allocation] : values) {
    out << sep << name << ": ";
    heap_.PrintAllocation(allocation, out);
  }
}

//
// State Operations
//

auto Interpreter::CurrentScope() -> Scope& {
  for (Nonnull<Action*> action : todo_) {
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
  std::optional<AllocationId> pointer = CurrentEnv().Get(name);
  if (!pointer) {
    FATAL_RUNTIME_ERROR(source_loc) << "could not find `" << name << "`";
  }
  return Address(*pointer);
}

void Interpreter::PrintState(llvm::raw_ostream& out) {
  out << "{\nstack: ";
  llvm::ListSeparator sep(" :: ");
  for (Nonnull<const Action*> action : todo_) {
    out << sep << *action;
  }
  out << "\nheap: " << heap_;
  if (!todo_.IsEmpty()) {
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
      return arena_->New<IntValue>(-cast<IntValue>(*args[0]).value());
    case Operator::Add:
      return arena_->New<IntValue>(cast<IntValue>(*args[0]).value() +
                                   cast<IntValue>(*args[1]).value());
    case Operator::Sub:
      return arena_->New<IntValue>(cast<IntValue>(*args[0]).value() -
                                   cast<IntValue>(*args[1]).value());
    case Operator::Mul:
      return arena_->New<IntValue>(cast<IntValue>(*args[0]).value() *
                                   cast<IntValue>(*args[1]).value());
    case Operator::Not:
      return arena_->New<BoolValue>(!cast<BoolValue>(*args[0]).value());
    case Operator::And:
      return arena_->New<BoolValue>(cast<BoolValue>(*args[0]).value() &&
                                    cast<BoolValue>(*args[1]).value());
    case Operator::Or:
      return arena_->New<BoolValue>(cast<BoolValue>(*args[0]).value() ||
                                    cast<BoolValue>(*args[1]).value());
    case Operator::Eq:
      return arena_->New<BoolValue>(ValueEqual(args[0], args[1], source_loc));
    case Operator::Ptr:
      return arena_->New<PointerType>(args[0]);
    case Operator::Deref:
      FATAL() << "dereference not implemented yet";
  }
}

void Interpreter::InitEnv(const Declaration& d, Env* env) {
  switch (d.kind()) {
    case Declaration::Kind::FunctionDeclaration: {
      const auto& func_def = cast<FunctionDeclaration>(d);
      Env new_env = *env;
      // Bring the deduced parameters into scope.
      for (Nonnull<const GenericBinding*> deduced :
           func_def.deduced_parameters()) {
        AllocationId a =
            heap_.AllocateValue(arena_->New<VariableType>(deduced->name()));
        new_env.Set(deduced->name(), a);
      }
      Nonnull<const FunctionValue*> f = arena_->New<FunctionValue>(&func_def);
      AllocationId a = heap_.AllocateValue(f);
      env->Set(func_def.name(), a);
      break;
    }

    case Declaration::Kind::ClassDeclaration: {
      const ClassDefinition& class_def = cast<ClassDeclaration>(d).definition();
      std::vector<NamedValue> fields;
      std::vector<NamedValue> methods;
      for (Nonnull<const Member*> m : class_def.members()) {
        switch (m->kind()) {
          case Member::Kind::FieldMember: {
            const BindingPattern& binding = cast<FieldMember>(*m).binding();
            const Expression& type_expression =
                cast<ExpressionPattern>(binding.type()).expression();
            auto type = InterpExp(Env(arena_), &type_expression);
            fields.push_back({.name = *binding.name(), .value = type});
            break;
          }
        }
      }
      auto st = arena_->New<NominalClassType>(
          class_def.name(), std::move(fields), std::move(methods));
      AllocationId a = heap_.AllocateValue(st);
      env->Set(class_def.name(), a);
      break;
    }

    case Declaration::Kind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(d);
      std::vector<NamedValue> alts;
      for (Nonnull<const ChoiceDeclaration::Alternative*> alternative :
           choice.alternatives()) {
        auto t = InterpExp(Env(arena_), &alternative->signature());
        alts.push_back({.name = alternative->name(), .value = t});
      }
      auto ct = arena_->New<ChoiceType>(choice.name(), std::move(alts));
      AllocationId a = heap_.AllocateValue(ct);
      env->Set(choice.name(), a);
      break;
    }

    case Declaration::Kind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(d);
      // Adds an entry in `globals` mapping the variable's name to the
      // result of evaluating the initializer.
      Nonnull<const Value*> v =
          Convert(InterpExp(*env, &var.initializer()), &var.static_type());
      AllocationId a = heap_.AllocateValue(v);
      env->Set(*var.binding().name(), a);
      break;
    }
  }
}

void Interpreter::InitGlobals(llvm::ArrayRef<Nonnull<Declaration*>> fs) {
  for (const auto d : fs) {
    InitEnv(*d, &globals_);
  }
}

auto Interpreter::UnwindTodoTop() -> Nonnull<Action*> {
  Nonnull<Action*> act = todo_.Pop();
  if (act->scope().has_value()) {
    CHECK(!act->scope()->deallocated);
    for (const auto& l : act->scope()->locals) {
      std::optional<AllocationId> a = act->scope()->values.Get(l);
      CHECK(a);
      heap_.Deallocate(*a);
    }
    act->scope()->deallocated = true;
  }
  return act;
}

auto Interpreter::CreateTuple(Nonnull<Action*> act,
                              Nonnull<const Expression*> exp)
    -> Nonnull<const Value*> {
  //    { { (v1,...,vn) :: C, E, F} :: S, H}
  // -> { { `(v1,...,vn) :: C, E, F} :: S, H}
  const auto& tup_lit = cast<TupleLiteral>(*exp);
  CHECK(act->results().size() == tup_lit.fields().size());
  return arena_->New<TupleValue>(act->results());
}

auto Interpreter::CreateStruct(const std::vector<FieldInitializer>& fields,
                               const std::vector<Nonnull<const Value*>>& values)
    -> Nonnull<const Value*> {
  CHECK(fields.size() == values.size());
  std::vector<NamedValue> elements;
  for (size_t i = 0; i < fields.size(); ++i) {
    elements.push_back({.name = fields[i].name(), .value = values[i]});
  }

  return arena_->New<StructValue>(std::move(elements));
}

auto Interpreter::PatternMatch(Nonnull<const Value*> p, Nonnull<const Value*> v,
                               SourceLocation source_loc)
    -> std::optional<Env> {
  switch (p->kind()) {
    case Value::Kind::BindingPlaceholderValue: {
      const auto& placeholder = cast<BindingPlaceholderValue>(*p);
      Env values(arena_);
      if (placeholder.name().has_value()) {
        AllocationId a = heap_.AllocateValue(v);
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
          Env values(arena_);
          for (size_t i = 0; i < p_tup.elements().size(); ++i) {
            std::optional<Env> matches = PatternMatch(
                p_tup.elements()[i], v_tup.elements()[i], source_loc);
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
      Env values(arena_);
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
      return Env(arena_);
    default:
      if (ValueEqual(p, v, source_loc)) {
        return Env(arena_);
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
      heap_.Write(cast<PointerValue>(*pat).value(), val, source_loc);
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
          for (size_t i = 0; i < pat_tup.elements().size(); ++i) {
            PatternAssignment(pat_tup.elements()[i], val_tup.elements()[i],
                              source_loc);
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
  Nonnull<Action*> act = todo_.Top();
  const Expression& exp = cast<LValAction>(*act).expression();
  if (trace_) {
    llvm::outs() << "--- step lvalue " << exp << " (" << exp.source_loc()
                 << ") --->\n";
  }
  switch (exp.kind()) {
    case Expression::Kind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      Address pointer =
          GetFromEnv(exp.source_loc(), cast<IdentifierExpression>(exp).name());
      Nonnull<const Value*> v = arena_->New<PointerValue>(pointer);
      return Done{v};
    }
    case Expression::Kind::FieldAccessExpression: {
      if (act->pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        return Spawn{arena_->New<LValAction>(
            &cast<FieldAccessExpression>(exp).aggregate())};
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->results()[0]).value();
        Address field = aggregate.SubobjectAddress(
            cast<FieldAccessExpression>(exp).field());
        return Done{arena_->New<PointerValue>(field)};
      }
    }
    case Expression::Kind::IndexExpression: {
      if (act->pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{
            arena_->New<LValAction>(&cast<IndexExpression>(exp).aggregate())};

      } else if (act->pos() == 1) {
        return Spawn{arena_->New<ExpressionAction>(
            &cast<IndexExpression>(exp).offset())};
      } else {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = cast<PointerValue>(*act->results()[0]).value();
        std::string f =
            std::to_string(cast<IntValue>(*act->results()[1]).value());
        Address field = aggregate.SubobjectAddress(f);
        return Done{arena_->New<PointerValue>(field)};
      }
    }
    case Expression::Kind::TupleLiteral: {
      if (act->pos() <
          static_cast<int>(cast<TupleLiteral>(exp).fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        return Spawn{arena_->New<LValAction>(
            cast<TupleLiteral>(exp).fields()[act->pos()])};
      } else {
        return Done{CreateTuple(act, &exp)};
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
          << "Can't treat expression as lvalue: " << exp;
  }
}

auto Interpreter::Convert(Nonnull<const Value*> value,
                          Nonnull<const Value*> destination_type) const
    -> Nonnull<const Value*> {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::PointerValue:
    case Value::Kind::BoolValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::AutoType:
    case Value::Kind::StructType:
    case Value::Kind::NominalClassType:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringType:
    case Value::Kind::StringValue:
      // TODO: add `CHECK(TypeEqual(type, value->dynamic_type()))`, once we
      // have Value::dynamic_type.
      return value;
    case Value::Kind::StructValue: {
      const auto& struct_val = cast<StructValue>(*value);
      switch (destination_type->kind()) {
        case Value::Kind::StructType: {
          const auto& destination_struct_type =
              cast<StructType>(*destination_type);
          std::vector<NamedValue> new_elements;
          for (const auto& [field_name, field_type] :
               destination_struct_type.fields()) {
            std::optional<Nonnull<const Value*>> old_value =
                struct_val.FindField(field_name);
            new_elements.push_back(
                {.name = field_name, .value = Convert(*old_value, field_type)});
          }
          return arena_->New<StructValue>(std::move(new_elements));
        }
        case Value::Kind::NominalClassType:
          return arena_->New<NominalClassValue>(destination_type, value);
        default:
          FATAL() << "Can't convert value " << *value << " to type "
                  << *destination_type;
      }
    }
    case Value::Kind::TupleValue: {
      const auto& tuple = cast<TupleValue>(value);
      const auto& destination_tuple_type = cast<TupleValue>(destination_type);
      CHECK(tuple->elements().size() ==
            destination_tuple_type->elements().size());
      std::vector<Nonnull<const Value*>> new_elements;
      for (size_t i = 0; i < tuple->elements().size(); ++i) {
        new_elements.push_back(Convert(tuple->elements()[i],
                                       destination_tuple_type->elements()[i]));
      }
      return arena_->New<TupleValue>(std::move(new_elements));
    }
  }
}

auto Interpreter::StepExp() -> Transition {
  Nonnull<Action*> act = todo_.Top();
  const Expression& exp = cast<ExpressionAction>(*act).expression();
  if (trace_) {
    llvm::outs() << "--- step exp " << exp << " (" << exp.source_loc()
                 << ") --->\n";
  }
  switch (exp.kind()) {
    case Expression::Kind::IndexExpression: {
      if (act->pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        return Spawn{arena_->New<ExpressionAction>(
            &cast<IndexExpression>(exp).aggregate())};
      } else if (act->pos() == 1) {
        return Spawn{arena_->New<ExpressionAction>(
            &cast<IndexExpression>(exp).offset())};
      } else {
        //    { { v :: [][i] :: C, E, F} :: S, H}
        // -> { { v_i :: C, E, F} : S, H}
        const auto& tuple = cast<TupleValue>(*act->results()[0]);
        int i = cast<IntValue>(*act->results()[1]).value();
        if (i < 0 || i >= static_cast<int>(tuple.elements().size())) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "index " << i << " out of range in " << tuple;
        }
        return Done{tuple.elements()[i]};
      }
    }
    case Expression::Kind::TupleLiteral: {
      if (act->pos() <
          static_cast<int>(cast<TupleLiteral>(exp).fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        return Spawn{arena_->New<ExpressionAction>(
            cast<TupleLiteral>(exp).fields()[act->pos()])};
      } else {
        return Done{CreateTuple(act, &exp)};
      }
    }
    case Expression::Kind::StructLiteral: {
      const auto& literal = cast<StructLiteral>(exp);
      if (act->pos() < static_cast<int>(literal.fields().size())) {
        return Spawn{arena_->New<ExpressionAction>(
            &literal.fields()[act->pos()].expression())};
      } else {
        return Done{CreateStruct(literal.fields(), act->results())};
      }
    }
    case Expression::Kind::StructTypeLiteral: {
      const auto& struct_type = cast<StructTypeLiteral>(exp);
      if (act->pos() < static_cast<int>(struct_type.fields().size())) {
        return Spawn{arena_->New<ExpressionAction>(
            &struct_type.fields()[act->pos()].expression())};
      } else {
        std::vector<NamedValue> fields;
        for (size_t i = 0; i < struct_type.fields().size(); ++i) {
          fields.push_back({struct_type.fields()[i].name(), act->results()[i]});
        }
        return Done{arena_->New<StructType>(std::move(fields))};
      }
    }
    case Expression::Kind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(exp);
      if (act->pos() == 0) {
        //    { { e.f :: C, E, F} :: S, H}
        // -> { { e :: [].f :: C, E, F} :: S, H}
        return Spawn{arena_->New<ExpressionAction>(&access.aggregate())};
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        return Done{act->results()[0]->GetField(
            arena_, FieldPath(access.field()), exp.source_loc())};
      }
    }
    case Expression::Kind::IdentifierExpression: {
      CHECK(act->pos() == 0);
      const auto& ident = cast<IdentifierExpression>(exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp.source_loc(), ident.name());
      return Done{heap_.Read(pointer, exp.source_loc())};
    }
    case Expression::Kind::IntLiteral:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena_->New<IntValue>(cast<IntLiteral>(exp).value())};
    case Expression::Kind::BoolLiteral:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena_->New<BoolValue>(cast<BoolLiteral>(exp).value())};
    case Expression::Kind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(exp);
      if (act->pos() != static_cast<int>(op.arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        Nonnull<const Expression*> arg = op.arguments()[act->pos()];
        return Spawn{arena_->New<ExpressionAction>(arg)};
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        return Done{EvalPrim(op.op(), act->results(), exp.source_loc())};
      }
    }
    case Expression::Kind::CallExpression:
      if (act->pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        return Spawn{arena_->New<ExpressionAction>(
            &cast<CallExpression>(exp).function())};
      } else if (act->pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        return Spawn{arena_->New<ExpressionAction>(
            &cast<CallExpression>(exp).argument())};
      } else if (act->pos() == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        switch (act->results()[0]->kind()) {
          case Value::Kind::AlternativeConstructorValue: {
            const auto& alt =
                cast<AlternativeConstructorValue>(*act->results()[0]);
            return Done{arena_->New<AlternativeValue>(
                alt.alt_name(), alt.choice_name(), act->results()[1])};
          }
          case Value::Kind::FunctionValue:
            return CallFunction{
                .function =
                    &cast<FunctionValue>(*act->results()[0]).declaration(),
                .args = act->results()[1],
                .source_loc = exp.source_loc()};
          default:
            FATAL_RUNTIME_ERROR(exp.source_loc())
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
      switch (cast<IntrinsicExpression>(exp).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          Address pointer = GetFromEnv(exp.source_loc(), "format_str");
          Nonnull<const Value*> pointee = heap_.Read(pointer, exp.source_loc());
          CHECK(pointee->kind() == Value::Kind::StringValue);
          // TODO: This could eventually use something like llvm::formatv.
          llvm::outs() << cast<StringValue>(*pointee).value();
          return Done{TupleValue::Empty()};
      }

    case Expression::Kind::IntTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena_->New<IntType>()};
    }
    case Expression::Kind::BoolTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena_->New<BoolType>()};
    }
    case Expression::Kind::TypeTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena_->New<TypeType>()};
    }
    case Expression::Kind::FunctionTypeLiteral: {
      if (act->pos() == 0) {
        return Spawn{arena_->New<ExpressionAction>(
            &cast<FunctionTypeLiteral>(exp).parameter())};
      } else if (act->pos() == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        return Spawn{arena_->New<ExpressionAction>(
            &cast<FunctionTypeLiteral>(exp).return_type())};
      } else {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        return Done{arena_->New<FunctionType>(
            std::vector<Nonnull<const GenericBinding*>>(), act->results()[0],
            act->results()[1])};
      }
    }
    case Expression::Kind::ContinuationTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena_->New<ContinuationType>()};
    }
    case Expression::Kind::StringLiteral:
      CHECK(act->pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return Done{arena_->New<StringValue>(cast<StringLiteral>(exp).value())};
    case Expression::Kind::StringTypeLiteral: {
      CHECK(act->pos() == 0);
      return Done{arena_->New<StringType>()};
    }
  }  // switch (exp->kind)
}

auto Interpreter::StepPattern() -> Transition {
  Nonnull<Action*> act = todo_.Top();
  const Pattern& pattern = cast<PatternAction>(*act).pattern();
  if (trace_) {
    llvm::outs() << "--- step pattern " << pattern << " ("
                 << pattern.source_loc() << ") --->\n";
  }
  switch (pattern.kind()) {
    case Pattern::Kind::AutoPattern: {
      CHECK(act->pos() == 0);
      return Done{arena_->New<AutoType>()};
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(pattern);
      if (act->pos() == 0) {
        return Spawn{arena_->New<PatternAction>(&binding.type())};
      } else {
        return Done{arena_->New<BindingPlaceholderValue>(binding.name(),
                                                         act->results()[0])};
      }
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(pattern);
      if (act->pos() < static_cast<int>(tuple.fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        return Spawn{arena_->New<PatternAction>(tuple.fields()[act->pos()])};
      } else {
        return Done{arena_->New<TupleValue>(act->results())};
      }
    }
    case Pattern::Kind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(pattern);
      if (act->pos() == 0) {
        return Spawn{arena_->New<ExpressionAction>(&alternative.choice_type())};
      } else if (act->pos() == 1) {
        return Spawn{arena_->New<PatternAction>(&alternative.arguments())};
      } else {
        CHECK(act->pos() == 2);
        const auto& choice_type = cast<ChoiceType>(*act->results()[0]);
        return Done{arena_->New<AlternativeValue>(
            alternative.alternative_name(), choice_type.name(),
            act->results()[1])};
      }
    }
    case Pattern::Kind::ExpressionPattern:
      return Delegate{arena_->New<ExpressionAction>(
          &cast<ExpressionPattern>(pattern).expression())};
  }
}

static auto IsRunAction(Nonnull<Action*> action) -> bool {
  const auto* statement = dyn_cast<StatementAction>(action);
  return statement != nullptr && llvm::isa<Run>(statement->statement());
}

auto Interpreter::StepStmt() -> Transition {
  Nonnull<Action*> act = todo_.Top();
  const Statement& stmt = cast<StatementAction>(*act).statement();
  if (trace_) {
    llvm::outs() << "--- step stmt ";
    stmt.PrintDepth(1, llvm::outs());
    llvm::outs() << " (" << stmt.source_loc() << ") --->\n";
  }
  switch (stmt.kind()) {
    case Statement::Kind::Match: {
      const auto& match_stmt = cast<Match>(stmt);
      if (act->pos() == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        act->StartScope(Scope(CurrentEnv()));
        return Spawn{arena_->New<ExpressionAction>(&match_stmt.expression())};
      } else {
        int clause_num = act->pos() - 1;
        if (clause_num >= static_cast<int>(match_stmt.clauses().size())) {
          return Done{};
        }
        auto c = match_stmt.clauses()[clause_num];
        std::optional<Env> matches =
            PatternMatch(&c.pattern().value(),
                         Convert(act->results()[0], &c.pattern().static_type()),
                         stmt.source_loc());
        if (matches) {  // We have a match, start the body.
          // Ensure we don't process any more clauses.
          act->set_pos(match_stmt.clauses().size() + 1);

          for (const auto& [name, value] : *matches) {
            act->scope()->values.Set(name, value);
            act->scope()->locals.push_back(name);
          }
          return Spawn{arena_->New<StatementAction>(&c.statement())};
        } else {
          return RunAgain{};
        }
      }
    }
    case Statement::Kind::While:
      if (act->pos() % 2 == 0) {
        //    { { (while (e) s) :: C, E, F} :: S, H}
        // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
        act->Clear();
        return Spawn{
            arena_->New<ExpressionAction>(&cast<While>(stmt).condition())};
      } else {
        Nonnull<const Value*> condition =
            Convert(act->results().back(), arena_->New<BoolType>());
        if (cast<BoolValue>(*condition).value()) {
          //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
          // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
          return Spawn{arena_->New<StatementAction>(&cast<While>(stmt).body())};
        } else {
          //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
          // -> { { C, E, F } :: S, H}
          return Done{};
        }
      }
    case Statement::Kind::Break: {
      CHECK(act->pos() == 0);
      //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { C, E', F} :: S, H}
      return UnwindPast{.ast_node = &cast<Break>(stmt).loop()};
    }
    case Statement::Kind::Continue: {
      CHECK(act->pos() == 0);
      //    { { continue; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { (while (e) s) :: C, E', F} :: S, H}
      return UnwindTo{.ast_node = &cast<Continue>(stmt).loop()};
    }
    case Statement::Kind::Block: {
      const auto& block = cast<Block>(stmt);
      if (act->pos() >= static_cast<int>(block.statements().size())) {
        // If the position is past the end of the block, end processing. Note
        // that empty blocks immediately end.
        return Done{};
      }
      // Initialize a scope when starting a block.
      if (act->pos() == 0) {
        act->StartScope(Scope(CurrentEnv()));
      }
      // Process the next statement in the block. The position will be
      // incremented as part of Spawn.
      return Spawn{
          arena_->New<StatementAction>(block.statements()[act->pos()])};
    }
    case Statement::Kind::VariableDefinition: {
      const auto& definition = cast<VariableDefinition>(stmt);
      if (act->pos() == 0) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        return Spawn{arena_->New<ExpressionAction>(&definition.init())};
      } else {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        Nonnull<const Value*> v =
            Convert(act->results()[0], &definition.pattern().static_type());
        Nonnull<const Value*> p =
            &cast<VariableDefinition>(stmt).pattern().value();

        std::optional<Env> matches = PatternMatch(p, v, stmt.source_loc());
        CHECK(matches)
            << stmt.source_loc()
            << ": internal error in variable definition, match failed";
        for (const auto& [name, value] : *matches) {
          Scope& current_scope = CurrentScope();
          current_scope.values.Set(name, value);
          current_scope.locals.push_back(name);
        }
        return Done{};
      }
    }
    case Statement::Kind::ExpressionStatement:
      if (act->pos() == 0) {
        //    { {e :: C, E, F} :: S, H}
        // -> { {e :: C, E, F} :: S, H}
        return Spawn{arena_->New<ExpressionAction>(
            &cast<ExpressionStatement>(stmt).expression())};
      } else {
        return Done{};
      }
    case Statement::Kind::Assign: {
      const auto& assign = cast<Assign>(stmt);
      if (act->pos() == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        return Spawn{arena_->New<LValAction>(&assign.lhs())};
      } else if (act->pos() == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        return Spawn{arena_->New<ExpressionAction>(&assign.rhs())};
      } else {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        auto pat = act->results()[0];
        auto val = Convert(act->results()[1], &assign.lhs().static_type());
        PatternAssignment(pat, val, stmt.source_loc());
        return Done{};
      }
    }
    case Statement::Kind::If:
      if (act->pos() == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        return Spawn{
            arena_->New<ExpressionAction>(&cast<If>(stmt).condition())};
      } else {
        Nonnull<const Value*> condition =
            Convert(act->results()[0], arena_->New<BoolType>());
        if (cast<BoolValue>(*condition).value()) {
          //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
          //      S, H}
          // -> { { then_stmt :: C, E, F } :: S, H}
          return Delegate{
              arena_->New<StatementAction>(&cast<If>(stmt).then_block())};
        } else if (cast<If>(stmt).else_block()) {
          //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
          //      S, H}
          // -> { { else_stmt :: C, E, F } :: S, H}
          return Delegate{
              arena_->New<StatementAction>(*cast<If>(stmt).else_block())};
        } else {
          return Done{};
        }
      }
    case Statement::Kind::Return:
      if (act->pos() == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        return Spawn{
            arena_->New<ExpressionAction>(&cast<Return>(stmt).expression())};
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        // TODO(geoffromer): convert the result to the function's return type,
        // once #880 gives us a way to find that type.
        const FunctionDeclaration& function = cast<Return>(stmt).function();
        return UnwindPast{.ast_node = *function.body(),
                          .result = act->results()[0]};
      }
    case Statement::Kind::Continuation: {
      CHECK(act->pos() == 0);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      auto continuation_stack = arena_->New<std::vector<Nonnull<Action*>>>();
      continuation_stack->push_back(
          arena_->New<StatementAction>(&cast<Continuation>(stmt).body()));
      continuation_stack->push_back(
          arena_->New<ScopeAction>(Scope(CurrentEnv())));
      AllocationId continuation_address = heap_.AllocateValue(
          arena_->New<ContinuationValue>(continuation_stack));
      // Bind the continuation object to the continuation variable
      CurrentScope().values.Set(
          cast<Continuation>(stmt).continuation_variable(),
          continuation_address);
      return Done{};
    }
    case Statement::Kind::Run: {
      auto& run = cast<Run>(stmt);
      if (act->pos() == 0) {
        // Evaluate the argument of the run statement.
        return Spawn{arena_->New<ExpressionAction>(&run.argument())};
      } else if (act->pos() == 1) {
        // Push the continuation onto the current stack.
        std::vector<Nonnull<Action*>>& continuation_vector =
            cast<const ContinuationValue>(*act->results()[0]).stack();
        while (!continuation_vector.empty()) {
          todo_.Push(continuation_vector.back());
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
      todo_.Pop();
      std::vector<Nonnull<Action*>> paused;
      while (!IsRunAction(todo_.Top())) {
        paused.push_back(todo_.Pop());
      }
      const auto& continuation =
          cast<const ContinuationValue>(*todo_.Top()->results()[0]);
      CHECK(continuation.stack().empty());
      // Update the continuation with the paused stack.
      continuation.stack() = std::move(paused);
      return ManualTransition{};
  }
}

class Interpreter::DoTransition {
 public:
  // Does not take ownership of interpreter.
  explicit DoTransition(Interpreter* interpreter) : interpreter(interpreter) {}

  void operator()(const Done& done) {
    Nonnull<Action*> act = interpreter->UnwindTodoTop();
    switch (act->kind()) {
      case Action::Kind::ExpressionAction:
      case Action::Kind::LValAction:
      case Action::Kind::PatternAction:
        CHECK(done.result.has_value());
        interpreter->todo_.Top()->AddResult(*done.result);
        break;
      case Action::Kind::StatementAction:
        CHECK(!done.result.has_value());
        break;
      case Action::Kind::ScopeAction:
        if (done.result.has_value()) {
          interpreter->todo_.Top()->AddResult(*done.result);
        }
        break;
    }
  }

  void operator()(const Spawn& spawn) {
    Nonnull<Action*> action = interpreter->todo_.Top();
    action->set_pos(action->pos() + 1);
    interpreter->todo_.Push(spawn.child);
  }

  void operator()(const Delegate& delegate) {
    Nonnull<Action*> act = interpreter->todo_.Pop();
    if (act->scope().has_value()) {
      delegate.delegate->StartScope(*act->scope());
    }
    interpreter->todo_.Push(delegate.delegate);
  }

  void operator()(const RunAgain&) {
    Nonnull<Action*> action = interpreter->todo_.Top();
    action->set_pos(action->pos() + 1);
  }

  void operator()(const UnwindTo& unwind_to) { DoUnwindTo(unwind_to.ast_node); }

  void operator()(const UnwindPast& unwind_past) {
    DoUnwindTo(unwind_past.ast_node);
    // Unwind past the statement and return a result if needed.
    interpreter->UnwindTodoTop();
    if (unwind_past.result.has_value()) {
      interpreter->todo_.Top()->AddResult(*unwind_past.result);
    }
  }

  void operator()(const CallFunction& call) {
    Nonnull<Action*> action = interpreter->todo_.Top();
    action->set_pos(action->pos() + 1);
    Nonnull<const Value*> converted_args = interpreter->Convert(
        call.args, &call.function->param_pattern().static_type());
    std::optional<Env> matches =
        interpreter->PatternMatch(&call.function->param_pattern().value(),
                                  converted_args, call.source_loc);
    CHECK(matches.has_value())
        << "internal error in call_function, pattern match failed";
    // Create the new frame and push it on the stack
    Scope new_scope(interpreter->globals_);
    for (const auto& [name, value] : *matches) {
      new_scope.values.Set(name, value);
      new_scope.locals.push_back(name);
    }
    interpreter->todo_.Push(
        interpreter->arena_->New<ScopeAction>(std::move(new_scope)));
    CHECK(call.function->body()) << "Calling a function that's missing a body";
    interpreter->todo_.Push(
        interpreter->arena_->New<StatementAction>(*call.function->body()));
  }

  void operator()(const ManualTransition&) {}

 private:
  // Unwinds to the indicated node.
  void DoUnwindTo(Nonnull<const Statement*> ast_node) {
    while (true) {
      if (const auto* statement_action =
              dyn_cast<StatementAction>(interpreter->todo_.Top());
          statement_action != nullptr &&
          &statement_action->statement() == ast_node) {
        break;
      }
      interpreter->UnwindTodoTop();
    }
  }

  Nonnull<Interpreter*> interpreter;
};

// State transition.
void Interpreter::Step() {
  Nonnull<Action*> act = todo_.Top();
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

auto Interpreter::ExecuteAction(Nonnull<Action*> action, Env values,
                                bool trace_steps) -> Nonnull<const Value*> {
  todo_ = {};
  todo_.Push(arena_->New<ScopeAction>(Scope(values)));
  todo_.Push(action);

  while (todo_.Count() > 1) {
    Step();
    if (trace_steps) {
      PrintState(llvm::outs());
    }
  }
  CHECK(todo_.Top()->results().size() == 1);
  return todo_.Top()->results()[0];
}

auto Interpreter::InterpProgram(llvm::ArrayRef<Nonnull<Declaration*>> fs,
                                Nonnull<const Expression*> call_main) -> int {
  // Check that the interpreter is in a clean state.
  CHECK(globals_.IsEmpty());
  CHECK(todo_.IsEmpty());

  if (trace_) {
    llvm::outs() << "********** initializing globals **********\n";
  }
  InitGlobals(fs);

  if (trace_) {
    llvm::outs() << "********** calling main function **********\n";
    PrintState(llvm::outs());
  }

  return cast<IntValue>(*ExecuteAction(arena_->New<ExpressionAction>(call_main),
                                       globals_, trace_))
      .value();
}

auto Interpreter::InterpExp(Env values, Nonnull<const Expression*> e)
    -> Nonnull<const Value*> {
  return ExecuteAction(arena_->New<ExpressionAction>(e), values,
                       /*trace_steps=*/false);
}

auto Interpreter::InterpPattern(Env values, Nonnull<const Pattern*> p)
    -> Nonnull<const Value*> {
  return ExecuteAction(arena_->New<PatternAction>(p), values,
                       /*trace_steps=*/false);
}

}  // namespace Carbon
