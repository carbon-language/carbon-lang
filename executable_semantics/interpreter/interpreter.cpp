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

auto Interpreter::CurrentEnv() -> Env { return todo_.CurrentScope().values(); }

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
  out << "{\nstack: " << todo_;
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
    case DeclarationKind::FunctionDeclaration: {
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

    case DeclarationKind::ClassDeclaration: {
      const auto& class_decl = cast<ClassDeclaration>(d);
      std::vector<NamedValue> fields;
      std::vector<NamedValue> methods;
      for (Nonnull<const Member*> m : class_decl.members()) {
        switch (m->kind()) {
          case MemberKind::FieldMember: {
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
          class_decl.name(), std::move(fields), std::move(methods));
      AllocationId a = heap_.AllocateValue(st);
      env->Set(class_decl.name(), a);
      break;
    }

    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(d);
      std::vector<NamedValue> alts;
      for (Nonnull<const AlternativeSignature*> alternative :
           choice.alternatives()) {
        auto t = InterpExp(Env(arena_), &alternative->signature());
        alts.push_back({.name = alternative->name(), .value = t});
      }
      auto ct = arena_->New<ChoiceType>(choice.name(), std::move(alts));
      AllocationId a = heap_.AllocateValue(ct);
      env->Set(choice.name(), a);
      break;
    }

    case DeclarationKind::VariableDeclaration: {
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

void Interpreter::StepLvalue() {
  Action& act = todo_.CurrentAction();
  const Expression& exp = cast<LValAction>(act).expression();
  if (trace_) {
    llvm::outs() << "--- step lvalue " << exp << " (" << exp.source_loc()
                 << ") --->\n";
  }
  switch (exp.kind()) {
    case ExpressionKind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      Address pointer =
          GetFromEnv(exp.source_loc(), cast<IdentifierExpression>(exp).name());
      Nonnull<const Value*> v = arena_->New<LValue>(pointer);
      return todo_.FinishAction(v);
    }
    case ExpressionKind::FieldAccessExpression: {
      if (act.pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<LValAction>(
            &cast<FieldAccessExpression>(exp).aggregate()));
      } else {
        //    { v :: [].f :: C, E, F} :: S, H}
        // -> { { &v.f :: C, E, F} :: S, H }
        Address aggregate = cast<LValue>(*act.results()[0]).address();
        Address field = aggregate.SubobjectAddress(
            cast<FieldAccessExpression>(exp).field());
        return todo_.FinishAction(arena_->New<LValue>(field));
      }
    }
    case ExpressionKind::IndexExpression: {
      if (act.pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<LValAction>(
            &cast<IndexExpression>(exp).aggregate()));

      } else if (act.pos() == 1) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<IndexExpression>(exp).offset()));
      } else {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address aggregate = cast<LValue>(*act.results()[0]).address();
        std::string f =
            std::to_string(cast<IntValue>(*act.results()[1]).value());
        Address field = aggregate.SubobjectAddress(f);
        return todo_.FinishAction(arena_->New<LValue>(field));
      }
    }
    case ExpressionKind::TupleLiteral:
    case ExpressionKind::StructLiteral:
    case ExpressionKind::StructTypeLiteral:
    case ExpressionKind::IntLiteral:
    case ExpressionKind::BoolLiteral:
    case ExpressionKind::CallExpression:
    case ExpressionKind::PrimitiveOperatorExpression:
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::FunctionTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
    case ExpressionKind::StringLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::IntrinsicExpression:
      FATAL() << "Can't treat expression as lvalue: " << exp;
  }
}

auto Interpreter::Convert(Nonnull<const Value*> value,
                          Nonnull<const Value*> destination_type) const
    -> Nonnull<const Value*> {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::LValue:
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

void Interpreter::StepExp() {
  Action& act = todo_.CurrentAction();
  const Expression& exp = cast<ExpressionAction>(act).expression();
  if (trace_) {
    llvm::outs() << "--- step exp " << exp << " (" << exp.source_loc()
                 << ") --->\n";
  }
  switch (exp.kind()) {
    case ExpressionKind::IndexExpression: {
      if (act.pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<IndexExpression>(exp).aggregate()));
      } else if (act.pos() == 1) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<IndexExpression>(exp).offset()));
      } else {
        //    { { v :: [][i] :: C, E, F} :: S, H}
        // -> { { v_i :: C, E, F} : S, H}
        const auto& tuple = cast<TupleValue>(*act.results()[0]);
        int i = cast<IntValue>(*act.results()[1]).value();
        if (i < 0 || i >= static_cast<int>(tuple.elements().size())) {
          FATAL_RUNTIME_ERROR_NO_LINE()
              << "index " << i << " out of range in " << tuple;
        }
        return todo_.FinishAction(tuple.elements()[i]);
      }
    }
    case ExpressionKind::TupleLiteral: {
      if (act.pos() <
          static_cast<int>(cast<TupleLiteral>(exp).fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            cast<TupleLiteral>(exp).fields()[act.pos()]));
      } else {
        return todo_.FinishAction(arena_->New<TupleValue>(act.results()));
      }
    }
    case ExpressionKind::StructLiteral: {
      const auto& literal = cast<StructLiteral>(exp);
      if (act.pos() < static_cast<int>(literal.fields().size())) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &literal.fields()[act.pos()].expression()));
      } else {
        return todo_.FinishAction(
            CreateStruct(literal.fields(), act.results()));
      }
    }
    case ExpressionKind::StructTypeLiteral: {
      const auto& struct_type = cast<StructTypeLiteral>(exp);
      if (act.pos() < static_cast<int>(struct_type.fields().size())) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &struct_type.fields()[act.pos()].expression()));
      } else {
        std::vector<NamedValue> fields;
        for (size_t i = 0; i < struct_type.fields().size(); ++i) {
          fields.push_back({struct_type.fields()[i].name(), act.results()[i]});
        }
        return todo_.FinishAction(arena_->New<StructType>(std::move(fields)));
      }
    }
    case ExpressionKind::FieldAccessExpression: {
      const auto& access = cast<FieldAccessExpression>(exp);
      if (act.pos() == 0) {
        //    { { e.f :: C, E, F} :: S, H}
        // -> { { e :: [].f :: C, E, F} :: S, H}
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&access.aggregate()));
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        return todo_.FinishAction(act.results()[0]->GetField(
            arena_, FieldPath(access.field()), exp.source_loc()));
      }
    }
    case ExpressionKind::IdentifierExpression: {
      CHECK(act.pos() == 0);
      const auto& ident = cast<IdentifierExpression>(exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      Address pointer = GetFromEnv(exp.source_loc(), ident.name());
      return todo_.FinishAction(heap_.Read(pointer, exp.source_loc()));
    }
    case ExpressionKind::IntLiteral:
      CHECK(act.pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return todo_.FinishAction(
          arena_->New<IntValue>(cast<IntLiteral>(exp).value()));
    case ExpressionKind::BoolLiteral:
      CHECK(act.pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return todo_.FinishAction(
          arena_->New<BoolValue>(cast<BoolLiteral>(exp).value()));
    case ExpressionKind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(exp);
      if (act.pos() != static_cast<int>(op.arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        Nonnull<const Expression*> arg = op.arguments()[act.pos()];
        return todo_.Spawn(std::make_unique<ExpressionAction>(arg));
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        return todo_.FinishAction(
            EvalPrim(op.op(), act.results(), exp.source_loc()));
      }
    }
    case ExpressionKind::CallExpression:
      if (act.pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<CallExpression>(exp).function()));
      } else if (act.pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<CallExpression>(exp).argument()));
      } else if (act.pos() == 2) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        switch (act.results()[0]->kind()) {
          case Value::Kind::AlternativeConstructorValue: {
            const auto& alt =
                cast<AlternativeConstructorValue>(*act.results()[0]);
            return todo_.FinishAction(arena_->New<AlternativeValue>(
                alt.alt_name(), alt.choice_name(), act.results()[1]));
          }
          case Value::Kind::FunctionValue: {
            const FunctionDeclaration& function =
                cast<FunctionValue>(*act.results()[0]).declaration();
            Nonnull<const Value*> converted_args = Convert(
                act.results()[1], &function.param_pattern().static_type());
            std::optional<Env> matches =
                PatternMatch(&function.param_pattern().value(), converted_args,
                             exp.source_loc());
            CHECK(matches.has_value())
                << "internal error in call_function, pattern match failed";
            Scope new_scope(globals_, &heap_);
            for (const auto& [name, value] : *matches) {
              new_scope.AddLocal(name, value);
            }
            CHECK(function.body().has_value())
                << "Calling a function that's missing a body";
            return todo_.Spawn(
                std::make_unique<StatementAction>(*function.body()),
                std::move(new_scope));
          }
          default:
            FATAL_RUNTIME_ERROR(exp.source_loc())
                << "in call, expected a function, not " << *act.results()[0];
        }
      } else if (act.pos() == 3) {
        if (act.results().size() < 3) {
          // Control fell through without explicit return.
          return todo_.FinishAction(TupleValue::Empty());
        } else {
          return todo_.FinishAction(act.results()[2]);
        }
      } else {
        FATAL() << "in handle_value with Call pos " << act.pos();
      }
    case ExpressionKind::IntrinsicExpression:
      CHECK(act.pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      switch (cast<IntrinsicExpression>(exp).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print:
          Address pointer = GetFromEnv(exp.source_loc(), "format_str");
          Nonnull<const Value*> pointee = heap_.Read(pointer, exp.source_loc());
          CHECK(pointee->kind() == Value::Kind::StringValue);
          // TODO: This could eventually use something like llvm::formatv.
          llvm::outs() << cast<StringValue>(*pointee).value();
          return todo_.FinishAction(TupleValue::Empty());
      }

    case ExpressionKind::IntTypeLiteral: {
      CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<IntType>());
    }
    case ExpressionKind::BoolTypeLiteral: {
      CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<BoolType>());
    }
    case ExpressionKind::TypeTypeLiteral: {
      CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<TypeType>());
    }
    case ExpressionKind::FunctionTypeLiteral: {
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<FunctionTypeLiteral>(exp).parameter()));
      } else if (act.pos() == 1) {
        //    { { pt :: fn [] -> e :: C, E, F} :: S, H}
        // -> { { e :: fn pt -> []) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<FunctionTypeLiteral>(exp).return_type()));
      } else {
        //    { { rt :: fn pt -> [] :: C, E, F} :: S, H}
        // -> { fn pt -> rt :: {C, E, F} :: S, H}
        return todo_.FinishAction(arena_->New<FunctionType>(
            std::vector<Nonnull<const GenericBinding*>>(), act.results()[0],
            act.results()[1]));
      }
    }
    case ExpressionKind::ContinuationTypeLiteral: {
      CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<ContinuationType>());
    }
    case ExpressionKind::StringLiteral:
      CHECK(act.pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return todo_.FinishAction(
          arena_->New<StringValue>(cast<StringLiteral>(exp).value()));
    case ExpressionKind::StringTypeLiteral: {
      CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<StringType>());
    }
  }  // switch (exp->kind)
}

void Interpreter::StepPattern() {
  Action& act = todo_.CurrentAction();
  const Pattern& pattern = cast<PatternAction>(act).pattern();
  if (trace_) {
    llvm::outs() << "--- step pattern " << pattern << " ("
                 << pattern.source_loc() << ") --->\n";
  }
  switch (pattern.kind()) {
    case PatternKind::AutoPattern: {
      CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<AutoType>());
    }
    case PatternKind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(pattern);
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<PatternAction>(&binding.type()));
      } else {
        return todo_.FinishAction(arena_->New<BindingPlaceholderValue>(
            binding.name(), act.results()[0]));
      }
    }
    case PatternKind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(pattern);
      if (act.pos() < static_cast<int>(tuple.fields().size())) {
        //    { { vk :: (f1=v1,..., fk=[],fk+1=ek+1,...) :: C, E, F} :: S,
        //    H}
        // -> { { ek+1 :: (f1=v1,..., fk=vk, fk+1=[],...) :: C, E, F} :: S,
        // H}
        return todo_.Spawn(
            std::make_unique<PatternAction>(tuple.fields()[act.pos()]));
      } else {
        return todo_.FinishAction(arena_->New<TupleValue>(act.results()));
      }
    }
    case PatternKind::AlternativePattern: {
      const auto& alternative = cast<AlternativePattern>(pattern);
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&alternative.choice_type()));
      } else if (act.pos() == 1) {
        return todo_.Spawn(
            std::make_unique<PatternAction>(&alternative.arguments()));
      } else {
        CHECK(act.pos() == 2);
        const auto& choice_type = cast<ChoiceType>(*act.results()[0]);
        return todo_.FinishAction(arena_->New<AlternativeValue>(
            alternative.alternative_name(), choice_type.name(),
            act.results()[1]));
      }
    }
    case PatternKind::ExpressionPattern:
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<ExpressionPattern>(pattern).expression()));
      } else {
        return todo_.FinishAction(act.results()[0]);
      }
  }
}

void Interpreter::StepStmt() {
  Action& act = todo_.CurrentAction();
  const Statement& stmt = cast<StatementAction>(act).statement();
  if (trace_) {
    llvm::outs() << "--- step stmt ";
    stmt.PrintDepth(1, llvm::outs());
    llvm::outs() << " (" << stmt.source_loc() << ") --->\n";
  }
  switch (stmt.kind()) {
    case StatementKind::Match: {
      const auto& match_stmt = cast<Match>(stmt);
      if (act.pos() == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        act.StartScope(Scope(CurrentEnv(), &heap_));
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&match_stmt.expression()));
      } else {
        int clause_num = act.pos() - 1;
        if (clause_num >= static_cast<int>(match_stmt.clauses().size())) {
          return todo_.FinishAction();
        }
        auto c = match_stmt.clauses()[clause_num];
        std::optional<Env> matches =
            PatternMatch(&c.pattern().value(),
                         Convert(act.results()[0], &c.pattern().static_type()),
                         stmt.source_loc());
        if (matches) {  // We have a match, start the body.
          // Ensure we don't process any more clauses.
          act.set_pos(match_stmt.clauses().size() + 1);

          for (const auto& [name, value] : *matches) {
            act.scope()->AddLocal(name, value);
          }
          return todo_.Spawn(std::make_unique<StatementAction>(&c.statement()));
        } else {
          return todo_.RunAgain();
        }
      }
    }
    case StatementKind::While:
      if (act.pos() % 2 == 0) {
        //    { { (while (e) s) :: C, E, F} :: S, H}
        // -> { { e :: (while ([]) s) :: C, E, F} :: S, H}
        act.Clear();
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&cast<While>(stmt).condition()));
      } else {
        Nonnull<const Value*> condition =
            Convert(act.results().back(), arena_->New<BoolType>());
        if (cast<BoolValue>(*condition).value()) {
          //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
          // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
          return todo_.Spawn(
              std::make_unique<StatementAction>(&cast<While>(stmt).body()));
        } else {
          //    { {false :: (while ([]) s) :: C, E, F} :: S, H}
          // -> { { C, E, F } :: S, H}
          return todo_.FinishAction();
        }
      }
    case StatementKind::Break: {
      CHECK(act.pos() == 0);
      //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { C, E', F} :: S, H}
      return todo_.UnwindPast(&cast<Break>(stmt).loop());
    }
    case StatementKind::Continue: {
      CHECK(act.pos() == 0);
      //    { { continue; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { (while (e) s) :: C, E', F} :: S, H}
      return todo_.UnwindTo(&cast<Continue>(stmt).loop());
    }
    case StatementKind::Block: {
      const auto& block = cast<Block>(stmt);
      if (act.pos() >= static_cast<int>(block.statements().size())) {
        // If the position is past the end of the block, end processing. Note
        // that empty blocks immediately end.
        return todo_.FinishAction();
      }
      // Initialize a scope when starting a block.
      if (act.pos() == 0) {
        act.StartScope(Scope(CurrentEnv(), &heap_));
      }
      // Process the next statement in the block. The position will be
      // incremented as part of Spawn.
      return todo_.Spawn(
          std::make_unique<StatementAction>(block.statements()[act.pos()]));
    }
    case StatementKind::VariableDefinition: {
      const auto& definition = cast<VariableDefinition>(stmt);
      if (act.pos() == 0) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&definition.init()));
      } else {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        Nonnull<const Value*> v =
            Convert(act.results()[0], &definition.pattern().static_type());
        Nonnull<const Value*> p =
            &cast<VariableDefinition>(stmt).pattern().value();

        std::optional<Env> matches = PatternMatch(p, v, stmt.source_loc());
        CHECK(matches)
            << stmt.source_loc()
            << ": internal error in variable definition, match failed";
        for (const auto& [name, value] : *matches) {
          Scope& current_scope = todo_.CurrentScope();
          current_scope.AddLocal(name, value);
        }
        return todo_.FinishAction();
      }
    }
    case StatementKind::ExpressionStatement:
      if (act.pos() == 0) {
        //    { {e :: C, E, F} :: S, H}
        // -> { {e :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<ExpressionStatement>(stmt).expression()));
      } else {
        return todo_.FinishAction();
      }
    case StatementKind::Assign: {
      const auto& assign = cast<Assign>(stmt);
      if (act.pos() == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<LValAction>(&assign.lhs()));
      } else if (act.pos() == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(&assign.rhs()));
      } else {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        const auto& lval = cast<LValue>(*act.results()[0]);
        Nonnull<const Value*> rval =
            Convert(act.results()[1], &assign.lhs().static_type());
        heap_.Write(lval.address(), rval, stmt.source_loc());
        return todo_.FinishAction();
      }
    }
    case StatementKind::If:
      if (act.pos() == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&cast<If>(stmt).condition()));
      } else if (act.pos() == 1) {
        Nonnull<const Value*> condition =
            Convert(act.results()[0], arena_->New<BoolType>());
        if (cast<BoolValue>(*condition).value()) {
          //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
          //      S, H}
          // -> { { then_stmt :: C, E, F } :: S, H}
          return todo_.Spawn(
              std::make_unique<StatementAction>(&cast<If>(stmt).then_block()));
        } else if (cast<If>(stmt).else_block()) {
          //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
          //      S, H}
          // -> { { else_stmt :: C, E, F } :: S, H}
          return todo_.Spawn(
              std::make_unique<StatementAction>(*cast<If>(stmt).else_block()));
        } else {
          return todo_.FinishAction();
        }
      } else {
        return todo_.FinishAction();
      }
    case StatementKind::Return:
      if (act.pos() == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<Return>(stmt).expression()));
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        const FunctionDeclaration& function = cast<Return>(stmt).function();
        return todo_.UnwindPast(
            *function.body(),
            Convert(act.results()[0], &function.return_term().static_type()));
      }
    case StatementKind::Continuation: {
      CHECK(act.pos() == 0);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      auto fragment = arena_->New<ContinuationValue::StackFragment>();
      stack_fragments_.push_back(fragment);
      std::vector<std::unique_ptr<Action>> reversed_todo;
      reversed_todo.push_back(
          std::make_unique<StatementAction>(&cast<Continuation>(stmt).body()));
      reversed_todo.push_back(
          std::make_unique<ScopeAction>(Scope(CurrentEnv(), &heap_)));
      fragment->StoreReversed(std::move(reversed_todo));
      AllocationId continuation_address =
          heap_.AllocateValue(arena_->New<ContinuationValue>(fragment));
      // Bind the continuation object to the continuation variable
      todo_.CurrentScope().AddLocal(
          cast<Continuation>(stmt).continuation_variable(),
          continuation_address);
      return todo_.FinishAction();
    }
    case StatementKind::Run: {
      auto& run = cast<Run>(stmt);
      if (act.pos() == 0) {
        // Evaluate the argument of the run statement.
        return todo_.Spawn(std::make_unique<ExpressionAction>(&run.argument()));
      } else if (act.pos() == 1) {
        // Push the continuation onto the current stack.
        return todo_.Resume(cast<const ContinuationValue>(act.results()[0]));
      } else {
        return todo_.FinishAction();
      }
    }
    case StatementKind::Await:
      CHECK(act.pos() == 0);
      return todo_.Suspend();
  }
}

// State transition.
void Interpreter::Step() {
  Action& act = todo_.CurrentAction();
  switch (act.kind()) {
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
    case Action::Kind::ScopeAction:
      FATAL() << "ScopeAction escaped ActionStack";
  }  // switch
}

auto Interpreter::ExecuteAction(std::unique_ptr<Action> action, Env values,
                                bool trace_steps) -> Nonnull<const Value*> {
  todo_.Start(std::move(action), Scope(values, &heap_));

  while (!todo_.IsEmpty()) {
    Step();
    if (trace_steps) {
      PrintState(llvm::outs());
    }
  }

  // Clean up any remaining suspended continuations.
  for (Nonnull<ContinuationValue::StackFragment*> fragment : stack_fragments_) {
    fragment->Clear();
  }

  return todo_.result();
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

  return cast<IntValue>(
             *ExecuteAction(std::make_unique<ExpressionAction>(call_main),
                            globals_, trace_))
      .value();
}

auto Interpreter::InterpExp(Env values, Nonnull<const Expression*> e)
    -> Nonnull<const Value*> {
  return ExecuteAction(std::make_unique<ExpressionAction>(e), values,
                       /*trace_steps=*/false);
}

auto Interpreter::InterpPattern(Env values, Nonnull<const Pattern*> p)
    -> Nonnull<const Value*> {
  return ExecuteAction(std::make_unique<PatternAction>(p), values,
                       /*trace_steps=*/false);
}

}  // namespace Carbon
