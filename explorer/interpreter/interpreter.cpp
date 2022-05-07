// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/interpreter.h"

#include <iterator>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "common/check.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/common/arena.h"
#include "explorer/common/error_builders.h"
#include "explorer/interpreter/action.h"
#include "explorer/interpreter/action_stack.h"
#include "explorer/interpreter/stack.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {

// Constructs an ActionStack suitable for the specified phase.
static auto MakeTodo(Phase phase, Nonnull<Heap*> heap) -> ActionStack {
  switch (phase) {
    case Phase::CompileTime:
      return ActionStack();
    case Phase::RunTime:
      return ActionStack(heap);
  }
}

// An Interpreter represents an instance of the Carbon abstract machine. It
// manages the state of the abstract machine, and executes the steps of Actions
// passed to it.
class Interpreter {
 public:
  // Constructs an Interpreter which allocates values on `arena`, and prints
  // traces if `trace` is true. `phase` indicates whether it executes at
  // compile time or run time.
  Interpreter(Phase phase, Nonnull<Arena*> arena,
              std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
      : arena_(arena),
        heap_(arena),
        todo_(MakeTodo(phase, &heap_)),
        trace_stream_(trace_stream),
        phase_(phase) {}

  ~Interpreter();

  // Runs all the steps of `action`.
  // It's not safe to call `RunAllSteps()` or `result()` after an error.
  auto RunAllSteps(std::unique_ptr<Action> action) -> ErrorOr<Success>;

  // The result produced by the `action` argument of the most recent
  // RunAllSteps call. Cannot be called if `action` was an action that doesn't
  // produce results.
  auto result() const -> Nonnull<const Value*> { return todo_.result(); }

 private:
  auto Step() -> ErrorOr<Success>;

  // State transitions for expressions.
  auto StepExp() -> ErrorOr<Success>;
  // State transitions for lvalues.
  auto StepLvalue() -> ErrorOr<Success>;
  // State transitions for patterns.
  auto StepPattern() -> ErrorOr<Success>;
  // State transition for statements.
  auto StepStmt() -> ErrorOr<Success>;
  // State transition for declarations.
  auto StepDeclaration() -> ErrorOr<Success>;

  auto CreateStruct(const std::vector<FieldInitializer>& fields,
                    const std::vector<Nonnull<const Value*>>& values)
      -> Nonnull<const Value*>;

  auto EvalPrim(Operator op, const std::vector<Nonnull<const Value*>>& args,
                SourceLocation source_loc) -> ErrorOr<Nonnull<const Value*>>;

  // Returns the result of converting `value` to type `destination_type`.
  auto Convert(Nonnull<const Value*> value,
               Nonnull<const Value*> destination_type,
               SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const Value*>>;

  // Evaluate an impl expression to produce a witness, or signal an
  // error.
  //
  // An impl expression is either
  // 1) an IdentifierExpression whose value_node is an impl declaration, or
  // 2) an InstantiateImpl expression.
  auto EvalImplExp(Nonnull<const Expression*> exp) const
      -> ErrorOr<Nonnull<const Witness*>>;

  // Instantiate a type by replacing all type variables that occur inside the
  // type by the current values of those variables.
  //
  // For example, suppose T=i32 and U=Bool. Then
  //     __Fn (Point(T)) -> Point(U)
  // becomes
  //     __Fn (Point(i32)) -> Point(Bool)
  auto InstantiateType(Nonnull<const Value*> type,
                       SourceLocation source_loc) const
      -> ErrorOr<Nonnull<const Value*>>;

  // Call the function `fun` with the given `arg` and the `witnesses`
  // for the function's impl bindings.
  auto CallFunction(const CallExpression& call, Nonnull<const Value*> fun,
                    Nonnull<const Value*> arg, const ImplWitnessMap& witnesses)
      -> ErrorOr<Success>;

  void PrintState(llvm::raw_ostream& out);

  Phase phase() const { return phase_; }

  Nonnull<Arena*> arena_;

  Heap heap_;
  ActionStack todo_;

  // The underlying states of continuation values. All StackFragments created
  // during execution are tracked here, in order to safely deallocate the
  // contents of any non-completed continuations at the end of execution.
  std::vector<Nonnull<ContinuationValue::StackFragment*>> stack_fragments_;

  std::optional<Nonnull<llvm::raw_ostream*>> trace_stream_;
  Phase phase_;
};

Interpreter::~Interpreter() {
  // Clean up any remaining suspended continuations.
  for (Nonnull<ContinuationValue::StackFragment*> fragment : stack_fragments_) {
    fragment->Clear();
  }
}

//
// State Operations
//

void Interpreter::PrintState(llvm::raw_ostream& out) {
  out << "{\nstack: " << todo_;
  out << "\nheap: " << heap_;
  if (!todo_.IsEmpty()) {
    out << "\nvalues: ";
    todo_.PrintScopes(out);
  }
  out << "\n}\n";
}

auto Interpreter::EvalPrim(Operator op,
                           const std::vector<Nonnull<const Value*>>& args,
                           SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
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
      return arena_->New<BoolValue>(ValueEqual(args[0], args[1]));
    case Operator::Ptr:
      return arena_->New<PointerType>(args[0]);
    case Operator::Deref:
      return heap_.Read(cast<PointerValue>(*args[0]).address(), source_loc);
    case Operator::AddressOf:
      return arena_->New<PointerValue>(cast<LValue>(*args[0]).address());
  }
}

auto Interpreter::CreateStruct(const std::vector<FieldInitializer>& fields,
                               const std::vector<Nonnull<const Value*>>& values)
    -> Nonnull<const Value*> {
  CARBON_CHECK(fields.size() == values.size());
  std::vector<NamedValue> elements;
  for (size_t i = 0; i < fields.size(); ++i) {
    elements.push_back({.name = fields[i].name(), .value = values[i]});
  }

  return arena_->New<StructValue>(std::move(elements));
}

auto PatternMatch(Nonnull<const Value*> p, Nonnull<const Value*> v,
                  SourceLocation source_loc,
                  std::optional<Nonnull<RuntimeScope*>> bindings,
                  BindingMap& generic_args,
                  std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
    -> bool {
  if (trace_stream) {
    **trace_stream << "match pattern " << *p << "\nwith value " << *v << "\n";
  }
  switch (p->kind()) {
    case Value::Kind::BindingPlaceholderValue: {
      CARBON_CHECK(bindings.has_value());
      const auto& placeholder = cast<BindingPlaceholderValue>(*p);
      if (placeholder.value_node().has_value()) {
        (*bindings)->Initialize(*placeholder.value_node(), v);
      }
      return true;
    }
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*p);
      generic_args[&var_type.binding()] = v;
      return true;
    }
    case Value::Kind::TupleValue:
      switch (v->kind()) {
        case Value::Kind::TupleValue: {
          const auto& p_tup = cast<TupleValue>(*p);
          const auto& v_tup = cast<TupleValue>(*v);
          CARBON_CHECK(p_tup.elements().size() == v_tup.elements().size());
          for (size_t i = 0; i < p_tup.elements().size(); ++i) {
            if (!PatternMatch(p_tup.elements()[i], v_tup.elements()[i],
                              source_loc, bindings, generic_args,
                              trace_stream)) {
              return false;
            }
          }  // for
          return true;
        }
        default:
          CARBON_FATAL() << "expected a tuple value in pattern, not " << *v;
      }
    case Value::Kind::StructValue: {
      const auto& p_struct = cast<StructValue>(*p);
      const auto& v_struct = cast<StructValue>(*v);
      CARBON_CHECK(p_struct.elements().size() == v_struct.elements().size());
      for (size_t i = 0; i < p_struct.elements().size(); ++i) {
        CARBON_CHECK(p_struct.elements()[i].name ==
                     v_struct.elements()[i].name);
        if (!PatternMatch(p_struct.elements()[i].value,
                          v_struct.elements()[i].value, source_loc, bindings,
                          generic_args, trace_stream)) {
          return false;
        }
      }
      return true;
    }
    case Value::Kind::AlternativeValue:
      switch (v->kind()) {
        case Value::Kind::AlternativeValue: {
          const auto& p_alt = cast<AlternativeValue>(*p);
          const auto& v_alt = cast<AlternativeValue>(*v);
          if (p_alt.choice_name() != v_alt.choice_name() ||
              p_alt.alt_name() != v_alt.alt_name()) {
            return false;
          }
          return PatternMatch(&p_alt.argument(), &v_alt.argument(), source_loc,
                              bindings, generic_args, trace_stream);
        }
        default:
          CARBON_FATAL() << "expected a choice alternative in pattern, not "
                         << *v;
      }
    case Value::Kind::FunctionType:
      switch (v->kind()) {
        case Value::Kind::FunctionType: {
          const auto& p_fn = cast<FunctionType>(*p);
          const auto& v_fn = cast<FunctionType>(*v);
          if (!PatternMatch(&p_fn.parameters(), &v_fn.parameters(), source_loc,
                            bindings, generic_args, trace_stream)) {
            return false;
          }
          if (!PatternMatch(&p_fn.return_type(), &v_fn.return_type(),
                            source_loc, bindings, generic_args, trace_stream)) {
            return false;
          }
          return true;
        }
        default:
          return false;
      }
    case Value::Kind::AutoType:
      // `auto` matches any type, without binding any new names. We rely
      // on the typechecker to ensure that `v` is a type.
      return true;
    default:
      return ValueEqual(p, v);
  }
}

auto Interpreter::StepLvalue() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Expression& exp = cast<LValAction>(act).expression();
  if (trace_stream_) {
    **trace_stream_ << "--- step lvalue " << exp << " (" << exp.source_loc()
                    << ") --->\n";
  }
  switch (exp.kind()) {
    case ExpressionKind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(cast<IdentifierExpression>(exp).value_node(),
                            exp.source_loc()));
      CARBON_CHECK(isa<LValue>(value)) << *value;
      return todo_.FinishAction(value);
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
    case ExpressionKind::CompoundFieldAccessExpression: {
      const auto& access = cast<CompoundFieldAccessExpression>(exp);
      if (act.pos() == 0) {
        //    { {e.(f) :: C, E, F} :: S, H}
        // -> { e :: [].(f) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<LValAction>(&access.object()));
      } else {
        //    { v :: [].(f) :: C, E, F} :: S, H}
        // -> { { &v.(f) :: C, E, F} :: S, H }
        CARBON_CHECK(!access.member().interface().has_value())
            << "unexpected lvalue interface member";
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> val,
            Convert(act.results()[0], *access.member().base_type(),
                    exp.source_loc()));
        Address object = cast<LValue>(*val).address();
        Address field = object.SubobjectAddress(access.member().name());
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
    case ExpressionKind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(exp);
      if (op.op() != Operator::Deref) {
        CARBON_FATAL()
            << "Can't treat primitive operator expression as lvalue: " << exp;
      }
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(op.arguments()[0]));
      } else {
        const auto& res = cast<PointerValue>(*act.results()[0]);
        return todo_.FinishAction(arena_->New<LValue>(res.address()));
      }
      break;
    }
    case ExpressionKind::TupleLiteral:
    case ExpressionKind::StructLiteral:
    case ExpressionKind::StructTypeLiteral:
    case ExpressionKind::IntLiteral:
    case ExpressionKind::BoolLiteral:
    case ExpressionKind::CallExpression:
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
    case ExpressionKind::FunctionTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
    case ExpressionKind::StringLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::IntrinsicExpression:
    case ExpressionKind::IfExpression:
    case ExpressionKind::ArrayTypeLiteral:
    case ExpressionKind::InstantiateImpl:
      CARBON_FATAL() << "Can't treat expression as lvalue: " << exp;
    case ExpressionKind::UnimplementedExpression:
      CARBON_FATAL() << "Unimplemented: " << exp;
  }
}

auto Interpreter::EvalImplExp(Nonnull<const Expression*> exp) const
    -> ErrorOr<Nonnull<const Witness*>> {
  switch (exp->kind()) {
    case ExpressionKind::InstantiateImpl: {
      const InstantiateImpl& inst_impl = cast<InstantiateImpl>(*exp);
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Witness*> gen_impl,
                              EvalImplExp(inst_impl.generic_impl()));
      ImplWitnessMap witnesses;
      for (auto& [bind, impl_exp] : inst_impl.impls()) {
        CARBON_ASSIGN_OR_RETURN(witnesses[bind], EvalImplExp(impl_exp));
      }
      return arena_->New<Witness>(&gen_impl->declaration(),
                                  inst_impl.type_args(), witnesses);
    }
    case ExpressionKind::IdentifierExpression: {
      const auto& ident = cast<IdentifierExpression>(*exp);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(ident.value_node(), ident.source_loc()));
      if (const auto* lvalue = dyn_cast<LValue>(value)) {
        CARBON_ASSIGN_OR_RETURN(
            value, heap_.Read(lvalue->address(), exp->source_loc()));
      }
      return cast<Witness>(value);
    }
    default: {
      CARBON_FATAL() << "EvalImplExp, unexpected expression: " << *exp;
    }
  }
}

auto Interpreter::InstantiateType(Nonnull<const Value*> type,
                                  SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  switch (type->kind()) {
    case Value::Kind::VariableType: {
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(&cast<VariableType>(*type).binding(), source_loc));
      if (const auto* lvalue = dyn_cast<LValue>(value)) {
        CARBON_ASSIGN_OR_RETURN(value,
                                heap_.Read(lvalue->address(), source_loc));
      }
      return value;
    }
    case Value::Kind::NominalClassType: {
      const auto& class_type = cast<NominalClassType>(*type);
      BindingMap inst_type_args;
      for (const auto& [ty_var, ty_arg] : class_type.type_args()) {
        CARBON_ASSIGN_OR_RETURN(inst_type_args[ty_var],
                                InstantiateType(ty_arg, source_loc));
      }
      std::map<Nonnull<const ImplBinding*>, Nonnull<const Witness*>> witnesses;
      for (const auto& [bind, impl_exp] : class_type.impls()) {
        CARBON_ASSIGN_OR_RETURN(witnesses[bind], EvalImplExp(impl_exp));
      }
      return arena_->New<NominalClassType>(&class_type.declaration(),
                                           inst_type_args, witnesses);
    }
    default:
      return type;
  }
}

auto Interpreter::Convert(Nonnull<const Value*> value,
                          Nonnull<const Value*> destination_type,
                          SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::PointerValue:
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
    case Value::Kind::InterfaceType:
    case Value::Kind::Witness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringType:
    case Value::Kind::StringValue:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::StaticArrayType:
    case Value::Kind::MemberName:
      // TODO: add `CARBON_CHECK(TypeEqual(type, value->dynamic_type()))`, once
      // we have Value::dynamic_type.
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
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> val,
                Convert(*old_value, field_type, source_loc));
            new_elements.push_back({.name = field_name, .value = val});
          }
          return arena_->New<StructValue>(std::move(new_elements));
        }
        case Value::Kind::NominalClassType: {
          // Instantiate the `destintation_type` to obtain the runtime
          // type of the object.
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> inst_dest,
              InstantiateType(destination_type, source_loc));
          return arena_->New<NominalClassValue>(inst_dest, value);
        }
        default:
          CARBON_FATAL() << "Can't convert value " << *value << " to type "
                         << *destination_type;
      }
    }
    case Value::Kind::TupleValue: {
      const auto& tuple = cast<TupleValue>(value);
      std::vector<Nonnull<const Value*>> destination_element_types;
      switch (destination_type->kind()) {
        case Value::Kind::TupleValue:
          destination_element_types =
              cast<TupleValue>(destination_type)->elements();
          break;
        case Value::Kind::StaticArrayType: {
          const auto& array_type = cast<StaticArrayType>(*destination_type);
          destination_element_types.resize(array_type.size(),
                                           &array_type.element_type());
          break;
        }
        default:
          CARBON_FATAL() << "Can't convert value " << *value << " to type "
                         << *destination_type;
      }
      CARBON_CHECK(tuple->elements().size() ==
                   destination_element_types.size());
      std::vector<Nonnull<const Value*>> new_elements;
      for (size_t i = 0; i < tuple->elements().size(); ++i) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> val,
            Convert(tuple->elements()[i], destination_element_types[i],
                    source_loc));
        new_elements.push_back(val);
      }
      return arena_->New<TupleValue>(std::move(new_elements));
    }
  }
}

auto Interpreter::CallFunction(const CallExpression& call,
                               Nonnull<const Value*> fun,
                               Nonnull<const Value*> arg,
                               const ImplWitnessMap& witnesses)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    **trace_stream_ << "calling function: " << *fun << "\n";
  }
  switch (fun->kind()) {
    case Value::Kind::AlternativeConstructorValue: {
      const auto& alt = cast<AlternativeConstructorValue>(*fun);
      return todo_.FinishAction(arena_->New<AlternativeValue>(
          alt.alt_name(), alt.choice_name(), arg));
    }
    case Value::Kind::FunctionValue: {
      const FunctionValue& fun_val = cast<FunctionValue>(*fun);
      const FunctionDeclaration& function = fun_val.declaration();
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> converted_args,
          Convert(arg, &function.param_pattern().static_type(),
                  call.source_loc()));
      RuntimeScope function_scope(&heap_);
      // Bring the class type arguments into scope.
      for (const auto& [bind, val] : fun_val.type_args()) {
        function_scope.Initialize(bind, val);
      }
      // Bring the deduced type arguments into scope.
      for (const auto& [bind, val] : call.deduced_args()) {
        function_scope.Initialize(bind, val);
      }
      // Bring the impl witness tables into scope.
      for (const auto& [impl_bind, witness] : witnesses) {
        function_scope.Initialize(impl_bind, witness);
      }
      for (const auto& [impl_bind, witness] : fun_val.witnesses()) {
        function_scope.Initialize(impl_bind, witness);
      }
      BindingMap generic_args;
      CARBON_CHECK(PatternMatch(&function.param_pattern().value(),
                                converted_args, call.source_loc(),
                                &function_scope, generic_args, trace_stream_));
      CARBON_CHECK(function.body().has_value())
          << "Calling a function that's missing a body";
      return todo_.Spawn(std::make_unique<StatementAction>(*function.body()),
                         std::move(function_scope));
    }
    case Value::Kind::BoundMethodValue: {
      const auto& m = cast<BoundMethodValue>(*fun);
      const FunctionDeclaration& method = m.declaration();
      CARBON_CHECK(method.is_method());
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> converted_args,
          Convert(arg, &method.param_pattern().static_type(),
                  call.source_loc()));
      RuntimeScope method_scope(&heap_);
      BindingMap generic_args;
      CARBON_CHECK(PatternMatch(&method.me_pattern().value(), m.receiver(),
                                call.source_loc(), &method_scope, generic_args,
                                trace_stream_));
      CARBON_CHECK(PatternMatch(&method.param_pattern().value(), converted_args,
                                call.source_loc(), &method_scope, generic_args,
                                trace_stream_));
      // Bring the class type arguments into scope.
      for (const auto& [bind, val] : m.type_args()) {
        method_scope.Initialize(bind, val);
      }

      // Bring the impl witness tables into scope.
      for (const auto& [impl_bind, witness] : m.witnesses()) {
        method_scope.Initialize(impl_bind, witness);
      }
      CARBON_CHECK(method.body().has_value())
          << "Calling a method that's missing a body";
      return todo_.Spawn(std::make_unique<StatementAction>(*method.body()),
                         std::move(method_scope));
    }
    case Value::Kind::ParameterizedEntityName: {
      const auto& name = cast<ParameterizedEntityName>(*fun);
      const Declaration& decl = name.declaration();
      RuntimeScope params_scope(&heap_);
      BindingMap generic_args;
      CARBON_CHECK(PatternMatch(&name.params().value(), arg, call.source_loc(),
                                &params_scope, generic_args, trace_stream_));
      switch (decl.kind()) {
        case DeclarationKind::ClassDeclaration: {
          switch (phase()) {
            case Phase::RunTime:
              return todo_.FinishAction(arena_->New<NominalClassType>(
                  &cast<ClassDeclaration>(decl), generic_args, witnesses));
            case Phase::CompileTime:
              return todo_.FinishAction(arena_->New<NominalClassType>(
                  &cast<ClassDeclaration>(decl), generic_args, call.impls()));
          }
        }
        case DeclarationKind::InterfaceDeclaration: {
          switch (phase()) {
            case Phase::RunTime:
              return todo_.FinishAction(arena_->New<InterfaceType>(
                  &cast<InterfaceDeclaration>(decl), generic_args, witnesses));
            case Phase::CompileTime:
              return todo_.FinishAction(
                  arena_->New<InterfaceType>(&cast<InterfaceDeclaration>(decl),
                                             generic_args, call.impls()));
          }
        }
        default:
          CARBON_FATAL() << "unknown kind of ParameterizedEntityName " << decl;
      }
    }
    default:
      return RuntimeError(call.source_loc())
             << "in call, expected a function, not " << *fun;
  }
}

auto Interpreter::StepExp() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Expression& exp = cast<ExpressionAction>(act).expression();
  if (trace_stream_) {
    **trace_stream_ << "--- step exp " << exp << " (" << exp.source_loc()
                    << ") --->\n";
  }
  switch (exp.kind()) {
    case ExpressionKind::InstantiateImpl: {
      const InstantiateImpl& inst_impl = cast<InstantiateImpl>(exp);
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(inst_impl.generic_impl()));
      } else if (act.pos() - 1 < int(inst_impl.impls().size())) {
        auto iter = inst_impl.impls().begin();
        std::advance(iter, act.pos() - 1);
        return todo_.Spawn(std::make_unique<ExpressionAction>(iter->second));
      } else {
        Nonnull<const Witness*> generic_witness =
            cast<Witness>(act.results()[0]);
        ImplWitnessMap witnesses;
        int i = 0;
        for (const auto& [impl_bind, impl_exp] : inst_impl.impls()) {
          witnesses[impl_bind] = cast<Witness>(act.results()[i + 1]);
          ++i;
        }
        return todo_.FinishAction(arena_->New<Witness>(
            &generic_witness->declaration(), inst_impl.type_args(), witnesses));
      }
    }
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
          return RuntimeError(exp.source_loc())
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
        if (const auto* member_name_type =
                dyn_cast<TypeOfMemberName>(&access.static_type())) {
          CARBON_CHECK(phase() == Phase::CompileTime)
              << "should not form MemberNames at runtime";
          std::optional<const InterfaceType*> iface_result = std::nullopt;
          std::optional<const Value*> type_result = std::nullopt;
          if (auto* iface_type = dyn_cast<InterfaceType>(act.results()[0])) {
            iface_result = iface_type;
          } else {
            type_result = act.results()[0];
            if (access.impl().has_value()) {
              iface_result =
                  cast<InterfaceType>(access.impl().value()->interface());
            }
          }
          MemberName* member_name =
              arena_->New<MemberName>(type_result, iface_result, access.field(),
                                      &member_name_type->declaration());
          return todo_.FinishAction(member_name);
        } else {
          std::optional<Nonnull<const Witness*>> witness = std::nullopt;
          if (access.impl().has_value()) {
            CARBON_ASSIGN_OR_RETURN(
                auto witness_addr,
                todo_.ValueOfNode(*access.impl(), access.source_loc()));
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> witness_value,
                heap_.Read(llvm::cast<LValue>(witness_addr)->address(),
                           access.source_loc()));
            witness = cast<Witness>(witness_value);
          }
          FieldPath::Component field(access.field(), witness);
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> member,
              act.results()[0]->GetField(arena_, FieldPath(field),
                                         exp.source_loc()));
          return todo_.FinishAction(member);
        }
      }
    }
    case ExpressionKind::CompoundFieldAccessExpression: {
      const auto& access = cast<CompoundFieldAccessExpression>(exp);
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&access.object()));
      } else if (const auto* member_name_type =
                     dyn_cast<TypeOfMemberName>(&access.static_type())) {
        CARBON_CHECK(phase() == Phase::CompileTime)
            << "should not form MemberNames at runtime";
        CARBON_CHECK(!access.member().base_type().has_value())
            << "compound member access forming a member name should be "
               "performing impl lookup";
        auto* member_name = arena_->New<MemberName>(
            act.results()[0], access.member().interface(),
            access.member().name(), &access.member().declaration());
        return todo_.FinishAction(member_name);
      } else if (act.pos() == 1 && access.impl().has_value()) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(access.impl().value()));
      } else {
        Nonnull<const Value*> object = act.results()[0];
        std::optional<Nonnull<const Witness*>> witness = std::nullopt;
        if (access.impl().has_value()) {
          witness = cast<Witness>(act.results()[1]);
        } else {
          CARBON_CHECK(access.member().base_type().has_value())
              << "compound access should have base type or impl";
          CARBON_ASSIGN_OR_RETURN(
              object,
              Convert(object, *access.member().base_type(), exp.source_loc()));
        }
        FieldPath::Component field(access.member().name(), witness);
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> member,
            object->GetField(arena_, FieldPath(field), exp.source_loc()));
        return todo_.FinishAction(member);
      }
    }
    case ExpressionKind::IdentifierExpression: {
      CARBON_CHECK(act.pos() == 0);
      const auto& ident = cast<IdentifierExpression>(exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(ident.value_node(), ident.source_loc()));
      if (const auto* lvalue = dyn_cast<LValue>(value)) {
        CARBON_ASSIGN_OR_RETURN(
            value, heap_.Read(lvalue->address(), exp.source_loc()));
      }
      return todo_.FinishAction(value);
    }
    case ExpressionKind::IntLiteral:
      CARBON_CHECK(act.pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return todo_.FinishAction(
          arena_->New<IntValue>(cast<IntLiteral>(exp).value()));
    case ExpressionKind::BoolLiteral:
      CARBON_CHECK(act.pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return todo_.FinishAction(
          arena_->New<BoolValue>(cast<BoolLiteral>(exp).value()));
    case ExpressionKind::PrimitiveOperatorExpression: {
      const auto& op = cast<PrimitiveOperatorExpression>(exp);
      if (act.pos() != static_cast<int>(op.arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        Nonnull<const Expression*> arg = op.arguments()[act.pos()];
        if (op.op() == Operator::AddressOf) {
          return todo_.Spawn(std::make_unique<LValAction>(arg));
        } else {
          return todo_.Spawn(std::make_unique<ExpressionAction>(arg));
        }
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> value,
            EvalPrim(op.op(), act.results(), exp.source_loc()));
        return todo_.FinishAction(value);
      }
    }
    case ExpressionKind::CallExpression: {
      const CallExpression& call = cast<CallExpression>(exp);
      // Don't evaluate the impls at compile time?
      unsigned int num_impls =
          phase() == Phase::CompileTime ? 0 : call.impls().size();
      if (act.pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&call.function()));
      } else if (act.pos() == 1) {
        //    { { v :: [](e) :: C, E, F} :: S, H}
        // -> { { e :: v([]) :: C, E, F} :: S, H}
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&call.argument()));
      } else if (num_impls > 0 && act.pos() < 2 + int(num_impls)) {
        auto iter = call.impls().begin();
        std::advance(iter, act.pos() - 2);
        return todo_.Spawn(std::make_unique<ExpressionAction>(iter->second));
      } else if (act.pos() == 2 + int(num_impls)) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        ImplWitnessMap witnesses;
        if (num_impls > 0) {
          int i = 2;
          for (const auto& [impl_bind, impl_exp] : call.impls()) {
            witnesses[impl_bind] = cast<Witness>(act.results()[i]);
            ++i;
          }
        }
        return CallFunction(call, act.results()[0], act.results()[1],
                            witnesses);
      } else if (act.pos() == 3 + int(num_impls)) {
        if (act.results().size() < 3 + num_impls) {
          // Control fell through without explicit return.
          return todo_.FinishAction(TupleValue::Empty());
        } else {
          return todo_.FinishAction(act.results()[2 + int(num_impls)]);
        }
      } else {
        CARBON_FATAL() << "in StepExp with Call pos " << act.pos();
      }
    }
    case ExpressionKind::IntrinsicExpression: {
      const auto& intrinsic = cast<IntrinsicExpression>(exp);
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&intrinsic.args()));
      }
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      switch (cast<IntrinsicExpression>(exp).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print: {
          const auto& args = cast<TupleValue>(*act.results()[0]);
          // TODO: This could eventually use something like llvm::formatv.
          llvm::outs() << cast<StringValue>(*args.elements()[0]).value();
          return todo_.FinishAction(TupleValue::Empty());
        }
      }
    }
    case ExpressionKind::IntTypeLiteral: {
      CARBON_CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<IntType>());
    }
    case ExpressionKind::BoolTypeLiteral: {
      CARBON_CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<BoolType>());
    }
    case ExpressionKind::TypeTypeLiteral: {
      CARBON_CHECK(act.pos() == 0);
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
            act.results()[1], std::vector<Nonnull<const ImplBinding*>>()));
      }
    }
    case ExpressionKind::ContinuationTypeLiteral: {
      CARBON_CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<ContinuationType>());
    }
    case ExpressionKind::StringLiteral:
      CARBON_CHECK(act.pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return todo_.FinishAction(
          arena_->New<StringValue>(cast<StringLiteral>(exp).value()));
    case ExpressionKind::StringTypeLiteral: {
      CARBON_CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<StringType>());
    }
    case ExpressionKind::IfExpression: {
      const auto& if_expr = cast<IfExpression>(exp);
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&if_expr.condition()));
      } else if (act.pos() == 1) {
        const auto& condition = cast<BoolValue>(*act.results()[0]);
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            condition.value() ? &if_expr.then_expression()
                              : &if_expr.else_expression()));
      } else {
        return todo_.FinishAction(act.results()[1]);
      }
      break;
    }
    case ExpressionKind::UnimplementedExpression:
      CARBON_FATAL() << "Unimplemented: " << exp;
    case ExpressionKind::ArrayTypeLiteral: {
      const auto& array_literal = cast<ArrayTypeLiteral>(exp);
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &array_literal.element_type_expression()));
      } else if (act.pos() == 1) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &array_literal.size_expression()));
      } else {
        return todo_.FinishAction(arena_->New<StaticArrayType>(
            act.results()[0], cast<IntValue>(act.results()[1])->value()));
      }
    }
  }  // switch (exp->kind)
}

auto Interpreter::StepPattern() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Pattern& pattern = cast<PatternAction>(act).pattern();
  if (trace_stream_) {
    **trace_stream_ << "--- step pattern " << pattern << " ("
                    << pattern.source_loc() << ") --->\n";
  }
  switch (pattern.kind()) {
    case PatternKind::AutoPattern: {
      CARBON_CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<AutoType>());
    }
    case PatternKind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(pattern);
      if (binding.name() != AnonymousName) {
        return todo_.FinishAction(
            arena_->New<BindingPlaceholderValue>(&binding));
      } else {
        return todo_.FinishAction(arena_->New<BindingPlaceholderValue>());
      }
    }
    case PatternKind::GenericBinding: {
      const auto& binding = cast<GenericBinding>(pattern);
      return todo_.FinishAction(arena_->New<VariableType>(&binding));
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
        CARBON_CHECK(act.pos() == 2);
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
    case PatternKind::VarPattern:
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<PatternAction>(
            &cast<VarPattern>(pattern).pattern()));
      } else {
        return todo_.FinishAction(act.results()[0]);
      }
  }
}

auto Interpreter::StepStmt() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Statement& stmt = cast<StatementAction>(act).statement();
  if (trace_stream_) {
    **trace_stream_ << "--- step stmt ";
    stmt.PrintDepth(1, **trace_stream_);
    **trace_stream_ << " (" << stmt.source_loc() << ") --->\n";
  }
  switch (stmt.kind()) {
    case StatementKind::Match: {
      const auto& match_stmt = cast<Match>(stmt);
      if (act.pos() == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        act.StartScope(RuntimeScope(&heap_));
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&match_stmt.expression()));
      } else {
        int clause_num = act.pos() - 1;
        if (clause_num >= static_cast<int>(match_stmt.clauses().size())) {
          return todo_.FinishAction();
        }
        auto c = match_stmt.clauses()[clause_num];
        RuntimeScope matches(&heap_);
        BindingMap generic_args;
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> val,
            Convert(act.results()[0], &c.pattern().static_type(),
                    stmt.source_loc()));
        if (PatternMatch(&c.pattern().value(), val, stmt.source_loc(), &matches,
                         generic_args, trace_stream_)) {
          // Ensure we don't process any more clauses.
          act.set_pos(match_stmt.clauses().size() + 1);
          todo_.MergeScope(std::move(matches));
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
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> condition,
            Convert(act.results().back(), arena_->New<BoolType>(),
                    stmt.source_loc()));
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
      CARBON_CHECK(act.pos() == 0);
      //    { { break; :: ... :: (while (e) s) :: C, E, F} :: S, H}
      // -> { { C, E', F} :: S, H}
      return todo_.UnwindPast(&cast<Break>(stmt).loop());
    }
    case StatementKind::Continue: {
      CARBON_CHECK(act.pos() == 0);
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
        act.StartScope(RuntimeScope(&heap_));
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
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> v,
            Convert(act.results()[0], &definition.pattern().static_type(),
                    stmt.source_loc()));
        Nonnull<const Value*> p =
            &cast<VariableDefinition>(stmt).pattern().value();

        RuntimeScope matches(&heap_);
        BindingMap generic_args;
        CARBON_CHECK(PatternMatch(p, v, stmt.source_loc(), &matches,
                                  generic_args, trace_stream_))
            << stmt.source_loc()
            << ": internal error in variable definition, match failed";
        todo_.MergeScope(std::move(matches));
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
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> rval,
            Convert(act.results()[1], &assign.lhs().static_type(),
                    stmt.source_loc()));
        CARBON_RETURN_IF_ERROR(
            heap_.Write(lval.address(), rval, stmt.source_loc()));
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
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> condition,
            Convert(act.results()[0], arena_->New<BoolType>(),
                    stmt.source_loc()));
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
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> return_value,
            Convert(act.results()[0], &function.return_term().static_type(),
                    stmt.source_loc()));
        return todo_.UnwindPast(*function.body(), return_value);
      }
    case StatementKind::Continuation: {
      CARBON_CHECK(act.pos() == 0);
      const auto& continuation = cast<Continuation>(stmt);
      // Create a continuation object by creating a frame similar the
      // way one is created in a function call.
      auto fragment = arena_->New<ContinuationValue::StackFragment>();
      stack_fragments_.push_back(fragment);
      todo_.InitializeFragment(*fragment, &continuation.body());
      // Bind the continuation object to the continuation variable
      todo_.Initialize(&cast<Continuation>(stmt),
                       arena_->New<ContinuationValue>(fragment));
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
      CARBON_CHECK(act.pos() == 0);
      return todo_.Suspend();
  }
}

auto Interpreter::StepDeclaration() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Declaration& decl = cast<DeclarationAction>(act).declaration();
  if (trace_stream_) {
    **trace_stream_ << "--- step declaration (" << decl.source_loc()
                    << ") --->\n";
  }
  switch (decl.kind()) {
    case DeclarationKind::VariableDeclaration: {
      const auto& var_decl = cast<VariableDeclaration>(decl);
      if (var_decl.has_initializer()) {
        if (act.pos() == 0) {
          return todo_.Spawn(
              std::make_unique<ExpressionAction>(&var_decl.initializer()));
        } else {
          todo_.Initialize(&var_decl.binding(), act.results()[0]);
          return todo_.FinishAction();
        }
      } else {
        return todo_.FinishAction();
      }
    }
    case DeclarationKind::FunctionDeclaration:
    case DeclarationKind::ClassDeclaration:
    case DeclarationKind::ChoiceDeclaration:
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::SelfDeclaration:
      // These declarations have no run-time effects.
      return todo_.FinishAction();
  }
}

// State transition.
auto Interpreter::Step() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  switch (act.kind()) {
    case Action::Kind::LValAction:
      CARBON_RETURN_IF_ERROR(StepLvalue());
      break;
    case Action::Kind::ExpressionAction:
      CARBON_RETURN_IF_ERROR(StepExp());
      break;
    case Action::Kind::PatternAction:
      CARBON_RETURN_IF_ERROR(StepPattern());
      break;
    case Action::Kind::StatementAction:
      CARBON_RETURN_IF_ERROR(StepStmt());
      break;
    case Action::Kind::DeclarationAction:
      CARBON_RETURN_IF_ERROR(StepDeclaration());
      break;
    case Action::Kind::ScopeAction:
      CARBON_FATAL() << "ScopeAction escaped ActionStack";
  }  // switch
  return Success();
}

auto Interpreter::RunAllSteps(std::unique_ptr<Action> action)
    -> ErrorOr<Success> {
  if (trace_stream_) {
    PrintState(**trace_stream_);
  }
  todo_.Start(std::move(action));
  while (!todo_.IsEmpty()) {
    CARBON_RETURN_IF_ERROR(Step());
    if (trace_stream_) {
      PrintState(**trace_stream_);
    }
  }
  return Success();
}

auto InterpProgram(const AST& ast, Nonnull<Arena*> arena,
                   std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
    -> ErrorOr<int> {
  Interpreter interpreter(Phase::RunTime, arena, trace_stream);
  if (trace_stream) {
    **trace_stream << "********** initializing globals **********\n";
  }

  for (Nonnull<Declaration*> declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(interpreter.RunAllSteps(
        std::make_unique<DeclarationAction>(declaration)));
  }

  if (trace_stream) {
    **trace_stream << "********** calling main function **********\n";
  }

  CARBON_RETURN_IF_ERROR(interpreter.RunAllSteps(
      std::make_unique<ExpressionAction>(*ast.main_call)));

  return cast<IntValue>(*interpreter.result()).value();
}

auto InterpExp(Nonnull<const Expression*> e, Nonnull<Arena*> arena,
               std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
    -> ErrorOr<Nonnull<const Value*>> {
  Interpreter interpreter(Phase::CompileTime, arena, trace_stream);
  CARBON_RETURN_IF_ERROR(
      interpreter.RunAllSteps(std::make_unique<ExpressionAction>(e)));
  return interpreter.result();
}

auto InterpPattern(Nonnull<const Pattern*> p, Nonnull<Arena*> arena,
                   std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
    -> ErrorOr<Nonnull<const Value*>> {
  Interpreter interpreter(Phase::CompileTime, arena, trace_stream);
  CARBON_RETURN_IF_ERROR(
      interpreter.RunAllSteps(std::make_unique<PatternAction>(p)));
  return interpreter.result();
}

}  // namespace Carbon
