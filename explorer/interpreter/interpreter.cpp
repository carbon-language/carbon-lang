// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/interpreter.h"

#include <llvm/Support/raw_ostream.h>

#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <variant>
#include <vector>

#include "common/check.h"
#include "common/error.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/element.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/value.h"
#include "explorer/common/arena.h"
#include "explorer/common/error_builders.h"
#include "explorer/common/source_location.h"
#include "explorer/interpreter/action.h"
#include "explorer/interpreter/action_stack.h"
#include "explorer/interpreter/stack.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {

// Limits for various overflow conditions.
static constexpr int64_t MaxTodoSize = 1e3;
static constexpr int64_t MaxStepsTaken = 1e6;
static constexpr int64_t MaxArenaAllocated = 1e9;

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
              Nonnull<TraceStream*> trace_stream,
              Nonnull<llvm::raw_ostream*> print_stream)
      : arena_(arena),
        heap_(arena),
        todo_(MakeTodo(phase, &heap_)),
        trace_stream_(trace_stream),
        print_stream_(print_stream),
        phase_(phase) {}

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
  auto StepLocation() -> ErrorOr<Success>;
  // State transitions for witnesses.
  auto StepWitness() -> ErrorOr<Success>;
  // State transition for statements.
  auto StepStmt() -> ErrorOr<Success>;
  // State transition for declarations.
  auto StepDeclaration() -> ErrorOr<Success>;
  // State transition for object destruction.
  auto StepCleanUp() -> ErrorOr<Success>;
  auto StepDestroy() -> ErrorOr<Success>;
  // State transition for type instantiation.
  auto StepInstantiateType() -> ErrorOr<Success>;

  auto CreateStruct(const std::vector<FieldInitializer>& fields,
                    const std::vector<Nonnull<const Value*>>& values)
      -> Nonnull<const Value*>;

  auto EvalPrim(Operator op, Nonnull<const Value*> static_type,
                const std::vector<Nonnull<const Value*>>& args,
                SourceLocation source_loc) -> ErrorOr<Nonnull<const Value*>>;

  // Returns the result of converting `value` to type `destination_type`.
  auto Convert(Nonnull<const Value*> value,
               Nonnull<const Value*> destination_type,
               SourceLocation source_loc) -> ErrorOr<Nonnull<const Value*>>;

  // Create a class value and its base class(es) from an init struct.
  auto ConvertStructToClass(Nonnull<const StructValue*> init,
                            Nonnull<const NominalClassType*> class_type,
                            SourceLocation source_loc)
      -> ErrorOr<Nonnull<NominalClassValue*>>;

  // Evaluate an expression immediately, recursively, and return its result.
  //
  // TODO: Stop using this.
  auto EvalRecursively(std::unique_ptr<Action> action)
      -> ErrorOr<Nonnull<const Value*>>;

  // Evaluate an associated constant by evaluating its witness and looking
  // inside the impl for the corresponding value.
  //
  // TODO: This approach doesn't provide values that are known because they
  // appear in constraints:
  //
  //   interface Iface { let N:! i32; }
  //   fn PickType(N: i32) -> type { return i32; }
  //   fn F[T:! Iface where .N == 5](x: T) {
  //     var x: PickType(T.N) = 0;
  //   }
  //
  // ... will fail because we can't resolve T.N to 5 at compile time.
  auto EvalAssociatedConstant(Nonnull<const AssociatedConstant*> assoc,
                              SourceLocation source_loc)
      -> ErrorOr<Nonnull<const Value*>>;

  // Instantiate a type by replacing all type variables that occur inside the
  // type by the current values of those variables.
  //
  // For example, suppose T=i32 and U=bool. Then
  //     __Fn (Point(T)) -> Point(U)
  // becomes
  //     __Fn (Point(i32)) -> Point(bool)
  //
  // TODO: This should be an Action.
  auto InstantiateType(Nonnull<const Value*> type, SourceLocation source_loc)
      -> ErrorOr<Nonnull<const Value*>>;

  // Instantiate a set of bindings by replacing all type variables that occur
  // within it by the current values of those variables.
  auto InstantiateBindings(Nonnull<const Bindings*> bindings,
                           SourceLocation source_loc)
      -> ErrorOr<Nonnull<const Bindings*>>;

  // Instantiate a witness by replacing all type variables and impl binding
  // references that occur within it by the current values of those variables.
  auto InstantiateWitness(Nonnull<const Witness*> witness)
      -> ErrorOr<Nonnull<const Witness*>>;

  // Call the function `fun` with the given `arg` and the `witnesses`
  // for the function's impl bindings.
  auto CallFunction(const CallExpression& call, Nonnull<const Value*> fun,
                    Nonnull<const Value*> arg, ImplWitnessMap&& witnesses)
      -> ErrorOr<Success>;

  auto CallDestructor(Nonnull<const DestructorDeclaration*> fun,
                      Nonnull<const Value*> receiver) -> ErrorOr<Success>;

  void TraceState();

  auto phase() const -> Phase { return phase_; }

  Nonnull<Arena*> arena_;

  Heap heap_;
  ActionStack todo_;

  Nonnull<TraceStream*> trace_stream_;

  // The stream for the Print intrinsic.
  Nonnull<llvm::raw_ostream*> print_stream_;

  Phase phase_;

  // The number of steps taken by the interpreter. Used for infinite loop
  // detection.
  int64_t steps_taken_ = 0;
};

//
// State Operations
//

void Interpreter::TraceState() {
  *trace_stream_ << "{\nstack: " << todo_ << "\nmemory: " << heap_ << "\n}\n";
}

auto Interpreter::EvalPrim(Operator op, Nonnull<const Value*> /*static_type*/,
                           const std::vector<Nonnull<const Value*>>& args,
                           SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  switch (op) {
    case Operator::Neg:
    case Operator::Add:
    case Operator::Sub:
    case Operator::Div:
    case Operator::Mul: {
      llvm::APInt op0(64, cast<IntValue>(*args[0]).value());
      llvm::APInt result;
      if (op == Operator::Neg) {
        result = -op0;
      } else {
        llvm::APInt op1(64, cast<IntValue>(*args[1]).value());
        if (op == Operator::Add) {
          result = op0 + op1;
        } else if (op == Operator::Sub) {
          result = op0 - op1;
        } else if (op == Operator::Mul) {
          result = op0 * op1;
        } else if (op == Operator::Div) {
          if (op1.getSExtValue() == 0) {
            return ProgramError(source_loc) << "division by zero";
          }
          result = op0.sdiv(op1);
        }
      }
      if (result.isSignedIntN(32)) {
        return arena_->New<IntValue>(result.getSExtValue());
      } else {
        return ProgramError(source_loc) << "integer overflow";
      }
    }
    case Operator::Mod: {
      const auto& lhs = cast<IntValue>(*args[0]).value();
      const auto& rhs = cast<IntValue>(*args[1]).value();
      if (rhs == 0) {
        return ProgramError(source_loc) << "division by zero";
      }
      return arena_->New<IntValue>(lhs % rhs);
    }
    case Operator::Not:
      return arena_->New<BoolValue>(!cast<BoolValue>(*args[0]).value());
    case Operator::And:
      return arena_->New<BoolValue>(cast<BoolValue>(*args[0]).value() &&
                                    cast<BoolValue>(*args[1]).value());
    case Operator::Or:
      return arena_->New<BoolValue>(cast<BoolValue>(*args[0]).value() ||
                                    cast<BoolValue>(*args[1]).value());
    case Operator::Ptr:
      return arena_->New<PointerType>(args[0]);
    case Operator::Deref:
      return heap_.Read(cast<PointerValue>(*args[0]).address(), source_loc);
    case Operator::AddressOf:
      return arena_->New<PointerValue>(cast<LocationValue>(*args[0]).address());
    case Operator::As:
    case Operator::Eq:
    case Operator::NotEq:
    case Operator::Less:
    case Operator::LessEq:
    case Operator::Greater:
    case Operator::GreaterEq:
    case Operator::BitwiseAnd:
    case Operator::BitwiseOr:
    case Operator::BitwiseXor:
    case Operator::BitShiftLeft:
    case Operator::BitShiftRight:
    case Operator::Complement:
      CARBON_FATAL() << "operator " << OperatorToString(op)
                     << " should always be rewritten";
  }
}

auto Interpreter::CreateStruct(const std::vector<FieldInitializer>& fields,
                               const std::vector<Nonnull<const Value*>>& values)
    -> Nonnull<const Value*> {
  CARBON_CHECK(fields.size() == values.size());
  std::vector<NamedValue> elements;
  for (size_t i = 0; i < fields.size(); ++i) {
    elements.push_back({fields[i].name(), values[i]});
  }

  return arena_->New<StructValue>(std::move(elements));
}

auto PatternMatch(Nonnull<const Value*> p, Nonnull<const Value*> v,
                  SourceLocation source_loc,
                  std::optional<Nonnull<RuntimeScope*>> bindings,
                  BindingMap& generic_args, Nonnull<TraceStream*> trace_stream,
                  Nonnull<Arena*> arena) -> bool {
  if (trace_stream->is_enabled()) {
    *trace_stream << "match pattern " << *p << "\nwith value " << *v << "\n";
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
    case Value::Kind::AddrValue: {
      const auto& addr = cast<AddrValue>(*p);
      CARBON_CHECK(v->kind() == Value::Kind::LocationValue);
      const auto& location = cast<LocationValue>(*v);
      return PatternMatch(
          &addr.pattern(), arena->New<PointerValue>(location.address()),
          source_loc, bindings, generic_args, trace_stream, arena);
    }
    case Value::Kind::VariableType: {
      const auto& var_type = cast<VariableType>(*p);
      generic_args[&var_type.binding()] = v;
      return true;
    }
    case Value::Kind::TupleType:
    case Value::Kind::TupleValue:
      switch (v->kind()) {
        case Value::Kind::TupleType:
        case Value::Kind::TupleValue: {
          const auto& p_tup = cast<TupleValueBase>(*p);
          const auto& v_tup = cast<TupleValueBase>(*v);
          CARBON_CHECK(p_tup.elements().size() == v_tup.elements().size());
          for (size_t i = 0; i < p_tup.elements().size(); ++i) {
            if (!PatternMatch(p_tup.elements()[i], v_tup.elements()[i],
                              source_loc, bindings, generic_args, trace_stream,
                              arena)) {
              return false;
            }
          }  // for
          return true;
        }
        case Value::Kind::UninitializedValue: {
          const auto& p_tup = cast<TupleValueBase>(*p);
          for (const auto& ele : p_tup.elements()) {
            if (!PatternMatch(ele, arena->New<UninitializedValue>(ele),
                              source_loc, bindings, generic_args, trace_stream,
                              arena)) {
              return false;
            }
          }
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
                          generic_args, trace_stream, arena)) {
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
          if (&p_alt.alternative() != &v_alt.alternative()) {
            return false;
          }
          CARBON_CHECK(p_alt.argument().has_value() ==
                       v_alt.argument().has_value());
          if (!p_alt.argument().has_value()) {
            return true;
          }
          return PatternMatch(*p_alt.argument(), *v_alt.argument(), source_loc,
                              bindings, generic_args, trace_stream, arena);
        }
        default:
          CARBON_FATAL() << "expected a choice alternative in pattern, not "
                         << *v;
      }
    case Value::Kind::UninitializedValue:
      CARBON_FATAL() << "uninitialized value is not allowed in pattern " << *v;
    case Value::Kind::FunctionType:
      switch (v->kind()) {
        case Value::Kind::FunctionType: {
          const auto& p_fn = cast<FunctionType>(*p);
          const auto& v_fn = cast<FunctionType>(*v);
          if (!PatternMatch(&p_fn.parameters(), &v_fn.parameters(), source_loc,
                            bindings, generic_args, trace_stream, arena)) {
            return false;
          }
          if (!PatternMatch(&p_fn.return_type(), &v_fn.return_type(),
                            source_loc, bindings, generic_args, trace_stream,
                            arena)) {
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
      return ValueEqual(p, v, std::nullopt);
  }
}

auto Interpreter::StepLocation() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Expression& exp = cast<LocationAction>(act).expression();
  if (trace_stream_->is_enabled()) {
    *trace_stream_ << "--- step location " << exp << " ." << act.pos() << "."
                   << " (" << exp.source_loc() << ") --->\n";
  }
  switch (exp.kind()) {
    case ExpressionKind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(cast<IdentifierExpression>(exp).value_node(),
                            exp.source_loc()));
      CARBON_CHECK(isa<LocationValue>(value)) << *value;
      return todo_.FinishAction(value);
    }
    case ExpressionKind::SimpleMemberAccessExpression: {
      const auto& access = cast<SimpleMemberAccessExpression>(exp);
      const auto constant_value = access.constant_value();
      if (auto rewrite = access.rewritten_form()) {
        return todo_.ReplaceWith(std::make_unique<LocationAction>(*rewrite));
      }
      if (act.pos() == 0) {
        //    { {e.f :: C, E, F} :: S, H}
        // -> { e :: [].f :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<LocationAction>(&access.object()));
      } else if (act.pos() == 1 && constant_value) {
        return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
            *constant_value, access.source_loc()));
      } else {
        if (constant_value) {
          return todo_.FinishAction(act.results().back());
        } else {
          //    { v :: [].f :: C, E, F} :: S, H}
          // -> { { &v.f :: C, E, F} :: S, H }
          Address object = cast<LocationValue>(*act.results()[0]).address();
          Address member = object.ElementAddress(&access.member());
          return todo_.FinishAction(arena_->New<LocationValue>(member));
        }
      }
    }
    case ExpressionKind::CompoundMemberAccessExpression: {
      const auto& access = cast<CompoundMemberAccessExpression>(exp);
      const auto constant_value = access.constant_value();
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<LocationAction>(&access.object()));
      }
      if (act.pos() == 1 && constant_value) {
        return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
            *constant_value, access.source_loc()));
      } else {
        if (constant_value) {
          return todo_.FinishAction(act.results().back());
        }
        CARBON_CHECK(!access.member().interface().has_value())
            << "unexpected location interface member";
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> val,
            Convert(act.results()[0], *access.member().base_type(),
                    exp.source_loc()));
        Address object = cast<LocationValue>(*val).address();
        Address field = object.ElementAddress(&access.member().member());
        return todo_.FinishAction(arena_->New<LocationValue>(field));
      }
    }
    case ExpressionKind::BaseAccessExpression: {
      const auto& access = cast<BaseAccessExpression>(exp);
      if (act.pos() == 0) {
        // Get LocationValue for expression.
        return todo_.Spawn(std::make_unique<LocationAction>(&access.object()));
      } else {
        // Append `.base` element to the address, and return the new
        // LocationValue.
        Address object = cast<LocationValue>(*act.results()[0]).address();
        Address base = object.ElementAddress(&access.element());
        return todo_.FinishAction(arena_->New<LocationValue>(base));
      }
    }
    case ExpressionKind::IndexExpression: {
      if (act.pos() == 0) {
        //    { {e[i] :: C, E, F} :: S, H}
        // -> { e :: [][i] :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<LocationAction>(
            &cast<IndexExpression>(exp).object()));

      } else if (act.pos() == 1) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<IndexExpression>(exp).offset()));
      } else {
        //    { v :: [][i] :: C, E, F} :: S, H}
        // -> { { &v[i] :: C, E, F} :: S, H }
        Address object = cast<LocationValue>(*act.results()[0]).address();
        const auto index = cast<IntValue>(*act.results()[1]).value();
        Address field = object.ElementAddress(
            arena_->New<PositionalElement>(index, &exp.static_type()));
        return todo_.FinishAction(arena_->New<LocationValue>(field));
      }
    }
    case ExpressionKind::OperatorExpression: {
      const auto& op = cast<OperatorExpression>(exp);
      if (auto rewrite = op.rewritten_form()) {
        return todo_.ReplaceWith(std::make_unique<LocationAction>(*rewrite));
      }
      if (op.op() != Operator::Deref) {
        CARBON_FATAL()
            << "Can't treat primitive operator expression as location: " << exp;
      }
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(op.arguments()[0]));
      } else {
        const auto& res = cast<PointerValue>(*act.results()[0]);
        return todo_.FinishAction(arena_->New<LocationValue>(res.address()));
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
    case ExpressionKind::StringLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::ValueLiteral:
    case ExpressionKind::IntrinsicExpression:
    case ExpressionKind::IfExpression:
    case ExpressionKind::WhereExpression:
    case ExpressionKind::DotSelfExpression:
    case ExpressionKind::ArrayTypeLiteral:
    case ExpressionKind::BuiltinConvertExpression:
      CARBON_FATAL() << "Can't treat expression as location: " << exp;
    case ExpressionKind::UnimplementedExpression:
      CARBON_FATAL() << "Unimplemented: " << exp;
  }
}

auto Interpreter::EvalRecursively(std::unique_ptr<Action> action)
    -> ErrorOr<Nonnull<const Value*>> {
  if (trace_stream_->is_enabled()) {
    *trace_stream_ << "--- recursive eval\n";
    TraceState();
  }
  todo_.BeginRecursiveAction();
  CARBON_RETURN_IF_ERROR(todo_.Spawn(std::move(action)));
  // Note that the only `RecursiveAction` we can encounter here is our own --
  // if a nested action begins a recursive action, it will run until that
  // action is finished and popped off the queue before returning to us.
  while (!isa<RecursiveAction>(todo_.CurrentAction())) {
    CARBON_RETURN_IF_ERROR(Step());
    if (trace_stream_->is_enabled()) {
      TraceState();
    }
  }
  if (trace_stream_->is_enabled()) {
    *trace_stream_ << "--- recursive eval done\n";
  }
  Nonnull<const Value*> result =
      cast<RecursiveAction>(todo_.CurrentAction()).results()[0];
  CARBON_RETURN_IF_ERROR(todo_.FinishAction());
  return result;
}

auto Interpreter::EvalAssociatedConstant(
    Nonnull<const AssociatedConstant*> assoc, SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  // Instantiate the associated constant.
  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> interface,
                          InstantiateType(&assoc->interface(), source_loc));
  CARBON_ASSIGN_OR_RETURN(Nonnull<const Witness*> witness,
                          InstantiateWitness(&assoc->witness()));

  const auto* impl_witness = dyn_cast<ImplWitness>(witness);
  if (!impl_witness) {
    CARBON_CHECK(phase() == Phase::CompileTime)
        << "symbolic witnesses should only be formed at compile time";
    CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> base,
                            InstantiateType(&assoc->base(), source_loc));
    return arena_->New<AssociatedConstant>(base, cast<InterfaceType>(interface),
                                           &assoc->constant(), witness);
  }

  // We have an impl. Extract the value from it.
  Nonnull<const ConstraintType*> constraint =
      impl_witness->declaration().constraint_type();
  std::optional<Nonnull<const Value*>> result;
  for (const auto& rewrite : constraint->rewrite_constraints()) {
    if (&rewrite.constant->constant() == &assoc->constant() &&
        TypeEqual(&rewrite.constant->interface(), interface, std::nullopt)) {
      // TODO: The value might depend on the parameters of the impl. We need to
      // substitute impl_witness->type_args() into the value.
      result = rewrite.converted_replacement;
      break;
    }
  }
  if (!result) {
    CARBON_FATAL() << impl_witness->declaration() << " with constraint "
                   << *constraint
                   << " is missing value for associated constant "
                   << *interface << "." << assoc->constant().binding().name();
  }
  return *result;
}

auto Interpreter::InstantiateType(Nonnull<const Value*> type,
                                  SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  switch (type->kind()) {
    case Value::Kind::VariableType: {
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(&cast<VariableType>(*type).binding(), source_loc));
      if (const auto* location = dyn_cast<LocationValue>(value)) {
        CARBON_ASSIGN_OR_RETURN(value,
                                heap_.Read(location->address(), source_loc));
      }
      return value;
    }
    case Value::Kind::InterfaceType: {
      const auto& interface_type = cast<InterfaceType>(*type);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Bindings*> bindings,
          InstantiateBindings(&interface_type.bindings(), source_loc));
      return arena_->New<InterfaceType>(&interface_type.declaration(),
                                        bindings);
    }
    case Value::Kind::NamedConstraintType: {
      const auto& constraint_type = cast<NamedConstraintType>(*type);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Bindings*> bindings,
          InstantiateBindings(&constraint_type.bindings(), source_loc));
      return arena_->New<NamedConstraintType>(&constraint_type.declaration(),
                                              bindings);
    }
    case Value::Kind::ChoiceType: {
      const auto& choice_type = cast<ChoiceType>(*type);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Bindings*> bindings,
          InstantiateBindings(&choice_type.bindings(), source_loc));
      return arena_->New<ChoiceType>(&choice_type.declaration(), bindings);
    }
    case Value::Kind::AssociatedConstant: {
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> type_value,
          EvalAssociatedConstant(cast<AssociatedConstant>(type), source_loc));
      return type_value;
    }
    default:
      return type;
  }
}

auto Interpreter::InstantiateBindings(Nonnull<const Bindings*> bindings,
                                      SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Bindings*>> {
  BindingMap args = bindings->args();
  for (auto& [var, arg] : args) {
    CARBON_ASSIGN_OR_RETURN(arg, InstantiateType(arg, source_loc));
  }

  ImplWitnessMap witnesses = bindings->witnesses();
  for (auto& [bind, witness] : witnesses) {
    CARBON_ASSIGN_OR_RETURN(witness,
                            InstantiateWitness(cast<Witness>(witness)));
  }

  if (args == bindings->args() && witnesses == bindings->witnesses()) {
    return bindings;
  }
  return arena_->New<Bindings>(std::move(args), std::move(witnesses));
}

auto Interpreter::InstantiateWitness(Nonnull<const Witness*> witness)
    -> ErrorOr<Nonnull<const Witness*>> {
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<const Value*> value,
      EvalRecursively(std::make_unique<WitnessAction>(witness)));
  return cast<Witness>(value);
}

auto Interpreter::ConvertStructToClass(
    Nonnull<const StructValue*> init_struct,
    Nonnull<const NominalClassType*> class_type, SourceLocation source_loc)
    -> ErrorOr<Nonnull<NominalClassValue*>> {
  std::vector<NamedValue> struct_values;
  std::optional<Nonnull<const NominalClassValue*>> base_instance;
  // Instantiate the `destination_type` to obtain the runtime
  // type of the object.
  CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> inst_class,
                          InstantiateType(class_type, source_loc));
  for (const auto& field : init_struct->elements()) {
    if (field.name == NominalClassValue::BaseField) {
      CARBON_CHECK(class_type->base().has_value())
          << "Invalid 'base' field for class '"
          << class_type->declaration().name() << "' without base class.";
      CARBON_ASSIGN_OR_RETURN(
          auto base,
          Convert(field.value, class_type->base().value(), source_loc));
      base_instance = cast<NominalClassValue>(base);
    } else {
      struct_values.push_back(field);
    }
  }
  CARBON_CHECK(!cast<NominalClassType>(inst_class)->base() || base_instance)
      << "Invalid conversion for `" << *inst_class << "`: base class missing";
  auto* converted_init_struct =
      arena_->New<StructValue>(std::move(struct_values));
  Nonnull<const NominalClassValue** const> class_value_ptr =
      base_instance ? (*base_instance)->class_value_ptr()
                    : arena_->New<const NominalClassValue*>();
  return arena_->New<NominalClassValue>(inst_class, converted_init_struct,
                                        base_instance, class_value_ptr);
}

auto Interpreter::Convert(Nonnull<const Value*> value,
                          Nonnull<const Value*> destination_type,
                          SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Value*>> {
  switch (value->kind()) {
    case Value::Kind::IntValue:
    case Value::Kind::FunctionValue:
    case Value::Kind::DestructorValue:
    case Value::Kind::BoundMethodValue:
    case Value::Kind::LocationValue:
    case Value::Kind::BoolValue:
    case Value::Kind::NominalClassValue:
    case Value::Kind::AlternativeValue:
    case Value::Kind::UninitializedValue:
    case Value::Kind::IntType:
    case Value::Kind::BoolType:
    case Value::Kind::TypeType:
    case Value::Kind::FunctionType:
    case Value::Kind::PointerType:
    case Value::Kind::TupleType:
    case Value::Kind::StructType:
    case Value::Kind::AutoType:
    case Value::Kind::NominalClassType:
    case Value::Kind::MixinPseudoType:
    case Value::Kind::InterfaceType:
    case Value::Kind::NamedConstraintType:
    case Value::Kind::ConstraintType:
    case Value::Kind::ImplWitness:
    case Value::Kind::BindingWitness:
    case Value::Kind::ConstraintWitness:
    case Value::Kind::ConstraintImplWitness:
    case Value::Kind::ParameterizedEntityName:
    case Value::Kind::ChoiceType:
    case Value::Kind::VariableType:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::StringType:
    case Value::Kind::StringValue:
    case Value::Kind::TypeOfMixinPseudoType:
    case Value::Kind::TypeOfParameterizedEntityName:
    case Value::Kind::TypeOfMemberName:
    case Value::Kind::TypeOfNamespaceName:
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
            new_elements.push_back({field_name, val});
          }
          return arena_->New<StructValue>(std::move(new_elements));
        }
        case Value::Kind::NominalClassType: {
          CARBON_ASSIGN_OR_RETURN(
              auto class_value,
              ConvertStructToClass(cast<StructValue>(value),
                                   cast<NominalClassType>(destination_type),
                                   source_loc));
          return class_value;
        }
        case Value::Kind::TypeType:
        case Value::Kind::ConstraintType:
        case Value::Kind::NamedConstraintType:
        case Value::Kind::InterfaceType: {
          CARBON_CHECK(struct_val.elements().empty())
              << "only empty structs convert to `type`";
          return arena_->New<StructType>();
        }
        default: {
          CARBON_CHECK(IsValueKindDependent(destination_type) ||
                       isa<TypeType, ConstraintType>(destination_type))
              << "Can't convert value " << *value << " to type "
              << *destination_type;
          return value;
        }
      }
    }
    case Value::Kind::TupleValue: {
      const auto* tuple = cast<TupleValue>(value);
      std::vector<Nonnull<const Value*>> destination_element_types;
      switch (destination_type->kind()) {
        case Value::Kind::TupleType:
          destination_element_types =
              cast<TupleType>(destination_type)->elements();
          break;
        case Value::Kind::StaticArrayType: {
          const auto& array_type = cast<StaticArrayType>(*destination_type);
          destination_element_types.resize(array_type.size(),
                                           &array_type.element_type());
          break;
        }
        case Value::Kind::TypeType:
        case Value::Kind::ConstraintType:
        case Value::Kind::NamedConstraintType:
        case Value::Kind::InterfaceType: {
          std::vector<Nonnull<const Value*>> new_elements;
          Nonnull<const Value*> type_type = arena_->New<TypeType>();
          for (Nonnull<const Value*> value : tuple->elements()) {
            CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> value_as_type,
                                    Convert(value, type_type, source_loc));
            new_elements.push_back(value_as_type);
          }
          return arena_->New<TupleType>(std::move(new_elements));
        }
        default: {
          CARBON_CHECK(IsValueKindDependent(destination_type) ||
                       isa<TypeType, ConstraintType>(destination_type))
              << "Can't convert value " << *value << " to type "
              << *destination_type;
          return value;
        }
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
    case Value::Kind::AssociatedConstant: {
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          EvalAssociatedConstant(cast<AssociatedConstant>(value), source_loc));
      if (const auto* new_const = dyn_cast<AssociatedConstant>(value)) {
        // TODO: Detect whether conversions are required in type-checking.
        if (isa<TypeType, ConstraintType, NamedConstraintType, InterfaceType>(
                destination_type) &&
            isa<TypeType, ConstraintType, NamedConstraintType, InterfaceType>(
                new_const->constant().static_type())) {
          // No further conversions are required.
          return value;
        }
        // We need to convert this, and we don't know how because we don't have
        // the value yet.
        return ProgramError(source_loc)
               << "value of associated constant " << *value << " is not known";
      }
      return Convert(value, destination_type, source_loc);
    }
    case Value::Kind::PointerValue: {
      if (destination_type->kind() != Value::Kind::PointerType ||
          cast<PointerType>(destination_type)->pointee_type().kind() !=
              Value::Kind::NominalClassType) {
        // No conversion needed.
        return value;
      }

      // Get pointee value.
      const auto* src_ptr = cast<PointerValue>(value);
      CARBON_ASSIGN_OR_RETURN(const auto* pointee,
                              heap_.Read(src_ptr->address(), source_loc))
      CARBON_CHECK(pointee->kind() == Value::Kind::NominalClassValue)
          << "Unexpected pointer type";

      // Conversion logic for subtyping for function arguments only.
      // TODO: Drop when able to rewrite subtyping in TypeChecker for arguments.
      const auto* dest_ptr = cast<PointerType>(destination_type);
      std::optional<Nonnull<const NominalClassValue*>> class_subobj =
          cast<NominalClassValue>(pointee);
      auto new_addr = src_ptr->address();
      while (class_subobj) {
        if (TypeEqual(&(*class_subobj)->type(), &dest_ptr->pointee_type(),
                      std::nullopt)) {
          return arena_->New<PointerValue>(new_addr);
        }
        class_subobj = (*class_subobj)->base();
        new_addr = new_addr.ElementAddress(
            arena_->New<BaseElement>(&dest_ptr->pointee_type()));
      }

      // Unable to resolve, return as-is.
      // TODO: Produce error instead once we can properly substitute
      // parameterized types for pointers in function call parameters.
      return value;
    }
  }
}

auto Interpreter::CallDestructor(Nonnull<const DestructorDeclaration*> fun,
                                 Nonnull<const Value*> receiver)
    -> ErrorOr<Success> {
  const DestructorDeclaration& method = *fun;
  CARBON_CHECK(method.is_method());
  RuntimeScope method_scope(&heap_);
  BindingMap generic_args;

  // TODO: move this logic into PatternMatch, and call it here.
  const auto* p = &method.self_pattern().value();
  const auto* placeholder = dyn_cast<BindingPlaceholderValue>(p);
  if (!placeholder) {
    // TODO: Fix this, probably merging logic with CallFunction.
    // https://github.com/carbon-language/carbon-lang/issues/2802
    return ProgramError(fun->source_loc())
           << "destructors currently don't support `addr self` bindings";
  }
  if (placeholder->value_node().has_value()) {
    method_scope.Bind(*placeholder->value_node(), receiver);
  }
  CARBON_CHECK(method.body().has_value())
      << "Calling a method that's missing a body";

  auto act = std::make_unique<StatementAction>(*method.body());
  return todo_.Spawn(std::unique_ptr<Action>(std::move(act)),
                     std::move(method_scope));
}

auto Interpreter::CallFunction(const CallExpression& call,
                               Nonnull<const Value*> fun,
                               Nonnull<const Value*> arg,
                               ImplWitnessMap&& witnesses) -> ErrorOr<Success> {
  if (trace_stream_->is_enabled()) {
    *trace_stream_ << "calling function: " << *fun << "\n";
  }
  switch (fun->kind()) {
    case Value::Kind::AlternativeConstructorValue: {
      const auto& alt = cast<AlternativeConstructorValue>(*fun);
      return todo_.FinishAction(arena_->New<AlternativeValue>(
          &alt.choice(), &alt.alternative(), cast<TupleValue>(arg)));
    }
    case Value::Kind::FunctionValue:
    case Value::Kind::BoundMethodValue: {
      const auto* func_val = cast<FunctionOrMethodValue>(fun);

      const FunctionDeclaration& function = func_val->declaration();
      if (!function.body().has_value()) {
        return ProgramError(call.source_loc())
               << "attempt to call function `" << function.name()
               << "` that has not been defined";
      }
      if (!function.is_type_checked()) {
        return ProgramError(call.source_loc())
               << "attempt to call function `" << function.name()
               << "` that has not been fully type-checked";
      }

      RuntimeScope binding_scope(&heap_);

      // Bring the deduced arguments and their witnesses into scope.
      for (const auto& [bind, val] : call.deduced_args()) {
        CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> inst_val,
                                InstantiateType(val, call.source_loc()));
        binding_scope.Initialize(bind->original(), inst_val);
      }
      for (const auto& [impl_bind, witness] : witnesses) {
        binding_scope.Initialize(impl_bind->original(), witness);
      }

      // Bring the arguments that are determined by the function value into
      // scope. This includes the arguments for the class of which the function
      // is a member.
      for (const auto& [bind, val] : func_val->type_args()) {
        binding_scope.Initialize(bind->original(), val);
      }
      for (const auto& [impl_bind, witness] : func_val->witnesses()) {
        binding_scope.Initialize(impl_bind->original(), witness);
      }

      // Enter the binding scope to make any deduced arguments visible before
      // we resolve the self type and parameter type.
      todo_.CurrentAction().StartScope(std::move(binding_scope));

      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> converted_args,
          Convert(arg, &function.param_pattern().static_type(),
                  call.source_loc()));

      RuntimeScope function_scope(&heap_);
      BindingMap generic_args;

      // Bind the receiver to the `self` parameter, if there is one.
      if (const auto* method_val = dyn_cast<BoundMethodValue>(func_val)) {
        CARBON_CHECK(function.is_method());
        const auto* self_pattern = &function.self_pattern().value();
        if (const auto* placeholder =
                dyn_cast<BindingPlaceholderValue>(self_pattern)) {
          // TODO: move this logic into PatternMatch
          if (placeholder->value_node().has_value()) {
            function_scope.Bind(*placeholder->value_node(),
                                method_val->receiver());
          }
        } else {
          CARBON_CHECK(PatternMatch(self_pattern, method_val->receiver(),
                                    call.source_loc(), &function_scope,
                                    generic_args, trace_stream_, this->arena_));
        }
      }

      // Bind the arguments to the parameters.
      CARBON_CHECK(PatternMatch(
          &function.param_pattern().value(), converted_args, call.source_loc(),
          &function_scope, generic_args, trace_stream_, this->arena_));
      return todo_.Spawn(std::make_unique<StatementAction>(*function.body()),
                         std::move(function_scope));
    }
    case Value::Kind::ParameterizedEntityName: {
      const auto& name = cast<ParameterizedEntityName>(*fun);
      const Declaration& decl = name.declaration();
      RuntimeScope params_scope(&heap_);
      BindingMap generic_args;
      CARBON_CHECK(PatternMatch(&name.params().value(), arg, call.source_loc(),
                                &params_scope, generic_args, trace_stream_,
                                this->arena_));
      Nonnull<const Bindings*> bindings =
          arena_->New<Bindings>(std::move(generic_args), std::move(witnesses));
      switch (decl.kind()) {
        case DeclarationKind::ClassDeclaration: {
          const auto& class_decl = cast<ClassDeclaration>(decl);
          return todo_.FinishAction(arena_->New<NominalClassType>(
              &class_decl, bindings, class_decl.base_type(), VTable()));
        }
        case DeclarationKind::InterfaceDeclaration:
          return todo_.FinishAction(arena_->New<InterfaceType>(
              &cast<InterfaceDeclaration>(decl), bindings));
        case DeclarationKind::ConstraintDeclaration:
          return todo_.FinishAction(arena_->New<NamedConstraintType>(
              &cast<ConstraintDeclaration>(decl), bindings));
        case DeclarationKind::ChoiceDeclaration:
          return todo_.FinishAction(arena_->New<ChoiceType>(
              &cast<ChoiceDeclaration>(decl), bindings));
        default:
          CARBON_FATAL() << "unknown kind of ParameterizedEntityName " << decl;
      }
    }
    default:
      return ProgramError(call.source_loc())
             << "in call, expected a function, not " << *fun;
  }
}

// Returns true if the format string is okay to pass to formatv. This only
// supports `{{` and `{N}` as special syntax.
static auto ValidateFormatString(SourceLocation source_loc,
                                 const char* format_string, int num_args)
    -> ErrorOr<Success> {
  const char* cursor = format_string;
  while (true) {
    switch (*cursor) {
      case '\0':
        // End of string.
        return Success();
      case '{':
        // `{` is a special character.
        ++cursor;
        switch (*cursor) {
          case '\0':
            return ProgramError(source_loc)
                   << "`{` must be followed by a second `{` or index in `"
                   << format_string << "`";
          case '{':
            // Escaped `{`.
            ++cursor;
            break;
          case '}':
            return ProgramError(source_loc)
                   << "Invalid `{}` in `" << format_string << "`";
          default:
            int index = 0;
            while (*cursor != '}') {
              if (*cursor == '\0') {
                return ProgramError(source_loc)
                       << "Index incomplete in `" << format_string << "`";
              }
              if (*cursor < '0' || *cursor > '9') {
                return ProgramError(source_loc)
                       << "Non-numeric character in index at offset "
                       << cursor - format_string << " in `" << format_string
                       << "`";
              }
              index = (10 * index) + (*cursor - '0');
              if (index >= num_args) {
                return ProgramError(source_loc)
                       << "Index invalid with argument count of " << num_args
                       << " at offset " << cursor - format_string << " in `"
                       << format_string << "`";
              }
              ++cursor;
            }
            // Move past the `}`.
            ++cursor;
        }
        break;
      default:
        // Arbitrary text.
        ++cursor;
    }
  }
  llvm_unreachable("Loop returns directly");
}

auto Interpreter::StepInstantiateType() -> ErrorOr<Success> {
  const Action& act = todo_.CurrentAction();
  const Nonnull<const Value*> type = cast<TypeInstantiationAction>(act).type();
  SourceLocation source_loc = cast<TypeInstantiationAction>(act).source_loc();

  switch (type->kind()) {
    case Value::Kind::NominalClassType: {
      const auto& class_type = cast<NominalClassType>(*type);
      std::optional<Nonnull<const NominalClassType*>> base = class_type.base();
      if (act.pos() == 0 && base.has_value()) {
        return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
            base.value(), source_loc));
      } else {
        if (base.has_value()) {
          base = cast<NominalClassType>(act.results().back());
        }
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Bindings*> bindings,
            InstantiateBindings(&class_type.bindings(), source_loc));
        return todo_.FinishAction(arena_->New<NominalClassType>(
            &class_type.declaration(), bindings, base, class_type.vtable()));
      }
    }
    case Value::Kind::PointerType: {
      const auto* ptr = cast<PointerType>(type);
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
            &ptr->pointee_type(), source_loc));
      } else {
        const auto* actual_type = act.results().back();
        return todo_.FinishAction(arena_->New<PointerType>(actual_type));
      }
    }
    default:
      CARBON_ASSIGN_OR_RETURN(auto inst_type, InstantiateType(type, source_loc))
      return todo_.FinishAction(inst_type);
  }
}

auto Interpreter::StepExp() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Expression& exp = cast<ExpressionAction>(act).expression();
  if (trace_stream_->is_enabled()) {
    *trace_stream_ << "--- step exp " << exp << " ." << act.pos() << "."
                   << " (" << exp.source_loc() << ") --->\n";
  }
  switch (exp.kind()) {
    case ExpressionKind::IndexExpression: {
      if (act.pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<IndexExpression>(exp).object()));
      } else if (act.pos() == 1) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<IndexExpression>(exp).offset()));
      } else {
        //    { { v :: [][i] :: C, E, F} :: S, H}
        // -> { { v_i :: C, E, F} : S, H}
        const auto& tuple = cast<TupleValue>(*act.results()[0]);
        int i = cast<IntValue>(*act.results()[1]).value();
        if (i < 0 || i >= static_cast<int>(tuple.elements().size())) {
          return ProgramError(exp.source_loc())
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
    case ExpressionKind::SimpleMemberAccessExpression: {
      const auto& access = cast<SimpleMemberAccessExpression>(exp);
      if (auto rewrite = access.rewritten_form()) {
        return todo_.ReplaceWith(std::make_unique<ExpressionAction>(*rewrite));
      }
      if (act.pos() == 0) {
        // First, evaluate the first operand.
        if (access.is_addr_me_method()) {
          return todo_.Spawn(
              std::make_unique<LocationAction>(&access.object()));
        } else {
          return todo_.Spawn(
              std::make_unique<ExpressionAction>(&access.object()));
        }
      } else {
        if (auto constant_value = access.constant_value()) {
          if (act.pos() == 1) {
            return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
                *constant_value, access.source_loc()));
          } else {
            return todo_.FinishAction(act.results().back());
          }
        } else if (const auto* member_name_type =
                       dyn_cast<TypeOfMemberName>(&access.static_type())) {
          // The result is a member name, such as in `Type.field_name`. Form a
          // suitable member name value.
          CARBON_CHECK(phase() == Phase::CompileTime)
              << "should not form MemberNames at runtime";
          auto found_in_interface = access.found_in_interface();
          if (act.pos() == 1 && found_in_interface) {
            return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
                *found_in_interface, exp.source_loc()));
          } else {
            if (found_in_interface) {
              found_in_interface = cast<InterfaceType>(act.results().back());
            }
            std::optional<const Value*> type_result;
            if (!isa<InterfaceType, NamedConstraintType, ConstraintType>(
                    act.results()[0])) {
              type_result = act.results()[0];
            }
            MemberName* member_name = arena_->New<MemberName>(
                type_result, found_in_interface, member_name_type->member());
            return todo_.FinishAction(member_name);
          }
        } else {
          // The result is the value of the named field, such as in
          // `value.field_name`. Extract the value within the given object.
          auto impl_has_value = access.impl().has_value();
          if (act.pos() == 1) {
            // Next, if we're accessing an interface member, evaluate the `impl`
            // expression to find the corresponding witness.
            if (impl_has_value) {
              return todo_.Spawn(
                  std::make_unique<WitnessAction>(access.impl().value()));
            } else {
              return todo_.RunAgain();
            }
          } else if (act.pos() == 2) {
            if (auto found_in_interface = access.found_in_interface()) {
              return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
                  *found_in_interface, exp.source_loc()));
            } else {
              return todo_.RunAgain();
            }
          } else if (act.pos() == 3) {
            if (access.is_type_access()) {
              return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
                  &access.object().static_type(), access.source_loc()));
            } else {
              return todo_.RunAgain();
            }
          } else {
            auto found_in_interface = access.found_in_interface();
            if (found_in_interface) {
              found_in_interface = cast<InterfaceType>(
                  impl_has_value ? act.results()[2] : act.results()[1]);
            }
            std::optional<Nonnull<const Witness*>> witness;
            if (access.impl().has_value()) {
              witness = cast<Witness>(act.results()[1]);
            }
            ElementPath::Component member(&access.member(), found_in_interface,
                                          witness);
            const Value* aggregate;
            if (access.is_type_access()) {
              aggregate = act.results().back();
            } else if (const auto* location =
                           dyn_cast<LocationValue>(act.results()[0])) {
              CARBON_ASSIGN_OR_RETURN(
                  aggregate,
                  this->heap_.Read(location->address(), exp.source_loc()));
            } else {
              aggregate = act.results()[0];
            }
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> member_value,
                aggregate->GetElement(arena_, ElementPath(member),
                                      exp.source_loc(), act.results()[0]));
            return todo_.FinishAction(member_value);
          }
        }
      }
    }
    case ExpressionKind::CompoundMemberAccessExpression: {
      const auto& access = cast<CompoundMemberAccessExpression>(exp);
      bool forming_member_name = isa<TypeOfMemberName>(&access.static_type());
      if (act.pos() == 0) {
        // First, evaluate the first operand.
        if (access.is_addr_me_method()) {
          return todo_.Spawn(
              std::make_unique<LocationAction>(&access.object()));
        } else {
          return todo_.Spawn(
              std::make_unique<ExpressionAction>(&access.object()));
        }
      } else {
        if (auto constant_value = access.constant_value()) {
          if (act.pos() == 1) {
            return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
                *constant_value, access.source_loc()));
          } else {
            return todo_.FinishAction(act.results().back());
          }
        } else if (forming_member_name) {
          CARBON_CHECK(phase() == Phase::CompileTime)
              << "should not form MemberNames at runtime";
          if (auto found_in_interface = access.member().interface();
              found_in_interface && act.pos() == 1) {
            return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
                *found_in_interface, exp.source_loc()));
          } else {
            // If we're forming a member name, we must be in the outer
            // evaluation in `Type.(Interface.method)`. Produce the same method
            // name with its `type` field set.
            if (found_in_interface) {
              found_in_interface = cast<InterfaceType>(act.results().back());
            }
            CARBON_CHECK(!access.member().base_type().has_value())
                << "compound member access forming a member name should be "
                   "performing impl lookup";
            auto* member_name = arena_->New<MemberName>(
                act.results()[0], found_in_interface, access.member().member());
            return todo_.FinishAction(member_name);
          }
        } else {
          auto impl_has_value = access.impl().has_value();
          if (act.pos() == 1) {
            if (impl_has_value) {
              // Next, if we're accessing an interface member, evaluate the
              // `impl` expression to find the corresponding witness.
              return todo_.Spawn(
                  std::make_unique<WitnessAction>(access.impl().value()));
            } else {
              return todo_.RunAgain();
            }
          } else if (act.pos() == 2) {
            if (auto found_in_interface = access.member().interface()) {
              return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
                  *found_in_interface, exp.source_loc()));
            } else {
              return todo_.RunAgain();
            }
          } else if (act.pos() == 3) {
            if (access.is_type_access()) {
              return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
                  &access.object().static_type(), access.source_loc()));
            } else {
              return todo_.RunAgain();
            }
          } else {
            // Access the object to find the named member.
            auto found_in_interface = access.member().interface();
            if (found_in_interface) {
              found_in_interface = cast<InterfaceType>(
                  impl_has_value ? act.results()[2] : act.results()[1]);
            }

            Nonnull<const Value*> object = act.results()[0];
            if (access.is_type_access()) {
              object = act.results().back();
            }
            std::optional<Nonnull<const Witness*>> witness;
            if (access.impl().has_value()) {
              witness = cast<Witness>(act.results()[1]);
            } else {
              CARBON_CHECK(access.member().base_type().has_value())
                  << "compound access should have base type or impl";
              CARBON_ASSIGN_OR_RETURN(
                  object, Convert(object, *access.member().base_type(),
                                  exp.source_loc()));
            }
            ElementPath::Component field(&access.member().member(),
                                         found_in_interface, witness);
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> member,
                object->GetElement(arena_, ElementPath(field), exp.source_loc(),
                                   object));
            return todo_.FinishAction(member);
          }
        }
      }
    }
    case ExpressionKind::BaseAccessExpression: {
      const auto& access = cast<BaseAccessExpression>(exp);
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&access.object()));
      } else {
        ElementPath::Component base_elt(&access.element(), std::nullopt,
                                        std::nullopt);
        const Value* value = act.results()[0];
        CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> base_value,
                                value->GetElement(arena_, ElementPath(base_elt),
                                                  exp.source_loc(), value));
        return todo_.FinishAction(base_value);
      }
    }
    case ExpressionKind::IdentifierExpression: {
      CARBON_CHECK(act.pos() == 0);
      const auto& ident = cast<IdentifierExpression>(exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(ident.value_node(), ident.source_loc()));
      if (const auto* location = dyn_cast<LocationValue>(value)) {
        CARBON_ASSIGN_OR_RETURN(
            value, heap_.Read(location->address(), exp.source_loc()));
      }
      return todo_.FinishAction(value);
    }
    case ExpressionKind::DotSelfExpression: {
      CARBON_CHECK(act.pos() == 0);
      const auto& dot_self = cast<DotSelfExpression>(exp);
      return todo_.FinishAction(*dot_self.self_binding().symbolic_identity());
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
    case ExpressionKind::OperatorExpression: {
      const auto& op = cast<OperatorExpression>(exp);
      if (auto rewrite = op.rewritten_form()) {
        return todo_.ReplaceWith(std::make_unique<ExpressionAction>(*rewrite));
      }
      if (act.pos() != static_cast<int>(op.arguments().size())) {
        //    { {v :: op(vs,[],e,es) :: C, E, F} :: S, H}
        // -> { {e :: op(vs,v,[],es) :: C, E, F} :: S, H}
        Nonnull<const Expression*> arg = op.arguments()[act.pos()];
        if (op.op() == Operator::AddressOf) {
          return todo_.Spawn(std::make_unique<LocationAction>(arg));
        } else if ((op.op() == Operator::And || op.op() == Operator::Or) &&
                   act.pos() == 1) {
          // Short-circuit evaluation for 'and' & 'or'
          const auto* operand_value =
              cast<BoolValue>(act.results()[act.pos() - 1]);
          if ((op.op() == Operator::Or && operand_value->value()) ||
              (op.op() == Operator::And && !operand_value->value())) {
            return todo_.FinishAction(operand_value);
          }
          // No short-circuit, fall through to evaluate 2nd operand.
        }
        return todo_.Spawn(std::make_unique<ExpressionAction>(arg));
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> value,
                                EvalPrim(op.op(), &op.static_type(),
                                         act.results(), exp.source_loc()));
        return todo_.FinishAction(value);
      }
    }
    case ExpressionKind::CallExpression: {
      const auto& call = cast<CallExpression>(exp);
      unsigned int num_witnesses = call.witnesses().size();
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
      } else if (num_witnesses > 0 &&
                 act.pos() < 2 + static_cast<int>(num_witnesses)) {
        auto iter = call.witnesses().begin();
        std::advance(iter, act.pos() - 2);
        return todo_.Spawn(
            std::make_unique<WitnessAction>(cast<Witness>(iter->second)));
      } else if (act.pos() == 2 + static_cast<int>(num_witnesses)) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        ImplWitnessMap witnesses;
        if (num_witnesses > 0) {
          int i = 2;
          for (const auto& [impl_bind, impl_exp] : call.witnesses()) {
            witnesses[impl_bind] = act.results()[i];
            ++i;
          }
        }
        return CallFunction(call, act.results()[0], act.results()[1],
                            std::move(witnesses));
      } else if (act.pos() == 3 + static_cast<int>(num_witnesses)) {
        if (act.results().size() < 3 + num_witnesses) {
          // Control fell through without explicit return.
          return todo_.FinishAction(TupleValue::Empty());
        } else {
          return todo_.FinishAction(
              act.results()[2 + static_cast<int>(num_witnesses)]);
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
      const auto& args = cast<TupleValue>(*act.results()[0]).elements();
      switch (cast<IntrinsicExpression>(exp).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print: {
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> format_string_value,
              Convert(args[0], arena_->New<StringType>(), exp.source_loc()));
          const char* format_string =
              cast<StringValue>(*format_string_value).value().c_str();
          int num_format_args = args.size() - 1;
          CARBON_RETURN_IF_ERROR(ValidateFormatString(
              intrinsic.source_loc(), format_string, num_format_args));
          switch (num_format_args) {
            case 0:
              *print_stream_ << llvm::formatv(format_string);
              break;
            case 1: {
              if ((*args[1]).kind() == Value::Kind::UninitializedValue) {
                return ProgramError(exp.source_loc())
                       << "Printing uninitialized value";
              }
              *print_stream_ << llvm::formatv(format_string,
                                              cast<IntValue>(*args[1]).value());
              break;
            }
            default:
              CARBON_FATAL() << "Too many format args: " << num_format_args;
          }
          // Implicit newline; currently no way to disable it.
          *print_stream_ << "\n";
          return todo_.FinishAction(TupleValue::Empty());
        }
        case IntrinsicExpression::Intrinsic::Assert: {
          CARBON_CHECK(args.size() == 2);
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> condition,
              Convert(args[0], arena_->New<BoolType>(), exp.source_loc()));
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> string_value,
              Convert(args[1], arena_->New<StringType>(), exp.source_loc()));
          bool condition_value = cast<BoolValue>(condition)->value();
          if (!condition_value) {
            return ProgramError(exp.source_loc()) << *string_value;
          }
          return todo_.FinishAction(TupleValue::Empty());
        }
        case IntrinsicExpression::Intrinsic::Alloc: {
          CARBON_CHECK(args.size() == 1);
          Address addr(heap_.AllocateValue(args[0]));
          return todo_.FinishAction(arena_->New<PointerValue>(addr));
        }
        case IntrinsicExpression::Intrinsic::Dealloc: {
          CARBON_CHECK(args.size() == 1);
          CARBON_CHECK(act.pos() > 0);
          const auto* ptr = cast<PointerValue>(args[0]);
          CARBON_ASSIGN_OR_RETURN(const auto* pointee,
                                  heap_.Read(ptr->address(), exp.source_loc()));
          if (const auto* class_value = dyn_cast<NominalClassValue>(pointee)) {
            // Handle destruction from base class pointer.
            const auto* child_class_value = *class_value->class_value_ptr();
            bool is_subtyped = child_class_value != class_value;
            if (is_subtyped) {
              // Error if destructor is not virtual.
              const auto& class_type =
                  cast<NominalClassType>(class_value->type());
              const auto& class_decl = class_type.declaration();
              if ((*class_decl.destructor())->virt_override() ==
                  VirtualOverride::None) {
                return ProgramError(exp.source_loc())
                       << "Deallocating a derived class from base class "
                          "pointer requires a virtual destructor";
              }
            }
            const Address obj_addr = is_subtyped
                                         ? ptr->address().DowncastedAddress()
                                         : ptr->address();
            if (act.pos() == 1) {
              return todo_.Spawn(std::make_unique<DestroyAction>(
                  arena_->New<LocationValue>(obj_addr), child_class_value));
            } else {
              heap_.Deallocate(obj_addr);
              return todo_.FinishAction(TupleValue::Empty());
            }
          } else {
            if (act.pos() == 1) {
              return todo_.Spawn(std::make_unique<DestroyAction>(
                  arena_->New<LocationValue>(ptr->address()), pointee));
            } else {
              heap_.Deallocate(ptr->address());
              return todo_.FinishAction(TupleValue::Empty());
            }
          }
        }
        case IntrinsicExpression::Intrinsic::Rand: {
          CARBON_CHECK(args.size() == 2);
          const int64_t low = cast<IntValue>(*args[0]).value();
          const int64_t high = cast<IntValue>(*args[1]).value();
          if (low >= high) {
            return ProgramError(exp.source_loc())
                   << "Rand inputs must be ordered for a non-empty range: "
                   << low << " must be less than " << high;
          }
          // Use 64-bit to handle large ranges where `high - low` might exceed
          // int32_t maximums.
          static std::mt19937_64 generator(12);
          const int64_t range = high - low;
          // We avoid using std::uniform_int_distribution because it's not
          // reproducible across builds/platforms.
          int64_t r = (generator() % range) + low;
          CARBON_CHECK(r >= std::numeric_limits<int32_t>::min() &&
                       r <= std::numeric_limits<int32_t>::max())
              << "Non-int32 result: " << r;
          CARBON_CHECK(r >= low && r <= high) << "Out-of-range result: " << r;
          return todo_.FinishAction(arena_->New<IntValue>(r));
        }
        case IntrinsicExpression::Intrinsic::ImplicitAs: {
          CARBON_CHECK(args.size() == 1);
          // Build a constraint type that constrains its .Self type to satisfy
          // the "ImplicitAs" intrinsic constraint. This involves creating a
          // number of objects that all point to each other.
          // TODO: Factor out a simple version of ConstraintTypeBuilder and
          // use it from here.
          auto* self_binding = arena_->New<GenericBinding>(
              exp.source_loc(), ".Self",
              arena_->New<TypeTypeLiteral>(exp.source_loc()),
              GenericBinding::BindingKind::Checked);
          auto* self = arena_->New<VariableType>(self_binding);
          auto* impl_binding = arena_->New<ImplBinding>(
              exp.source_loc(), self_binding, std::nullopt);
          impl_binding->set_symbolic_identity(
              arena_->New<BindingWitness>(impl_binding));
          self_binding->set_symbolic_identity(self);
          self_binding->set_value(self);
          self_binding->set_impl_binding(impl_binding);
          IntrinsicConstraint constraint = {
              .type = self,
              .kind = IntrinsicConstraint::ImplicitAs,
              .arguments = args};
          auto* result = arena_->New<ConstraintType>(
              self_binding, std::vector<ImplsConstraint>{},
              std::vector<IntrinsicConstraint>{std::move(constraint)},
              std::vector<EqualityConstraint>{},
              std::vector<RewriteConstraint>{}, std::vector<LookupContext>{});
          impl_binding->set_interface(result);
          return todo_.FinishAction(result);
        }
        case IntrinsicExpression::Intrinsic::ImplicitAsConvert: {
          CARBON_CHECK(args.size() == 2);
          CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> result,
                                  Convert(args[0], args[1], exp.source_loc()));
          return todo_.FinishAction(result);
        }
        case IntrinsicExpression::Intrinsic::IntEq: {
          CARBON_CHECK(args.size() == 2);
          auto lhs = cast<IntValue>(*args[0]).value();
          auto rhs = cast<IntValue>(*args[1]).value();
          auto* result = arena_->New<BoolValue>(lhs == rhs);
          return todo_.FinishAction(result);
        }
        case IntrinsicExpression::Intrinsic::StrEq: {
          CARBON_CHECK(args.size() == 2);
          const auto& lhs = cast<StringValue>(*args[0]).value();
          const auto& rhs = cast<StringValue>(*args[1]).value();
          auto* result = arena_->New<BoolValue>(lhs == rhs);
          return todo_.FinishAction(result);
        }
        case IntrinsicExpression::Intrinsic::IntCompare: {
          CARBON_CHECK(args.size() == 2);
          auto lhs = cast<IntValue>(*args[0]).value();
          auto rhs = cast<IntValue>(*args[1]).value();
          if (lhs < rhs) {
            auto* result = arena_->New<IntValue>(-1);
            return todo_.FinishAction(result);
          }
          if (lhs == rhs) {
            auto* result = arena_->New<IntValue>(0);
            return todo_.FinishAction(result);
          }
          auto* result = arena_->New<IntValue>(1);
          return todo_.FinishAction(result);
        }
        case IntrinsicExpression::Intrinsic::StrCompare: {
          CARBON_CHECK(args.size() == 2);
          const auto& lhs = cast<StringValue>(*args[0]).value();
          const auto& rhs = cast<StringValue>(*args[1]).value();
          if (lhs < rhs) {
            auto* result = arena_->New<IntValue>(-1);
            return todo_.FinishAction(result);
          }
          if (lhs == rhs) {
            auto* result = arena_->New<IntValue>(0);
            return todo_.FinishAction(result);
          }
          auto* result = arena_->New<IntValue>(1);
          return todo_.FinishAction(result);
        }
        case IntrinsicExpression::Intrinsic::IntBitComplement: {
          CARBON_CHECK(args.size() == 1);
          return todo_.FinishAction(
              arena_->New<IntValue>(~cast<IntValue>(*args[0]).value()));
        }
        case IntrinsicExpression::Intrinsic::IntBitAnd: {
          CARBON_CHECK(args.size() == 2);
          return todo_.FinishAction(
              arena_->New<IntValue>(cast<IntValue>(*args[0]).value() &
                                    cast<IntValue>(*args[1]).value()));
        }
        case IntrinsicExpression::Intrinsic::IntBitOr: {
          CARBON_CHECK(args.size() == 2);
          return todo_.FinishAction(
              arena_->New<IntValue>(cast<IntValue>(*args[0]).value() |
                                    cast<IntValue>(*args[1]).value()));
        }
        case IntrinsicExpression::Intrinsic::IntBitXor: {
          CARBON_CHECK(args.size() == 2);
          return todo_.FinishAction(
              arena_->New<IntValue>(cast<IntValue>(*args[0]).value() ^
                                    cast<IntValue>(*args[1]).value()));
        }
        case IntrinsicExpression::Intrinsic::IntLeftShift: {
          CARBON_CHECK(args.size() == 2);
          const auto& lhs = cast<IntValue>(*args[0]).value();
          const auto& rhs = cast<IntValue>(*args[1]).value();
          if (rhs >= 0 && rhs < 32) {
            return todo_.FinishAction(
                arena_->New<IntValue>(static_cast<uint32_t>(lhs) << rhs));
          }
          return ProgramError(exp.source_loc()) << "Integer overflow";
        }
        case IntrinsicExpression::Intrinsic::IntRightShift: {
          CARBON_CHECK(args.size() == 2);
          const auto& lhs = cast<IntValue>(*args[0]).value();
          const auto& rhs = cast<IntValue>(*args[1]).value();
          if (rhs >= 0 && rhs < 32) {
            return todo_.FinishAction(arena_->New<IntValue>(lhs >> rhs));
          }
          return ProgramError(exp.source_loc()) << "Integer overflow";
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
    case ExpressionKind::StringLiteral:
      CARBON_CHECK(act.pos() == 0);
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      return todo_.FinishAction(
          arena_->New<StringValue>(cast<StringLiteral>(exp).value()));
    case ExpressionKind::StringTypeLiteral: {
      CARBON_CHECK(act.pos() == 0);
      return todo_.FinishAction(arena_->New<StringType>());
    }
    case ExpressionKind::FunctionTypeLiteral:
    case ExpressionKind::StructTypeLiteral:
    case ExpressionKind::ArrayTypeLiteral:
    case ExpressionKind::ValueLiteral: {
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
            &exp.static_type(), exp.source_loc()));
      } else {
        const auto* value = &cast<ConstantValueLiteral>(exp).constant_value();
        Nonnull<const Value*> destination = act.results().back();
        CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> result,
                                Convert(value, destination, exp.source_loc()));
        return todo_.FinishAction(result);
      }
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
    case ExpressionKind::WhereExpression: {
      auto rewrite = cast<WhereExpression>(exp).rewritten_form();
      CARBON_CHECK(rewrite) << "where expression should be rewritten";
      return todo_.ReplaceWith(std::make_unique<ExpressionAction>(*rewrite));
    }
    case ExpressionKind::BuiltinConvertExpression: {
      const auto& convert_expr = cast<BuiltinConvertExpression>(exp);
      if (auto rewrite = convert_expr.rewritten_form()) {
        return todo_.ReplaceWith(std::make_unique<ExpressionAction>(*rewrite));
      }
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            convert_expr.source_expression()));
      } else if (act.pos() == 1) {
        return todo_.Spawn(std::make_unique<TypeInstantiationAction>(
            &convert_expr.static_type(), convert_expr.source_loc()));
      } else {
        // TODO: Remove all calls to Convert other than this one. We shouldn't
        // need them any more.
        Nonnull<const Value*> destination = act.results().back();
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> result,
            Convert(act.results()[0], destination, convert_expr.source_loc()));
        return todo_.FinishAction(result);
      }
    }
    case ExpressionKind::UnimplementedExpression:
      CARBON_FATAL() << "Unimplemented: " << exp;
  }  // switch (exp->kind)
}

auto Interpreter::StepWitness() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Witness* witness = cast<WitnessAction>(act).witness();
  if (trace_stream_->is_enabled()) {
    *trace_stream_ << "--- step witness " << *witness << " ." << act.pos()
                   << ". --->\n";
  }
  switch (witness->kind()) {
    case Value::Kind::BindingWitness: {
      const ImplBinding* binding = cast<BindingWitness>(witness)->binding();
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(binding, binding->type_var()->source_loc()));
      if (const auto* location = dyn_cast<LocationValue>(value)) {
        // TODO: Why do we store values for impl bindings on the heap?
        CARBON_ASSIGN_OR_RETURN(
            value,
            heap_.Read(location->address(), binding->type_var()->source_loc()));
      }
      return todo_.FinishAction(value);
    }

    case Value::Kind::ConstraintWitness: {
      llvm::ArrayRef<Nonnull<const Witness*>> witnesses =
          cast<ConstraintWitness>(witness)->witnesses();
      if (act.pos() < static_cast<int>(witnesses.size())) {
        return todo_.Spawn(
            std::make_unique<WitnessAction>(witnesses[act.pos()]));
      }
      std::vector<Nonnull<const Witness*>> new_witnesses;
      new_witnesses.reserve(witnesses.size());
      for (const auto* witness : act.results()) {
        new_witnesses.push_back(cast<Witness>(witness));
      }
      return todo_.FinishAction(
          arena_->New<ConstraintWitness>(std::move(new_witnesses)));
    }

    case Value::Kind::ConstraintImplWitness: {
      const auto* constraint_impl = cast<ConstraintImplWitness>(witness);
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<WitnessAction>(
            constraint_impl->constraint_witness()));
      }
      return todo_.FinishAction(ConstraintImplWitness::Make(
          arena_, cast<Witness>(act.results()[0]), constraint_impl->index()));
    }

    case Value::Kind::ImplWitness: {
      const auto* impl_witness = cast<ImplWitness>(witness);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Bindings*> new_bindings,
          InstantiateBindings(&impl_witness->bindings(),
                              impl_witness->declaration().source_loc()));
      return todo_.FinishAction(
          new_bindings == &impl_witness->bindings()
              ? impl_witness
              : arena_->New<ImplWitness>(&impl_witness->declaration(),
                                         new_bindings));
    }

    default:
      CARBON_FATAL() << "unexpected kind of witness " << *witness;
  }
}

auto Interpreter::StepStmt() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Statement& stmt = cast<StatementAction>(act).statement();
  if (trace_stream_->is_enabled()) {
    *trace_stream_ << "--- step stmt ";
    stmt.PrintDepth(1, trace_stream_->stream());
    *trace_stream_ << " ." << act.pos() << ". "
                   << "(" << stmt.source_loc() << ") --->\n";
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
                         generic_args, trace_stream_, this->arena_)) {
          // Ensure we don't process any more clauses.
          act.set_pos(match_stmt.clauses().size() + 1);
          todo_.MergeScope(std::move(matches));
          return todo_.Spawn(std::make_unique<StatementAction>(&c.statement()));
        } else {
          return todo_.RunAgain();
        }
      }
    }
    case StatementKind::For: {
      constexpr int TargetVarPosInResult = 0;
      constexpr int CurrentIndexPosInResult = 1;
      constexpr int EndIndexPosInResult = 2;
      const auto* loop_var = &cast<BindingPlaceholderValue>(
          cast<For>(stmt).variable_declaration().value());
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&cast<For>(stmt).loop_target()));
      }
      if (act.pos() == 1) {
        const auto* source_array =
            cast<TupleValue>(act.results()[TargetVarPosInResult]);

        int start_index = 0;
        auto end_index = static_cast<int>(source_array->elements().size());
        if (end_index == 0) {
          return todo_.FinishAction();
        }
        act.AddResult(arena_->New<IntValue>(start_index));
        act.AddResult(arena_->New<IntValue>(end_index));
        todo_.Initialize(*(loop_var->value_node()),
                         source_array->elements()[start_index]);
        act.ReplaceResult(CurrentIndexPosInResult,
                          arena_->New<IntValue>(start_index + 1));
        return todo_.Spawn(
            std::make_unique<StatementAction>(&cast<For>(stmt).body()));
      }
      if (act.pos() >= 2) {
        auto current_index =
            cast<IntValue>(act.results()[CurrentIndexPosInResult])->value();
        auto end_index =
            cast<IntValue>(act.results()[EndIndexPosInResult])->value();

        if (current_index < end_index) {
          const auto* source_array =
              cast<const TupleValue>(act.results()[TargetVarPosInResult]);

          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> assigned_array_element,
              todo_.ValueOfNode(*(loop_var->value_node()), stmt.source_loc()));

          const auto* location = cast<LocationValue>(assigned_array_element);
          CARBON_RETURN_IF_ERROR(heap_.Write(
              location->address(), source_array->elements()[current_index],
              stmt.source_loc()));

          act.ReplaceResult(CurrentIndexPosInResult,
                            arena_->New<IntValue>(current_index + 1));
          return todo_.Spawn(
              std::make_unique<StatementAction>(&cast<For>(stmt).body()));
        }
      }
      return todo_.FinishAction();
    }
    case StatementKind::While:
      // TODO: Rewrite While to use ReplaceResult to store condition result.
      //       This will remove the inconsistency between the while and for
      //       loops.
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
      const auto* dest_type = &definition.pattern().static_type();
      if (act.pos() == 0 && definition.has_init()) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&definition.init()));
      } else {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        Nonnull<const Value*> p =
            &cast<VariableDefinition>(stmt).pattern().value();
        Nonnull<const Value*> v;
        if (definition.has_init()) {
          CARBON_ASSIGN_OR_RETURN(
              v, Convert(act.results()[0], dest_type, stmt.source_loc()));
        } else if (dest_type->kind() == Value::Kind::StaticArrayType) {
          const auto& array = cast<StaticArrayType>(dest_type);
          const auto& element_type = array->element_type();
          const auto size = array->size();

          std::vector<Nonnull<const Value*>> elements;
          elements.reserve(size);
          for (size_t i = 0; i < size; i++) {
            elements.push_back(arena_->New<UninitializedValue>(&element_type));
          }
          v = arena_->New<TupleValueBase>(Value::Kind::TupleValue, elements);
        } else {
          v = arena_->New<UninitializedValue>(p);
        }

        RuntimeScope matches(&heap_);
        BindingMap generic_args;
        CARBON_CHECK(PatternMatch(p, v, stmt.source_loc(), &matches,
                                  generic_args, trace_stream_, this->arena_))
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
      if (auto rewrite = assign.rewritten_form()) {
        if (act.pos() == 0) {
          return todo_.Spawn(std::make_unique<ExpressionAction>(*rewrite));
        } else {
          return todo_.FinishAction();
        }
      }
      if (act.pos() == 0) {
        //    { {(lv = e) :: C, E, F} :: S, H}
        // -> { {lv :: ([] = e) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<LocationAction>(&assign.lhs()));
      } else if (act.pos() == 1) {
        //    { { a :: ([] = e) :: C, E, F} :: S, H}
        // -> { { e :: (a = []) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(&assign.rhs()));
      } else {
        //    { { v :: (a = []) :: C, E, F} :: S, H}
        // -> { { C, E, F} :: S, H(a := v)}
        const auto& lval = cast<LocationValue>(*act.results()[0]);
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> rval,
            Convert(act.results()[1], &assign.lhs().static_type(),
                    stmt.source_loc()));
        CARBON_RETURN_IF_ERROR(
            heap_.Write(lval.address(), rval, stmt.source_loc()));
        return todo_.FinishAction();
      }
    }
    case StatementKind::IncrementDecrement: {
      const auto& inc_dec = cast<IncrementDecrement>(stmt);
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(*inc_dec.rewritten_form()));
      } else {
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
    case StatementKind::ReturnVar: {
      const auto& ret_var = cast<ReturnVar>(stmt);
      const ValueNodeView& value_node = ret_var.value_node();
      if (trace_stream_->is_enabled()) {
        *trace_stream_ << "--- step returned var "
                       << cast<BindingPattern>(value_node.base()).name() << " ."
                       << act.pos() << "."
                       << " (" << stmt.source_loc() << ") --->\n";
      }
      CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> value,
                              todo_.ValueOfNode(value_node, stmt.source_loc()));
      if (const auto* location = dyn_cast<LocationValue>(value)) {
        CARBON_ASSIGN_OR_RETURN(
            value, heap_.Read(location->address(), ret_var.source_loc()));
      }
      const CallableDeclaration& function = cast<Return>(stmt).function();
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> return_value,
          Convert(value, &function.return_term().static_type(),
                  stmt.source_loc()));
      return todo_.UnwindPast(*function.body(), return_value);
    }
    case StatementKind::ReturnExpression:
      if (act.pos() == 0) {
        //    { {return e :: C, E, F} :: S, H}
        // -> { {e :: return [] :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &cast<ReturnExpression>(stmt).expression()));
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        const CallableDeclaration& function = cast<Return>(stmt).function();
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> return_value,
            Convert(act.results()[0], &function.return_term().static_type(),
                    stmt.source_loc()));
        return todo_.UnwindPast(*function.body(), return_value);
      }
  }
}

auto Interpreter::StepDeclaration() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Declaration& decl = cast<DeclarationAction>(act).declaration();
  if (trace_stream_->is_enabled()) {
    *trace_stream_ << "--- step decl ";
    decl.PrintID(trace_stream_->stream());
    *trace_stream_ << " ." << act.pos() << ". "
                   << "(" << decl.source_loc() << ") --->\n";
  }
  switch (decl.kind()) {
    case DeclarationKind::VariableDeclaration: {
      const auto& var_decl = cast<VariableDeclaration>(decl);
      const auto* var_type = &var_decl.binding().static_type();
      if (var_decl.has_initializer()) {
        if (act.pos() == 0) {
          return todo_.Spawn(
              std::make_unique<ExpressionAction>(&var_decl.initializer()));
        } else {
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> v,
              Convert(act.results()[0], &var_decl.binding().static_type(),
                      var_decl.source_loc()));
          todo_.Initialize(&var_decl.binding(), v);
          return todo_.FinishAction();
        }
      } else if (var_type->kind() == Value::Kind::StaticArrayType) {
        const auto& array = cast<StaticArrayType>(var_type);
        const auto& element_type = array->element_type();
        const auto size = array->size();

        std::vector<Nonnull<const Value*>> elements;
        elements.reserve(size);
        for (size_t i = 0; i < size; i++) {
          elements.push_back(arena_->New<UninitializedValue>(&element_type));
        }

        Nonnull<const Value*> v =
            arena_->New<TupleValueBase>(Value::Kind::TupleValue, elements);
        todo_.Initialize(&var_decl.binding(), v);
        return todo_.FinishAction();
      } else {
        Nonnull<const Value*> v =
            arena_->New<UninitializedValue>(&var_decl.binding().value());
        todo_.Initialize(&var_decl.binding(), v);
        return todo_.FinishAction();
      }
    }
    case DeclarationKind::NamespaceDeclaration:
    case DeclarationKind::DestructorDeclaration:
    case DeclarationKind::FunctionDeclaration:
    case DeclarationKind::ClassDeclaration:
    case DeclarationKind::MixinDeclaration:
    case DeclarationKind::MixDeclaration:
    case DeclarationKind::ChoiceDeclaration:
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration:
    case DeclarationKind::InterfaceExtendsDeclaration:
    case DeclarationKind::InterfaceImplDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration:
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::MatchFirstDeclaration:
    case DeclarationKind::SelfDeclaration:
    case DeclarationKind::AliasDeclaration:
      // These declarations have no run-time effects.
      return todo_.FinishAction();
  }
}

auto Interpreter::StepDestroy() -> ErrorOr<Success> {
  const Action& act = todo_.CurrentAction();
  const auto& destroy_act = cast<DestroyAction>(act);
  switch (destroy_act.value()->kind()) {
    case Value::Kind::NominalClassValue: {
      const auto* class_obj = cast<NominalClassValue>(destroy_act.value());
      const auto& class_decl =
          cast<NominalClassType>(class_obj->type()).declaration();
      const int member_count = class_decl.members().size();
      if (act.pos() == 0) {
        // Run the destructor, if there is one.
        if (auto destructor = class_decl.destructor()) {
          return CallDestructor(*destructor, class_obj);
        } else {
          return todo_.RunAgain();
        }
      } else if (act.pos() <= member_count) {
        // Destroy members.
        const int index = class_decl.members().size() - act.pos();
        const auto& member = class_decl.members()[index];
        if (const auto* var = dyn_cast<VariableDeclaration>(member)) {
          const Address object = destroy_act.location()->address();
          const Address var_addr =
              object.ElementAddress(arena_->New<NamedElement>(var));
          const auto v = heap_.Read(var_addr, SourceLocation("destructor", 1));
          CARBON_CHECK(v.ok())
              << "Failed to read member `" << var->binding().name()
              << "` from class `" << class_decl.name() << "`";
          return todo_.Spawn(std::make_unique<DestroyAction>(
              arena_->New<LocationValue>(var_addr), *v));
        } else {
          return todo_.RunAgain();
        }
      } else if (act.pos() == member_count + 1) {
        // Destroy the parent, if there is one.
        if (auto base = class_obj->base()) {
          const Address obj_addr = destroy_act.location()->address();
          const Address base_addr =
              obj_addr.ElementAddress(arena_->New<BaseElement>(class_obj));
          return todo_.Spawn(std::make_unique<DestroyAction>(
              arena_->New<LocationValue>(base_addr), base.value()));
        } else {
          return todo_.RunAgain();
        }
      } else {
        todo_.Pop();
        return Success();
      }
    }
    case Value::Kind::TupleValue: {
      const auto* tuple = cast<TupleValue>(destroy_act.value());
      const auto element_count = tuple->elements().size();
      if (static_cast<size_t>(act.pos()) < element_count) {
        const size_t index = element_count - act.pos() - 1;
        const auto& item = tuple->elements()[index];
        const auto object_addr = destroy_act.location()->address();
        Address field_address = object_addr.ElementAddress(
            arena_->New<PositionalElement>(index, item));
        if (item->kind() == Value::Kind::NominalClassValue ||
            item->kind() == Value::Kind::TupleValue) {
          return todo_.Spawn(std::make_unique<DestroyAction>(
              arena_->New<LocationValue>(field_address), item));
        } else {
          // The tuple element's type is an integral type (e.g., i32)
          // or the type doesn't support destruction.
          return todo_.RunAgain();
        }
      } else {
        todo_.Pop();
        return Success();
      }
    }
    default:
      // These declarations have no run-time effects.
      todo_.Pop();
      return Success();
  }
  CARBON_FATAL() << "Unreachable";
}

auto Interpreter::StepCleanUp() -> ErrorOr<Success> {
  const Action& act = todo_.CurrentAction();
  const auto& cleanup = cast<CleanUpAction>(act);
  if (act.pos() < cleanup.allocations_count() * 2) {
    const size_t alloc_index = cleanup.allocations_count() - act.pos() / 2 - 1;
    auto allocation = act.scope()->allocations()[alloc_index];
    if (act.pos() % 2 == 0) {
      auto* location = arena_->New<LocationValue>(Address(allocation));
      auto value =
          heap_.Read(location->address(), SourceLocation("destructor", 1));
      // Step over uninitialized values.
      if (value.ok()) {
        return todo_.Spawn(std::make_unique<DestroyAction>(location, *value));
      } else {
        return todo_.RunAgain();
      }
    } else {
      heap_.Deallocate(allocation);
      return todo_.RunAgain();
    }
  }
  todo_.Pop();
  return Success();
}

// State transition.
auto Interpreter::Step() -> ErrorOr<Success> {
  // Check for various overflow conditions before stepping.
  if (todo_.size() > MaxTodoSize) {
    return ProgramError(SourceLocation("overflow", 1))
           << "stack overflow: too many interpreter actions on stack";
  }
  if (++steps_taken_ > MaxStepsTaken) {
    return ProgramError(SourceLocation("overflow", 1))
           << "possible infinite loop: too many interpreter steps executed";
  }
  if (arena_->allocated() > MaxArenaAllocated) {
    return ProgramError(SourceLocation("overflow", 1))
           << "out of memory: exceeded arena allocation limit";
  }

  Action& act = todo_.CurrentAction();
  switch (act.kind()) {
    case Action::Kind::LocationAction:
      CARBON_RETURN_IF_ERROR(StepLocation());
      break;
    case Action::Kind::ExpressionAction:
      CARBON_RETURN_IF_ERROR(StepExp());
      break;
    case Action::Kind::WitnessAction:
      CARBON_RETURN_IF_ERROR(StepWitness());
      break;
    case Action::Kind::StatementAction:
      CARBON_RETURN_IF_ERROR(StepStmt());
      break;
    case Action::Kind::DeclarationAction:
      CARBON_RETURN_IF_ERROR(StepDeclaration());
      break;
    case Action::Kind::CleanUpAction:
      CARBON_RETURN_IF_ERROR(StepCleanUp());
      break;
    case Action::Kind::DestroyAction:
      CARBON_RETURN_IF_ERROR(StepDestroy());
      break;
    case Action::Kind::TypeInstantiationAction:
      CARBON_RETURN_IF_ERROR(StepInstantiateType());
      break;
    case Action::Kind::ScopeAction:
      CARBON_FATAL() << "ScopeAction escaped ActionStack";
    case Action::Kind::RecursiveAction:
      CARBON_FATAL() << "Tried to step a RecursiveAction";
  }  // switch
  return Success();
}

auto Interpreter::RunAllSteps(std::unique_ptr<Action> action)
    -> ErrorOr<Success> {
  if (trace_stream_->is_enabled()) {
    TraceState();
  }
  todo_.Start(std::move(action));
  while (!todo_.empty()) {
    CARBON_RETURN_IF_ERROR(Step());
    if (trace_stream_->is_enabled()) {
      TraceState();
    }
  }
  return Success();
}

auto InterpProgram(const AST& ast, Nonnull<Arena*> arena,
                   Nonnull<TraceStream*> trace_stream,
                   Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<int> {
  Interpreter interpreter(Phase::RunTime, arena, trace_stream, print_stream);
  if (trace_stream->is_enabled()) {
    *trace_stream << "********** initializing globals **********\n";
  }

  for (Nonnull<Declaration*> declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(interpreter.RunAllSteps(
        std::make_unique<DeclarationAction>(declaration)));
  }

  if (trace_stream->is_enabled()) {
    *trace_stream << "********** calling main function **********\n";
  }

  CARBON_RETURN_IF_ERROR(interpreter.RunAllSteps(
      std::make_unique<ExpressionAction>(*ast.main_call)));

  return cast<IntValue>(*interpreter.result()).value();
}

auto InterpExp(Nonnull<const Expression*> e, Nonnull<Arena*> arena,
               Nonnull<TraceStream*> trace_stream,
               Nonnull<llvm::raw_ostream*> print_stream)
    -> ErrorOr<Nonnull<const Value*>> {
  Interpreter interpreter(Phase::CompileTime, arena, trace_stream,
                          print_stream);
  CARBON_RETURN_IF_ERROR(
      interpreter.RunAllSteps(std::make_unique<ExpressionAction>(e)));
  return interpreter.result();
}

}  // namespace Carbon
