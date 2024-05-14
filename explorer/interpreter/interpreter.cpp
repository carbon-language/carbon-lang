// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/interpreter.h"

#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "common/check.h"
#include "common/error.h"
#include "explorer/ast/address.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/element.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/expression_category.h"
#include "explorer/ast/value.h"
#include "explorer/base/arena.h"
#include "explorer/base/error_builders.h"
#include "explorer/base/print_as_id.h"
#include "explorer/base/source_location.h"
#include "explorer/base/trace_stream.h"
#include "explorer/interpreter/action.h"
#include "explorer/interpreter/action_stack.h"
#include "explorer/interpreter/heap.h"
#include "explorer/interpreter/pattern_match.h"
#include "explorer/interpreter/type_utils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {

// Limits for various overflow conditions.
static constexpr int64_t MaxTodoSize = 1e3;
static constexpr int64_t MaxStepsTaken = 1e6;
static constexpr int64_t MaxArenaAllocated = 1e9;

// Constructs an ActionStack suitable for the specified phase.
static auto MakeTodo(Phase phase, Nonnull<Heap*> heap,
                     Nonnull<TraceStream*> trace_stream) -> ActionStack {
  switch (phase) {
    case Phase::CompileTime:
      return ActionStack(trace_stream);
    case Phase::RunTime:
      return ActionStack(trace_stream, heap);
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
        heap_(trace_stream, arena),
        todo_(MakeTodo(phase, &heap_, trace_stream)),
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

  // State transitions for expressions value generation.
  auto StepValueExp() -> ErrorOr<Success>;
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
      -> ErrorOr<Nonnull<const NominalClassValue*>>;

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
  auto InstantiateWitness(Nonnull<const Witness*> witness,
                          SourceLocation source_loc)
      -> ErrorOr<Nonnull<const Witness*>>;

  // Call the function `fun` with the given `arg` and the `witnesses`
  // for the function's impl bindings.
  auto CallFunction(const CallExpression& call, Nonnull<const Value*> fun,
                    Nonnull<const Value*> arg, ImplWitnessMap&& witnesses,
                    std::optional<AllocationId> location_received)
      -> ErrorOr<Success>;

  // Call the destructor method in `fun`, with any self argument bound to
  // `receiver`.
  auto CallDestructor(Nonnull<const DestructorDeclaration*> fun,
                      ExpressionResult receiver) -> ErrorOr<Success>;

  // If the given method or destructor `decl` has a self argument, bind it to
  // `receiver`.
  void BindSelfIfPresent(Nonnull<const CallableDeclaration*> decl,
                         ExpressionResult receiver, RuntimeScope& method_scope,
                         BindingMap& generic_args,
                         const SourceLocation& source_location);

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
    case Operator::Deref: {
      CARBON_ASSIGN_OR_RETURN(
          const auto* value,
          heap_.Read(cast<PointerValue>(*args[0]).address(), source_loc));
      return arena_->New<ReferenceExpressionValue>(
          value, cast<PointerValue>(*args[0]).address());
    }
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
  std::vector<NamedValue> elements;
  for (const auto [field, value] : llvm::zip_equal(fields, values)) {
    elements.push_back({field.name(), value});
  }

  return arena_->New<StructValue>(std::move(elements));
}

auto Interpreter::StepLocation() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Expression& exp = cast<LocationAction>(act).expression();

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
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
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
            std::make_unique<ValueExpressionAction>(op.arguments()[0]));
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
  todo_.BeginRecursiveAction();
  CARBON_RETURN_IF_ERROR(todo_.Spawn(std::move(action)));
  // Note that the only `RecursiveAction` we can encounter here is our own --
  // if a nested action begins a recursive action, it will run until that
  // action is finished and popped off the queue before returning to us.
  while (!isa<RecursiveAction>(todo_.CurrentAction())) {
    CARBON_RETURN_IF_ERROR(Step());
  }
  if (trace_stream_->is_enabled()) {
    trace_stream_->End() << "recursive eval done\n";
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
                          InstantiateWitness(&assoc->witness(), source_loc));

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
  if (trace_stream_->is_enabled()) {
    trace_stream_->Start() << "instantiating type `" << *type << "` ("
                           << source_loc << ")\n";
  }

  const Value* value = nullptr;
  switch (type->kind()) {
    case Value::Kind::VariableType: {
      CARBON_ASSIGN_OR_RETURN(
          value,
          todo_.ValueOfNode(&cast<VariableType>(*type).binding(), source_loc));
      if (const auto* location = dyn_cast<LocationValue>(value)) {
        CARBON_ASSIGN_OR_RETURN(value,
                                heap_.Read(location->address(), source_loc));
      }
      break;
    }
    case Value::Kind::InterfaceType: {
      const auto& interface_type = cast<InterfaceType>(*type);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Bindings*> bindings,
          InstantiateBindings(&interface_type.bindings(), source_loc));
      value =
          arena_->New<InterfaceType>(&interface_type.declaration(), bindings);
      break;
    }
    case Value::Kind::NamedConstraintType: {
      const auto& constraint_type = cast<NamedConstraintType>(*type);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Bindings*> bindings,
          InstantiateBindings(&constraint_type.bindings(), source_loc));
      value = arena_->New<NamedConstraintType>(&constraint_type.declaration(),
                                               bindings);
      break;
    }
    case Value::Kind::ChoiceType: {
      const auto& choice_type = cast<ChoiceType>(*type);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Bindings*> bindings,
          InstantiateBindings(&choice_type.bindings(), source_loc));
      value = arena_->New<ChoiceType>(&choice_type.declaration(), bindings);
      break;
    }
    case Value::Kind::AssociatedConstant: {
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> type_value,
          EvalAssociatedConstant(cast<AssociatedConstant>(type), source_loc));
      value = type_value;
      break;
    }
    default:
      value = type;
      break;
  }

  if (trace_stream_->is_enabled()) {
    trace_stream_->End() << "instantiated type `" << *type << "` as `" << *value
                         << "` (" << source_loc << ")\n";
  }

  return value;
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
    CARBON_ASSIGN_OR_RETURN(
        witness, InstantiateWitness(cast<Witness>(witness), source_loc));
  }

  if (args == bindings->args() && witnesses == bindings->witnesses()) {
    return bindings;
  }
  return arena_->New<Bindings>(std::move(args), std::move(witnesses));
}

auto Interpreter::InstantiateWitness(Nonnull<const Witness*> witness,
                                     SourceLocation source_loc)
    -> ErrorOr<Nonnull<const Witness*>> {
  CARBON_ASSIGN_OR_RETURN(
      Nonnull<const Value*> value,
      EvalRecursively(std::make_unique<WitnessAction>(witness, source_loc)));
  return cast<Witness>(value);
}

auto Interpreter::ConvertStructToClass(
    Nonnull<const StructValue*> init_struct,
    Nonnull<const NominalClassType*> class_type, SourceLocation source_loc)
    -> ErrorOr<Nonnull<const NominalClassValue*>> {
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
          CARBON_CHECK(array_type.has_size());
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
      std::vector<Nonnull<const Value*>> new_elements;
      for (const auto [element, dest_type] :
           llvm::zip_equal(tuple->elements(), destination_element_types)) {
        CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> val,
                                Convert(element, dest_type, source_loc));
        new_elements.push_back(val);
      }
      return arena_->New<TupleValue>(std::move(new_elements));
    }
    case Value::Kind::VariableType: {
      std::optional<Nonnull<const Value*>> source_type;
      // While type-checking a `where` expression, we can evaluate a reference
      // to its self binding before we know its type. In this case, the self
      // binding is always a type.
      //
      // TODO: Add a conversion kind to BuiltinConvertExpression so that we
      // don't need to look at the types and reconstruct what kind of
      // conversion is being performed from here.
      if (cast<VariableType>(value)->binding().is_type_checked()) {
        CARBON_ASSIGN_OR_RETURN(
            source_type,
            InstantiateType(&cast<VariableType>(value)->binding().static_type(),
                            source_loc));
      }
      if (isa<TypeType, ConstraintType, NamedConstraintType, InterfaceType>(
              destination_type) &&
          (!source_type ||
           isa<TypeType, ConstraintType, NamedConstraintType, InterfaceType>(
               *source_type))) {
        // No further conversions are required.
        return value;
      }
      // We need to convert this, and we don't know how because we don't have
      // the value yet.
      return ProgramError(source_loc)
             << "value of generic binding " << *value << " is not known";
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
    case Value::Kind::ReferenceExpressionValue: {
      const auto* expr_value = cast<ReferenceExpressionValue>(value);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> converted,
          Convert(expr_value->value(), destination_type, source_loc));
      if (converted == expr_value->value()) {
        return expr_value;
      } else {
        return converted;
      }
    }
  }
}

auto Interpreter::CallFunction(const CallExpression& call,
                               Nonnull<const Value*> fun,
                               Nonnull<const Value*> arg,
                               ImplWitnessMap&& witnesses,
                               std::optional<AllocationId> location_received)
    -> ErrorOr<Success> {
  if (trace_stream_->is_enabled()) {
    trace_stream_->Call() << "calling function: " << *fun << "\n";
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

      // Enter the binding scope to make any deduced arguments visible before
      // we resolve the self type and parameter type.
      auto& binding_scope = todo_.CurrentAction().scope().value();

      // Bring the deduced arguments and their witnesses into scope.
      for (const auto& [bind, val] : call.deduced_args()) {
        CARBON_ASSIGN_OR_RETURN(Nonnull<const Value*> inst_val,
                                InstantiateType(val, call.source_loc()));
        binding_scope.BindValue(bind->original(), inst_val);
      }
      for (const auto& [impl_bind, witness] : witnesses) {
        binding_scope.BindValue(impl_bind->original(), witness);
      }

      // Bring the arguments that are determined by the function value into
      // scope. This includes the arguments for the class of which the function
      // is a member.
      for (const auto& [bind, val] : func_val->type_args()) {
        binding_scope.BindValue(bind->original(), val);
      }
      for (const auto& [impl_bind, witness] : func_val->witnesses()) {
        binding_scope.BindValue(impl_bind->original(), witness);
      }

      RuntimeScope function_scope(&heap_);
      BindingMap generic_args;

      // Bind the receiver to the `self` parameter, if there is one.
      if (const auto* method_val = dyn_cast<BoundMethodValue>(func_val)) {
        BindSelfIfPresent(&function,
                          ExpressionResult::Value(method_val->receiver()),
                          function_scope, generic_args, call.source_loc());
      }

      CARBON_ASSIGN_OR_RETURN(
          Nonnull<const Value*> converted_args,
          Convert(arg, &function.param_pattern().static_type(),
                  call.source_loc()));

      // Bind the arguments to the parameters.
      bool success = PatternMatch(&function.param_pattern().value(),
                                  ExpressionResult::Value(converted_args),
                                  call.source_loc(), &function_scope,
                                  generic_args, trace_stream_, this->arena_);
      CARBON_CHECK(success) << "Failed to bind arguments to parameters";
      return todo_.Spawn(std::make_unique<StatementAction>(*function.body(),
                                                           location_received),
                         std::move(function_scope));
    }
    case Value::Kind::ParameterizedEntityName: {
      const auto& name = cast<ParameterizedEntityName>(*fun);
      const Declaration& decl = name.declaration();
      RuntimeScope params_scope(&heap_);
      BindingMap generic_args;
      CARBON_CHECK(PatternMatch(&name.params().value(),
                                ExpressionResult::Value(arg), call.source_loc(),
                                &params_scope, generic_args, trace_stream_,
                                this->arena_));
      Nonnull<const Bindings*> bindings =
          arena_->New<Bindings>(std::move(generic_args), std::move(witnesses));
      switch (decl.kind()) {
        case DeclarationKind::ClassDeclaration: {
          const auto& class_decl = cast<ClassDeclaration>(decl);
          return todo_.FinishAction(arena_->New<NominalClassType>(
              &class_decl, bindings, class_decl.base_type(), EmptyVTable()));
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

auto Interpreter::CallDestructor(Nonnull<const DestructorDeclaration*> fun,
                                 ExpressionResult receiver)
    -> ErrorOr<Success> {
  const DestructorDeclaration& method = *fun;
  CARBON_CHECK(method.is_method());

  RuntimeScope method_scope(&heap_);
  BindingMap generic_args;
  BindSelfIfPresent(fun, receiver, method_scope, generic_args,
                    SourceLocation::DiagnosticsIgnored());

  CARBON_CHECK(method.body().has_value())
      << "Calling a method that's missing a body";

  auto act = std::make_unique<StatementAction>(*method.body(), std::nullopt);
  return todo_.Spawn(std::unique_ptr<Action>(std::move(act)),
                     std::move(method_scope));
}

void Interpreter::BindSelfIfPresent(Nonnull<const CallableDeclaration*> decl,
                                    ExpressionResult receiver,
                                    RuntimeScope& method_scope,
                                    BindingMap& generic_args,
                                    const SourceLocation& source_location) {
  CARBON_CHECK(decl->is_method());
  const auto* self_pattern = &decl->self_pattern().value();
  if (const auto* placeholder =
          dyn_cast<BindingPlaceholderValue>(self_pattern)) {
    // Immutable self with `[self: Self]`
    if (placeholder->value_node().has_value()) {
      bool success =
          PatternMatch(placeholder, receiver, source_location, &method_scope,
                       generic_args, trace_stream_, this->arena_);
      CARBON_CHECK(success) << "Failed to bind self";
    }
  } else {
    // Mutable self with `[addr self: Self*]`
    CARBON_CHECK(isa<AddrValue>(self_pattern));
    ExpressionResult v = receiver;
    // See if we need to make a LocationValue from the address provided in the
    // ExpressionResult
    if (receiver.value()->kind() != Value::Kind::LocationValue) {
      CARBON_CHECK(receiver.expression_category() ==
                   ExpressionCategory::Reference);
      CARBON_CHECK(receiver.address().has_value());
      v = ExpressionResult::Value(
          arena_->New<LocationValue>(receiver.address().value()));
    }
    bool success = PatternMatch(self_pattern, v, source_location, &method_scope,
                                generic_args, trace_stream_, this->arena_);
    CARBON_CHECK(success) << "Failed to bind addr self";
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
            &class_type.declaration(), bindings, base, &class_type.vtable()));
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

auto Interpreter::StepValueExp() -> ErrorOr<Success> {
  auto& act = cast<ValueExpressionAction>(todo_.CurrentAction());

  if (act.pos() == 0) {
    return todo_.Spawn(std::make_unique<ExpressionAction>(
        &act.expression(), /*preserve_nested_categories=*/false,
        act.location_received()));
  } else {
    CARBON_CHECK(act.results().size() == 1);
    if (const auto* expr_value =
            dyn_cast<ReferenceExpressionValue>(act.results()[0])) {
      // Unwrap the ExpressionAction to only keep the resulting
      // `Value*`.
      return todo_.FinishAction(expr_value->value());
    } else {
      return todo_.FinishAction(act.results()[0]);
    }
  }
}

auto Interpreter::StepExp() -> ErrorOr<Success> {
  auto& act = cast<ExpressionAction>(todo_.CurrentAction());
  const Expression& exp = act.expression();

  switch (exp.kind()) {
    case ExpressionKind::IndexExpression: {
      if (act.pos() == 0) {
        //    { { e[i] :: C, E, F} :: S, H}
        // -> { { e :: [][i] :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
            &cast<IndexExpression>(exp).object()));
      } else if (act.pos() == 1) {
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
            &cast<IndexExpression>(exp).offset()));
      } else {
        //    { { v :: [][i] :: C, E, F} :: S, H}
        // -> { { v_i :: C, E, F} : S, H}
        CARBON_ASSIGN_OR_RETURN(
            auto converted,
            Convert(act.results()[0],
                    &cast<IndexExpression>(exp).object().static_type(),
                    exp.source_loc()));
        const auto& tuple = cast<TupleValue>(*converted);
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
        const auto* field = cast<TupleLiteral>(exp).fields()[act.pos()];
        if (act.preserve_nested_categories()) {
          return todo_.Spawn(std::make_unique<ExpressionAction>(field, false));
        } else {
          return todo_.Spawn(std::make_unique<ValueExpressionAction>(field));
        }
      } else {
        return todo_.FinishAction(arena_->New<TupleValue>(act.results()));
      }
    }
    case ExpressionKind::StructLiteral: {
      const auto& literal = cast<StructLiteral>(exp);
      if (act.pos() < static_cast<int>(literal.fields().size())) {
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
            &literal.fields()[act.pos()].expression()));
      } else {
        return todo_.FinishAction(
            CreateStruct(literal.fields(), act.results()));
      }
    }
    case ExpressionKind::SimpleMemberAccessExpression: {
      const auto& access = cast<SimpleMemberAccessExpression>(exp);
      if (auto rewrite = access.rewritten_form()) {
        return todo_.ReplaceWith(std::make_unique<ExpressionAction>(
            *rewrite, act.preserve_nested_categories(),
            act.location_received()));
      }
      if (act.pos() == 0) {
        // First, evaluate the first operand.
        if (access.is_addr_me_method()) {
          return todo_.Spawn(
              std::make_unique<LocationAction>(&access.object()));
        } else {
          return todo_.Spawn(std::make_unique<ExpressionAction>(
              &access.object(), /*preserve_nested_categories=*/false));
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
            const auto* result =
                act.results()[0]->kind() ==
                        Value::Kind::ReferenceExpressionValue
                    ? cast<ReferenceExpressionValue>(act.results()[0])->value()
                    : act.results()[0];
            if (!isa<InterfaceType, NamedConstraintType, ConstraintType>(
                    result)) {
              type_result = result;
            }
            const auto* member_name = arena_->New<MemberName>(
                type_result, found_in_interface, &member_name_type->member());
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
              return todo_.Spawn(std::make_unique<WitnessAction>(
                  access.impl().value(), access.source_loc()));
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
            std::optional<Nonnull<const Value*>> me_value;
            std::optional<Address> lhs_address;
            if (access.is_type_access()) {
              aggregate = act.results().back();
            } else if (const auto* location =
                           dyn_cast<LocationValue>(act.results()[0])) {
              lhs_address = location->address();
              me_value = act.results()[0];
              CARBON_ASSIGN_OR_RETURN(
                  aggregate,
                  this->heap_.Read(location->address(), exp.source_loc()));
            } else if (const auto* expr_value =
                           dyn_cast<ReferenceExpressionValue>(
                               act.results()[0])) {
              lhs_address = expr_value->address();
              aggregate = expr_value->value();
              me_value = aggregate;
            } else {
              aggregate = act.results()[0];
              me_value = aggregate;
            }
            CARBON_ASSIGN_OR_RETURN(
                Nonnull<const Value*> member_value,
                aggregate->GetElement(arena_, ElementPath(member),
                                      exp.source_loc(), me_value));
            if (lhs_address) {
              return todo_.FinishAction(arena_->New<ReferenceExpressionValue>(
                  member_value, lhs_address->ElementAddress(member.element())));
            } else {
              return todo_.FinishAction(member_value);
            }
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
              std::make_unique<ValueExpressionAction>(&access.object()));
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
            auto* member_name =
                arena_->New<MemberName>(act.results()[0], found_in_interface,
                                        &access.member().member());
            return todo_.FinishAction(member_name);
          }
        } else {
          auto impl_has_value = access.impl().has_value();
          if (act.pos() == 1) {
            if (impl_has_value) {
              // Next, if we're accessing an interface member, evaluate the
              // `impl` expression to find the corresponding witness.
              return todo_.Spawn(std::make_unique<WitnessAction>(
                  access.impl().value(), access.source_loc()));
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
            std::make_unique<ValueExpressionAction>(&access.object()));
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
        if (ident.expression_category() == ExpressionCategory::Reference) {
          return todo_.FinishAction(arena_->New<ReferenceExpressionValue>(
              value, location->address()));
        }
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
        return todo_.ReplaceWith(std::make_unique<ExpressionAction>(
            *rewrite, act.preserve_nested_categories(),
            act.location_received()));
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
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(arg));
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
      CARBON_CHECK(call.argument().kind() == ExpressionKind::TupleLiteral);
      const auto& args = cast<TupleLiteral>(call.argument());
      const int num_args = args.fields().size();
      const int num_witnesses = call.witnesses().size();
      const int function_call_pos = 1 + num_args + num_witnesses;
      if (act.pos() == 0) {
        //    { {e1(e2) :: C, E, F} :: S, H}
        // -> { {e1 :: [](e2) :: C, E, F} :: S, H}
        act.StartScope(RuntimeScope(&heap_));
        return todo_.Spawn(
            std::make_unique<ValueExpressionAction>(&call.function()));
      } else if (act.pos() < 1 + num_args) {
        const auto* field = args.fields()[act.pos() - 1];
        std::optional<AllocationId> alloc;
        if (field->expression_category() == ExpressionCategory::Initializing) {
          alloc = heap_.AllocateValue(
              arena_->New<UninitializedValue>(&field->static_type()));
          act.scope()->BindLifetimeToScope(Address(*alloc));
        }
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(field, false, alloc));
      } else if (act.pos() < function_call_pos) {
        auto iter = call.witnesses().begin();
        std::advance(iter, act.pos() - 1 - num_args);
        return todo_.Spawn(std::make_unique<WitnessAction>(
            cast<Witness>(iter->second), call.source_loc()));
      } else if (act.pos() == function_call_pos) {
        //    { { v2 :: v1([]) :: C, E, F} :: S, H}
        // -> { {C',E',F'} :: {C, E, F} :: S, H}
        // Prepare parameters tuple.
        std::vector<Nonnull<const Value*>> param_values;
        for (const auto& arg_result :
             llvm::ArrayRef(act.results()).slice(1, num_args)) {
          param_values.push_back(arg_result);
        }
        const auto* param_tuple = arena_->New<TupleValue>(param_values);
        // Prepare witnesses.
        ImplWitnessMap witnesses;
        if (num_witnesses > 0) {
          for (const auto [witness, result] : llvm::zip(
                   call.witnesses(),
                   llvm::ArrayRef(act.results()).drop_front(1 + num_args))) {
            witnesses[witness.first] = result;
          }
        }
        return CallFunction(call, act.results()[0], param_tuple,
                            std::move(witnesses), act.location_received());
      } else if (act.pos() == 1 + function_call_pos) {
        if (static_cast<int>(act.results().size()) < 1 + function_call_pos) {
          // Control fell through without explicit return.
          return todo_.FinishAction(TupleValue::Empty());
        } else {
          return todo_.FinishAction(act.results()[function_call_pos]);
        }
      } else {
        CARBON_FATAL() << "in StepValueExp with Call pos " << act.pos();
      }
    }
    case ExpressionKind::IntrinsicExpression: {
      const auto& intrinsic = cast<IntrinsicExpression>(exp);
      if (auto rewrite = intrinsic.rewritten_form()) {
        return todo_.ReplaceWith(std::make_unique<ExpressionAction>(
            *rewrite, act.preserve_nested_categories(),
            act.location_received()));
      }
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ValueExpressionAction>(&intrinsic.args()));
      }
      // { {n :: C, E, F} :: S, H} -> { {n' :: C, E, F} :: S, H}
      const auto& args = cast<TupleValue>(*act.results()[0]).elements();
      switch (cast<IntrinsicExpression>(exp).intrinsic()) {
        case IntrinsicExpression::Intrinsic::Print: {
          if (phase_ != Phase::RunTime) {
            return ProgramError(exp.source_loc())
                   << "Print called before run time";
          }
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
              CARBON_RETURN_IF_ERROR(heap_.Deallocate(obj_addr));
              return todo_.FinishAction(TupleValue::Empty());
            }
          } else {
            if (act.pos() == 1) {
              return todo_.Spawn(std::make_unique<DestroyAction>(
                  arena_->New<LocationValue>(ptr->address()), pointee));
            } else {
              CARBON_RETURN_IF_ERROR(heap_.Deallocate(ptr->address()));
              return todo_.FinishAction(TupleValue::Empty());
            }
          }
        }
        case IntrinsicExpression::Intrinsic::PrintAllocs: {
          CARBON_CHECK(args.empty());
          heap_.Print(*print_stream_);
          *print_stream_ << "\n";
          return todo_.FinishAction(TupleValue::Empty());
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
          IntrinsicConstraint constraint(self, IntrinsicConstraint::ImplicitAs,
                                         args);
          auto* result = arena_->New<ConstraintType>(
              self_binding, std::vector<ImplsConstraint>{},
              std::vector<IntrinsicConstraint>{std::move(constraint)},
              std::vector<EqualityConstraint>{},
              std::vector<RewriteConstraint>{}, std::vector<LookupContext>{});
          impl_binding->set_interface(result);
          return todo_.FinishAction(result);
        }
        case IntrinsicExpression::Intrinsic::ImplicitAsConvert: {
          CARBON_FATAL()
              << "__intrinsic_implicit_as_convert should have been rewritten";
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
            std::make_unique<ValueExpressionAction>(&if_expr.condition()));
      } else if (act.pos() == 1) {
        const auto& condition = cast<BoolValue>(*act.results()[0]);
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
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
      return todo_.ReplaceWith(std::make_unique<ExpressionAction>(
          *rewrite, act.preserve_nested_categories(), act.location_received()));
    }
    case ExpressionKind::BuiltinConvertExpression: {
      const auto& convert_expr = cast<BuiltinConvertExpression>(exp);
      if (auto rewrite = convert_expr.rewritten_form()) {
        return todo_.ReplaceWith(std::make_unique<ExpressionAction>(
            *rewrite, act.preserve_nested_categories(),
            act.location_received()));
      }
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
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
  auto& act = cast<WitnessAction>(todo_.CurrentAction());
  const Witness* witness = act.witness();

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
        return todo_.Spawn(std::make_unique<WitnessAction>(witnesses[act.pos()],
                                                           act.source_loc()));
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
            constraint_impl->constraint_witness(), act.source_loc()));
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
  auto& act = cast<StatementAction>(todo_.CurrentAction());
  const Statement& stmt = act.statement();

  if (trace_stream_->is_enabled()) {
    trace_stream_->Source() << "statement at (" << stmt.source_loc() << ")\n";
    *trace_stream_ << "```\n" << stmt << "\n```\n";
  }

  switch (stmt.kind()) {
    case StatementKind::Match: {
      const auto& match_stmt = cast<Match>(stmt);
      if (act.pos() == 0) {
        //    { { (match (e) ...) :: C, E, F} :: S, H}
        // -> { { e :: (match ([]) ...) :: C, E, F} :: S, H}
        act.StartScope(RuntimeScope(&heap_));
        return todo_.Spawn(
            std::make_unique<ValueExpressionAction>(&match_stmt.expression()));
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
        if (PatternMatch(&c.pattern().value(), ExpressionResult::Value(val),
                         stmt.source_loc(), &matches, generic_args,
                         trace_stream_, this->arena_)) {
          // Ensure we don't process any more clauses.
          act.set_pos(match_stmt.clauses().size() + 1);
          todo_.MergeScope(std::move(matches));
          return todo_.Spawn(
              std::make_unique<StatementAction>(&c.statement(), std::nullopt));
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
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
            &cast<For>(stmt).loop_target()));
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
        return todo_.Spawn(std::make_unique<StatementAction>(
            &cast<For>(stmt).body(), std::nullopt));
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
          return todo_.Spawn(std::make_unique<StatementAction>(
              &cast<For>(stmt).body(), std::nullopt));
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
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
            &cast<While>(stmt).condition()));
      } else {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> condition,
            Convert(act.results().back(), arena_->New<BoolType>(),
                    stmt.source_loc()));
        if (cast<BoolValue>(*condition).value()) {
          //    { {true :: (while ([]) s) :: C, E, F} :: S, H}
          // -> { { s :: (while (e) s) :: C, E, F } :: S, H}
          return todo_.Spawn(std::make_unique<StatementAction>(
              &cast<While>(stmt).body(), std::nullopt));
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
      return todo_.Spawn(std::make_unique<StatementAction>(
          block.statements()[act.pos()], act.location_received()));
    }
    case StatementKind::VariableDefinition: {
      const auto& definition = cast<VariableDefinition>(stmt);
      const bool has_initializing_expr =
          definition.has_init() &&
          definition.init().kind() == ExpressionKind::CallExpression &&
          definition.init().expression_category() ==
              ExpressionCategory::Initializing;
      auto init_location = (act.location_received() && definition.is_returned())
                               ? act.location_received()
                               : act.location_created();
      if (act.pos() == 0 && definition.has_init()) {
        //    { {(var x = e) :: C, E, F} :: S, H}
        // -> { {e :: (var x = []) :: C, E, F} :: S, H}
        if (has_initializing_expr && !init_location) {
          // Allocate storage for initializing expression.
          const auto allocation_id =
              heap_.AllocateValue(arena_->New<UninitializedValue>(
                  &definition.init().static_type()));
          act.set_location_created(allocation_id);
          init_location = allocation_id;
          RuntimeScope scope(&heap_);
          scope.BindLifetimeToScope(Address(allocation_id));
          todo_.MergeScope(std::move(scope));
        }
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            &definition.init(), /*preserve_nested_categories=*/false,
            init_location));
      } else {
        //    { { v :: (x = []) :: C, E, F} :: S, H}
        // -> { { C, E(x := a), F} :: S, H(a := copy(v))}
        Nonnull<const Value*> p = &definition.pattern().value();
        Nonnull<const Value*> v;
        std::optional<Address> v_location;
        ExpressionCategory expr_category =
            definition.has_init() ? definition.init().expression_category()
                                  : ExpressionCategory::Value;
        if (definition.has_init()) {
          Nonnull<const Value*> result = act.results()[0];
          std::optional<Nonnull<const ReferenceExpressionValue*>> v_expr =
              (result->kind() == Value::Kind::ReferenceExpressionValue)
                  ? std::optional{cast<ReferenceExpressionValue>(result)}
                  : std::nullopt;
          const auto init_location = act.location_created();
          v = v_expr ? (*v_expr)->value() : result;
          if (expr_category == ExpressionCategory::Reference) {
            CARBON_CHECK(v_expr) << "Expecting ReferenceExpressionValue from "
                                    "reference expression";
            v_location = (*v_expr)->address();
            CARBON_CHECK(v_location)
                << "Expecting a valid address from reference expression";
          } else if (has_initializing_expr && init_location &&
                     heap_.is_initialized(*init_location)) {
            // Bind even if a conversion is necessary.
            v_location = Address(*init_location);
            CARBON_ASSIGN_OR_RETURN(
                result, heap_.Read(*v_location, definition.source_loc()));
            CARBON_CHECK(v == result);
          } else {
            // TODO: Prevent copies for Value expressions from Reference
            // expression, once able to prevent mutations.
            if (init_location && act.location_created()) {
              // Location provided to initializing expression was not used.
              heap_.Discard(*init_location);
            }
            expr_category = ExpressionCategory::Value;
            const auto* dest_type = &definition.pattern().static_type();
            CARBON_ASSIGN_OR_RETURN(v,
                                    Convert(v, dest_type, stmt.source_loc()));
          }
        } else {
          v = arena_->New<UninitializedValue>(p);
        }

        // If declaring a returned var, bind name to the location provided to
        // initializing expression, if any.
        RuntimeScope scope(&heap_);
        if (definition.is_returned() && init_location) {
          CARBON_CHECK(p->kind() == Value::Kind::BindingPlaceholderValue);
          const auto value_node =
              cast<BindingPlaceholderValue>(*p).value_node();
          CARBON_CHECK(value_node);
          const auto address = Address(*init_location);
          scope.Bind(*value_node, address);
          CARBON_RETURN_IF_ERROR(heap_.Write(address, v, stmt.source_loc()));
        } else {
          BindingMap generic_args;
          bool matched =
              PatternMatch(p, ExpressionResult(v, v_location, expr_category),
                           stmt.source_loc(), &scope, generic_args,
                           trace_stream_, this->arena_);
          CARBON_CHECK(matched)
              << stmt.source_loc()
              << ": internal error in variable definition, match failed";
        }
        todo_.MergeScope(std::move(scope));
        return todo_.FinishAction();
      }
    }
    case StatementKind::ExpressionStatement:
      if (act.pos() == 0) {
        //    { {e :: C, E, F} :: S, H}
        // -> { {e :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
            &cast<ExpressionStatement>(stmt).expression()));
      } else {
        return todo_.FinishAction();
      }
    case StatementKind::Assign: {
      const auto& assign = cast<Assign>(stmt);
      if (auto rewrite = assign.rewritten_form()) {
        if (act.pos() == 0) {
          return todo_.Spawn(std::make_unique<ValueExpressionAction>(*rewrite));
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
        return todo_.Spawn(
            std::make_unique<ValueExpressionAction>(&assign.rhs()));
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
            std::make_unique<ValueExpressionAction>(*inc_dec.rewritten_form()));
      } else {
        return todo_.FinishAction();
      }
    }
    case StatementKind::If:
      if (act.pos() == 0) {
        //    { {(if (e) then_stmt else else_stmt) :: C, E, F} :: S, H}
        // -> { { e :: (if ([]) then_stmt else else_stmt) :: C, E, F} :: S, H}
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
            &cast<If>(stmt).condition()));
      } else if (act.pos() == 1) {
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> condition,
            Convert(act.results()[0], arena_->New<BoolType>(),
                    stmt.source_loc()));
        if (cast<BoolValue>(*condition).value()) {
          //    { {true :: if ([]) then_stmt else else_stmt :: C, E, F} ::
          //      S, H}
          // -> { { then_stmt :: C, E, F } :: S, H}
          return todo_.Spawn(std::make_unique<StatementAction>(
              &cast<If>(stmt).then_block(), std::nullopt));
        } else if (cast<If>(stmt).else_block()) {
          //    { {false :: if ([]) then_stmt else else_stmt :: C, E, F} ::
          //      S, H}
          // -> { { else_stmt :: C, E, F } :: S, H}
          return todo_.Spawn(std::make_unique<StatementAction>(
              *cast<If>(stmt).else_block(), std::nullopt));
        } else {
          return todo_.FinishAction();
        }
      } else {
        return todo_.FinishAction();
      }
    case StatementKind::ReturnVar: {
      const auto& ret_var = cast<ReturnVar>(stmt);
      const ValueNodeView& value_node = ret_var.value_node();
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
        return todo_.Spawn(std::make_unique<ValueExpressionAction>(
            &cast<ReturnExpression>(stmt).expression()));
      } else {
        //    { {v :: return [] :: C, E, F} :: {C', E', F'} :: S, H}
        // -> { {v :: C', E', F'} :: S, H}
        const CallableDeclaration& function = cast<Return>(stmt).function();
        CARBON_ASSIGN_OR_RETURN(
            Nonnull<const Value*> return_value,
            Convert(act.results()[0], &function.return_term().static_type(),
                    stmt.source_loc()));
        // Write to initialized storage location, if any.
        if (const auto location = act.location_received()) {
          CARBON_RETURN_IF_ERROR(
              heap_.Write(Address(*location), return_value, stmt.source_loc()));
        }
        return todo_.UnwindPast(*function.body(), return_value);
      }
  }
}

auto Interpreter::StepDeclaration() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Declaration& decl = cast<DeclarationAction>(act).declaration();

  if (trace_stream_->is_enabled()) {
    trace_stream_->Source() << "declaration at (" << decl.source_loc() << ")\n";
    *trace_stream_ << "```\n" << decl << "\n```\n";
  }

  switch (decl.kind()) {
    case DeclarationKind::VariableDeclaration: {
      const auto& var_decl = cast<VariableDeclaration>(decl);
      if (var_decl.has_initializer()) {
        if (act.pos() == 0) {
          return todo_.Spawn(
              std::make_unique<ValueExpressionAction>(&var_decl.initializer()));
        } else {
          CARBON_ASSIGN_OR_RETURN(
              Nonnull<const Value*> v,
              Convert(act.results()[0], &var_decl.binding().static_type(),
                      var_decl.source_loc()));
          todo_.Initialize(&var_decl.binding(), v);
          return todo_.FinishAction();
        }
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
    case DeclarationKind::InterfaceExtendDeclaration:
    case DeclarationKind::InterfaceRequireDeclaration:
    case DeclarationKind::AssociatedConstantDeclaration:
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::MatchFirstDeclaration:
    case DeclarationKind::SelfDeclaration:
    case DeclarationKind::AliasDeclaration:
    case DeclarationKind::ExtendBaseDeclaration:
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
          return CallDestructor(
              *destructor, ExpressionResult::Reference(
                               class_obj, destroy_act.location()->address()));
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
          const auto v = heap_.Read(var_addr, var->source_loc());
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
    if (heap_.is_discarded(allocation)) {
      // Initializing expressions can generate discarded allocations.
      return todo_.RunAgain();
    }
    if (act.pos() % 2 == 0) {
      auto* location = arena_->New<LocationValue>(Address(allocation));
      auto value = heap_.Read(location->address(), *cleanup.source_loc());
      // Step over uninitialized values.
      if (value.ok()) {
        return todo_.Spawn(std::make_unique<DestroyAction>(location, *value));
      } else {
        return todo_.RunAgain();
      }
    } else {
      CARBON_RETURN_IF_ERROR(heap_.Deallocate(allocation));
      return todo_.RunAgain();
    }
  }
  todo_.Pop();
  return Success();
}

// State transition.
auto Interpreter::Step() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();

  if (trace_stream_->is_enabled()) {
    trace_stream_->Start() << "step " << act << " (" << act.source_loc()
                           << ") --->\n";
  }

  auto error_builder = [&] {
    if (auto loc = act.source_loc()) {
      return ProgramError(*loc);
    }
    return ErrorBuilder();
  };

  // Check for various overflow conditions before stepping.
  if (todo_.size() > MaxTodoSize) {
    return error_builder()
           << "stack overflow: too many interpreter actions on stack";
  }
  if (++steps_taken_ > MaxStepsTaken) {
    return error_builder()
           << "possible infinite loop: too many interpreter steps executed";
  }
  if (arena_->allocated() > MaxArenaAllocated) {
    return error_builder() << "out of memory: exceeded arena allocation limit";
  }

  switch (act.kind()) {
    case Action::Kind::LocationAction:
      CARBON_RETURN_IF_ERROR(StepLocation());
      break;
    case Action::Kind::ValueExpressionAction:
      CARBON_RETURN_IF_ERROR(StepValueExp());
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
  todo_.Start(std::move(action));
  while (!todo_.empty()) {
    CARBON_RETURN_IF_ERROR(Step());
  }
  return Success();
}

auto InterpProgram(const AST& ast, Nonnull<Arena*> arena,
                   Nonnull<TraceStream*> trace_stream,
                   Nonnull<llvm::raw_ostream*> print_stream) -> ErrorOr<int> {
  Interpreter interpreter(Phase::RunTime, arena, trace_stream, print_stream);
  if (trace_stream->is_enabled()) {
    trace_stream->SubHeading("initializing globals");
  }

  SetFileContext set_file_ctx(*trace_stream,
                              ast.declarations.front()->source_loc());
  for (Nonnull<Declaration*> declaration : ast.declarations) {
    set_file_ctx.update_source_loc(declaration->source_loc());
    CARBON_RETURN_IF_ERROR(interpreter.RunAllSteps(
        std::make_unique<DeclarationAction>(declaration)));
  }

  if (trace_stream->is_enabled()) {
    trace_stream->SubHeading("calling main function");
  }

  CARBON_CHECK(ast.main_call);
  set_file_ctx.update_source_loc(ast.main_call.value()->source_loc());
  CARBON_RETURN_IF_ERROR(interpreter.RunAllSteps(
      std::make_unique<ValueExpressionAction>(*ast.main_call)));

  return cast<IntValue>(*interpreter.result()).value();
}

auto InterpExp(Nonnull<const Expression*> e, Nonnull<Arena*> arena,
               Nonnull<TraceStream*> trace_stream,
               Nonnull<llvm::raw_ostream*> print_stream)
    -> ErrorOr<Nonnull<const Value*>> {
  Interpreter interpreter(Phase::CompileTime, arena, trace_stream,
                          print_stream);
  CARBON_RETURN_IF_ERROR(
      interpreter.RunAllSteps(std::make_unique<ValueExpressionAction>(e)));
  return interpreter.result();
}

}  // namespace Carbon
