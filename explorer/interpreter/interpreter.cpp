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
#include "explorer/common/error.h"
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
  Interpreter(Phase phase, Nonnull<Arena*> arena, bool trace)
      : arena_(arena),
        heap_(arena),
        todo_(MakeTodo(phase, &heap_)),
        trace_(trace),
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

  void PrintState(llvm::raw_ostream& out);

  Phase phase() const { return phase_; }

  Nonnull<Arena*> arena_;

  Heap heap_;
  ActionStack todo_;

  // The underlying states of continuation values. All StackFragments created
  // during execution are tracked here, in order to safely deallocate the
  // contents of any non-completed continuations at the end of execution.
  std::vector<Nonnull<ContinuationValue::StackFragment*>> stack_fragments_;

  bool trace_;
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
  CHECK(fields.size() == values.size());
  std::vector<NamedValue> elements;
  for (size_t i = 0; i < fields.size(); ++i) {
    elements.push_back({.name = fields[i].name(), .value = values[i]});
  }

  return arena_->New<StructValue>(std::move(elements));
}

auto PatternMatch(Nonnull<const Value*> p, Nonnull<const Value*> v,
                  SourceLocation source_loc,
                  std::optional<Nonnull<RuntimeScope*>> bindings,
                  BindingMap& generic_args, Nonnull<Arena*> arena) -> bool {
  switch (p->kind()) {
    case Value::Kind::BindingPlaceholderValue: {
      CHECK(bindings.has_value());
      const auto& placeholder = cast<BindingPlaceholderValue>(*p);
      if (placeholder.value_node().has_value()) {
        (*bindings)->Initialize(*placeholder.value_node(), v);
      }
      return true;
    }
    case Value::Kind::AddrValue: {
      const auto& addr = cast<AddrValue>(*p);
      CHECK(v->kind() == Value::Kind::LValue);
      const auto& lvalue = cast<LValue>(*v);
      return PatternMatch(&addr.pattern(),
                          arena->New<PointerValue>(lvalue.address()),
                          source_loc, bindings, generic_args, arena);
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
          CHECK(p_tup.elements().size() == v_tup.elements().size());
          for (size_t i = 0; i < p_tup.elements().size(); ++i) {
            if (!PatternMatch(p_tup.elements()[i], v_tup.elements()[i],
                              source_loc, bindings, generic_args, arena)) {
              return false;
            }
          }  // for
          return true;
        }
        default:
          FATAL() << "expected a tuple value in pattern, not " << *v;
      }
    case Value::Kind::StructValue: {
      const auto& p_struct = cast<StructValue>(*p);
      const auto& v_struct = cast<StructValue>(*v);
      CHECK(p_struct.elements().size() == v_struct.elements().size());
      for (size_t i = 0; i < p_struct.elements().size(); ++i) {
        CHECK(p_struct.elements()[i].name == v_struct.elements()[i].name);
        if (!PatternMatch(p_struct.elements()[i].value,
                          v_struct.elements()[i].value, source_loc, bindings,
                          generic_args, arena)) {
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
                              bindings, generic_args, arena);
        }
        default:
          FATAL() << "expected a choice alternative in pattern, not " << *v;
      }
    case Value::Kind::FunctionType:
      switch (v->kind()) {
        case Value::Kind::FunctionType: {
          const auto& p_fn = cast<FunctionType>(*p);
          const auto& v_fn = cast<FunctionType>(*v);
          if (!PatternMatch(&p_fn.parameters(), &v_fn.parameters(), source_loc,
                            bindings, generic_args, arena)) {
            return false;
          }
          if (!PatternMatch(&p_fn.return_type(), &v_fn.return_type(),
                            source_loc, bindings, generic_args, arena)) {
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
  if (trace_) {
    llvm::outs() << "--- step lvalue " << exp << " (" << exp.source_loc()
                 << ") --->\n";
  }
  switch (exp.kind()) {
    case ExpressionKind::IdentifierExpression: {
      //    { {x :: C, E, F} :: S, H}
      // -> { {E(x) :: C, E, F} :: S, H}
      ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(cast<IdentifierExpression>(exp).value_node(),
                            exp.source_loc()));
      CHECK(isa<LValue>(value)) << *value;
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
        FATAL() << "Can't treat primitive operator expression as lvalue: "
                << exp;
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
      FATAL() << "Can't treat expression as lvalue: " << exp;
    case ExpressionKind::UnimplementedExpression:
      FATAL() << "Unimplemented: " << exp;
  }
}

auto Interpreter::InstantiateType(Nonnull<const Value*> type,
                                  SourceLocation source_loc) const
    -> ErrorOr<Nonnull<const Value*>> {
  if (trace_) {
    llvm::outs() << "instantiating: " << *type << "\n";
  }
  switch (type->kind()) {
    case Value::Kind::VariableType: {
      if (trace_) {
        llvm::outs() << "case VariableType\n";
      }
      ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(&cast<VariableType>(*type).binding(), source_loc));
      if (const auto* lvalue = dyn_cast<LValue>(value)) {
        ASSIGN_OR_RETURN(value, heap_.Read(lvalue->address(), source_loc));
      }
      return value;
    }
    case Value::Kind::NominalClassType: {
      if (trace_) {
        llvm::outs() << "case NominalClassType\n";
      }
      const auto& class_type = cast<NominalClassType>(*type);
      BindingMap inst_type_args;
      for (const auto& [ty_var, ty_arg] : class_type.type_args()) {
        ASSIGN_OR_RETURN(inst_type_args[ty_var],
                         InstantiateType(ty_arg, source_loc));
      }
      if (trace_) {
        llvm::outs() << "finished instantiating ty_arg\n";
      }
      std::map<Nonnull<const ImplBinding*>, Nonnull<const Witness*>> witnesses;
      for (const auto& [bind, impl] : class_type.impls()) {
        ASSIGN_OR_RETURN(Nonnull<const Value*> witness_addr,
                         todo_.ValueOfNode(impl, source_loc));
        if (trace_) {
          llvm::outs() << "witness_addr: " << *witness_addr << "\n";
        }
        // If the witness came directly from an `impl` declaration (via
        // `constant_value`), then it is a `Witness`. If the witness
        // came from the runtime scope, then the `Witness` got wrapped
        // in an `LValue` because that's what
        // `RuntimeScope::Initialize` does.
        Nonnull<const Witness*> witness;
        if (llvm::isa<Witness>(witness_addr)) {
          witness = cast<Witness>(witness_addr);
        } else if (llvm::isa<LValue>(witness_addr)) {
          ASSIGN_OR_RETURN(
              Nonnull<const Value*> witness_value,
              heap_.Read(llvm::cast<LValue>(witness_addr)->address(),
                         source_loc));
          witness = cast<Witness>(witness_value);
        } else {
          FATAL() << "expected a witness or LValue of a witness";
        }
        witnesses[bind] = witness;
      }
      if (trace_) {
        llvm::outs() << "finished finding witnesses\n";
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
    case Value::Kind::ChoiceType:
    case Value::Kind::ContinuationType:
    case Value::Kind::VariableType:
    case Value::Kind::BindingPlaceholderValue:
    case Value::Kind::AddrValue:
    case Value::Kind::AlternativeConstructorValue:
    case Value::Kind::ContinuationValue:
    case Value::Kind::StringType:
    case Value::Kind::StringValue:
    case Value::Kind::TypeOfClassType:
    case Value::Kind::TypeOfInterfaceType:
    case Value::Kind::TypeOfChoiceType:
    case Value::Kind::StaticArrayType:
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
            ASSIGN_OR_RETURN(Nonnull<const Value*> val,
                             Convert(*old_value, field_type, source_loc));
            new_elements.push_back({.name = field_name, .value = val});
          }
          return arena_->New<StructValue>(std::move(new_elements));
        }
        case Value::Kind::NominalClassType: {
          // Instantiate the `destintation_type` to obtain the runtime
          // type of the object.
          ASSIGN_OR_RETURN(Nonnull<const Value*> inst_dest,
                           InstantiateType(destination_type, source_loc));
          return arena_->New<NominalClassValue>(inst_dest, value);
        }
        default:
          FATAL() << "Can't convert value " << *value << " to type "
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
          FATAL() << "Can't convert value " << *value << " to type "
                  << *destination_type;
      }
      CHECK(tuple->elements().size() == destination_element_types.size());
      std::vector<Nonnull<const Value*>> new_elements;
      for (size_t i = 0; i < tuple->elements().size(); ++i) {
        ASSIGN_OR_RETURN(Nonnull<const Value*> val,
                         Convert(tuple->elements()[i],
                                 destination_element_types[i], source_loc));
        new_elements.push_back(val);
      }
      return arena_->New<TupleValue>(std::move(new_elements));
    }
  }
}

auto Interpreter::StepExp() -> ErrorOr<Success> {
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
          return FATAL_RUNTIME_ERROR_NO_LINE()
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
        if (access.is_field_addr_me_method()) {
          return todo_.Spawn(std::make_unique<LValAction>(&access.aggregate()));
        }
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(&access.aggregate()));
      } else {
        //    { { v :: [].f :: C, E, F} :: S, H}
        // -> { { v_f :: C, E, F} : S, H}
        std::optional<Nonnull<const Witness*>> witness = std::nullopt;
        if (access.impl().has_value()) {
          ASSIGN_OR_RETURN(
              auto witness_addr,
              todo_.ValueOfNode(*access.impl(), access.source_loc()));
          ASSIGN_OR_RETURN(
              Nonnull<const Value*> witness_value,
              heap_.Read(llvm::cast<LValue>(witness_addr)->address(),
                         access.source_loc()));
          witness = cast<Witness>(witness_value);
        }
        FieldPath::Component field(access.field(), witness);
        const Value* aggregate;
        if (const auto* lvalue = dyn_cast<LValue>(act.results()[0])) {
          ASSIGN_OR_RETURN(
              aggregate, this->heap_.Read(lvalue->address(), exp.source_loc()));
        } else {
          aggregate = act.results()[0];
        }
        ASSIGN_OR_RETURN(
            Nonnull<const Value*> member,
            aggregate->GetField(arena_, FieldPath(field), exp.source_loc(),
                                act.results()[0]));
        return todo_.FinishAction(member);
      }
    }
    case ExpressionKind::IdentifierExpression: {
      CHECK(act.pos() == 0);
      const auto& ident = cast<IdentifierExpression>(exp);
      // { {x :: C, E, F} :: S, H} -> { {H(E(x)) :: C, E, F} :: S, H}
      ASSIGN_OR_RETURN(
          Nonnull<const Value*> value,
          todo_.ValueOfNode(ident.value_node(), ident.source_loc()));
      if (const auto* lvalue = dyn_cast<LValue>(value)) {
        ASSIGN_OR_RETURN(value,
                         heap_.Read(lvalue->address(), exp.source_loc()));
      }
      return todo_.FinishAction(value);
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
        if (op.op() == Operator::AddressOf) {
          return todo_.Spawn(std::make_unique<LValAction>(arg));
        } else {
          return todo_.Spawn(std::make_unique<ExpressionAction>(arg));
        }
      } else {
        //    { {v :: op(vs,[]) :: C, E, F} :: S, H}
        // -> { {eval_prim(op, (vs,v)) :: C, E, F} :: S, H}
        ASSIGN_OR_RETURN(Nonnull<const Value*> value,
                         EvalPrim(op.op(), act.results(), exp.source_loc()));
        return todo_.FinishAction(value);
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
            const FunctionValue& fun_val =
                cast<FunctionValue>(*act.results()[0]);
            const FunctionDeclaration& function = fun_val.declaration();
            if (trace_) {
              llvm::outs() << "*** call function " << function.name() << "\n";
            }
            ASSIGN_OR_RETURN(Nonnull<const Value*> converted_args,
                             Convert(act.results()[1],
                                     &function.param_pattern().static_type(),
                                     exp.source_loc()));
            RuntimeScope function_scope(&heap_);
            // Bring the class type arguments into scope.
            for (const auto& [bind, val] : fun_val.type_args()) {
              function_scope.Initialize(bind, val);
            }
            // Bring the deduced type arguments into scope.
            for (const auto& [bind, val] :
                 cast<CallExpression>(exp).deduced_args()) {
              function_scope.Initialize(bind, val);
            }

            // Bring the impl witness tables into scope.
            for (const auto& [impl_bind, impl_node] :
                 cast<CallExpression>(exp).impls()) {
              ASSIGN_OR_RETURN(Nonnull<const Value*> witness,
                               todo_.ValueOfNode(impl_node, exp.source_loc()));
              if (witness->kind() == Value::Kind::LValue) {
                const auto& lval = cast<LValue>(*witness);
                ASSIGN_OR_RETURN(witness,
                                 heap_.Read(lval.address(), exp.source_loc()));
              }
              function_scope.Initialize(impl_bind, witness);
            }
            for (const auto& [impl_bind, witness] : fun_val.witnesses()) {
              function_scope.Initialize(impl_bind, witness);
            }
            BindingMap generic_args;
            CHECK(PatternMatch(&function.param_pattern().value(),
                               converted_args, exp.source_loc(),
                               &function_scope, generic_args, this->arena_));
            CHECK(function.body().has_value())
                << "Calling a function that's missing a body";
            return todo_.Spawn(
                std::make_unique<StatementAction>(*function.body()),
                std::move(function_scope));
          }
          case Value::Kind::BoundMethodValue: {
            const auto& m = cast<BoundMethodValue>(*act.results()[0]);
            const FunctionDeclaration& method = m.declaration();
            CHECK(method.is_method());
            ASSIGN_OR_RETURN(
                Nonnull<const Value*> converted_args,
                Convert(act.results()[1], &method.param_pattern().static_type(),
                        exp.source_loc()));
            RuntimeScope method_scope(&heap_);
            BindingMap generic_args;
            CHECK(PatternMatch(&method.me_pattern().value(), m.receiver(),
                               exp.source_loc(), &method_scope, generic_args,
                               this->arena_));
            CHECK(PatternMatch(&method.param_pattern().value(), converted_args,
                               exp.source_loc(), &method_scope, generic_args,
                               this->arena_));
            // Bring the class type arguments into scope.
            for (const auto& [bind, val] : m.type_args()) {
              method_scope.Initialize(bind, val);
            }

            // Bring the impl witness tables into scope.
            for (const auto& [impl_bind, witness] : m.witnesses()) {
              method_scope.Initialize(impl_bind, witness);
            }
            CHECK(method.body().has_value())
                << "Calling a method that's missing a body";
            return todo_.Spawn(
                std::make_unique<StatementAction>(*method.body()),
                std::move(method_scope));
          }
          case Value::Kind::NominalClassType: {
            const NominalClassType& class_type =
                cast<NominalClassType>(*act.results()[0]);
            const ClassDeclaration& class_decl = class_type.declaration();
            RuntimeScope type_params_scope(&heap_);
            BindingMap generic_args;
            if (class_decl.type_params().has_value()) {
              CHECK(PatternMatch(&(*class_decl.type_params())->value(),
                                 act.results()[1], exp.source_loc(),
                                 &type_params_scope, generic_args,
                                 this->arena_));
              switch (phase()) {
                case Phase::RunTime: {
                  std::map<Nonnull<const ImplBinding*>, const Witness*>
                      witnesses;
                  for (const auto& [impl_bind, impl_node] :
                       cast<CallExpression>(exp).impls()) {
                    ASSIGN_OR_RETURN(
                        Nonnull<const Value*> witness,
                        todo_.ValueOfNode(impl_node, exp.source_loc()));
                    if (witness->kind() == Value::Kind::LValue) {
                      const LValue& lval = cast<LValue>(*witness);
                      ASSIGN_OR_RETURN(witness, heap_.Read(lval.address(),
                                                           exp.source_loc()));
                    }
                    witnesses[impl_bind] = &cast<Witness>(*witness);
                  }
                  Nonnull<NominalClassType*> inst_class =
                      arena_->New<NominalClassType>(&class_type.declaration(),
                                                    generic_args, witnesses);
                  return todo_.FinishAction(inst_class);
                }
                case Phase::CompileTime: {
                  Nonnull<NominalClassType*> inst_class =
                      arena_->New<NominalClassType>(
                          &class_type.declaration(), generic_args,
                          cast<CallExpression>(exp).impls());
                  return todo_.FinishAction(inst_class);
                }
              }
            } else {
              FATAL() << "instantiation of non-generic class " << class_type;
            }
          }
          default:
            return FATAL_RUNTIME_ERROR(exp.source_loc())
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
            act.results()[1], std::vector<Nonnull<const ImplBinding*>>()));
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
    case ExpressionKind::IfExpression: {
      const auto& if_expr = cast<IfExpression>(exp);
      if (act.pos() == 0) {
        return todo_.Spawn(
            std::make_unique<ExpressionAction>(if_expr.condition()));
      } else if (act.pos() == 1) {
        const auto& condition = cast<BoolValue>(*act.results()[0]);
        return todo_.Spawn(std::make_unique<ExpressionAction>(
            condition.value() ? if_expr.then_expression()
                              : if_expr.else_expression()));
      } else {
        return todo_.FinishAction(act.results()[1]);
      }
      break;
    }
    case ExpressionKind::UnimplementedExpression:
      FATAL() << "Unimplemented: " << exp;
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
    case PatternKind::VarPattern:
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<PatternAction>(
            &cast<VarPattern>(pattern).pattern()));
      } else {
        return todo_.FinishAction(act.results()[0]);
      }
    case PatternKind::AddrBindingPattern:
      const auto& addr = cast<AddrBindingPattern>(pattern);
      if (act.pos() == 0) {
        return todo_.Spawn(std::make_unique<PatternAction>(&addr.binding()));
      } else {
        return todo_.FinishAction(arena_->New<AddrValue>(act.results()[0]));
      }
      break;
  }
}

auto Interpreter::StepStmt() -> ErrorOr<Success> {
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
        ASSIGN_OR_RETURN(Nonnull<const Value*> val,
                         Convert(act.results()[0], &c.pattern().static_type(),
                                 stmt.source_loc()));
        if (PatternMatch(&c.pattern().value(), val, stmt.source_loc(), &matches,
                         generic_args, this->arena_)) {
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
        ASSIGN_OR_RETURN(Nonnull<const Value*> condition,
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
        ASSIGN_OR_RETURN(
            Nonnull<const Value*> v,
            Convert(act.results()[0], &definition.pattern().static_type(),
                    stmt.source_loc()));
        Nonnull<const Value*> p =
            &cast<VariableDefinition>(stmt).pattern().value();

        RuntimeScope matches(&heap_);
        BindingMap generic_args;
        CHECK(PatternMatch(p, v, stmt.source_loc(), &matches, generic_args,
                           this->arena_))
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
        ASSIGN_OR_RETURN(Nonnull<const Value*> rval,
                         Convert(act.results()[1], &assign.lhs().static_type(),
                                 stmt.source_loc()));
        RETURN_IF_ERROR(heap_.Write(lval.address(), rval, stmt.source_loc()));
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
        ASSIGN_OR_RETURN(Nonnull<const Value*> condition,
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
        ASSIGN_OR_RETURN(
            Nonnull<const Value*> return_value,
            Convert(act.results()[0], &function.return_term().static_type(),
                    stmt.source_loc()));
        return todo_.UnwindPast(*function.body(), return_value);
      }
    case StatementKind::Continuation: {
      CHECK(act.pos() == 0);
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
      CHECK(act.pos() == 0);
      return todo_.Suspend();
  }
}

auto Interpreter::StepDeclaration() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  const Declaration& decl = cast<DeclarationAction>(act).declaration();
  if (trace_) {
    llvm::outs() << "--- step declaration (" << decl.source_loc() << ") --->\n";
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
      // These declarations have no run-time effects.
      return todo_.FinishAction();
  }
}

// State transition.
auto Interpreter::Step() -> ErrorOr<Success> {
  Action& act = todo_.CurrentAction();
  switch (act.kind()) {
    case Action::Kind::LValAction:
      RETURN_IF_ERROR(StepLvalue());
      break;
    case Action::Kind::ExpressionAction:
      RETURN_IF_ERROR(StepExp());
      break;
    case Action::Kind::PatternAction:
      RETURN_IF_ERROR(StepPattern());
      break;
    case Action::Kind::StatementAction:
      RETURN_IF_ERROR(StepStmt());
      break;
    case Action::Kind::DeclarationAction:
      RETURN_IF_ERROR(StepDeclaration());
      break;
    case Action::Kind::ScopeAction:
      FATAL() << "ScopeAction escaped ActionStack";
  }  // switch
  return Success();
}

auto Interpreter::RunAllSteps(std::unique_ptr<Action> action)
    -> ErrorOr<Success> {
  if (trace_) {
    PrintState(llvm::outs());
  }
  todo_.Start(std::move(action));
  while (!todo_.IsEmpty()) {
    RETURN_IF_ERROR(Step());
    if (trace_) {
      PrintState(llvm::outs());
    }
  }
  return Success();
}

auto InterpProgram(const AST& ast, Nonnull<Arena*> arena, bool trace)
    -> ErrorOr<int> {
  Interpreter interpreter(Phase::RunTime, arena, trace);
  if (trace) {
    llvm::outs() << "********** initializing globals **********\n";
  }

  for (Nonnull<Declaration*> declaration : ast.declarations) {
    RETURN_IF_ERROR(interpreter.RunAllSteps(
        std::make_unique<DeclarationAction>(declaration)));
  }

  if (trace) {
    llvm::outs() << "********** calling main function **********\n";
  }

  RETURN_IF_ERROR(interpreter.RunAllSteps(
      std::make_unique<ExpressionAction>(*ast.main_call)));

  return cast<IntValue>(*interpreter.result()).value();
}

auto InterpExp(Nonnull<const Expression*> e, Nonnull<Arena*> arena, bool trace)
    -> ErrorOr<Nonnull<const Value*>> {
  Interpreter interpreter(Phase::CompileTime, arena, trace);
  RETURN_IF_ERROR(
      interpreter.RunAllSteps(std::make_unique<ExpressionAction>(e)));
  return interpreter.result();
}

auto InterpPattern(Nonnull<const Pattern*> p, Nonnull<Arena*> arena, bool trace)
    -> ErrorOr<Nonnull<const Value*>> {
  Interpreter interpreter(Phase::CompileTime, arena, trace);
  RETURN_IF_ERROR(interpreter.RunAllSteps(std::make_unique<PatternAction>(p)));
  return interpreter.result();
}

}  // namespace Carbon
