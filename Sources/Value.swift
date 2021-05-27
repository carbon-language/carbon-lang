// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

protocol Value {
  /// The actual type of this value.
  ///
  /// The interpreter should only depend on this where dynamic dispatch is
  /// intended.  Otherwise, use `ExecutableProgram.staticType[e]` to look up the
  /// type from the expression creating the value.
  ///
  /// This name uses `snake_case` because `.dynamicType` is a deprecated
  /// built-in property name.
  var dynamic_type: Type { get }

  /// The parts of this value that can be individually bound to variables.
  var parts: Tuple<Value> { get }
}

struct FunctionValue: Value, Equatable {
  let dynamic_type: Type
  let code: FunctionDefinition
  var parts: Tuple<Value> { Tuple() }
}

typealias IntValue = Int
extension IntValue: Value {
  var dynamic_type: Type { .int }
  var parts: Tuple<Value> { Tuple() }
}

typealias BoolValue = Bool
extension BoolValue: Value {
  var dynamic_type: Type { .bool }
  var parts: Tuple<Value> { Tuple() }
}

struct ChoiceValue: Value {
  let dynamic_type_: ASTIdentity<ChoiceDefinition>
  let discriminator: ASTIdentity<Alternative>
  let payload: Tuple<Value>

  var dynamic_type: Type { .choice(dynamic_type_) }

  var parts: Tuple<Value> {
    Tuple(
      [.position(0): dynamic_type,
       .position(1): Type.alternative(discriminator, parent: dynamic_type_),
       .position(2): payload])
  }
}

extension ChoiceValue: CustomStringConvertible {
  var description: String {
    "\(dynamic_type).\(discriminator.structure.name)\(payload)"
  }
}

// TODO: Alternative => AlternativeDefinition?
