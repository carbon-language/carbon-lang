// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

protocol Value {
  /// The type of this value.
  // This is available for diagnostic purposes; semantics mustn't depend on it.
  var type: Type { get }

  /// The parts of this value that can be individually bound to variables.
  var parts: Tuple<Value> { get }
}

struct FunctionValue: Value, Equatable {
  let type: Type
  let code: FunctionDefinition
  var parts: Tuple<Value> { Tuple() }
}

typealias IntValue = Int
extension IntValue: Value {
  var type: Type { .int }
  var parts: Tuple<Value> { Tuple() }
}

typealias BoolValue = Bool
extension BoolValue: Value {
  var type: Type { .bool }
  var parts: Tuple<Value> { Tuple() }
}

struct ChoiceValue: Value {
  let type_: ASTIdentity<ChoiceDefinition>
  let discriminator: ASTIdentity<Alternative>
  let payload: Tuple<Value>

  var alternativeType: Type { .alternative(discriminator, parent: type_) }

  var type: Type { .choice(type_) }
  var parts: Tuple<Value> {
    Tuple(
      [.position(0): type,
       .position(1): Type.alternative(discriminator, parent: type_),
       .position(2): payload])
  }
}

extension ChoiceValue: CustomStringConvertible {
  var description: String {
    "\(type).\(discriminator.structure.name)\(payload)"
  }
}

// TODO: Alternative => AlternativeDefinition?
