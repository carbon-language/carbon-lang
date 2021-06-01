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

  /// Deserializes a new instance from `memory` at `location`.
  init(from location: Address, in memory: Memory)
}

protocol CompoundValue: Value {
  /// Creates an instance from the given parts.
  init(parts: Tuple<Value>)
}

extension CompoundValue {
  /// Deserializes a new instance from `memory` at `location`.
  init(from location: Address, in memory: Memory) {
    self.init(parts: memory.substructure(at: location).mapFields { memory[$0] })
  }
}

protocol AtomicValue: Value {}
extension AtomicValue {
  var parts: Tuple<Value> { Tuple() }

  /// Returns the value stored at the given location
  init(from location: Address, in memory: Memory) {
    self = memory.atom(at: location) as! Self
  }
}

struct FunctionValue: AtomicValue, Equatable {
  let dynamic_type: Type
  let code: FunctionDefinition
}

typealias IntValue = Int
extension IntValue: AtomicValue {
  var dynamic_type: Type { .int }
}

typealias BoolValue = Bool
extension BoolValue: AtomicValue {
  var dynamic_type: Type { .bool }
}

struct ChoiceValue: CompoundValue {
  let dynamic_type_: ASTIdentity<ChoiceDefinition>
  let discriminator: ASTIdentity<Alternative>
  let payload: Tuple<Value>

  var dynamic_type: Type { .choice(dynamic_type_) }

  init(
    type: ASTIdentity<ChoiceDefinition>,
    discriminator: ASTIdentity<Alternative>,
    payload: Tuple<Value>
  ) {
    dynamic_type_ = type
    self.discriminator = discriminator
    self.payload = payload
  }

  init(parts: Tuple<Value>) {
    guard
      case .choice(let parent) = parts[0] as! Type,
      case .alternative(let discriminator) = parts[1] as! Type
    else {
      UNREACHABLE()
    }
    self.dynamic_type_ = parent
    self.discriminator = discriminator
    self.payload = parts[2] as! Tuple<Value>
 }

  var parts: Tuple<Value> {
    Tuple(
      [.position(0): dynamic_type,
       .position(1): Type.alternative(discriminator),
       .position(2): payload])
  }
}

extension ChoiceValue: CustomStringConvertible {
  var description: String {
    "\(dynamic_type).\(discriminator.structure.name)\(payload)"
  }
}

// TODO: Alternative => AlternativeDefinition?

struct StructValue: CompoundValue {
  let dynamic_type_: ASTIdentity<StructDefinition>
  let payload: Tuple<Value>

  var dynamic_type: Type { .struct(dynamic_type_) }

  init(
    type: ASTIdentity<StructDefinition>,
    payload: Tuple<Value>
  ) {
    dynamic_type_ = type
    self.payload = payload
  }

  init(parts: Tuple<Value>) {
    guard
      case .struct(let parent) = parts[0] as! Type
    else {
      UNREACHABLE()
    }
    self.dynamic_type_ = parent
    self.payload = parts[1] as! Tuple<Value>
 }

  var parts: Tuple<Value> {
    Tuple(
      [.position(0): dynamic_type,
       .position(1): payload])
  }
}
