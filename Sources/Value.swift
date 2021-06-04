// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct TypeID<T>: Hashable {
  func hash(into h: inout Hasher) { ObjectIdentifier(T.self).hash(into: &h) }
}

protocol FieldAccess {
  associatedtype Field
  subscript(_: FieldID) -> Field? { get set }
}

extension FieldAccess {
  subscript(n: Int) -> Field? {
    get { self[.position(n)] }
    set { self[.position(n)] = newValue }
  }

  subscript(fieldName: Identifier) -> Field? {
    get { self[.label(fieldName)] }
    set { self[.label(fieldName)] = newValue }
  }
}

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
  subscript(_: FieldID) -> Value? { get set }
}

extension Value {
  subscript<T: Value>(downcastTo _: TypeID<T>) -> T? {
    get { self as? T }
    set {
      self = newValue as! Self
    }
  }

  var upcastToValue: Value {
    get { self }
    set { self = newValue as! Self }
  }
}

protocol AtomicValue: Value, FieldAccess {}

extension AtomicValue {
  subscript(field: FieldID) -> Value? {
    get { nil }
    set {
      if newValue != nil {
        fatal("Value \(self) of atomic type"
                + " \(self.dynamic_type) has no field \(field)")
      }
    }
  }
}

protocol CompoundValue: Value, FieldAccess {
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
  var payload: Tuple<Value>

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

  subscript(field: FieldID) -> Value? {
    get { payload[field] }
    set { payload[field] = newValue! }
  }
}

extension ChoiceValue: CustomStringConvertible {
  var description: String {
    "\(dynamic_type).\(discriminator.structure.name)\(payload)"
  }
}

struct StructValue: CompoundValue {
  let dynamic_type_: ASTIdentity<StructDefinition>
  var payload: Tuple<Value>

  var dynamic_type: Type { .struct(dynamic_type_) }

  init(
    type: ASTIdentity<StructDefinition>,
    payload: Tuple<Value>
  ) {
    dynamic_type_ = type
    self.payload = payload
  }

  subscript(field: FieldID) -> Value? {
    get { payload[field] }
    set { payload[field] = newValue! }
  }
}

struct AlternativeValue: AtomicValue {
  init(_ t: ASTIdentity<Alternative>) { dynamic_type_ = t }

  let dynamic_type_: ASTIdentity<Alternative>
  var dynamic_type: Type { .alternative(dynamic_type_) }
}

struct Uninitialized: AtomicValue {
  let dynamic_type: Type
}

// TODO: Alternative => AlternativeDefinition?
