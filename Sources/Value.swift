// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A singular-valued type that identifies `T`.
///
/// You can think of this as a `Hashable`, dynamically-non-polymorphic version
/// of `T.Type`; which is both non-`Hashable`, and can be used to represent the
/// dynamic type of some supertype instance.
struct TypeID<T>: Hashable {
  func hash(into h: inout Hasher) { ObjectIdentifier(T.self).hash(into: &h) }
}

/// A thing that can be (mutably) subscripted with `FieldID`s (and for
/// convenience, `Int` and `Identifier`).
///
/// - Note: This protocol does not represent a concept; it is simply mechanism.
protocol FieldAccess {
  /// The type of thing accessed by `FieldID`.
  associatedtype Field

  /// Accesses the given field.
  subscript(_: FieldID) -> Field? { get set }
}

extension FieldAccess {
  /// Accesses the field `.position(n)`.
  subscript(n: Int) -> Field? {
    get { self[.position(n)] }
    set { self[.position(n)] = newValue }
  }

  /// Accesses the field `.label(fieldName)`.
  subscript(fieldName: Identifier) -> Field? {
    get { self[.label(fieldName)] }
    set { self[.label(fieldName)] = newValue }
  }
}

/// A Carbon value.
///
/// `Value` is the supertype of all Swift types that represent Carbon instances.
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

  /// Accesses the given field.
  subscript(_: FieldID) -> Value? { get set }
}

/// Type conversions that can be used via keypaths for `Address` formation.
extension Value {
  /// Accesses `self` if it is an instance of `T`, and `nil` otherwise.
  ///
  /// Preconditions: Written values must either be a non-`nil` of type `Self?`,
  /// or `U?.nil`, where `U.self != Self`.
  subscript<T: Value>(downcastTo _: TypeID<T>) -> T? {
    get { self as? T }
    set {
      guard let v = newValue else {
        sanityCheck(self as? T == nil)
        return
      }
      self = v as! Self
    }
  }

  /// `self` as an instance of its supertype `Value`.
  ///
  /// Because `self` is already of type `Value`, this property is only useful in
  /// keypath formation.
  var upcastToValue: Value {
    get { self }
    set { self = newValue as! Self }
  }
}

/// Swift representation of Carbon types having no fields.
protocol AtomicValue: Value, FieldAccess {}

extension AtomicValue {
  /// Accesses the given field.
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

/// Swift representation of Carbon types having fields.
protocol CompoundValue: Value, FieldAccess {}

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

/// A (singular) value of alternative type.
struct AlternativeValue: AtomicValue {
  init(_ t: ASTIdentity<Alternative>) { dynamic_type_ = t }

  let dynamic_type_: ASTIdentity<Alternative>
  var dynamic_type: Type { .alternative(dynamic_type_) }
}

// TODO: Alternative => AlternativeDefinition?
