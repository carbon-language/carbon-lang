// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

struct TypeID<T>: Hashable {
  func hash(into h: inout Hasher) { ObjectIdentifier(T.self).hash(into: &h) }
}

@dynamicMemberLookup
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

  subscript(_: FieldID) -> Value { get set }
  var fieldIDs: [FieldID] { get }
}

extension Value {
  subscript(dynamicMember fieldName: String) -> Value {
    get {
      self[Identifier(text: fieldName, site: .empty)]
    }
    set {
      self[Identifier(text: fieldName, site: .empty)] = newValue
    }
  }

  subscript(n: Int) -> Value {
    get { self[.position(n)] }
    set { self[.position(n)] = newValue }
  }

  subscript(fieldName: Identifier) -> Value {
    get { self[.label(fieldName)] }
    set { self[.label(fieldName)] = newValue }
  }

  subscript(field: FieldID) -> Value {
    get {
       fatal("Value \(self) has no field \(field)")
    }
    set {
      fatal("Value \(self) has no field \(field)") ?? ()
    }
  }

  subscript<T: Value>(downcastTo _: TypeID<T>) -> T {
    get { self as! T }
    set {
      self = newValue as! Self
    }
  }

  var upcastToValue: Value {
    get { self }
    set {
      self = newValue as! Self
    }
  }

  var fieldIDs: [FieldID] { [] }
}

struct FunctionValue: Value, Equatable {
  let dynamic_type: Type
  let code: FunctionDefinition
}

typealias IntValue = Int
extension IntValue: Value {
  var dynamic_type: Type { .int }
}

typealias BoolValue = Bool
extension BoolValue: Value {
  var dynamic_type: Type { .bool }
}

struct ChoiceValue: Value {
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
}

extension ChoiceValue: CustomStringConvertible {
  var description: String {
    "\(dynamic_type).\(discriminator.structure.name)\(payload)"
  }
}

struct StructValue: Value {
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
}

struct AlternativeValue: Value {
  init(_ t: ASTIdentity<Alternative>) { dynamic_type_ = t }

  let dynamic_type_: ASTIdentity<Alternative>
  var dynamic_type: Type { .alternative(dynamic_type_) }
}

struct Uninitialized: Value {
  let dynamic_type: Type
}

// TODO: Alternative => AlternativeDefinition?
