// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A type value in the interpreter and typesystem.
indirect enum Type: Equatable {
  case
    int, bool, type,
    function(FunctionType),
    tuple(TupleType),
    alternative(ASTIdentity<Alternative>),
    `struct`(ASTIdentity<StructDefinition>),
    choice(ASTIdentity<ChoiceDefinition>),

    error // Placeholder indicating failed type deduction.

  enum Kind: Int {
    case int, bool, type, function, tuple, alternative, `struct`, choice, error
  }

  /// Describes the category of type `self` is.
  var kind: Kind {
    switch self {
    case .int: return .int
    case .bool: return .bool
    case .type: return .type
    case .function: return .function
    case .tuple: return .tuple
    case .alternative: return .alternative
    case .struct: return .struct
    case .choice: return .choice
    case .error: return .error
    }
  }

  /// Creates an instance corresponding to `d` if it declares a type, or `nil`
  /// otherwise.
  init?(_ d: Declaration) {
    if let s = d as? StructDefinition { self = .struct(s.identity) }
    else if let c = d as? ChoiceDefinition { self = .choice(c.identity) }
    else { return nil }
  }

  /// Creates an instance corresponding to `v` if it is a type value, or `nil`
  /// otherwise.
  init?(_ v: Value) {
    if let r = (v as? Type) {
      self = r
      return
    }

    // If the value is a tuple, check that all its elements are types.
    if let elements = (v as? TupleValue) {
      let typeElements = elements.compactMapFields { Type($0) }
      if typeElements.count == elements.count {
        self = .tuple(typeElements)
        return
      }
    }

    return nil
  }

  /// `true` iff instances of this type are themselves types.
  var isMetatype: Bool {
    switch self {
    case .type:
      return true
    case .tuple(let t):
      return t.fields.allSatisfy(\.isMetatype)
    default:
      return false
    }
  }

  /// Convenience accessor for `.function` case.
  var function: FunctionType? {
    get {
      if case let .function(f) = self { return f } else { return nil }
    }
    set {
      guard let f = newValue else { return }
      self = .function(f)
    }
  }

  /// Convenience accessor for `.tuple` case.
  var tuple: TupleType? {
    get {
      if case .tuple(let r) = self { return r } else { return nil }
    }
    set {
      guard let n = newValue else { return }
      self = .tuple(n)
    }
  }

  /// The Swift type used to represent values of this type
  var swiftType: Value.Type {
    switch self {
    case .int: return Int.self
    case .bool: return Bool.self
    case .type: return Type.self
    case .function: return FunctionValue.self
    case .tuple: return Tuple<Value>.self
    case .alternative: return Type.self
    case .struct: return StructValue.self
    case .choice: return ChoiceValue.self
    case .error: return Type.self
    }
  }

  static var void: Type { .tuple(.void) }
}

extension Type: CompoundValue {
  var dynamic_type: Type { .type }

  /// Accesses the named subparts of this value, or `nil` if no such subpart
  /// exists.
  ///
  /// Writing `nil` into an existing subpart is a precondition violation.
  subscript(field: FieldID) -> Value? {
    get {
      switch self {
      case let .tuple(t): return t[field]
      case .int, .bool, .type, .alternative, .struct, .error, .choice, .function:
        return nil
      }
    }
    set {
      guard let v = newValue else {
        sanityCheck(self[field] == nil)
        return
      }

      switch self {
      case .tuple(var t):
        t[field] = Type(v)!
        self = .tuple(t)
        return
      case .int, .bool, .type, .alternative, .struct, .error, .choice, .function:
        fatal("Value \(self) has no field \(field)")
      }
    }
  }
}

/// Representation for the `Type.function` case.
struct FunctionType: Equatable {
  var parameterTypes: TupleType
  var returnType: Type
}

extension Type: CustomStringConvertible {
  var description: String {
    switch self {
    case .int: return "Int"
    case .bool: return "Bool"
    case .type: return "Type"
    case let .function(f):
      return "fnty \(f.parameterTypes) -> \(f.returnType)"
    case let .tuple(t): return "\(t)"
    case let .alternative(id):
      return "<Choice>.\(id.structure.name.text)"
    case let .struct(d): return d.structure.name.text
    case let .choice(d): return d.structure.name.text
    case .error:
      return "<<error type>>"
    }
  }
}

// TODO: make most subscripts require validity to reduce unwrapping? discuss.
