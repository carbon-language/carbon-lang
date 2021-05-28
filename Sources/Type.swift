// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A type value in the interpreter and typesystem.
indirect enum Type: Equatable {
  case
    int, bool, type,
    function(parameterTypes: TupleType, returnType: Type),
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
  var function: (parameterTypes: TupleType, returnType: Type)? {
    if case .function(parameterTypes: let p, returnType: let r) = self {
      return (p, r)
    } else { return nil }
  }

  /// Convenience accessor for `.tuple` case.
  var tuple: TupleType? {
    if case .tuple(let r) = self { return r } else { return nil }
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
    case .struct: UNIMPLEMENTED()
    case .choice: return ChoiceValue.self
    case .error: return Type.self
    }
  }

  static var void: Type { .tuple(.void) }
}

extension Type: CompoundValue {
  var dynamic_type: Type { .type }

  var parts: Tuple<Value> {
    switch self {
    case .int, .bool, .type, .error:
      return .init([.position(0): kind.rawValue])

    case let .alternative(discriminator):
      return Tuple<Value>(
        [.position(0): kind.rawValue, .position(1): discriminator.structure])

    case let .struct(id):
      return .init([.position(0): kind.rawValue, .position(1): id.structure])

    case let .choice(id):
      return .init([.position(0): kind.rawValue, .position(1): id.structure])

    case let .function(parameterTypes: p, returnType: r):
      return Tuple<Value>(
        [.position(0): kind.rawValue,
         .position(1): p.mapFields { $0 }, .position(2): r])

    case let .tuple(t):
      return Tuple<Value>(
        [.position(0): kind.rawValue, .position(1): t.mapFields { $0 }])
    }
  }

  init(parts: Tuple<Value>) {
    switch Kind(rawValue: parts[0] as! Int)! {
    case .int: self = .int
    case .bool: self = .bool
    case .type: self = .type
    case .alternative:
      self = .alternative((parts[1] as! Alternative).identity)
    case .error: self = .error
    case .struct:
      self = .struct((parts[1] as! StructDefinition).identity)
    case .choice:
      self = .choice((parts[1] as! ChoiceDefinition).identity)
    case .function:
      self = .function(
        parameterTypes: (parts[1] as! TupleValue).mapFields { Type($0)! },
        returnType: Type(parts[2]!)!)

    case .tuple:
      self = .tuple((parts[1] as! TupleValue).mapFields { Type($0)! })
    }
  }
}

extension Type: CustomStringConvertible {
  var description: String {
    switch self {
    case .int: return "Int"
    case .bool: return "Bool"
    case .type: return "Type"
    case let .function(parameterTypes: p, returnType: r):
      return "fnty \(p) -> \(r)"
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

// TODO: make most subscripts require validity to reduce unwrapping.
