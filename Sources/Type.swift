// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A type value in the interpreter and typesystem.
indirect enum Type: Equatable {
  case
    int, bool, type,
    function(parameterTypes: TupleType, returnType: Type),
    tuple(TupleType),
    alternative(
      ASTIdentity<Alternative>, parent: ASTIdentity<ChoiceDefinition>),
    `struct`(ASTIdentity<StructDefinition>),
    `choice`(ASTIdentity<ChoiceDefinition>),

    error // Placeholder indicating failed type deduction.

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

  static var void: Type { .tuple(.void) }
}

extension Type: Value {
  var type: Type { .type }

  var parts: Tuple<Value> {
    switch self {
    case .int, .bool, .type, .struct, .choice, .alternative, .error:
      return .init()
    case let .function(parameterTypes: p, returnType: r):
      return Tuple<Value>([.position(0): p.mapFields { $0 }, .position(1): r])
    case let .tuple(t):
      return t.mapFields { $0 }
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
    case let .alternative(id, parent: parent):
      return "\(parent.structure.name.text).\(id.structure.name.text)"
    case let .struct(d): return d.structure.name.text
    case let .choice(d): return d.structure.name.text
    case .error:
      return "<<error type>>"
    }
  }
}
