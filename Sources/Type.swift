// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A type value in the interpreter and typesystem.
indirect enum Type: Equatable {
  case
    int, bool, type,
    function(parameterTypes: TupleType, returnType: Type),
    tuple(TupleType),
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
      let typeElements = elements.compactMapValues { $0 as? Type }
      if typeElements.count == elements.count {
        self = .tuple(typeElements)
        return
      }
    }

    return nil
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

  static var void: Type { .tuple([:]) }
}

extension Type: Value {
  var type: Type { .type }
}
