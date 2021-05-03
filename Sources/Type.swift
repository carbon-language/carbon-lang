// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A type value in the interpreter and typesystem.
indirect enum Type: Equatable {
  case
    int, bool, type,
    function(parameterTypes: [Type], returnType: Type),
    tuple(TupleType),
    `struct`(StructDefinition),
    `choice`(ChoiceDefinition),

    error // Placeholder indicating failed type deduction.

  /// Convenience accessor for `.function` case.
  var function: (parameterTypes: [Type], returnType: Type)? {
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
