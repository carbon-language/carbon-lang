// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A type value in the interpreter and typesystem.
indirect enum Type: Equatable {
  case
    int, bool, type,
    function(parameterTypes: [Type], returnType: Type),
    tuple([Type]),
    `struct`(StructDefinition, SourceRegion),
    `choice`(ChoiceDefinition, SourceRegion),

    error // Placeholder indicating failed type deduction.

  static var void: Type { .tuple([]) }
}

/// A thing that has a type.
enum Typed: Hashable {
  case
    declaration(Declaration),
    expression(Expression),
    functionParameter(Identifier)
}
