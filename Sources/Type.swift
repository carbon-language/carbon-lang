// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A type value in the interpreter and typesystem.
indirect enum Type: Hashable {
  case
    int, bool, type,
    function(parameterTypes: [Type], returnType: Type),
    tuple([Type]),
    `struct`(StructDefinition, SourceRegion),
    `choice`(ChoiceDefinition, SourceRegion)

  static var void: Type { .tuple([]) }
}
