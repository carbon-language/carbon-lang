// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

protocol Value {
  /// The type of this value.
  // This is available for diagnostic purposes; semantics mustn't depend on it.
  var type: Type { get }
}

struct FunctionValue: Value {
  let type: Type
  let code: FunctionDefinition
}

struct IntValue : Value {
  let type: Type
  let value: Int
}
