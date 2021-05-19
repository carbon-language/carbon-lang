// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

protocol Value {
  /// The type of this value.
  // This is available for diagnostic purposes; semantics mustn't depend on it.
  var type: Type { get }

  /// The parts of this value that can be individually bound to variables.
  var parts: Tuple<Value> { get }
}

extension Value {
  /// The parts of this value that can be individually bound to variables.
  ///
  /// By default, values are indivisible.
  var parts: Tuple<Value> { Tuple() }
}

struct FunctionValue: Value, Equatable {
  let type: Type
  let code: FunctionDefinition
}

typealias IntValue = Int
extension IntValue: Value {
  var type: Type { .int }
}

typealias BoolValue = Bool
extension BoolValue: Value {
  var type: Type { .bool }
}
