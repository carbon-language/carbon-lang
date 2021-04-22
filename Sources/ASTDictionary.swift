// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A `Dictionary`-like type that maps from AST node identity to `Value`
struct ASTDictionary<KeyNode: AST, Value> {
  /// Creates an empty instance.
  init() { }

  /// Accesses the value associated with `n.identity` if any, inserting or
  /// removing elements according to the semantics of `Dictionary.subscript`.
  subscript(n: KeyNode) -> Value? {
    get { storage[n.identity] }
    _modify { yield &storage[n.identity] }
    set { storage[n.identity] = newValue }
  }

  /// Underlying dictionary storage
  private var storage: [KeyNode.Identity: Value] = [:]
}
