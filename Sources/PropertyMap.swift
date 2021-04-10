// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A mapping from AST<Body> nodes, by their identity, to values of type `T`.
struct PropertyMap<Body: Hashable, Value> {
  /// Accesses the value associated with `k`.
  subscript(k: AST<Body>.Identity) -> Value {
    get { storage[k] ?? fatal("\(k.value): No such key.") }
    set { storage[k] = newValue }
  }

  /// Accesses the value associated with `n.identity`
  subscript(n: AST<Body>) -> Value {
    get {
      self[n.identity ?? fatal("mapping synthesized AST node.")]
    }
    set {
      self[n.identity ?? fatal("mapping synthesized AST node.")] = newValue
    }
  }

  private var storage: [AST<Body>.Identity: Value]
}
