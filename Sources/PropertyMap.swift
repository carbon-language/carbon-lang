// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A mapping from AST<Body> nodes, by their identity, to values of type `T`.
struct PropertyMap<K: Hashable, Value> {
  init() { }
  
  /// Accesses the value associated with `k`.
  subscript(k: K) -> Value {
    get { storage[k] ?? fatal("\(k): No such key.") }
    set { storage[k] = newValue }
  }

  private var storage: [K: Value] = [:]
}
