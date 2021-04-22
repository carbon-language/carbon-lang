// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A mapping from Key to Value that contains a stack of backup `Values`
struct StackDictionary<Key: Hashable, Value> {
  /// Creates an empty instance.
  init() { }

  /// Adds `newValue` to the top of the stack at `k`, creating the stack if it
  /// doesn't exist.
  mutating func push(_ newValue: Value, at k: Key) {
    storage[k, default: Stack()].push(newValue)
  }

  /// Removes and returns the top element of the stack at `k`, or `nil` if
  /// either no such stack, or no such element, exists.
  @discardableResult
  mutating func pop(at k: Key) -> Value? {
    storage[k]?.pop()
  }

  /// Accesses the top of the stack at `k`.
  subscript(k: Key) -> Value {
    get { storage[k]!.top }
    _modify { yield &storage[k, default: Stack()].top }
    set { storage[k, default: Stack()].top = newValue }
  }

  /// Accesses the top element of the stack at `k`, or `nil` if either no such
  /// stack, or no such element, exists.
  subscript(query k: Key) -> Value? {
    storage[k]?.queryTop
  }

  /// Removes the top `n` elements of the stack at `k`.
  mutating func removeTop(_ n: Int = 1, at k: Key) {
    storage[k]!.removeTop(n)
  }

  /// Underlying dictionary storage
  private var storage: [Key: Stack<Value>] = [:]
}
