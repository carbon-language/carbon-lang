// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// A last-in, first-out (LIFO) queue of T.
struct Stack<T> {
  /// Creates an empty stack.
  init() { elements = [] }
  
  /// The top element.
  ///
  /// - Requires: !isEmpty
  var top: T {
    get { elements.last! }
    _modify { yield &elements[elements.count - 1] }
    set { elements[elements.count - 1] = newValue }
  }

  /// The top element, or `nil` if `self.isEmpty`
  ///
  /// - Requires: !isEmpty
  var queryTop: T? { elements.last }

  /// The bottom element.
  ///
  /// - Requires: !isEmpty
  var bottom: T {
    get { elements.first! }
    _modify { yield &elements[0] }
    set { elements[0] = newValue }
  }

  /// The number of elements.
  var count: Int { elements.count }

  /// True iff `count == 0`.
  var isEmpty: Bool { elements.isEmpty }

  /// Pushes `x` onto the top of `self`.
  mutating func push(_ x: T) { elements.append(x) }

  /// Removes and returns the top element, or returns `nil` if `isEmpty`.
  mutating func pop() -> T? { elements.popLast() }

  /// Removes the top `n` elements.
  ///
  /// - Requires: `count >= n`.
  mutating func removeTop(_ n: Int = 1) { elements.removeLast(n) }

  /// The elements, starting with the bottom element and ending with the top one.
  private(set) var elements: [T]
}
