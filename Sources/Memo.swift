// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// The state of memoization of a computation, including an "in progress"
/// state that allows us to detect dependency cycles.
enum Memo<T: Equatable>: Equatable {
  case beingComputed, final(T)

  /// The payload of the `.final` case, if any, or `nil` if `self ==
  /// .beingComputed`.
  var final: T? {
    if case let .final(x) = self { return x } else { return nil }
  }
}
