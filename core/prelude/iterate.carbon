// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/iterate";

export import library "prelude/copy";
export import library "prelude/destroy";
export import library "prelude/types";
export import library "prelude/operators";

interface Iterate {
  // TODO: Support iterating ranges of non-copyable values.
  let ElementType: Copy & Destroy;
  let CursorType: Destroy;
  fn NewCursor(self) -> CursorType;
  fn Next(self, cursor: CursorType*) -> Optional(ElementType);
}

impl forall [T: Copy & Destroy, N: IntLiteral]
    array(T, N) as Iterate
    where .ElementType = T and .CursorType = i32 {
  fn NewCursor(unused self) -> i32 { return 0; }
  fn Next(self, cursor: i32*) -> Optional(T) {
    if (*cursor < N) {
      ++*cursor;
      return Optional(T).Some(self[*cursor - 1]);
    } else {
      return Optional(T).None();
    }
  }
}

// Iteration support for C++ ranges.

private interface CppRangeForIterate {
  let Iterator: type;
  let Sentinel: type;

  // TODO: self must be ref-qualified
  fn Begin(self) -> Iterator;
  fn End(self) -> Sentinel;
}

private constraint CppIterator {
  require impls Inc & Destroy;
  require impls CppUnsafeDeref where .Result impls Destroy;

  alias ValueType = CppUnsafeDeref.Result;
}

private constraint CppSentinelFor(T: CppIterator) {
  require impls Destroy;
  require T impls EqWith(Self);
}

private constraint CppRange {
  extend require impls CppRangeForIterate;
  // TODO: `Self.` should not be necessary here.
  require Self.(Iterator) impls CppIterator;
  require Self.(Sentinel) impls CppSentinelFor(Self.(Iterator));

  alias ValueType = Self.(Iterator).(CppIterator.ValueType);
}

impl forall [T: CppRange where .ValueType impls Copy]
    T as Iterate where
        .ElementType = T.ValueType and
        .CursorType = (T.Iterator, T.Sentinel) {
  fn NewCursor(self) -> CursorType {
    return (self.Begin(), self.End());
  }

  fn Next(unused self, cursor: CursorType*) -> Optional(ElementType) {
    if (cursor->0 != cursor->1) {
      returned var result: Optional(ElementType) = cursor->0.(CppUnsafeDeref.Op)();
      ++cursor->0;
      return var;
    }
    return Optional(ElementType).None();
  }
}
