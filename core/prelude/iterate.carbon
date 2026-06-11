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
  let ElementType:! Copy & Destroy;
  let CursorType:! Destroy;
  fn NewCursor(self) -> CursorType;
  fn Next(self, cursor: CursorType*) -> Optional(ElementType);
}

impl forall [T:! Copy & Destroy, N:! IntLiteral]
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

interface CppRangeForIterate {
  let Iterator:! type;
  let Sentinel:! type;

  // TODO: self must be ref-qualified
  fn Begin(self) -> Iterator;
  fn End(self) -> Sentinel;
}

impl forall [
  R:! CppRangeForIterate where
    .Iterator impls Copy & Destroy & CppUnsafeDeref & Inc & EqWith(.Sentinel) and
    .Iterator.(CppUnsafeDeref.Result) impls Copy & Destroy and
    .Sentinel impls Copy & Destroy
]
  R as Iterate
  where
    .ElementType = R.Iterator.(CppUnsafeDeref.Result) and
    .CursorType = (R.Iterator, R.Sentinel)
{
  alias ElementType = R.Iterator.(CppUnsafeDeref.Result);

  fn NewCursor(self) -> (R.Iterator, R.Sentinel) {
    return (self.Begin(), self.End());
  }

  fn Next(unused self, cursor: (R.Iterator, R.Sentinel)*) -> Optional(ElementType) {
    if (cursor->0 != cursor->1) {
      returned var result: Optional(ElementType) =
          Optional(ElementType).Some(cursor->0.(CppUnsafeDeref.Op)());
      ++cursor->0;
      return var;
    } else {
      return Optional(ElementType).None();
    }
  }
}
