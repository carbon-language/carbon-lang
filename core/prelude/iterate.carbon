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
  let CursorType:! type;
  fn NewCursor[self: Self]() -> CursorType;
  fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
}

impl forall [T:! Copy & Destroy, N:! IntLiteral()]
    array(T, N) as Iterate
    where .ElementType = T and .CursorType = i32 {
  fn NewCursor[unused self: Self]() -> i32 { return 0; }
  fn Next[self: Self](cursor: i32*) -> Optional(T) {
    if (*cursor < N) {
      ++*cursor;
      return Optional(T).Some(self[*cursor - 1]);
    } else {
      return Optional(T).None();
    }
  }
}

interface CppRangeForIterate {
  let Iterator:! Copy & Destroy & CppUnsafeDeref;
  let Sentinel:! Copy & Destroy;
  let ElementType:! Copy & Destroy;
  fn Begin[self: Self]() -> Iterator;
  fn End[self: Self]() -> Sentinel;
  fn Next(iterator: Iterator*) -> Optional(ElementType);
  fn NotEqual(iterator: Iterator, sentinel: Sentinel) -> bool;
}

impl forall [R:! CppRangeForIterate]
  R as Iterate
  where .ElementType = R.ElementType and .CursorType = (R.Iterator, R.Sentinel)
{
  fn NewCursor[self: Self]() -> (R.Iterator, R.Sentinel) {
    return (self.Begin(), self.End());
  }

  fn Next[unused self: Self](cursor: (R.Iterator, R.Sentinel)*) -> Optional(R.ElementType) {
    if (R.NotEqual(cursor->0, cursor->1)) {
      returned var result: Optional(R.ElementType) = Optional(R.ElementType).Some(&cursor->0.(Core.CppUnsafeDeref.Op)());
      R.Next(&cursor->0);
      return var;
    } else {
      return Optional(R.ElementType).None();
    }
  }
}
