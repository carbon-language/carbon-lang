// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/iterate";

export import library "prelude/destroy";
export import library "prelude/types";
export import library "prelude/operators";

interface Iterate {
  let ElementType:! type;
  let CursorType:! type;
  fn NewCursor[self: Self]() -> CursorType;
  fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
}

impl forall [T:! type, N:! IntLiteral()]
    array(T, N) as Iterate
    where .ElementType = T and .CursorType = i32 {
  fn NewCursor[self: Self]() -> i32 { return 0; }
  fn Next[self: Self](cursor: i32*) -> Optional(T) {
    if (*cursor < N) {
      ++*cursor;
      return Optional(T).Some(self[*cursor - 1]);
    } else {
      return Optional(T).None();
    }
  }
}
