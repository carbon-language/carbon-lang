// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/iterate";

export import library "prelude/types/optional";

interface Iterate {
  let ElementType:! type;
  let CursorType:! type;
  fn NewCursor[self: Self]() -> CursorType;
  fn Next[self: Self](cursor: CursorType*) -> Optional(ElementType);
}
