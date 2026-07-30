// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/slice";

import library "prelude/copy";
import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators/as";
import library "prelude/operators/index";
import library "prelude/types/uint";

private fn PointerOffset[T: type](p: T*, offset: u64) -> T* = "pointer.offset";

class Slice(T: type) {
  // TODO: Make this unsafe.
  fn Make(ptr: T*, size: u64) -> Self {
    return {.ptr = ptr, .size = size};
  }

  fn Size(self) -> u64 { return self.size; }
  fn Ptr(self) -> T* { return self.ptr; }

  fn Range(self, start: u64, count: u64) -> Self {
    return Make(PointerOffset(self.ptr, start), count);
  }

  impl as Copy {
    fn Op(self) -> Self { return {.ptr = self.ptr, .size = self.size}; }
  }

  // TODO: This should be an array iterator.
  private var ptr: T*;
  // TODO: This should be a word-sized integer.
  private var size: u64;
}

impl forall [T: type, Subscript: ImplicitAs(u64)]
    Slice(T) as IndirectIndexWith(Subscript)
    where .ElementType = T {
  fn Addr(self, subscript: Subscript) -> T* {
    return PointerOffset(self.Ptr(), subscript as u64);
  }
}

impl forall [T: type] Slice(T) as UnformedInit {}
