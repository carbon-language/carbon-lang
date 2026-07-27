// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/as.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/copy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/default.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/destroy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/uint.carbon

// --- min_prelude/parts/slice.carbon

package Core library "prelude/parts/slice";

import library "prelude/parts/as";
import library "prelude/parts/copy";
import library "prelude/parts/default";
import library "prelude/parts/destroy";
import library "prelude/parts/uint";

private fn PointerOffset[T: type](p: T*, offset: u64) -> T* = "pointer.offset";

class Slice(T: type) {
  fn Make(ptr: T*, size: u64) -> Self {
    return {.ptr = ptr, .size = size};
  }

  fn Size(self) -> u64 { return self.size; }
  fn Ptr(self) -> T* { return self.ptr; }

  fn Range(self, start: u64, count: u64) -> Self {
    return Make(PointerOffset(self.ptr, start), count);
  }

  impl as Copy {
    fn Op(self) -> Self {
      return {.ptr = self.ptr, .size = self.size};
    }
  }

  private var ptr: T*;
  private var size: u64;
}

impl forall [T: type, Subscript: ImplicitAs(u64)]
    Slice(T) as IndirectIndexWith(Subscript)
    where .ElementType = T {
  fn Addr[me: Self](subscript: Subscript) -> T* {
    return PointerOffset(me.Ptr(), subscript as u64);
  }
}

impl forall [T: type] Slice(T) as UnformedInit {}
