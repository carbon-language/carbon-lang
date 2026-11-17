// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/default.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/destroy.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/int.carbon

// --- min_prelude/parts/array.carbon

package Core library "prelude/parts/array";

export import library "prelude/parts/default";
export import library "prelude/parts/destroy";
export import library "prelude/parts/int";

// Arrays.

private fn MakeArray(element_type: type, bound: IntLiteral) -> type =
    "array.make_type";

class Array(T: type, N: IntLiteral) {
  adapt MakeArray(T, N);
}

match_first {
  impl forall [T: TrivialDestroy, N: IntLiteral] Array(T, N) as DestroyImpl {
    fn Op(ref self) = "no_op";
  }

  impl forall [T: Destroy] Array(T, 0) as DestroyImpl {
    fn Op(ref self) = "no_op";
  }

  impl forall [T: Destroy, N: IntLiteral] Array(T, N) as DestroyImpl {
    fn Op(ref self) {
      var i: i32 = N;
      while (i >= 0) {
        i -= 1;
        SelfDestruct(ref self[i]);
      }
    }
  }
}
