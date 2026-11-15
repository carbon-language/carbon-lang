// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package Core library "prelude/types/array";

import library "prelude/default";
import library "prelude/destroy";
import library "prelude/operators/arithmetic";
import library "prelude/operators/as";
import library "prelude/operators/comparison";
import library "prelude/types/int_literal";
import library "prelude/types/int";

private fn MakeArray(T: type, N: IntLiteral) -> type = "array.make_type";

// The type `array(T, N)`.
class Array(T: type, N: IntLiteral) {
  adapt MakeArray(T, N);
}

impl forall [T: UnformedInit, N: IntLiteral] Array(T, N) as UnformedInit {}

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
