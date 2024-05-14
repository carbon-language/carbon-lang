// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package ExplorerTest api;

interface I { fn F(); }

impl () as I {
  fn F() {}
}

impl forall [template T:! type] (T,) as I {
  fn F() {
    if (true) {
      T.(I.F)();
    } else {
      T.(I.F)();
    }
  }
}

fn Main() -> i32 {
  // This would cause an exponential number of template instantiations if we
  // didn't cache them.
  ((((((((((((((((((((((((((((((((),),),),),),),),),),),),),),),),),),),),),),),),),),),),),),)).(I.F)();
	return 0;
}

// CHECK:STDOUT: result: 0
