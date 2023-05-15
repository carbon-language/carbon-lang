// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: (C(i32) as A).FA()
// CHECK:STDOUT: 6
// CHECK:STDOUT: (C(i32) as A).FA()
// CHECK:STDOUT: 7
// CHECK:STDOUT: (C(T) as B).FB()
// CHECK:STDOUT: (C(T) as A).FA()
// CHECK:STDOUT: 3
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface A {
  let TA:! type;
  fn FA() -> TA;
}
interface B {
  let TB:! type;
  fn FB() -> TB;
}

class C(T:! type) {
  impl as A & B where .TA = i32 and .TB = i32 {
    fn FA() -> i32 {
      Print("(C(T) as A).FA()");
      // OK, know that TA is i32 here.
      let v: Self.(A.TA) = 1;
      let w: i32 = v;
      return w;
    }
    fn FB() -> i32 {
      Print("(C(T) as B).FB()");
      // OK, know that TB is i32 here.
      let v: Self.(B.TB) = 2;
      // Don't know that TA is i32; it could be specialized.
      // TODO: We should not accept this.
      let w: Self.(A.TA) = Self.(A.FA)();
      return v + w;
    }
  }
}

external impl C(i32) as A where .TA = (i32, i32) {
  fn FA() -> (i32, i32) {
    Print("(C(i32) as A).FA()");
    return (6, 7);
  }
}

fn Main() -> i32 {
  Print("{0}", C(i32).(A.FA)()[0]);
  Print("{0}", C(i32).(A.FA)()[1]);
  // TODO: The implementation of C(T) as B eagerly picked the (non-final)
  // implementation of C(T) as A, so this ends up using a different
  // implementation of C(i32) as A than the one we just used, violating
  // coherence.
  Print("{0}", C(i32).(B.FB)());
  return 0;
}
