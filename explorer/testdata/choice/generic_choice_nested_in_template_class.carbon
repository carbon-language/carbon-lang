// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: H 22
// CHECK:STDOUT: result: 0

package ExplorerTest api;


choice MyOptionalElement(T:! type) {
  None(),
  Element(T)
}

class MyOptional(T:! type){
   fn CreateEmpty() -> MyOptional(T){
       return { .element = MyOptionalElement(T).None() };
   }
   fn Create ( value: T ) -> MyOptional(T){
       return { .element = MyOptionalElement(T).Element(value) };
   }

    fn has_value[self: Self] () -> bool{
        var x: MyOptionalElement(T) = self.element;
        match(x){
            case MyOptionalElement(T).None() => { return false; }
        }
        return false;
    }

    fn get[self: Self] () -> T {
        var y: T;
        var x: MyOptionalElement(T) = self.element;
        match(x){
            case MyOptionalElement(T).Element( var x: T ) =>{
                return x;
            }
        }
        return y;
    }

   var element: MyOptionalElement(T);
}


fn Main() -> i32 {
  var f: MyOptional(i32) = MyOptional(i32).Create(22);
  var x: i32 = f.get();
  Print("H {0}",x);

  return 0;
}
