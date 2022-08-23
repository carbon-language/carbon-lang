// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: H 22
// CHECK: result: 0

package ExplorerTest api;


choice MyOptionalElement(T:! Type) {
  None(),
  Element(T)
}

class MyOptional(T:! Type){
   fn CreateEmpty() -> MyOptional(T){
       return { .element = MyOptionalElement(T).None() };
   }
   fn Create ( value: T ) -> MyOptional(T){
       return { .element = MyOptionalElement(T).Element(value) };
   }

    fn has_value[me: Self] () -> bool{
        var x: MyOptionalElement(T) = me.element;
        match(x){
            case MyOptionalElement(T).None() => { return false; }
        }
        return false;
    }

    fn get[me: Self] () -> T {
        var y: T;
        var x: MyOptionalElement(T) = me.element;
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
