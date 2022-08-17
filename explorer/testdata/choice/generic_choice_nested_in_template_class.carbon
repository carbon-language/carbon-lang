// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: result: 0

package ExplorerTest api;


choice OptionalElement(T:! Type) {
  None(),
  Element(T)
}

class Optional(T:! Type){
   fn CreateEmpty() -> Optional(T){
       return { .element = OptionalElement(T).None() };
   }
   fn Create ( value: T ) -> Optional(T){
       return { .element = OptionalElement(T).Element(value) };
   }

    fn has_value[me: Self] () -> bool{
        var x: OptionalElement(T) = me.element;
        match(x){
            case OptionalElement(T).None() => { return false; }
        }
        return false;
    }

    fn get[me: Self] () -> T {
        var y: T;
        var x: OptionalElement(T) = me.element;
        match(x){
            case OptionalElement(T).Element( var x: T ) =>{
                return x;
            }
        }
        return y;
    }

   var element: OptionalElement(T);
}


fn Main() -> i32 {
  var f: Optional(i32) = Optional(i32).Create(22);
  var x: i32 = f.get();
  Print("H {0}",x);

  return 0;
}
