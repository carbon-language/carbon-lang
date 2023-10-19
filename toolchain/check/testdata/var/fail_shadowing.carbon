// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  var x: i32 = 0;
  if (true) {
    // TODO: We should reject the use of the shadowed variable `x`, not this
    // shadowing declaration. Also the diagnostic text is wrong ("same scope").
    // CHECK:STDERR: fail_shadowing.carbon:[[@LINE+6]]:9: ERROR: Duplicate name being declared in the same scope.
    // CHECK:STDERR:     var x: i32 = 0;
    // CHECK:STDERR:         ^
    // CHECK:STDERR: fail_shadowing.carbon:[[@LINE-7]]:7: Name is previously declared here.
    // CHECK:STDERR:   var x: i32 = 0;
    // CHECK:STDERR:       ^
    var x: i32 = 0;

    x = 1;
  }
}

// CHECK:STDOUT: file "fail_shadowing.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.loc8: ref i32 = var "x"
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0
// CHECK:STDOUT:   assign %x.loc8, %.loc8
// CHECK:STDOUT:   %.loc9: bool = bool_literal true
// CHECK:STDOUT:   if %.loc9 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %x.loc18: ref i32 = var "x"
// CHECK:STDOUT:   %.loc18: i32 = int_literal 0
// CHECK:STDOUT:   assign %x.loc18, %.loc18
// CHECK:STDOUT:   %x.ref: ref i32 = name_reference "x", %x.loc8
// CHECK:STDOUT:   %.loc20: i32 = int_literal 1
// CHECK:STDOUT:   assign %x.ref, %.loc20
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
