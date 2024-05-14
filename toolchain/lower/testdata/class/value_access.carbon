// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: (i32, i32, i32);
}

fn F(c: C) -> i32 {
  // TODO: `c.a` is a value expression here, which forces a value binding as
  // part of the member access, creating a tuple value temporary. We could
  // defer performing the value binding to avoid creating this temporary.
  return c.a[1];
}

// CHECK:STDOUT: ; ModuleID = 'value_access.carbon'
// CHECK:STDOUT: source_filename = "value_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %c) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15_11.1.a = getelementptr inbounds { { i32, i32, i32 } }, ptr %c, i32 0, i32 0
// CHECK:STDOUT:   %.loc15_11.2.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %.loc15_11.1.a, i32 0, i32 0
// CHECK:STDOUT:   %.loc15_11.3 = load i32, ptr %.loc15_11.2.tuple.elem, align 4
// CHECK:STDOUT:   %.loc15_11.4.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %.loc15_11.1.a, i32 0, i32 1
// CHECK:STDOUT:   %.loc15_11.5 = load i32, ptr %.loc15_11.4.tuple.elem, align 4
// CHECK:STDOUT:   %.loc15_11.6.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %.loc15_11.1.a, i32 0, i32 2
// CHECK:STDOUT:   %.loc15_11.7 = load i32, ptr %.loc15_11.6.tuple.elem, align 4
// CHECK:STDOUT:   %tuple = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple1 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc15_11.3, ptr %tuple1, align 4
// CHECK:STDOUT:   %tuple2 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc15_11.5, ptr %tuple2, align 4
// CHECK:STDOUT:   %tuple3 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc15_11.7, ptr %tuple3, align 4
// CHECK:STDOUT:   %.loc15_15.tuple.index = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   %.loc15_15.tuple.index.load = load i32, ptr %.loc15_15.tuple.index, align 4
// CHECK:STDOUT:   ret i32 %.loc15_15.tuple.index.load
// CHECK:STDOUT: }
