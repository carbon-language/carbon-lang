// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A;

interface GetSelf {
  // TODO: Use `Self` once that works.
  fn Get[addr self: A*]() -> A*;
}

class A {
  extend impl as GetSelf {
    fn Get[addr self: Self*]() -> Self* {
      return self;
    }
  }
}

fn Call(a: A*) -> A* {
  return (*a).Get();
}

// CHECK:STDOUT: ; ModuleID = 'instance_method.carbon'
// CHECK:STDOUT: source_filename = "instance_method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @Get(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @Get.1(ptr %self) {
// CHECK:STDOUT:   ret ptr %self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @Call(ptr %a) {
// CHECK:STDOUT:   %Get.1 = call ptr @Get.1(ptr %a)
// CHECK:STDOUT:   ret ptr %Get.1
// CHECK:STDOUT: }
