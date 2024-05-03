// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A;

interface GetSelf {
  fn Get[addr self: Self*]() -> Self*;
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
// CHECK:STDOUT: define ptr @Get(ptr %self) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @Call(ptr %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Get.call = call ptr @Get(ptr %a)
// CHECK:STDOUT:   ret ptr %Get.call
// CHECK:STDOUT: }
