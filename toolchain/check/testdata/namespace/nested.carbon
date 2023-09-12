// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "nested.carbon" {

namespace Foo;
// CHECK:STDOUT:   %.loc[[@LINE-1]] = namespace {.Bar = %.loc[[@LINE+1]]}
namespace Foo.Bar;
// CHECK:STDOUT:   %.loc[[@LINE-1]] = namespace {.Wiz = %.loc[[@LINE+2]], .Baz = %.loc[[@LINE+6]]}

fn Foo.Bar.Wiz() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Wiz
}

fn Foo.Bar.Baz() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Baz
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Wiz() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Baz() {
// CHECK:STDOUT: !entry:
  Foo.Bar.Wiz();
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_14.1: type = tuple_type ()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_14.2: init () = call @Wiz()
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
