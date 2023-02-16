// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 4

package ExplorerTest api;

interface MyHashable {
  let Result:! type;
  fn Hash[self: Self]() -> Result;
}

constraint HashToIntConvertible {
  impl as MyHashable;
  impl Self.(MyHashable.Result) as ImplicitAs(i32);
}

class MyHashValue {
  external impl as ImplicitAs(i32) {
    fn Convert[self: Self]() -> i32 { return 4; }
  }
}

class Widget {
  external impl as MyHashable where .Result = MyHashValue {
    fn Hash[self: Self]() -> MyHashValue { return {}; }
  }
}

fn MakeSmallHash[T:! HashToIntConvertible](x: T) -> i32 {
  return x.(MyHashable.Hash)();
}

fn Main() -> i32 {
  var w: Widget = {};
  return MakeSmallHash(w);
}
