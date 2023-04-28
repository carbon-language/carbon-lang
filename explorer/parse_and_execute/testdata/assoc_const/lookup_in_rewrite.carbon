// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: 1
// CHECK:STDOUT: i32.Hash
// CHECK:STDOUT: 0
// CHECK:STDOUT: Potato.(Hashable.Hash)
// CHECK:STDOUT: 1
// CHECK:STDOUT: Potato.Hash
// CHECK:STDOUT: 2
// CHECK:STDOUT: Potato.Hash
// CHECK:STDOUT: 2
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface Hashable {
  fn Hash[self: Self]() -> i32;
}

class Potato {
  external impl as Hashable {
    fn Hash[self: Self]() -> i32 {
      Print("Potato.(Hashable.Hash)");
      return 1;
    }
  }

  fn Hash[self: Self]() -> i32 {
    Print("Potato.Hash");
    return 2;
  }
}

interface Maker {
  let Result:! Hashable;
  fn Make() -> Result;
}

external impl i32 as Hashable {
  fn Hash[self: Self]() -> i32 {
    Print("i32.Hash");
    return self;
  }
}

fn F[T:! Maker where .Result = i32](x: T) -> i32 {
  // OK, can treat T.Make() as an i32.
  return T.Make() + 1;
}

fn G[T:! Maker](x: T) -> i32 {
  // OK, Potato.(Hashable.Hash), not Potato.Hash.
  return T.Make().Hash();
}

fn H[T:! Maker where .Result = Potato](x: T) -> i32 {
  // OK, Potato.Hash, not Potato.(Hashable.Hash).
  return T.Make().Hash();
}

fn I[T:! Maker where .Result = Potato](x: T) -> i32 {
  var p: Potato = {};
  // OK, Potato.Hash, not Potato.(Hashable.Hash), even though we know Potato is
  // Hashable here.
  return p.Hash();
}

class IntFactory {
  impl as Maker where .Result = i32 {
    fn Make() -> i32 { return 0; }
  }
}

class PotatoFactory {
  impl as Maker where .Result = Potato {
    fn Make() -> Potato { return {}; }
  }
}

fn Main() -> i32 {
  var f: IntFactory = {};
  var g: PotatoFactory = {};
  Print("{0}", F(f));
  Print("{0}", G(f));
  Print("{0}", G(g));
  Print("{0}", H(g));
  Print("{0}", I(g));
  return 0;
}
