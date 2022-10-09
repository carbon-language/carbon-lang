// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK:result: 2

package ExplorerTest api;

interface Hashable {
  fn Hash[me: Self]() -> i32;
}

class Potato {
  external impl as Hashable {
    fn Hash[me: Self]() -> i32 {
      Print("Potato.(Hashable.Hash)");
      return 1;
    }
  }

  fn Hash[me: Self]() -> i32 {
    Print("Potato.Hash");
    return 2;
  }
}

interface Maker {
  let Result:! Hashable;
  fn Make() -> Result;
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

class PotatoFactory {
  impl as Maker where .Result = Potato {
    fn Make() -> Potato { return {}; }
  }
}

fn Main() -> i32 {
  var f: PotatoFactory = {};
  Print("{0}", F(f));
  Print("{0}", G(f));
  Print("{0}", H(f));
  Print("{0}", I(f));
  return 0;
}
