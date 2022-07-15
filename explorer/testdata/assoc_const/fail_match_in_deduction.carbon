// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

// TODO: Should this work?

package ExplorerTest api;

interface Vector {
  let Dim:! i32;
}
external impl (i32, i32, i32) as Vector where .Dim == 3 {}

class Point(Scalar:! Type, Dim:! i32) {}

fn F[Scalar:! Type, V:! Vector where .Dim == 3](p: Point(Scalar, V.Dim), v: V) {}

fn G[Scalar:! Type](p: Point(Scalar, 3)) {}
fn H[V:! Vector where .Dim == 3](v: V) {
  var p: Point(i32, V.Dim) = {};
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/assoc_const/fail_match_in_deduction.carbon:[[@LINE+1]]: mismatch in non-type values, `(V:! Vector where .Self.Dim == 3).Dim` != `3`
  G(p);
}

fn Main() -> i32 {
  var p: Point(i32, 3) = {};
  // Deduce Point(Scalar, V.Dim) from Point(i32, 3).
  F(p, (0, 0, 0));
  // Deduce Point(Scalar, 3) from Point(i32, V.Dim).
  H((0, 0, 0));
  return 0;
}
