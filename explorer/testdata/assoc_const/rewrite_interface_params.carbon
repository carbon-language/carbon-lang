// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:STDOUT: result: 0

package ExplorerTest api;

interface HasTypes {
  let A:! type;
  let B:! type;
  let C:! type;
  let D:! type;
  let E:! type;
  let F:! type;
}

interface HasParams(A:! type, B:! type, C:! type, D:! type) {
  let V:! HasTypes;
}

// Here we discover that there is a rewrite for `HasParams(...).V` only after
// substitution converts each one to `HasParam(X, X, X, X).V`.
fn F[X:! (HasTypes & HasParams(.Self, .Self, .Self, .Self)) where
       .Self is HasParams(.A, .A, .A, .A) and
       .Self is HasParams(.B, .B, .B, .B) and
       .Self is HasParams(.C, .C, .C, .C) and
       .Self is HasParams(.D, .D, .D, .D) and
       .Self is HasParams(.E, .E, .E, .E) and
       .F = .Self.(HasParams(.E, .E, .E, .E).V).A and
       .E = .Self.(HasParams(.D, .D, .D, .D).V).A and
       .D = .Self.(HasParams(.C, .C, .C, .C).V).A and
       .C = .Self.(HasParams(.B, .B, .B, .B).V).A and
       .B = .Self.(HasParams(.A, .A, .A, .A).V).A and
       .A = .Self and .V = .Self](x: X) -> X.F { return x; }

impl i32 as HasTypes
  where .A = .B and .B = .C and .C = .D and .D = .E and .E = .F and .F = i32 {}
impl i32 as HasParams(i32, i32, i32, i32) where .V = i32 {}

fn Main() -> i32 {
  return F(0);
}
