// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 6

package ExplorerTest api;

namespace N;

fn OuterI32() -> type { return i32; }
fn One() -> i32 { return 1; }

fn N.I32() -> type { return i32; }
fn N.Five() -> I32() { return 5; }
fn N.Six() -> OuterI32() { return Five() + One(); }

fn Main() -> i32 { return N.Six(); }
