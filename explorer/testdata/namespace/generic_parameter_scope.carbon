// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 0

package ExplorerTest api;

// Check that even generic parameters are looked up in the namespace of an
// entity.
namespace N;

interface N.Interface {}

class N.A(T:! Interface) {}
interface N.B(T:! Interface) {}
constraint N.C(T:! Interface) {}
__mixin N.D(T:! Interface) {}
choice N.E(T:! Interface) {}
fn N.F[T:! Interface](a: A(T)) {}

fn Main() -> i32 { return 0; }
