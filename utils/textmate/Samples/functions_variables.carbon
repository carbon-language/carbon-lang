// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

fn F;
fn F1();
fn Func(a: i32, b: A);
fn Generic[T:! type, U:! I](a: T, b: U) -> U;
fn Lambda(a: i32) -> i32 => return a + 5;
fn

    MultiLine[
        A:! type,
        B:! C1,
        C:! C2
    ](a: A(B, C),
      b: B,
      c: i32);
fn Invalid#+?Name();
x.(I.F)();

a: A;
a: A = (b as A);
a: i32;
a: array(A, 5);
a:! type;
a:! T = b;
a: A in some_list;
var a: T;
let b: T;
binding: T;
binding:! T;
var
    multiline: T;

fn F(ref a: A, const ref b:! B);
fn F() -> T;
fn F() -> T.U { return val; }
fn F() -> i32 => return val;
