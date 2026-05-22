// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

class C1;
class C2(T1:! type, T2:! C1) {}
class C3 {
    extend base: C1;
}
class
        (C4(
            (T1:! type),
            T2:!
                    (A.(B(E, F(G)).C)
                                        .D))) ;

class C5 {
    impl as I;
    extend impl as I;
}

alias a1 = C1;
alias a2 = i32;

adapt SomeClass;
adapt {.a: SomeClass, .b: i32};
adapt i32;

interface I;
impl C as (I(A, B, C.(D.E(F, G)).H)) {}
impl C as I where .A = (B where .Self = {}) {}
impl C as I where .A = {} and .B = C {}
impl Self as I;
impl as I;

constraint Constraint {
    require T impls I;
    require Self impls I;
    require impls I;
    extern require impls I;
}
