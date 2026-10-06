// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

class C1;
class C2(T1: type, T2: C1) {}
class C3 {
    extend base: C1;
}
class
        (C4(
            (T1: type),
            T2:
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
impl forall [T: I where .A = i32] T as J where .B = T.A {}
impl Self as I;
impl as I;

constraint Constraint {
    require T impls I;
    require Self impls I;
    require impls I;
    extern require impls I;
}

namespace N;

class N.C1 {
    fn N.C1.Make() -> Self;
}
alias N.A = N.C1;
fn N.C1.Other();
class Outer(T: type).Inner(U: type) {}
fn Outer(T: type).Inner(U: type).F[self: Self](x: T) -> U;
fn N.Base(N.T).Member() {}
interface N.I2;
constraint N.C2 {}

class C6 {
    fn destroy(ref self);
}
fn C6.destroy(ref self) {}

var p: C6* = &c6;
var q: typeof(p) = p;
var r: typeof(Make(p, (q))).Pointee = *p;

// --- the same names in type and expression positions

var o: Optional(Slice(i32)) = Optional(Slice(i32)).None();
let origin: Point = Point.Origin();
let t: type = Point;
var c: Core.Int(64) = x as Core.Int(64);
var v: Cpp.std.vector(i32) = Make(i32, Vector(Point));
var s: {.x: i32, .y: Point} = {.x = 1, .y = p};
var tup: (i32, Point) = (1, p);
fn Pick[T: Hashable & Eq where .Key = i32](t: T) -> T.Key;
fn Iter() -> Vector(T).Iterator;
if (T == Point and x impls Hashable) {}
alias lower = i32;
var l: lower = a * b;
