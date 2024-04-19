// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Negate(a: u64) -> u64 = "int.unegate";
fn TestNegate(a: u64) -> u64 { return Negate(a); }

fn Add(a: u64, b: u64) -> u64 = "int.uadd";
fn TestAdd(a: u64, b: u64) -> u64 { return Add(a, b); }

fn Sub(a: u64, b: u64) -> u64 = "int.usub";
fn TestSub(a: u64, b: u64) -> u64 { return Sub(a, b); }

fn Mul(a: u64, b: u64) -> u64 = "int.umul";
fn TestMul(a: u64, b: u64) -> u64 { return Mul(a, b); }

fn Div(a: u64, b: u64) -> u64 = "int.udiv";
fn TestDiv(a: u64, b: u64) -> u64 { return Div(a, b); }

fn Mod(a: u64, b: u64) -> u64 = "int.umod";
fn TestMod(a: u64, b: u64) -> u64 { return Mod(a, b); }

fn Complement(a: u64) -> u64 = "int.complement";
fn TestComplement(a: u64) -> u64 { return Complement(a); }

fn And(a: u64, b: u64) -> u64 = "int.and";
fn TestAnd(a: u64, b: u64) -> u64 { return And(a, b); }

fn Or(a: u64, b: u64) -> u64 = "int.or";
fn TestOr(a: u64, b: u64) -> u64 { return Or(a, b); }

fn Xor(a: u64, b: u64) -> u64 = "int.xor";
fn TestXor(a: u64, b: u64) -> u64 { return Xor(a, b); }

fn LeftShift(a: u64, b: u64) -> u64 = "int.left_shift";
fn TestLeftShift(a: u64, b: u64) -> u64 { return LeftShift(a, b); }

fn RightShift(a: u64, b: u64) -> u64 = "int.right_shift";
fn TestRightShift(a: u64, b: u64) -> u64 { return RightShift(a, b); }

fn Eq(a: u64, b: u64) -> bool = "int.eq";
fn TestEq(a: u64, b: u64) -> bool { return Eq(a, b); }

fn Neq(a: u64, b: u64) -> bool = "int.neq";
fn TestNeq(a: u64, b: u64) -> bool { return Neq(a, b); }

fn Less(a: u64, b: u64) -> bool = "int.less";
fn TestLess(a: u64, b: u64) -> bool { return Less(a, b); }

fn LessEq(a: u64, b: u64) -> bool = "int.less_eq";
fn TestLessEq(a: u64, b: u64) -> bool { return LessEq(a, b); }

fn Greater(a: u64, b: u64) -> bool = "int.greater";
fn TestGreater(a: u64, b: u64) -> bool { return Greater(a, b); }

fn GreaterEq(a: u64, b: u64) -> bool = "int.greater_eq";
fn TestGreaterEq(a: u64, b: u64) -> bool { return GreaterEq(a, b); }

// CHECK:STDOUT: ; ModuleID = 'uint.carbon'
// CHECK:STDOUT: source_filename = "uint.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestNegate(i64 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.unegate = sub i64 0, %a
// CHECK:STDOUT:   ret i64 %int.unegate
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestAdd(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.uadd = add i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.uadd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestSub(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.usub = sub i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.usub
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestMul(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.umul = mul i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.umul
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestDiv(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.udiv = udiv i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.udiv
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestMod(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.umod = urem i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.umod
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestComplement(i64 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.complement = xor i64 -1, %a
// CHECK:STDOUT:   ret i64 %int.complement
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestAnd(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.and = and i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.and
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestOr(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.or = or i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.or
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestXor(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.xor = xor i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.xor
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestLeftShift(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.left_shift = shl i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.left_shift
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @TestRightShift(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift = lshr i64 %a, %b
// CHECK:STDOUT:   ret i64 %int.right_shift
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestEq(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.eq = icmp eq i64 %a, %b
// CHECK:STDOUT:   ret i1 %int.eq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestNeq(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.neq = icmp ne i64 %a, %b
// CHECK:STDOUT:   ret i1 %int.neq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestLess(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less = icmp ult i64 %a, %b
// CHECK:STDOUT:   ret i1 %int.less
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestLessEq(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less_eq = icmp ule i64 %a, %b
// CHECK:STDOUT:   ret i1 %int.less_eq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestGreater(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp ugt i64 %a, %b
// CHECK:STDOUT:   ret i1 %int.greater
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestGreaterEq(i64 %a, i64 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater_eq = icmp uge i64 %a, %b
// CHECK:STDOUT:   ret i1 %int.greater_eq
// CHECK:STDOUT: }
