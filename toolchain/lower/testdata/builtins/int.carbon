// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Negate(a: i32) -> i32 = "int.snegate";
fn TestNegate(a: i32) -> i32 { return Negate(a); }

fn Add(a: i32, b: i32) -> i32 = "int.sadd";
fn TestAdd(a: i32, b: i32) -> i32 { return Add(a, b); }

fn Sub(a: i32, b: i32) -> i32 = "int.ssub";
fn TestSub(a: i32, b: i32) -> i32 { return Sub(a, b); }

fn Mul(a: i32, b: i32) -> i32 = "int.smul";
fn TestMul(a: i32, b: i32) -> i32 { return Mul(a, b); }

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";
fn TestDiv(a: i32, b: i32) -> i32 { return Div(a, b); }

fn Mod(a: i32, b: i32) -> i32 = "int.smod";
fn TestMod(a: i32, b: i32) -> i32 { return Mod(a, b); }

fn Complement(a: i32) -> i32 = "int.complement";
fn TestComplement(a: i32) -> i32 { return Complement(a); }

fn And(a: i32, b: i32) -> i32 = "int.and";
fn TestAnd(a: i32, b: i32) -> i32 { return And(a, b); }

fn Or(a: i32, b: i32) -> i32 = "int.or";
fn TestOr(a: i32, b: i32) -> i32 { return Or(a, b); }

fn Xor(a: i32, b: i32) -> i32 = "int.xor";
fn TestXor(a: i32, b: i32) -> i32 { return Xor(a, b); }

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
fn TestLeftShift(a: i32, b: i32) -> i32 { return LeftShift(a, b); }

fn RightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn TestRightShift(a: i32, b: i32) -> i32 { return RightShift(a, b); }

fn Eq(a: i32, b: i32) -> bool = "int.eq";
fn TestEq(a: i32, b: i32) -> bool { return Eq(a, b); }

fn Neq(a: i32, b: i32) -> bool = "int.neq";
fn TestNeq(a: i32, b: i32) -> bool { return Neq(a, b); }

fn Less(a: i32, b: i32) -> bool = "int.less";
fn TestLess(a: i32, b: i32) -> bool { return Less(a, b); }

fn LessEq(a: i32, b: i32) -> bool = "int.less_eq";
fn TestLessEq(a: i32, b: i32) -> bool { return LessEq(a, b); }

fn Greater(a: i32, b: i32) -> bool = "int.greater";
fn TestGreater(a: i32, b: i32) -> bool { return Greater(a, b); }

fn GreaterEq(a: i32, b: i32) -> bool = "int.greater_eq";
fn TestGreaterEq(a: i32, b: i32) -> bool { return GreaterEq(a, b); }

// CHECK:STDOUT: ; ModuleID = 'int.carbon'
// CHECK:STDOUT: source_filename = "int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestNegate(i32 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.snegate = sub i32 0, %a
// CHECK:STDOUT:   ret i32 %int.snegate
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestAdd(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.sadd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestSub(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.ssub = sub i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.ssub
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestMul(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.smul = mul i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.smul
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestDiv(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sdiv = sdiv i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.sdiv
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestMod(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.smod = srem i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.smod
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestComplement(i32 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.complement = xor i32 -1, %a
// CHECK:STDOUT:   ret i32 %int.complement
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestAnd(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.and = and i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.and
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestOr(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.or = or i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.or
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestXor(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.xor = xor i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.xor
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestLeftShift(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.left_shift = shl i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.left_shift
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TestRightShift(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift = ashr i32 %a, %b
// CHECK:STDOUT:   ret i32 %int.right_shift
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestEq(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.eq = icmp eq i32 %a, %b
// CHECK:STDOUT:   ret i1 %int.eq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestNeq(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.neq = icmp ne i32 %a, %b
// CHECK:STDOUT:   ret i1 %int.neq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestLess(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less = icmp slt i32 %a, %b
// CHECK:STDOUT:   ret i1 %int.less
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestLessEq(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less_eq = icmp sle i32 %a, %b
// CHECK:STDOUT:   ret i1 %int.less_eq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestGreater(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %a, %b
// CHECK:STDOUT:   ret i1 %int.greater
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestGreaterEq(i32 %a, i32 %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater_eq = icmp sge i32 %a, %b
// CHECK:STDOUT:   ret i1 %int.greater_eq
// CHECK:STDOUT: }
