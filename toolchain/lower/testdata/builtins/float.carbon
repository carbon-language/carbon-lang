// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Negate(a: f64) -> f64 = "float.negate";
fn TestNegate(a: f64) -> f64 { return Negate(a); }

fn Add(a: f64, b: f64) -> f64 = "float.add";
fn TestAdd(a: f64, b: f64) -> f64 { return Add(a, b); }

fn Sub(a: f64, b: f64) -> f64 = "float.sub";
fn TestSub(a: f64, b: f64) -> f64 { return Sub(a, b); }

fn Mul(a: f64, b: f64) -> f64 = "float.mul";
fn TestMul(a: f64, b: f64) -> f64 { return Mul(a, b); }

fn Div(a: f64, b: f64) -> f64 = "float.div";
fn TestDiv(a: f64, b: f64) -> f64 { return Div(a, b); }

fn Eq(a: f64, b: f64) -> bool = "float.eq";
fn TestEq(a: f64, b: f64) -> bool { return Eq(a, b); }

fn Neq(a: f64, b: f64) -> bool = "float.neq";
fn TestNeq(a: f64, b: f64) -> bool { return Neq(a, b); }

fn Less(a: f64, b: f64) -> bool = "float.less";
fn TestLess(a: f64, b: f64) -> bool { return Less(a, b); }

fn LessEq(a: f64, b: f64) -> bool = "float.less_eq";
fn TestLessEq(a: f64, b: f64) -> bool { return LessEq(a, b); }

fn Greater(a: f64, b: f64) -> bool = "float.greater";
fn TestGreater(a: f64, b: f64) -> bool { return Greater(a, b); }

fn GreaterEq(a: f64, b: f64) -> bool = "float.greater_eq";
fn TestGreaterEq(a: f64, b: f64) -> bool { return GreaterEq(a, b); }

// CHECK:STDOUT: ; ModuleID = 'float.carbon'
// CHECK:STDOUT: source_filename = "float.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestNegate(double %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.negate = fneg double %a
// CHECK:STDOUT:   ret double %float.negate
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestAdd(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.add = fadd double %a, %b
// CHECK:STDOUT:   ret double %float.add
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestSub(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.sub = fsub double %a, %b
// CHECK:STDOUT:   ret double %float.sub
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestMul(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.mul = fmul double %a, %b
// CHECK:STDOUT:   ret double %float.mul
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestDiv(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.div = fdiv double %a, %b
// CHECK:STDOUT:   ret double %float.div
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestEq(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.eq = fcmp oeq double %a, %b
// CHECK:STDOUT:   ret i1 %float.eq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestNeq(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.neq = fcmp one double %a, %b
// CHECK:STDOUT:   ret i1 %float.neq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestLess(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.less = fcmp olt double %a, %b
// CHECK:STDOUT:   ret i1 %float.less
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestLessEq(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.less_eq = fcmp ole double %a, %b
// CHECK:STDOUT:   ret i1 %float.less_eq
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestGreater(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.greater = fcmp ogt double %a, %b
// CHECK:STDOUT:   ret i1 %float.greater
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @TestGreaterEq(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.greater_eq = fcmp oge double %a, %b
// CHECK:STDOUT:   ret i1 %float.greater_eq
// CHECK:STDOUT: }
