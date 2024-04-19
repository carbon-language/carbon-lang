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

// CHECK:STDOUT: ; ModuleID = 'float.carbon'
// CHECK:STDOUT: source_filename = "float.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestNegate(double %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.negate.fneg = fneg double %a
// CHECK:STDOUT:   ret double %float.negate.fneg
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestAdd(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.add.fadd = fadd double %a, %b
// CHECK:STDOUT:   ret double %float.add.fadd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestSub(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.sub.fsub = fsub double %a, %b
// CHECK:STDOUT:   ret double %float.sub.fsub
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestMul(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.mul.fmul = fmul double %a, %b
// CHECK:STDOUT:   ret double %float.mul.fmul
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @TestDiv(double %a, double %b) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.div.fdiv = fdiv double %a, %b
// CHECK:STDOUT:   ret double %float.div.fdiv
// CHECK:STDOUT: }
