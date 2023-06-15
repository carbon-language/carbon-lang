// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: VarDeclaration
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: VarDeclaration - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetDeclaration
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: LetDeclaration - end
// CHECK:STDOUT:
// CHECK:STDOUT: VarDeclarationAndLetAssignment
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: VarDeclarationAndLetAssignment - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetParamInitializesVar-Passing LValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: LetParamInitializesVar-Passing LValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetParamInitializesLet-Passing LValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: LetParamInitializesLet-Passing LValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetParamInitializesVar-Passing RValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: LetParamInitializesVar-Passing RValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetParamInitializesLet-Passing RValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: LetParamInitializesLet-Passing RValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: VarParamInitializesVar-Passing RValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: VarParamInitializesVar-Passing RValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: VarParamInitializesLet-Passing RValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: VarParamInitializesLet-Passing RValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: VarBindingInitializingCall
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: VarBindingInitializingCall - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetBindingInitializingCall
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: LetBindingInitializingCall - end
// CHECK:STDOUT:
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: result: 0

package ExplorerTest api;


class A {
  destructor[self: Self] {
    Print("DESTRUCTOR CALLED");
  }
}

fn VarDeclaration() {
  var a: A = {};
}

fn LetDeclaration() {
  let a: A = {};
}

fn VarDeclarationAndLetAssignment() {
  var a: A = {};
  let b: A = a;
}

fn LetParamInitializesVar(a: A) {
  var b: A = a;
}

fn LetParamInitializesLet(a: A) {
  let b: A = a;
}

fn VarParamInitializesVar(var a: A) {
  var b: A = a;
}

fn VarParamInitializesLet(var a: A) {
  let b: A = a;
}

fn Foo() -> A {
  var a: A = {};
  return a;
}

fn VarBindingInitializingCall() {
  var a: A = Foo();
}

fn LetBindingInitializingCall() {
  let a: A = Foo();
}

fn Main() -> i32 {
  Print("VarDeclaration");
  VarDeclaration();
  Print("VarDeclaration - end\n");

  Print("LetDeclaration");
  LetDeclaration();
  Print("LetDeclaration - end\n");

  Print("VarDeclarationAndLetAssignment");
  VarDeclarationAndLetAssignment();
  Print("VarDeclarationAndLetAssignment - end\n");

  Print("LetParamInitializesVar-Passing LValue");
  var a1: A = {};
  LetParamInitializesVar(a1);
  Print("LetParamInitializesVar-Passing LValue - end\n");

  Print("LetParamInitializesLet-Passing LValue");
  var a2: A = {};
  LetParamInitializesLet(a2);
  Print("LetParamInitializesLet-Passing LValue - end\n");

  Print("LetParamInitializesVar-Passing RValue");
  LetParamInitializesVar({});
  Print("LetParamInitializesVar-Passing RValue - end\n");

  Print("LetParamInitializesLet-Passing RValue");
  LetParamInitializesLet({});
  Print("LetParamInitializesLet-Passing RValue - end\n");

  Print("VarParamInitializesVar-Passing RValue");
  VarParamInitializesVar({});
  Print("VarParamInitializesVar-Passing RValue - end\n");

  Print("VarParamInitializesLet-Passing RValue");
  VarParamInitializesLet({});
  Print("VarParamInitializesLet-Passing RValue - end\n");

  Print("VarBindingInitializingCall");
  VarBindingInitializingCall();
  Print("VarBindingInitializingCall - end\n");

  Print("LetBindingInitializingCall");
  LetBindingInitializingCall();
  Print("LetBindingInitializingCall - end\n");

  return 0;
}
