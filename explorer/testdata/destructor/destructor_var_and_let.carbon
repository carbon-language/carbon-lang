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
// CHECK:STDOUT: LetDeclaration - end
// CHECK:STDOUT:
// CHECK:STDOUT: VarAndLetDeclaration
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: VarAndLetDeclaration - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetParamVarDeclaration-Passing LValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: LetParamVarDeclaration-Passing LValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetParamLetDeclaration-Passing LValue
// CHECK:STDOUT: LetParamLetDeclaration-Passing LValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetParamVarDeclaration-Passing RValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: LetParamVarDeclaration-Passing RValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: LetParamLetDeclaration-Passing RValue
// CHECK:STDOUT: LetParamLetDeclaration-Passing RValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: VarParamVarDeclaration-Passing RValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: VarParamVarDeclaration-Passing RValue - end
// CHECK:STDOUT:
// CHECK:STDOUT: VarParamLetDeclaration-Passing RValue
// CHECK:STDOUT: DESTRUCTOR CALLED
// CHECK:STDOUT: VarParamLetDeclaration-Passing RValue - end
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

fn VarAndLetDeclaration() {
  var a: A = {};
  let b: A = a;
}

fn LetDeclaration() {
  let a: A = {};
}

fn LetParamVarDeclaration(a: A) {
  var b: A = a;
}

fn LetParamLetDeclaration(a: A) {
  let b: A = a;
}

fn VarParamVarDeclaration(var a: A) {
  var b: A = a;
}

fn VarParamLetDeclaration(var a: A) {
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

  Print("VarAndLetDeclaration");
  VarAndLetDeclaration();
  Print("VarAndLetDeclaration - end\n");

  Print("LetParamVarDeclaration-Passing LValue");
  var a1: A = {};
  LetParamVarDeclaration(a1);
  Print("LetParamVarDeclaration-Passing LValue - end\n");

  Print("LetParamLetDeclaration-Passing LValue");
  var a2: A = {};
  LetParamLetDeclaration(a2);
  Print("LetParamLetDeclaration-Passing LValue - end\n");

  Print("LetParamVarDeclaration-Passing RValue");
  LetParamVarDeclaration({});
  Print("LetParamVarDeclaration-Passing RValue - end\n");

  Print("LetParamLetDeclaration-Passing RValue");
  LetParamLetDeclaration({});
  Print("LetParamLetDeclaration-Passing RValue - end\n");

  Print("VarParamVarDeclaration-Passing RValue");
  VarParamVarDeclaration({});
  Print("VarParamVarDeclaration-Passing RValue - end\n");

  Print("VarParamLetDeclaration-Passing RValue");
  VarParamLetDeclaration({});
  Print("VarParamLetDeclaration-Passing RValue - end\n");

  Print("VarBindingInitializingCall");
  VarBindingInitializingCall();
  Print("VarBindingInitializingCall - end\n");

  Print("LetBindingInitializingCall");
  LetBindingInitializingCall();
  Print("LetBindingInitializingCall - end\n");

  return 0;
}
