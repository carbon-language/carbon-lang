// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/access_modifers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/access_modifers.carbon

// --- fail_private_field_access.carbon

library "[[@TEST_NAME]]";

class C {
  private var x: {};
  private let SOME_INTERNAL_CONSTANT: {} = {};

  private fn SomeInternalFunction() -> {} {
    return {};
  }

  fn Make() -> Self {
    return {.x = {}};
  }
}

fn Run() {
  let c: C = C.Make();
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE+7]]:15: error: cannot access private member `x` of type `C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let x: {} = c.x;
  // CHECK:STDERR:               ^~~
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE-17]]:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private var x: {};
  // CHECK:STDERR:               ^~~~~
  // CHECK:STDERR:
  let x: {} = c.x;
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE+7]]:3: error: cannot access private member `x` of type `C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c.x = {};
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE-25]]:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private var x: {};
  // CHECK:STDERR:               ^~~~~
  // CHECK:STDERR:
  c.x = {};
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE+7]]:3: error: cannot access private member `SOME_INTERNAL_CONSTANT` of type `C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c.SOME_INTERNAL_CONSTANT;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE-32]]:15: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private let SOME_INTERNAL_CONSTANT: {} = {};
  // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c.SOME_INTERNAL_CONSTANT;

  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE+7]]:3: error: cannot access private member `SomeInternalFunction` of type `C` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   c.SomeInternalFunction();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_private_field_access.carbon:[[@LINE-39]]:3: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   private fn SomeInternalFunction() -> {} {
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  c.SomeInternalFunction();
}

// --- fail_protected_field_access.carbon

library "[[@TEST_NAME]]";

class A {
  protected var x: {};
}

fn Run() {
  // CHECK:STDERR: fail_protected_field_access.carbon:[[@LINE+7]]:15: error: cannot access protected member `x` of type `A` [ClassInvalidMemberAccess]
  // CHECK:STDERR:   let x: {} = A.x;
  // CHECK:STDERR:               ^~~
  // CHECK:STDERR: fail_protected_field_access.carbon:[[@LINE-7]]:17: note: declared here [ClassMemberDeclaration]
  // CHECK:STDERR:   protected var x: {};
  // CHECK:STDERR:                 ^~~~~
  // CHECK:STDERR:
  let x: {} = A.x;
}

// --- instance_private_field_access_on_self.carbon

library "[[@TEST_NAME]]";

class C {
  //@dump-sem-ir-begin
  private var x: {};
  //@dump-sem-ir-end

  fn GetRadius[self: Self]() -> {} {
    return self.x;
  }

  private fn SomeInternalFunction() -> {} {
    return {};
  }

  fn Compute[self: Self]() -> {} {
    return self.SomeInternalFunction();
  }
}

// --- public_global_access.carbon

library "[[@TEST_NAME]]";

class A {
  let x: () = ();
}

let x: () = A.x;

// --- fail_global_access.carbon

library "[[@TEST_NAME]]";

class A {
  protected let x: () = ();
  private let y: {} = {};
}

// CHECK:STDERR: fail_global_access.carbon:[[@LINE+7]]:13: error: cannot access protected member `x` of type `A` [ClassInvalidMemberAccess]
// CHECK:STDERR: let x: () = A.x;
// CHECK:STDERR:             ^~~
// CHECK:STDERR: fail_global_access.carbon:[[@LINE-7]]:17: note: declared here [ClassMemberDeclaration]
// CHECK:STDERR:   protected let x: () = ();
// CHECK:STDERR:                 ^
// CHECK:STDERR:
let x: () = A.x;
// CHECK:STDERR: fail_global_access.carbon:[[@LINE+7]]:13: error: cannot access private member `y` of type `A` [ClassInvalidMemberAccess]
// CHECK:STDERR: let y: {} = A.y;
// CHECK:STDERR:             ^~~
// CHECK:STDERR: fail_global_access.carbon:[[@LINE-14]]:15: note: declared here [ClassMemberDeclaration]
// CHECK:STDERR:   private let y: {} = {};
// CHECK:STDERR:               ^
// CHECK:STDERR:
let y: {} = A.y;

// --- self_access.carbon

library "[[@TEST_NAME]]";

class A {
  private fn F() {}
  private fn G() { Self.F(); }
}

// CHECK:STDOUT: --- instance_private_field_access_on_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc6_19.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc6_19.2: type = converted %.loc6_19.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %.loc6_16: %C.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .x [private] = %.loc6_16
// CHECK:STDOUT:   .GetRadius = %GetRadius.decl
// CHECK:STDOUT:   .SomeInternalFunction [private] = %SomeInternalFunction.decl
// CHECK:STDOUT:   .Compute = %Compute.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
