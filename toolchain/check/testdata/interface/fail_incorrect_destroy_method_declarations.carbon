// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_incorrect_destroy_method_declarations.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_incorrect_destroy_method_declarations.carbon

// TODO: remove `SubobjectDestroy` when `require impls` adds a witness table entry.

// --- fail_too_few_assoc_functions_declared.carbon

package Core library "too_few_assoc_functions_declared";

// CHECK:STDERR: fail_too_few_assoc_functions_declared.carbon:[[@LINE+4]]:1: error: semantics TODO: `interface `Core.Destroy` needs exactly 3 associated functions` [SemanticsTodo]
// CHECK:STDERR: interface Destroy {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Destroy {
  final fn SubobjectDestroy(ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_too_many_assoc_functions_declared.carbon

package Core library "too_many_assoc_functions_declared";

// CHECK:STDERR: fail_too_many_assoc_functions_declared.carbon:[[@LINE+4]]:1: error: semantics TODO: `interface `Core.Destroy` needs exactly 3 associated functions` [SemanticsTodo]
// CHECK:STDERR: interface Destroy {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(ref self) {}
  final fn SelfDestruct(unused ref self) {}
  fn Extra(ref self) {}
}

// --- fail_op_associated_type.carbon

package Core library "op_associated_type";

interface Destroy {
  // CHECK:STDERR: fail_op_associated_type.carbon:[[@LINE+4]]:7: error: semantics TODO: `associated entity must be a method` [SemanticsTodo]
  // CHECK:STDERR:   let Op: type;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let Op: type;
  final fn SubobjectDestroy(ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_wrong_name.carbon

package Core library "op_wrong_name";

interface Destroy {
  // CHECK:STDERR: fail_op_wrong_name.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function #1 must be named `Op`` [SemanticsTodo]
  // CHECK:STDERR:   final fn SubobjectDestroy(ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SubobjectDestroy(ref self) {}
  fn Op(ref self);
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_no_params.carbon

package Core library "op_no_params";

interface Destroy {
  // CHECK:STDERR: fail_op_no_params.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must have exactly 1 parameter` [SemanticsTodo]
  // CHECK:STDERR:   fn Op();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  fn Op();
  final fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_two_params.carbon

package Core library "op_two_params";

interface Destroy {
  // CHECK:STDERR: fail_op_two_params.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must have exactly 1 parameter` [SemanticsTodo]
  // CHECK:STDERR:   fn Op(ref self, x: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn Op(ref self, x: Self);
  final fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_param_not_self.carbon

package Core library "op_param_not_self";

class C {}

interface Destroy {
  // CHECK:STDERR: fail_op_param_not_self.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   fn Op(c: C);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  fn Op(c: C);
  final fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_param_is_generic.carbon

package Core library "op_param_is_generic";

interface Destroy {
  // CHECK:STDERR: fail_op_param_is_generic.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   fn Op[T: type](ref c: T);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn Op[T: type](ref c: T);
  final fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_self_param_not_reference.carbon

package Core library "op_self_param_not_reference";

interface Destroy {
  // CHECK:STDERR: fail_op_self_param_not_reference.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   fn Op(self);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  fn Op(self);
  final fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_explicit_self_type.carbon

package Core library "op_explicit_self_type";

interface Destroy {
  // CHECK:STDERR: fail_op_explicit_self_type.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   fn Op(ref s: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn Op(ref s: Self);
  final fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_final.carbon

package Core library "op_final";

interface Destroy {
  // CHECK:STDERR: fail_op_final.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must not have an interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   final fn Op(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn Op(unused ref self) {}
  final fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_default.carbon

package Core library "op_default";

interface Destroy {
  // CHECK:STDERR: fail_op_default.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must not have an interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default fn Op(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  default fn Op(unused ref self) {}
  final fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_self_destruct_associated_type.carbon

package Core library "self_destruct_associated_type";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_associated_type.carbon:[[@LINE+4]]:7: error: semantics TODO: `associated entity must be a method` [SemanticsTodo]
  // CHECK:STDERR:   let SelfDestruct: type;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let SelfDestruct: type;
}

// --- fail_self_destruct_wrong_name.carbon

package Core library "self_destruct_wrong_name";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_wrong_name.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function #3 must be named `SelfDestruct`` [SemanticsTodo]
  // CHECK:STDERR:   final fn selfDestruct(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn selfDestruct(unused ref self) {}
}

// --- fail_self_destruct_no_params.carbon

package Core library "self_destruct_no_params";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_no_params.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must have exactly 1 parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SelfDestruct() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SelfDestruct() {}
}

// --- fail_self_destruct_two_params.carbon

package Core library "self_destruct_two_params";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_two_params.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must have exactly 1 parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SelfDestruct(unused ref self, unused x: Self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SelfDestruct(unused ref self, unused x: Self) {}
}

// --- fail_self_destruct_param_not_self.carbon

package Core library "self_destruct_param_not_self";

class C {}

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_param_not_self.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SelfDestruct(unused ref c: C) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SelfDestruct(unused ref c: C) {}
}

// --- fail_self_destruct_param_is_generic.carbon

package Core library "self_destruct_param_is_generic";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_param_is_generic.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SelfDestruct[T: type](unused ref c: T) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SelfDestruct[T: type](unused ref c: T) {}
}

// --- fail_self_destruct_self_param_not_reference.carbon

package Core library "self_destruct_self_param_not_reference";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_self_param_not_reference.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SelfDestruct(unused self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SelfDestruct(unused self) {}
}

// --- fail_self_destruct_explicit_self_type.carbon

package Core library "self_destruct_explicit_self_type";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_explicit_self_type.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SelfDestruct(unused ref s: Self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SelfDestruct(unused ref s: Self) {}
}

// --- fail_self_destruct_not_final.carbon

package Core library "self_destruct_not_final";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_not_final.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must be `final`` [SemanticsTodo]
  // CHECK:STDERR:   fn SelfDestruct(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn SelfDestruct(unused ref self) {}
}

// --- fail_self_destruct_default.carbon

package Core library "self_destruct_default";

interface Destroy {
  fn Op(ref self);
  final fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR: fail_self_destruct_default.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must be `final`` [SemanticsTodo]
  // CHECK:STDERR:   default fn SelfDestruct(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  default fn SelfDestruct(unused ref self) {}
}
