// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_incorrect_subobject_destroy_method_declarations.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_incorrect_subobject_destroy_method_declarations.carbon

// TODO: change to `SubobjectDestroy` when `require impls` adds a witness table entry.

// --- fail_op_not_declared.carbon

package Core library "op_not_declared";

// CHECK:STDERR: fail_op_not_declared.carbon:[[@LINE+4]]:1: error: semantics TODO: `interface `Core.Destroy` needs exactly 3 associated functions` [SemanticsTodo]
// CHECK:STDERR: interface Destroy {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
interface Destroy {
  fn Op(ref self);
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_associated_type.carbon

package Core library "op_associated_type";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_associated_type.carbon:[[@LINE+4]]:7: error: semantics TODO: `associated entity must be a method` [SemanticsTodo]
  // CHECK:STDERR:   let SubobjectDestroy: type;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let SubobjectDestroy: type;
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_wrong_name.carbon

package Core library "op_wrong_name";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_wrong_name.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function #2 must be named `SubobjectDestroy`` [SemanticsTodo]
  // CHECK:STDERR:   final fn SelfDestruct(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SelfDestruct(unused ref self) {}
  final fn SubobjectDestroy(unused ref self) {}
}

// --- fail_op_no_params.carbon

package Core library "op_no_params";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_no_params.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must have exactly 1 parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SubobjectDestroy() {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SubobjectDestroy() {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_two_params.carbon

package Core library "op_two_params";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_two_params.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must have exactly 1 parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SubobjectDestroy(unused ref self, unused x: Self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SubobjectDestroy(unused ref self, unused x: Self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_param_not_self.carbon

package Core library "op_param_not_self";

class C {}

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_param_not_self.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SubobjectDestroy(unused ref c: C) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SubobjectDestroy(unused ref c: C) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_param_is_generic.carbon

package Core library "op_param_is_generic";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_param_is_generic.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SubobjectDestroy[T: type](unused ref c: T) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SubobjectDestroy[T: type](unused ref c: T) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_self_param_not_reference.carbon

package Core library "op_self_param_not_reference";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_self_param_not_reference.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SubobjectDestroy(unused self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SubobjectDestroy(unused self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_explicit_self_type.carbon

package Core library "op_explicit_self_type";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_explicit_self_type.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must take `ref self` as its parameter` [SemanticsTodo]
  // CHECK:STDERR:   final fn SubobjectDestroy(unused ref s: Self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  final fn SubobjectDestroy(unused ref s: Self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_not_final.carbon

package Core library "op_not_final";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_not_final.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must be `final`` [SemanticsTodo]
  // CHECK:STDERR:   fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}

// --- fail_op_default.carbon

package Core library "op_default";

interface Destroy {
  fn Op(ref self);
  // CHECK:STDERR: fail_op_default.carbon:[[@LINE+4]]:3: error: semantics TODO: `associated function must be `final`` [SemanticsTodo]
  // CHECK:STDERR:   default fn SubobjectDestroy(unused ref self) {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  default fn SubobjectDestroy(unused ref self) {}
  final fn SelfDestruct(unused ref self) {}
}
