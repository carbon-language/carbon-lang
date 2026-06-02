// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self/fail_self_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self/fail_self_param.carbon

// --- fail_self_param.carbon
library "[[@TEST_NAME]]";

// `self` is not a valid parameter of a class declaration even when placed in
// the explicit parameter list.
// CHECK:STDERR: fail_self_param.carbon:[[@LINE+4]]:9: error: `self` parameter only allowed on functions [SelfParameterNotAllowed]
// CHECK:STDERR: class C(self:! type, x:! self) {}
// CHECK:STDERR:         ^~~~~~~~~~~
// CHECK:STDERR:
class C(self:! type, x:! self) {}
// CHECK:STDERR: fail_self_param.carbon:[[@LINE+7]]:8: error: 1 argument passed to generic class expecting 2 arguments [CallArgCountMismatch]
// CHECK:STDERR: var v: C(());
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR: fail_self_param.carbon:[[@LINE-4]]:1: note: calling generic class declared here [InCallToEntity]
// CHECK:STDERR: class C(self:! type, x:! self) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var v: C(());

// --- fail_self_param_implicit.carbon
library "[[@TEST_NAME]]";

// `self` in the implicit parameter list of a class declaration is reported as
// both being in the wrong list and not allowed on a class at all.
// CHECK:STDERR: fail_self_param_implicit.carbon:[[@LINE+8]]:9: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
// CHECK:STDERR: class C[self:! type](x:! self) {}
// CHECK:STDERR:         ^~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_self_param_implicit.carbon:[[@LINE+4]]:9: error: `self` parameter only allowed on functions [SelfParameterNotAllowed]
// CHECK:STDERR: class C[self:! type](x:! self) {}
// CHECK:STDERR:         ^~~~~~~~~~~
// CHECK:STDERR:
class C[self:! type](x:! self) {}
var v: C(());
