// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/redeclaration_introducer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/redeclaration_introducer.carbon

class A;
// CHECK:STDERR: redeclaration_introducer.carbon:[[@LINE+4]]:1: error: `base` not allowed on `class` declaration
// CHECK:STDERR: base class B;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
base class B;
// CHECK:STDERR: redeclaration_introducer.carbon:[[@LINE+4]]:1: error: `abstract` not allowed on `class` declaration
// CHECK:STDERR: abstract class C;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract class C;

class A {}
// CHECK:STDERR: redeclaration_introducer.carbon:[[@LINE+7]]:1: error: class redeclared with different inheritance kind
// CHECK:STDERR: base class B {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: redeclaration_introducer.carbon:[[@LINE-11]]:1: note: previously declared here
// CHECK:STDERR: base class B;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
base class B {}
// CHECK:STDERR: redeclaration_introducer.carbon:[[@LINE+6]]:1: error: class redeclared with different inheritance kind
// CHECK:STDERR: abstract class C {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: redeclaration_introducer.carbon:[[@LINE-14]]:1: note: previously declared here
// CHECK:STDERR: abstract class C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
abstract class C {}

// CHECK:STDOUT: --- redeclaration_introducer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl.loc11
// CHECK:STDOUT:     .B = %B.decl.loc16
// CHECK:STDOUT:     .C = %C.decl.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc11: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc16: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl.loc21: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %A.decl.loc23: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc31: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl.loc38: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc23: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.loc31: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc38: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
