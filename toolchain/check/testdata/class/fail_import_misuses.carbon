// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_import_misuses.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_import_misuses.carbon

// --- a.carbon

library "a";

class Empty {
}

class Incomplete;

// --- fail_b.carbon

library "b";

import library "a";

// CHECK:STDERR: fail_b.carbon:[[@LINE+10]]:1: ERROR: Redeclaration of `class Empty` is redundant.
// CHECK:STDERR: class Empty {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_b.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import library "a";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: a.carbon:4:1: Previously declared here.
// CHECK:STDERR: class Empty {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Empty {
}

// CHECK:STDERR: fail_b.carbon:[[@LINE+9]]:8: ERROR: Variable has incomplete type `Incomplete`.
// CHECK:STDERR: var a: Incomplete;
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR: fail_b.carbon:[[@LINE-18]]:1: In import.
// CHECK:STDERR: import library "a";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: a.carbon:7:1: Class was forward declared here.
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
var a: Incomplete;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty: type = class_type @Empty [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = class_decl @Empty [template = constants.%Empty] {}
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Empty {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Empty
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty: type = class_type @Empty [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = class_type @.1 [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Main//a, inst+3, loaded [template = constants.%Empty]
// CHECK:STDOUT:   %import_ref.2: type = import_ref Main//a, inst+6, loaded [template = constants.%Incomplete]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//a, inst+4, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = imports.%import_ref.1
// CHECK:STDOUT:     .Incomplete = imports.%import_ref.2
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.2] {}
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, imports.%import_ref.2 [template = constants.%Incomplete]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Empty {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
