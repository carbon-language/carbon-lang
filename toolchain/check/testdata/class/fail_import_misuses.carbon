// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

class Empty {
}

class Incomplete;

// --- fail_b.carbon

library "b" api;

import library "a";

// CHECK:STDERR: fail_b.carbon:[[@LINE+10]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class Empty {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_b.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import library "a";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: a.carbon:4:1: Name is previously declared here.
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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Empty = %import_ref.1
// CHECK:STDOUT:     .Incomplete = %import_ref.2
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = constants.%Empty]
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir1, inst+4, used [template = constants.%Incomplete]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+2, unused
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.2] {
// CHECK:STDOUT:     %Empty.decl: invalid = class_decl @Empty [template = constants.%Empty] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: invalid = class_decl @Incomplete [template = constants.%Incomplete] {}
// CHECK:STDOUT:   %Incomplete.ref: type = name_ref Incomplete, %import_ref.2 [template = constants.%Incomplete]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Empty {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
