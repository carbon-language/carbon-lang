// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

class ForwardDecl;

// --- fail_b.carbon

library "b" api;

import library "a";

// TODO: This should probably have a valid form.
// CHECK:STDERR: fail_b.carbon:[[@LINE+9]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: class ForwardDecl {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_b.carbon:[[@LINE-6]]:1: In import.
// CHECK:STDERR: import library "a";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: a.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: class ForwardDecl;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
class ForwardDecl {
}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardDecl: type = class_type @ForwardDecl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ForwardDecl = %ForwardDecl.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDecl.decl: type = class_decl @ForwardDecl [template = constants.%ForwardDecl] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDecl;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForwardDecl: type = class_type @ForwardDecl [template]
// CHECK:STDOUT:   %.1: type = class_type @.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ForwardDecl = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: type = import_ref ir1, inst+1, used [template = constants.%ForwardDecl]
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.1] {
// CHECK:STDOUT:     %ForwardDecl.decl: invalid = class_decl @ForwardDecl [template = constants.%ForwardDecl] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ForwardDecl;
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
