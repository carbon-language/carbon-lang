// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/imported_indirect.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/imported_indirect.carbon

// --- a.carbon

package Same library "[[@TEST_NAME]]";

namespace A;

class A.C;

// --- b.carbon

package Same library "[[@TEST_NAME]]";
import library "a";

namespace A.B;

fn F() -> A.C;

// --- c.carbon

package Same library "[[@TEST_NAME]]";
import library "b";

namespace A.B.C;

// --- d.carbon

package Same library "[[@TEST_NAME]]";
import library "c";

fn A.B.C.D() {};

// --- e.carbon

package Same library "[[@TEST_NAME]]";
import library "d";

var e: () = A.B.C.D();

// --- fail_named_indirectly_same_package.carbon

package Same library "[[@TEST_NAME]]";

import library "b";

// CHECK:STDERR: fail_named_indirectly_same_package.carbon:[[@LINE+13]]:10: error: function returns incomplete type `A.C` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn G() { F(); }
// CHECK:STDERR:          ^~~
// CHECK:STDERR: fail_named_indirectly_same_package.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: b.carbon:3:1: in import [InImport]
// CHECK:STDERR: a.carbon:6:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class A.C;
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_named_indirectly_same_package.carbon:[[@LINE-10]]:1: in import [InImport]
// CHECK:STDERR: b.carbon:7:8: note: return type declared here [IncompleteReturnTypeHere]
// CHECK:STDERR: fn F() -> A.C;
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR:
fn G() { F(); }

// --- fail_named_indirectly_different_package.carbon

package Other library "[[@TEST_NAME]]";

import Same library "b";

// CHECK:STDERR: fail_named_indirectly_different_package.carbon:[[@LINE+13]]:10: error: function returns incomplete type `Same.A.C` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn G() { Same.F(); }
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR: fail_named_indirectly_different_package.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: b.carbon:3:1: in import [InImport]
// CHECK:STDERR: a.carbon:6:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class A.C;
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_named_indirectly_different_package.carbon:[[@LINE-10]]:1: in import [InImport]
// CHECK:STDERR: b.carbon:7:8: note: return type declared here [IncompleteReturnTypeHere]
// CHECK:STDERR: fn F() -> A.C;
// CHECK:STDERR:        ^~~~~~
// CHECK:STDERR:
fn G() { Same.F(); }

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Same.A: <namespace> = import_ref Same//a, A, loaded
// CHECK:STDOUT:   %A: <namespace> = namespace %Same.A, [concrete] {
// CHECK:STDOUT:     .C = %Same.C
// CHECK:STDOUT:     .B = file.%B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same.C: type = import_ref Same//a, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%A
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %B: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: <namespace> = name_ref A, imports.%A [concrete = imports.%A]
// CHECK:STDOUT:     %C.ref: type = name_ref C, imports.%Same.C [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "a.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C;
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Same.A: <namespace> = import_ref Same//b, A, loaded
// CHECK:STDOUT:   %A: <namespace> = namespace %Same.A, [concrete] {
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same.F = import_ref Same//b, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%A
// CHECK:STDOUT:     .F = imports.%Same.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- d.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Same.A: <namespace> = import_ref Same//c, A, loaded
// CHECK:STDOUT:   %A: <namespace> = namespace %Same.A, [concrete] {
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%A
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- e.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Same.A: <namespace> = import_ref Same//d, A, loaded
// CHECK:STDOUT:   %A: <namespace> = namespace %Same.A, [concrete] {
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same.B: <namespace> = import_ref Same//d, B, loaded
// CHECK:STDOUT:   %B: <namespace> = namespace %Same.B, [concrete] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same.C: <namespace> = import_ref Same//d, C, loaded
// CHECK:STDOUT:   %C: <namespace> = namespace %Same.C, [concrete] {
// CHECK:STDOUT:     .D = %Same.D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same.D: %D.type = import_ref Same//d, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%A
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %empty_tuple.type = binding_pattern e
// CHECK:STDOUT:     %.loc5_1: %empty_tuple.type = var_pattern %e.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %.loc5_9.1: type = splice_block %.loc5_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_9.3: type = converted %.loc5_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() [from "d.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <namespace> = name_ref A, imports.%A [concrete = imports.%A]
// CHECK:STDOUT:   %B.ref: <namespace> = name_ref B, imports.%B [concrete = imports.%B]
// CHECK:STDOUT:   %C.ref: <namespace> = name_ref C, imports.%C [concrete = imports.%C]
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%Same.D [concrete = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%e.var, %D.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_named_indirectly_same_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Same.A: <namespace> = import_ref Same//b, A, loaded
// CHECK:STDOUT:   %A: <namespace> = namespace %Same.A, [concrete] {
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same.F: %F.type = import_ref Same//b, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%A
// CHECK:STDOUT:     .F = imports.%Same.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Same.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init <error> = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C [from "b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_named_indirectly_different_package.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same: <namespace> = namespace file.%Same.import, [concrete] {
// CHECK:STDOUT:     .F = %Same.F
// CHECK:STDOUT:     import Same//b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Same.F: %F.type = import_ref Same//b, F, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Same = imports.%Same
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Same.import = import Same
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C [from "b.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Same.ref: <namespace> = name_ref Same, imports.%Same [concrete = imports.%Same]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Same.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init <error> = call %F.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %C [from "b.carbon"];
// CHECK:STDOUT:
