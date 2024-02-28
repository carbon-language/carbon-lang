// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

package Same library "a" api;

namespace A;

// --- b.carbon

package Same library "b" api;
import library "a";

namespace A.B;

// --- c.carbon

package Same library "c" api;
import library "b";

namespace A.B.C;

// --- d.carbon

package Same library "d" api;
import library "c";

fn A.B.C.D() {};

// --- e.carbon

package Same library "e" api;
import library "d";

var e: () = A.B.C.D();

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .A = %.loc4
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace {
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .A = %.2
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %.2: <namespace> = namespace %import_ref, {
// CHECK:STDOUT:     .B = %.loc5
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %.loc5: <namespace> = namespace {
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .A = %.2
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+2, used
// CHECK:STDOUT:   %.2: <namespace> = namespace %import_ref.1, {
// CHECK:STDOUT:     .B = %.3
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+3, used
// CHECK:STDOUT:   %.3: <namespace> = namespace %import_ref.2, {
// CHECK:STDOUT:     .C = %.loc5
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %.loc5: <namespace> = namespace {
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- d.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .A = %.2
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+2, used
// CHECK:STDOUT:   %.2: <namespace> = namespace %import_ref.1, {
// CHECK:STDOUT:     .B = %.3
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+4, used
// CHECK:STDOUT:   %.3: <namespace> = namespace %import_ref.2, {
// CHECK:STDOUT:     .C = %.4
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.3: <namespace> = import_ref ir1, inst+5, used
// CHECK:STDOUT:   %.4: <namespace> = namespace %import_ref.3, {
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %D: <function> = fn_decl @D [template]
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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .A = %.2
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.1: <namespace> = import_ref ir1, inst+2, used
// CHECK:STDOUT:   %.2: <namespace> = namespace %import_ref.1, {
// CHECK:STDOUT:     .B = %.3
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.2: <namespace> = import_ref ir1, inst+4, used
// CHECK:STDOUT:   %.3: <namespace> = namespace %import_ref.2, {
// CHECK:STDOUT:     .C = %.4
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.3: <namespace> = import_ref ir1, inst+6, used
// CHECK:STDOUT:   %.4: <namespace> = namespace %import_ref.3, {
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+7, used [template = imports.%D]
// CHECK:STDOUT:   %.loc5_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc5_9.2: type = converted %.loc5_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <namespace> = name_ref A, file.%.2 [template = file.%.2]
// CHECK:STDOUT:   %B.ref: <namespace> = name_ref B, file.%.3 [template = file.%.3]
// CHECK:STDOUT:   %C.ref: <namespace> = name_ref C, file.%.4 [template = file.%.4]
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D]
// CHECK:STDOUT:   %.loc5: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%e.var, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
