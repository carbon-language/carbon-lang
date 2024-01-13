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
// CHECK:STDOUT:   package: <namespace> = namespace package, {.A = %.loc4}
// CHECK:STDOUT:   %.loc4: <namespace> = namespace A, {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.A = %.2}
// CHECK:STDOUT:   %.2: <namespace> = namespace A, {.B = %.loc5}
// CHECK:STDOUT:   %.loc5: <namespace> = namespace B, {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.A = %.2}
// CHECK:STDOUT:   %.2: <namespace> = namespace A, {.B = %.3}
// CHECK:STDOUT:   %.3: <namespace> = namespace B, {.C = %.loc5}
// CHECK:STDOUT:   %.loc5: <namespace> = namespace C, {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- d.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.A = %.2}
// CHECK:STDOUT:   %.2: <namespace> = namespace A, {.B = %.3}
// CHECK:STDOUT:   %.3: <namespace> = namespace B, {.C = %.4}
// CHECK:STDOUT:   %.4: <namespace> = namespace C, {.D = %D}
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
// CHECK:STDOUT:   package: <namespace> = namespace package, {.A = %.2, .e = %e}
// CHECK:STDOUT:   %.2: <namespace> = namespace A, {.B = %.3}
// CHECK:STDOUT:   %.3: <namespace> = namespace B, {.C = %.4}
// CHECK:STDOUT:   %.4: <namespace> = namespace C, {.D = %.5}
// CHECK:STDOUT:   %.5: <function> = fn_decl @.1 [template]
// CHECK:STDOUT:   %.loc5_9: () = tuple_literal ()
// CHECK:STDOUT:   %.6: type = converted %.loc5_9, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT:   %A.ref: <namespace> = name_ref A, %.2
// CHECK:STDOUT:   %B.ref: <namespace> = name_ref B, %.3
// CHECK:STDOUT:   %C.ref: <namespace> = name_ref C, %.4
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, %.5 [template = %.5]
// CHECK:STDOUT:   %.loc5_20: init () = call %D.ref()
// CHECK:STDOUT:   assign %e.var, %.loc5_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
