// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-sem-ir %s
//
// Check that the command-line flag to dump textual IR works.
//
// AUTOUPDATE

fn Foo(n: ()) -> ((), ()) {
  return (n, ());
}

// CHECK:STDOUT: --- textual_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = tuple_type ((), ()) [template]
// CHECK:STDOUT:   %.3: type = ptr_type ((), ()) [template]
// CHECK:STDOUT:   %.4: () = tuple_value () [template]
// CHECK:STDOUT:   %.5: ((), ()) = tuple_value (%.4, %.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Foo = %Foo
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo: <function> = fn_decl @Foo [template] {
// CHECK:STDOUT:     %.loc11_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_12.2: type = converted %.loc11_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %n.loc11_8.1: () = param n
// CHECK:STDOUT:     @Foo.%n: () = bind_name n, %n.loc11_8.1
// CHECK:STDOUT:     %.loc11_20.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_24.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc11_25.1: ((), ()) = tuple_literal (%.loc11_20.1, %.loc11_24.1)
// CHECK:STDOUT:     %.loc11_20.2: type = converted %.loc11_20.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_24.2: type = converted %.loc11_24.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %.loc11_25.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     @Foo.%return: ref ((), ()) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n: ()) -> %return: ((), ()) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: () = name_ref n, %n
// CHECK:STDOUT:   %.loc12_15.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_16.1: ((), ()) = tuple_literal (%n.ref, %.loc12_15.1)
// CHECK:STDOUT:   %.loc12_16.2: ref () = tuple_access %return, element0
// CHECK:STDOUT:   %.loc12_11.1: init () = tuple_init () to %.loc12_16.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_11.2: init () = converted %n.ref, %.loc12_11.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_16.3: ref () = tuple_access %return, element1
// CHECK:STDOUT:   %.loc12_15.2: init () = tuple_init () to %.loc12_16.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_15.3: init () = converted %.loc12_15.1, %.loc12_15.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_16.4: init ((), ()) = tuple_init (%.loc12_11.2, %.loc12_15.3) to %return [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_16.5: init ((), ()) = converted %.loc12_16.1, %.loc12_16.4 [template = constants.%.5]
// CHECK:STDOUT:   return %.loc12_16.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
