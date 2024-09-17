// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/in_namespace.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/in_namespace.carbon

class C { var v: (); }

namespace NS;
alias NS.a = C;

let b: NS.a = {.v = ()};

fn F() -> NS.a {
  return {.v = ()};
}

// CHECK:STDOUT: --- in_namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, %.1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.v: %.1} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%tuple) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %NS.ref.loc16: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %a.ref.loc16: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %NS.ref.loc18: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:     %a.ref.loc18: type = name_ref a, %a [template = constants.%C]
// CHECK:STDOUT:     @F.%return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc11_19.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc11_19.2: type = converted %.loc11_19.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_16: %.2 = field_decl v, element0 [template]
// CHECK:STDOUT:   %.loc11_22: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .v = %.loc11_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_17.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc19_18.1: %.3 = struct_literal (%.loc19_17.1)
// CHECK:STDOUT:   %.loc19_18.2: ref %.1 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc19_17.2: init %.1 = tuple_init () to %.loc19_18.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc19_18.3: init %.1 = converted %.loc19_17.1, %.loc19_17.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc19_18.4: init %C = class_init (%.loc19_18.3), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_19: init %C = converted %.loc19_18.1, %.loc19_18.4 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc19_19 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_22.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc16_23.1: %.3 = struct_literal (%.loc16_22.1)
// CHECK:STDOUT:   %.loc16_23.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc16_23.3: ref %.1 = class_element_access %.loc16_23.2, element0
// CHECK:STDOUT:   %.loc16_22.2: init %.1 = tuple_init () to %.loc16_23.3 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc16_23.4: init %.1 = converted %.loc16_22.1, %.loc16_22.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc16_23.5: init %C = class_init (%.loc16_23.4), %.loc16_23.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc16_23.6: ref %C = temporary %.loc16_23.2, %.loc16_23.5
// CHECK:STDOUT:   %.loc16_24.1: ref %C = converted %.loc16_23.1, %.loc16_23.6
// CHECK:STDOUT:   %.loc16_24.2: %C = bind_value %.loc16_24.1
// CHECK:STDOUT:   %b: %C = bind_name b, %.loc16_24.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
