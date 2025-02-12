// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/alias_of_alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/alias_of_alias.carbon

class C {}
alias a = C;
alias b = a;
alias c = b;
let d: c = {};

// CHECK:STDOUT: --- alias_of_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a: type = bind_alias a, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [concrete = constants.%C]
// CHECK:STDOUT:   %b: type = bind_alias b, %a [concrete = constants.%C]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [concrete = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %b [concrete = constants.%C]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %C = binding_pattern d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [concrete = constants.%C]
// CHECK:STDOUT:   %.loc15_13.1: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc15_13.2: init %C = class_init (), %.loc15_13.1 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc15_13.3: ref %C = temporary %.loc15_13.1, %.loc15_13.2
// CHECK:STDOUT:   %.loc15_13.4: ref %C = converted @__global_init.%.loc15, %.loc15_13.3
// CHECK:STDOUT:   %d: ref %C = bind_name d, %.loc15_13.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
