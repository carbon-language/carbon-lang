// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/choice/fail_invalid.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/choice/fail_invalid.carbon

// --- fail_no_alternative_construct.carbon
library "[[@TEST_NAME]]";

choice Never {}

fn F() {
  // TODO: Can we produce a better diagnostic?
  // CHECK:STDERR: fail_no_alternative_construct.carbon:[[@LINE+4]]:22: error: cannot initialize class with 1 field from struct with 0 fields [StructInitElementCountMismatch]
  // CHECK:STDERR:   let never: Never = {};
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR:
  let never: Never = {};
}

// CHECK:STDOUT: --- fail_no_alternative_construct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Never: type = class_type @Never [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (imports.%Core.import_ref.0b9), @impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.a28: <witness> = impl_witness %Destroy.impl_witness_table, @impl(%Never) [concrete]
// CHECK:STDOUT:   %Op.type.33d: type = fn_type @Op.2, @impl(%Never) [concrete]
// CHECK:STDOUT:   %Op.0b2: %Op.type.33d = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Never, (%Destroy.impl_witness.a28) [concrete]
// CHECK:STDOUT:   %.fb5: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.0b2, @Op.2(%Never) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Never = %Never.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Never.decl: type = class_decl @Never [concrete = constants.%Never] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Never {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.discriminant [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Never
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %never.patt: %Never = binding_pattern never
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_23.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Never.ref: type = name_ref Never, file.%Never.decl [concrete = constants.%Never]
// CHECK:STDOUT:   %impl.elem0: %.fb5 = impl_witness_access constants.%Destroy.impl_witness.a28, element0 [concrete = constants.%Op.0b2]
// CHECK:STDOUT:   %bound_method.loc11: <bound method> = bound_method %.loc11_23.2, %impl.elem0
// CHECK:STDOUT:   %.loc11_23.2: ref %Never = temporary_storage
// CHECK:STDOUT:   %.loc11_23.3: ref %Never = temporary %.loc11_23.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc11_23.4: ref %Never = converted %.loc11_23.1, %.loc11_23.3 [concrete = <error>]
// CHECK:STDOUT:   %never: ref %Never = bind_name never, %.loc11_23.4
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(constants.%Never) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %.loc11_23.2, %specific_fn
// CHECK:STDOUT:   %.loc11_23.5: %Never = bind_value %.loc11_23.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.1(%.loc11_23.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
