// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_decl_in_alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_decl_in_alias.carbon

namespace NS;

alias ns = NS;

// Aliases can't be used when declaring names.
// CHECK:STDERR: fail_decl_in_alias.carbon:[[@LINE+6]]:4: error: name qualifiers are only allowed for entities that provide a scope [QualifiedNameInNonScope]
// CHECK:STDERR: fn ns.A() -> i32 { return 0; }
// CHECK:STDERR:    ^~
// CHECK:STDERR: fail_decl_in_alias.carbon:[[@LINE-6]]:7: note: referenced non-scope entity declared here [QualifiedNameNonScopeEntity]
// CHECK:STDERR: alias ns = NS;
// CHECK:STDERR:       ^~
fn ns.A() -> i32 { return 0; }

// CHECK:STDOUT: --- fail_decl_in_alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.2: %.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .ns = %ns
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, %NS [template = %NS]
// CHECK:STDOUT:   %ns: <namespace> = bind_alias ns, %NS [template = %NS]
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.2] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc22_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc22_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_14.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc22_14.3: type = converted %int.make_type_signed, %.loc22_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_27: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc22_28.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc22_28.2: <bound method> = bound_method %.loc22_27, %.loc22_28.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc22_28.3: <specific function> = specific_function %.loc22_28.2, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc22_28.3(%.loc22_27) [template = constants.%.30]
// CHECK:STDOUT:   %.loc22_28.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.30]
// CHECK:STDOUT:   %.loc22_28.5: %i32 = converted %.loc22_27, %.loc22_28.4 [template = constants.%.30]
// CHECK:STDOUT:   return %.loc22_28.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
