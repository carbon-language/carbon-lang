// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/package_expr/syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/package_expr/syntax.carbon

// --- global.carbon

package Global;

var x: i32 = 0;
var y: i32 = package.x;

// --- inside_fn.carbon

package InsideFn;

var x: i32 = 0;

fn Main() {
  var x: i32 = 1;

  var y: i32 = package.x;
}

// --- namespace.carbon

package Namespace;

namespace NS;

class NS.C {
  fn Foo() {}
};

fn Main() {
  package.NS.C.Foo();
}

// CHECK:STDOUT: --- global.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 0 [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc4_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%.loc4_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8.2: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8.3: type = converted %int.make_type_signed.loc4, %.loc4_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %x.var: ref %i32 = var x
// CHECK:STDOUT:   %x: ref %i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc5_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8.3: type = converted %int.make_type_signed.loc5, %.loc5_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %y.var: ref %i32 = var y
// CHECK:STDOUT:   %y: ref %i32 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_14: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_15.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_15.2: <bound method> = bound_method %.loc4_14, %.loc4_15.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc4_15.3: <specific function> = specific_function %.loc4_15.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc4_15.3(%.loc4_14) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_15.4: init %i32 = converted %.loc4_14, %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   assign file.%x.var, %.loc4_15.4
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %x.ref: ref %i32 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc5: %i32 = bind_value %x.ref
// CHECK:STDOUT:   assign file.%y.var, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- inside_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 1 [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc4_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc4_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_8.3: type = converted %int.make_type_signed, %.loc4_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %x.var: ref %i32 = var x
// CHECK:STDOUT:   %x: ref %i32 = bind_name x, %x.var
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%.loc7_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10.2: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10.3: type = converted %int.make_type_signed.loc7, %.loc7_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %x.var: ref %i32 = var x
// CHECK:STDOUT:   %x: ref %i32 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc7_16: Core.IntLiteral = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc7_17.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_17.2: <bound method> = bound_method %.loc7_16, %.loc7_17.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc7_17.3: <specific function> = specific_function %.loc7_17.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc7_17.3(%.loc7_16) [template = constants.%.33]
// CHECK:STDOUT:   %.loc7_17.4: init %i32 = converted %.loc7_16, %int.convert_checked [template = constants.%.33]
// CHECK:STDOUT:   assign %x.var, %.loc7_17.4
// CHECK:STDOUT:   %.loc9_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%.loc9_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_10.2: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_10.3: type = converted %int.make_type_signed.loc9, %.loc9_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %y.var: ref %i32 = var y
// CHECK:STDOUT:   %y: ref %i32 = bind_name y, %y.var
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %x.ref: ref %i32 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc9_23: %i32 = bind_value %x.ref
// CHECK:STDOUT:   assign %y.var, %.loc9_23
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_14: Core.IntLiteral = int_value 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_15.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_15.2: <bound method> = bound_method %.loc4_14, %.loc4_15.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc4_15.3: <specific function> = specific_function %.loc4_15.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc4_15.3(%.loc4_14) [template = constants.%.29]
// CHECK:STDOUT:   %.loc4_15.4: init %i32 = converted %.loc4_14, %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   assign file.%x.var, %.loc4_15.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [template = constants.%Foo] {} {}
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Foo = %Foo.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %package.ref: <namespace> = name_ref package, package [template = package]
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %Foo.ref: %Foo.type = name_ref Foo, @C.%Foo.decl [template = constants.%Foo]
// CHECK:STDOUT:   %Foo.call: init %empty_tuple.type = call %Foo.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
