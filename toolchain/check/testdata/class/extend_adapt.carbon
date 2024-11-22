// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/extend_adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/extend_adapt.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class SomeClassAdapter;

class SomeClass {
  var a: i32;
  var b: i32;

  fn StaticMemberFunction();

  fn AdapterMethod[self: SomeClassAdapter]();
}

class SomeClassAdapter {
  extend adapt SomeClass;
}

fn TestStaticMemberFunction(a: SomeClassAdapter) {
  a.StaticMemberFunction();
}

fn TestAdapterMethod(a: SomeClassAdapter) {
  a.AdapterMethod();
}

// --- fail_todo_method_access.carbon

library "[[@TEST_NAME]]";

class SomeClass {
  fn F[self: Self]();
}

class SomeClassAdapter {
  extend adapt SomeClass;
}

fn F(a: SomeClassAdapter) {
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE+10]]:3: error: cannot implicitly convert from `SomeClassAdapter` to `SomeClass` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE+7]]:3: note: type `SomeClassAdapter` does not implement interface `ImplicitAs(SomeClass)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE-14]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[self: Self]();
  // CHECK:STDERR:        ^~~~~~~~~~
  // CHECK:STDERR:
  a.F();
}

// --- fail_todo_field_access.carbon

library "[[@TEST_NAME]]";

class SomeClass {
  var a: i32;
  var b: i32;
}

class SomeClassAdapter {
  extend adapt SomeClass;
}

fn F(a: SomeClassAdapter) -> i32 {
  // CHECK:STDERR: fail_todo_field_access.carbon:[[@LINE+7]]:10: error: cannot implicitly convert from `SomeClassAdapter` to `SomeClass` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   return a.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_field_access.carbon:[[@LINE+4]]:10: note: type `SomeClassAdapter` does not implement interface `ImplicitAs(SomeClass)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return a.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.b;
}

// --- fail_todo_adapt_struct.carbon

library "[[@TEST_NAME]]";

class StructAdapter {
  extend adapt {.a: i32, .b: i32};
}

fn F(a: StructAdapter) -> i32 {
  // TODO: This should be allowed.
  // CHECK:STDERR: fail_todo_adapt_struct.carbon:[[@LINE+4]]:10: error: name `b` not found [NameNotFound]
  // CHECK:STDERR:   return a.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.b;
}

// --- fail_todo_adapt_tuple.carbon

library "[[@TEST_NAME]]";

class TupleAdapter {
  extend adapt (i32, i32);
}

fn F(a: TupleAdapter) -> i32 {
  // TODO: This should be allowed.
  // CHECK:STDERR: fail_todo_adapt_tuple.carbon:[[@LINE+4]]:10: error: type `TupleAdapter` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
  // CHECK:STDERR:   return a.1;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.1;
}

// --- fail_adapt_builtin.carbon

library "[[@TEST_NAME]]";

fn MakeInt(N: Core.IntLiteral()) -> type = "int.make_type_signed";

class IntAdapter {
  extend adapt MakeInt(32);
}

fn F(a: IntAdapter) -> i32 {
  // Builtin types have no member names.
  // CHECK:STDERR: fail_adapt_builtin.carbon:[[@LINE+3]]:10: error: name `foo` not found [NameNotFound]
  // CHECK:STDERR:   return a.foo;
  // CHECK:STDERR:          ^~~~~
  return a.foo;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %SomeClass, %i32 [template]
// CHECK:STDOUT:   %StaticMemberFunction.type: type = fn_type @StaticMemberFunction [template]
// CHECK:STDOUT:   %StaticMemberFunction: %StaticMemberFunction.type = struct_value () [template]
// CHECK:STDOUT:   %AdapterMethod.type: type = fn_type @AdapterMethod [template]
// CHECK:STDOUT:   %AdapterMethod: %AdapterMethod.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %SomeClass [template]
// CHECK:STDOUT:   %TestStaticMemberFunction.type: type = fn_type @TestStaticMemberFunction [template]
// CHECK:STDOUT:   %TestStaticMemberFunction: %TestStaticMemberFunction.type = struct_value () [template]
// CHECK:STDOUT:   %TestAdapterMethod.type: type = fn_type @TestAdapterMethod [template]
// CHECK:STDOUT:   %TestAdapterMethod: %TestAdapterMethod.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl.loc4
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .TestStaticMemberFunction = %TestStaticMemberFunction.decl
// CHECK:STDOUT:     .TestAdapterMethod = %TestAdapterMethod.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SomeClassAdapter.decl.loc4: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [template = constants.%SomeClass] {} {}
// CHECK:STDOUT:   %SomeClassAdapter.decl.loc15: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %TestStaticMemberFunction.decl: %TestStaticMemberFunction.type = fn_decl @TestStaticMemberFunction [template = constants.%TestStaticMemberFunction] {
// CHECK:STDOUT:     %a.patt: %SomeClassAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %SomeClassAdapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.param: %SomeClassAdapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %SomeClassAdapter = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAdapterMethod.decl: %TestAdapterMethod.type = fn_decl @TestAdapterMethod [template = constants.%TestAdapterMethod] {
// CHECK:STDOUT:     %a.patt: %SomeClassAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %SomeClassAdapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.param: %SomeClassAdapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %SomeClassAdapter = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %SomeClass [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend %SomeClass.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %.loc7_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%.loc7_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10.2: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_10.3: type = converted %int.make_type_signed.loc7, %.loc7_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc8_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%.loc8_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.2: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_10.3: type = converted %int.make_type_signed.loc8, %.loc8_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %StaticMemberFunction.decl: %StaticMemberFunction.type = fn_decl @StaticMemberFunction [template = constants.%StaticMemberFunction] {} {}
// CHECK:STDOUT:   %AdapterMethod.decl: %AdapterMethod.type = fn_decl @AdapterMethod [template = constants.%AdapterMethod] {
// CHECK:STDOUT:     %self.patt: %SomeClassAdapter = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %SomeClassAdapter = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %self.param: %SomeClassAdapter = value_param runtime_param0
// CHECK:STDOUT:     %self: %SomeClassAdapter = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc7_8
// CHECK:STDOUT:   .b = %.loc8_8
// CHECK:STDOUT:   .StaticMemberFunction = %StaticMemberFunction.decl
// CHECK:STDOUT:   .AdapterMethod = %AdapterMethod.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @StaticMemberFunction();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdapterMethod[%self.param_patt: %SomeClassAdapter]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStaticMemberFunction(%a.param_patt: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %StaticMemberFunction.ref: %StaticMemberFunction.type = name_ref StaticMemberFunction, @SomeClass.%StaticMemberFunction.decl [template = constants.%StaticMemberFunction]
// CHECK:STDOUT:   %StaticMemberFunction.call: init %empty_tuple.type = call %StaticMemberFunction.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdapterMethod(%a.param_patt: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %AdapterMethod.ref: %AdapterMethod.type = name_ref AdapterMethod, @SomeClass.%AdapterMethod.decl [template = constants.%AdapterMethod]
// CHECK:STDOUT:   %.loc24: <bound method> = bound_method %a.ref, %AdapterMethod.ref
// CHECK:STDOUT:   %AdapterMethod.call: init %empty_tuple.type = call %.loc24(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_method_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %SomeClass [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [template = constants.%SomeClass] {} {}
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %a.patt: %SomeClassAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %SomeClassAdapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.param: %SomeClassAdapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %SomeClassAdapter = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %SomeClass = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %SomeClass = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SomeClass [template = constants.%SomeClass]
// CHECK:STDOUT:     %self.param: %SomeClass = value_param runtime_param0
// CHECK:STDOUT:     %self: %SomeClass = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %SomeClass [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend %SomeClass.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[%self.param_patt: %SomeClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a.param_patt: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @SomeClass.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc23_4: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc23_3: %SomeClass = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %.loc23_4(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %SomeClass, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %SomeClass [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
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
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [template = constants.%SomeClass] {} {}
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %SomeClassAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %SomeClassAdapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %.loc13_30.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc13_30.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_30.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_30.3: type = converted %int.make_type_signed, %.loc13_30.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %SomeClassAdapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %SomeClassAdapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %.loc5_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_10.3: type = converted %int.make_type_signed.loc5, %.loc5_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.2: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_10.3: type = converted %int.make_type_signed.loc6, %.loc6_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc7: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .b = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %SomeClass [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend %SomeClass.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %SomeClassAdapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %.2 = name_ref b, @SomeClass.%.loc6_8 [template = @SomeClass.%.loc6_8]
// CHECK:STDOUT:   %.loc21_11.1: %SomeClass = converted %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc21_11.2: %i32 = class_element_access <error>, element1 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adapt_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .StructAdapter = %StructAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %StructAdapter.decl: type = class_decl @StructAdapter [template = constants.%StructAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %StructAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %StructAdapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %StructAdapter.ref: type = name_ref StructAdapter, file.%StructAdapter.decl [template = constants.%StructAdapter]
// CHECK:STDOUT:     %.loc8_27.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc8_27.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_27.3: type = converted %int.make_type_signed, %.loc8_27.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %StructAdapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %StructAdapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %.loc5_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_21: init type = call constants.%Int(%.loc5_21.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_21.2: type = value_of_initializer %int.make_type_signed.loc5_21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_21.3: type = converted %int.make_type_signed.loc5_21, %.loc5_21.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_30.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_30: init type = call constants.%Int(%.loc5_30.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_30.2: type = value_of_initializer %int.make_type_signed.loc5_30 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_30.3: type = converted %int.make_type_signed.loc5_30, %.loc5_30.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_33: type = struct_type {.a: %i32, .b: %i32} [template = constants.%.2]
// CHECK:STDOUT:   adapt_decl %.2
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.2 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT:   extend %.loc5_33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %StructAdapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %StructAdapter = name_ref a, %a
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adapt_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TupleAdapter: type = class_type @TupleAdapter [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %tuple.type.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TupleAdapter = %TupleAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TupleAdapter.decl: type = class_decl @TupleAdapter [template = constants.%TupleAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %TupleAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %TupleAdapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %TupleAdapter.ref: type = name_ref TupleAdapter, file.%TupleAdapter.decl [template = constants.%TupleAdapter]
// CHECK:STDOUT:     %.loc8_26.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc8_26.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_26.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_26.3: type = converted %int.make_type_signed, %.loc8_26.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %TupleAdapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %TupleAdapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @TupleAdapter {
// CHECK:STDOUT:   %.loc5_17: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_17: init type = call constants.%Int(%.loc5_17) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5_22: init type = call constants.%Int(%.loc5_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_25: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc5_17, %int.make_type_signed.loc5_22)
// CHECK:STDOUT:   %.loc5_26.1: type = value_of_initializer %int.make_type_signed.loc5_17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_26.2: type = converted %int.make_type_signed.loc5_17, %.loc5_26.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_26.3: type = value_of_initializer %int.make_type_signed.loc5_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_26.4: type = converted %int.make_type_signed.loc5_22, %.loc5_26.3 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_26.5: type = converted %.loc5_25, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   adapt_decl %tuple.type.2
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %tuple.type.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%TupleAdapter
// CHECK:STDOUT:   extend %.loc5_26.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %TupleAdapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %TupleAdapter = name_ref a, %a
// CHECK:STDOUT:   %.loc14: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %MakeInt.type: type = fn_type @MakeInt [template]
// CHECK:STDOUT:   %MakeInt: %MakeInt.type = struct_value () [template]
// CHECK:STDOUT:   %IntAdapter: type = class_type @IntAdapter [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %i32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types, inst+7, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .MakeInt = %MakeInt.decl
// CHECK:STDOUT:     .IntAdapter = %IntAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %MakeInt.decl: %MakeInt.type = fn_decl @MakeInt [template = constants.%MakeInt] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_31.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_31.2: type = converted %int_literal.make_type, %.loc4_31.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntAdapter.decl: type = class_decl @IntAdapter [template = constants.%IntAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: %IntAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %IntAdapter = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntAdapter.ref: type = name_ref IntAdapter, file.%IntAdapter.decl [template = constants.%IntAdapter]
// CHECK:STDOUT:     %.loc10_24.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc10_24.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_24.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_24.3: type = converted %int.make_type_signed, %.loc10_24.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %IntAdapter = value_param runtime_param0
// CHECK:STDOUT:     %a: %IntAdapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @IntAdapter {
// CHECK:STDOUT:   %MakeInt.ref: %MakeInt.type = name_ref MakeInt, file.%MakeInt.decl [template = constants.%MakeInt]
// CHECK:STDOUT:   %.loc7_24: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %MakeInt.ref(%.loc7_24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_27.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_27.2: type = converted %int.make_type_signed, %.loc7_27.1 [template = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %i32 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%IntAdapter
// CHECK:STDOUT:   extend %.loc7_27.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeInt(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %IntAdapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %IntAdapter = name_ref a, %a
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
