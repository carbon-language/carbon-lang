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

library "basic";

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

library "fail_todo_method_access";

class SomeClass {
  fn F[self: Self]();
}

class SomeClassAdapter {
  extend adapt SomeClass;
}

fn F(a: SomeClassAdapter) {
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE+7]]:3: ERROR: Cannot implicitly convert from `SomeClassAdapter` to `SomeClass`.
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE-11]]:8: Initializing `self` parameter of method declared here.
  // CHECK:STDERR:   fn F[self: Self]();
  // CHECK:STDERR:        ^~~~
  // CHECK:STDERR:
  a.F();
}

// --- fail_todo_field_access.carbon

library "fail_todo_field_access";

class SomeClass {
  var a: i32;
  var b: i32;
}

class SomeClassAdapter {
  extend adapt SomeClass;
}

fn F(a: SomeClassAdapter) -> i32 {
  // CHECK:STDERR: fail_todo_field_access.carbon:[[@LINE+4]]:10: ERROR: Cannot implicitly convert from `SomeClassAdapter` to `SomeClass`.
  // CHECK:STDERR:   return a.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR:
  return a.b;
}

// --- fail_todo_adapt_non_class.carbon

library "fail_todo_adapt_non_class";

class StructAdapter {
  // CHECK:STDERR: fail_todo_adapt_non_class.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `extending non-class type`.
  // CHECK:STDERR:   extend adapt {.a: i32, .b: i32};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  extend adapt {.a: i32, .b: i32};
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %SomeClass, i32 [template]
// CHECK:STDOUT:   %StaticMemberFunction.type: type = fn_type @StaticMemberFunction [template]
// CHECK:STDOUT:   %StaticMemberFunction: %StaticMemberFunction.type = struct_value () [template]
// CHECK:STDOUT:   %AdapterMethod.type: type = fn_type @AdapterMethod [template]
// CHECK:STDOUT:   %AdapterMethod: %AdapterMethod.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %TestStaticMemberFunction.type: type = fn_type @TestStaticMemberFunction [template]
// CHECK:STDOUT:   %TestStaticMemberFunction: %TestStaticMemberFunction.type = struct_value () [template]
// CHECK:STDOUT:   %TestAdapterMethod.type: type = fn_type @TestAdapterMethod [template]
// CHECK:STDOUT:   %TestAdapterMethod: %TestAdapterMethod.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl.loc4
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .TestStaticMemberFunction = %TestStaticMemberFunction.decl
// CHECK:STDOUT:     .TestAdapterMethod = %TestAdapterMethod.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %SomeClassAdapter.decl.loc4: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {}
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [template = constants.%SomeClass] {}
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %SomeClassAdapter.decl.loc15: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {}
// CHECK:STDOUT:   %TestStaticMemberFunction.decl: %TestStaticMemberFunction.type = fn_decl @TestStaticMemberFunction [template = constants.%TestStaticMemberFunction] {
// CHECK:STDOUT:     %SomeClassAdapter.ref.loc19: type = name_ref SomeClassAdapter, %SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.loc19_29.1: %SomeClassAdapter = param a
// CHECK:STDOUT:     @TestStaticMemberFunction.%a: %SomeClassAdapter = bind_runtime_name a, %a.loc19_29.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAdapterMethod.decl: %TestAdapterMethod.type = fn_decl @TestAdapterMethod [template = constants.%TestAdapterMethod] {
// CHECK:STDOUT:     %SomeClassAdapter.ref.loc23: type = name_ref SomeClassAdapter, %SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.loc23_22.1: %SomeClassAdapter = param a
// CHECK:STDOUT:     @TestAdapterMethod.%a: %SomeClassAdapter = bind_runtime_name a, %a.loc23_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_10.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_10.2: type = converted %int.make_type_32.loc7, %.loc7_10.1 [template = i32]
// CHECK:STDOUT:   %.loc7_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_10.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_10.2: type = converted %int.make_type_32.loc8, %.loc8_10.1 [template = i32]
// CHECK:STDOUT:   %.loc8_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %StaticMemberFunction.decl: %StaticMemberFunction.type = fn_decl @StaticMemberFunction [template = constants.%StaticMemberFunction] {}
// CHECK:STDOUT:   %AdapterMethod.decl: %AdapterMethod.type = fn_decl @AdapterMethod [template = constants.%AdapterMethod] {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %self.loc12_20.1: %SomeClassAdapter = param self
// CHECK:STDOUT:     %self.loc12_20.2: %SomeClassAdapter = bind_runtime_name self, %self.loc12_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc7_8
// CHECK:STDOUT:   .b = %.loc8_8
// CHECK:STDOUT:   .StaticMemberFunction = %StaticMemberFunction.decl
// CHECK:STDOUT:   .AdapterMethod = %AdapterMethod.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @StaticMemberFunction();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdapterMethod[@SomeClass.%self.loc12_20.2: %SomeClassAdapter]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStaticMemberFunction(%a: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %StaticMemberFunction.ref: %StaticMemberFunction.type = name_ref StaticMemberFunction, @SomeClass.%StaticMemberFunction.decl [template = constants.%StaticMemberFunction]
// CHECK:STDOUT:   %StaticMemberFunction.call: init %.1 = call %StaticMemberFunction.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdapterMethod(%a: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %AdapterMethod.ref: %AdapterMethod.type = name_ref AdapterMethod, @SomeClass.%AdapterMethod.decl [template = constants.%AdapterMethod]
// CHECK:STDOUT:   %.loc24: <bound method> = bound_method %a.ref, %AdapterMethod.ref
// CHECK:STDOUT:   %AdapterMethod.call: init %.1 = call %.loc24(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_method_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [template = constants.%SomeClass] {}
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, %SomeClassAdapter.decl [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.loc12_6.1: %SomeClassAdapter = param a
// CHECK:STDOUT:     @F.2.%a: %SomeClassAdapter = bind_runtime_name a, %a.loc12_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SomeClass [template = constants.%SomeClass]
// CHECK:STDOUT:     %self.loc5_8.1: %SomeClass = param self
// CHECK:STDOUT:     %self.loc5_8.2: %SomeClass = bind_runtime_name self, %self.loc5_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[@SomeClass.%self.loc5_8.2: %SomeClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @SomeClass.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc20: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %F.call: init %.1 = call %.loc20(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %SomeClass, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [template = constants.%SomeClass] {}
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {}
// CHECK:STDOUT:   %import_ref.3: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, %SomeClassAdapter.decl [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.loc13_6.1: %SomeClassAdapter = param a
// CHECK:STDOUT:     @F.%a: %SomeClassAdapter = bind_runtime_name a, %a.loc13_6.1
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_30.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc13_30.2: type = converted %int.make_type_32, %.loc13_30.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc5_10.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_10.2: type = converted %int.make_type_32.loc5, %.loc5_10.1 [template = i32]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc6_10.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_10.2: type = converted %int.make_type_32.loc6, %.loc6_10.1 [template = i32]
// CHECK:STDOUT:   %.loc6_8: %.2 = field_decl b, element1 [template]
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: %SomeClassAdapter) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %.2 = name_ref b, @SomeClass.%.loc6_8 [template = @SomeClass.%.loc6_8]
// CHECK:STDOUT:   %.loc18: i32 = class_element_access <error>, element1 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adapt_non_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .StructAdapter = %StructAdapter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %StructAdapter.decl: type = class_decl @StructAdapter [template = constants.%StructAdapter] {}
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %int.make_type_32.loc8_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_21.1: type = value_of_initializer %int.make_type_32.loc8_21 [template = i32]
// CHECK:STDOUT:   %.loc8_21.2: type = converted %int.make_type_32.loc8_21, %.loc8_21.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc8_30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_30.1: type = value_of_initializer %int.make_type_32.loc8_30 [template = i32]
// CHECK:STDOUT:   %.loc8_30.2: type = converted %int.make_type_32.loc8_30, %.loc8_30.1 [template = i32]
// CHECK:STDOUT:   %.loc8_33: type = struct_type {.a: i32, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   adapt_decl %.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
