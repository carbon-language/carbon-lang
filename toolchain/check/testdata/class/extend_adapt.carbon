// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- basic.carbon

library "basic" api;

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

library "fail_todo_method_access" api;

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

library "fail_todo_field_access" api;

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

library "fail_todo_adapt_non_class" api;

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
// CHECK:STDOUT:   %.1: type = unbound_element_type SomeClass, i32 [template]
// CHECK:STDOUT:   %StaticMemberFunction: type = fn_type @StaticMemberFunction [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: StaticMemberFunction = struct_value () [template]
// CHECK:STDOUT:   %AdapterMethod: type = fn_type @AdapterMethod [template]
// CHECK:STDOUT:   %struct.2: AdapterMethod = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %TestStaticMemberFunction: type = fn_type @TestStaticMemberFunction [template]
// CHECK:STDOUT:   %struct.3: TestStaticMemberFunction = struct_value () [template]
// CHECK:STDOUT:   %TestAdapterMethod: type = fn_type @TestAdapterMethod [template]
// CHECK:STDOUT:   %struct.4: TestAdapterMethod = struct_value () [template]
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
// CHECK:STDOUT:   %SomeClassAdapter.decl.loc15: type = class_decl @SomeClassAdapter [template = constants.%SomeClassAdapter] {}
// CHECK:STDOUT:   %TestStaticMemberFunction.decl: TestStaticMemberFunction = fn_decl @TestStaticMemberFunction [template = constants.%struct.3] {
// CHECK:STDOUT:     %SomeClassAdapter.ref.loc19: type = name_ref SomeClassAdapter, %SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.loc19_29.1: SomeClassAdapter = param a
// CHECK:STDOUT:     @TestStaticMemberFunction.%a: SomeClassAdapter = bind_name a, %a.loc19_29.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAdapterMethod.decl: TestAdapterMethod = fn_decl @TestAdapterMethod [template = constants.%struct.4] {
// CHECK:STDOUT:     %SomeClassAdapter.ref.loc23: type = name_ref SomeClassAdapter, %SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.loc23_22.1: SomeClassAdapter = param a
// CHECK:STDOUT:     @TestAdapterMethod.%a: SomeClassAdapter = bind_name a, %a.loc23_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl SomeClass
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %.loc7: <unbound element of class SomeClass> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc8: <unbound element of class SomeClass> = field_decl b, element1 [template]
// CHECK:STDOUT:   %StaticMemberFunction.decl: StaticMemberFunction = fn_decl @StaticMemberFunction [template = constants.%struct.1] {}
// CHECK:STDOUT:   %AdapterMethod.decl: AdapterMethod = fn_decl @AdapterMethod [template = constants.%struct.2] {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl.loc4 [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %self.loc12_20.1: SomeClassAdapter = param self
// CHECK:STDOUT:     %self.loc12_20.2: SomeClassAdapter = bind_name self, %self.loc12_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc7
// CHECK:STDOUT:   .b = %.loc8
// CHECK:STDOUT:   .StaticMemberFunction = %StaticMemberFunction.decl
// CHECK:STDOUT:   .AdapterMethod = %AdapterMethod.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @StaticMemberFunction();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AdapterMethod[@SomeClass.%self.loc12_20.2: SomeClassAdapter]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestStaticMemberFunction(%a: SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %StaticMemberFunction.ref: StaticMemberFunction = name_ref StaticMemberFunction, @SomeClass.%StaticMemberFunction.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %StaticMemberFunction.call: init () = call %StaticMemberFunction.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdapterMethod(%a: SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %AdapterMethod.ref: AdapterMethod = name_ref AdapterMethod, @SomeClass.%AdapterMethod.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc24: <bound method> = bound_method %a.ref, %AdapterMethod.ref
// CHECK:STDOUT:   %AdapterMethod.call: init () = call %.loc24(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_method_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %F.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %F.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
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
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.2 [template = constants.%struct.2] {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, %SomeClassAdapter.decl [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.loc12_6.1: SomeClassAdapter = param a
// CHECK:STDOUT:     @F.2.%a: SomeClassAdapter = bind_name a, %a.loc12_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SomeClass [template = constants.%SomeClass]
// CHECK:STDOUT:     %self.loc5_8.1: SomeClass = param self
// CHECK:STDOUT:     %self.loc5_8.2: SomeClass = bind_name self, %self.loc5_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl SomeClass
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[@SomeClass.%self.loc5_8.2: SomeClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a: SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %F.ref: F = name_ref F, @SomeClass.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc20: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %F.call: init () = call %.loc20(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type SomeClass, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
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
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, %SomeClassAdapter.decl [template = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a.loc13_6.1: SomeClassAdapter = param a
// CHECK:STDOUT:     @F.%a: SomeClassAdapter = bind_name a, %a.loc13_6.1
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %.loc5: <unbound element of class SomeClass> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc6: <unbound element of class SomeClass> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc5
// CHECK:STDOUT:   .b = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [template = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl SomeClass
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: SomeClassAdapter) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %b.ref: <unbound element of class SomeClass> = name_ref b, @SomeClass.%.loc6 [template = @SomeClass.%.loc6]
// CHECK:STDOUT:   %.loc18: i32 = class_element_access <error>, element1 [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adapt_non_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.2: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .StructAdapter = %StructAdapter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %StructAdapter.decl: type = class_decl @StructAdapter [template = constants.%StructAdapter] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %.loc8: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   adapt_decl {.a: i32, .b: i32}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT:   has_error
// CHECK:STDOUT: }
// CHECK:STDOUT:
