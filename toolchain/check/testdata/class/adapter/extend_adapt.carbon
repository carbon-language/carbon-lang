// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/extend_adapt.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/extend_adapt.carbon

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
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `SomeClassAdapter` to `SomeClass` [ConversionFailure]
  // CHECK:STDERR:   a.F();
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_todo_method_access.carbon:[[@LINE+7]]:3: note: type `SomeClassAdapter` does not implement interface `Core.ImplicitAs(SomeClass)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_todo_field_access.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `SomeClassAdapter` to `SomeClass` [ConversionFailure]
  // CHECK:STDERR:   return a.b;
  // CHECK:STDERR:          ^~~
  // CHECK:STDERR: fail_todo_field_access.carbon:[[@LINE+4]]:10: note: type `SomeClassAdapter` does not implement interface `Core.ImplicitAs(SomeClass)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_todo_adapt_struct.carbon:[[@LINE+4]]:10: error: member name `b` not found in `StructAdapter` [MemberNameNotFoundInInstScope]
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
  // CHECK:STDERR: fail_adapt_builtin.carbon:[[@LINE+4]]:10: error: member name `foo` not found in `IntAdapter` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return a.foo;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a.foo;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [concrete]
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %SomeClass.elem: type = unbound_element_type %SomeClass, %i32 [concrete]
// CHECK:STDOUT:   %StaticMemberFunction.type: type = fn_type @StaticMemberFunction [concrete]
// CHECK:STDOUT:   %StaticMemberFunction: %StaticMemberFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %AdapterMethod.type: type = fn_type @AdapterMethod [concrete]
// CHECK:STDOUT:   %AdapterMethod: %AdapterMethod.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [concrete]
// CHECK:STDOUT:   %TestStaticMemberFunction.type: type = fn_type @TestStaticMemberFunction [concrete]
// CHECK:STDOUT:   %TestStaticMemberFunction: %TestStaticMemberFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %TestAdapterMethod.type: type = fn_type @TestAdapterMethod [concrete]
// CHECK:STDOUT:   %TestAdapterMethod: %TestAdapterMethod.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl.loc4
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .TestStaticMemberFunction = %TestStaticMemberFunction.decl
// CHECK:STDOUT:     .TestAdapterMethod = %TestAdapterMethod.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SomeClassAdapter.decl.loc4: type = class_decl @SomeClassAdapter [concrete = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [concrete = constants.%SomeClass] {} {}
// CHECK:STDOUT:   %SomeClassAdapter.decl.loc15: type = class_decl @SomeClassAdapter [concrete = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %TestStaticMemberFunction.decl: %TestStaticMemberFunction.type = fn_decl @TestStaticMemberFunction [concrete = constants.%TestStaticMemberFunction] {
// CHECK:STDOUT:     %a.patt: %SomeClassAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %SomeClassAdapter = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %SomeClassAdapter = value_param call_param0
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl.loc4 [concrete = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a: %SomeClassAdapter = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestAdapterMethod.decl: %TestAdapterMethod.type = fn_decl @TestAdapterMethod [concrete = constants.%TestAdapterMethod] {
// CHECK:STDOUT:     %a.patt: %SomeClassAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %SomeClassAdapter = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %SomeClassAdapter = value_param call_param0
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl.loc4 [concrete = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a: %SomeClassAdapter = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [concrete = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   .SomeClass = <poisoned>
// CHECK:STDOUT:   .StaticMemberFunction = <poisoned>
// CHECK:STDOUT:   .AdapterMethod = <poisoned>
// CHECK:STDOUT:   extend %SomeClass.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %.loc7_8: %SomeClass.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc7_3: %SomeClass.elem = var_pattern %.loc7_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc7: ref %SomeClass.elem = var <none>
// CHECK:STDOUT:   %.loc8_8: %SomeClass.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc8_3: %SomeClass.elem = var_pattern %.loc8_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc8: ref %SomeClass.elem = var <none>
// CHECK:STDOUT:   %StaticMemberFunction.decl: %StaticMemberFunction.type = fn_decl @StaticMemberFunction [concrete = constants.%StaticMemberFunction] {} {}
// CHECK:STDOUT:   %AdapterMethod.decl: %AdapterMethod.type = fn_decl @AdapterMethod [concrete = constants.%AdapterMethod] {
// CHECK:STDOUT:     %self.patt: %SomeClassAdapter = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %SomeClassAdapter = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %SomeClassAdapter = value_param call_param0
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl.loc4 [concrete = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %self: %SomeClassAdapter = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc7_8
// CHECK:STDOUT:   .b = %.loc8_8
// CHECK:STDOUT:   .StaticMemberFunction = %StaticMemberFunction.decl
// CHECK:STDOUT:   .SomeClassAdapter = <poisoned>
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
// CHECK:STDOUT:   %StaticMemberFunction.ref: %StaticMemberFunction.type = name_ref StaticMemberFunction, @SomeClass.%StaticMemberFunction.decl [concrete = constants.%StaticMemberFunction]
// CHECK:STDOUT:   %StaticMemberFunction.call: init %empty_tuple.type = call %StaticMemberFunction.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestAdapterMethod(%a.param_patt: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %AdapterMethod.ref: %AdapterMethod.type = name_ref AdapterMethod, @SomeClass.%AdapterMethod.decl [concrete = constants.%AdapterMethod]
// CHECK:STDOUT:   %AdapterMethod.bound: <bound method> = bound_method %a.ref, %AdapterMethod.ref
// CHECK:STDOUT:   %AdapterMethod.call: init %empty_tuple.type = call %AdapterMethod.bound(%a.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_method_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [concrete]
// CHECK:STDOUT:   %F.type.633: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.e19: %F.type.633 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [concrete = constants.%SomeClass] {} {}
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [concrete = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %a.patt: %SomeClassAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %SomeClassAdapter = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %SomeClassAdapter = value_param call_param0
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl [concrete = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a: %SomeClassAdapter = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %F.decl: %F.type.633 = fn_decl @F.1 [concrete = constants.%F.e19] {
// CHECK:STDOUT:     %self.patt: %SomeClass = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %SomeClass = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %SomeClass = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%SomeClass [concrete = constants.%SomeClass]
// CHECK:STDOUT:     %self: %SomeClass = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [concrete = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   .SomeClass = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %SomeClass.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[%self.param_patt: %SomeClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a.param_patt: %SomeClassAdapter) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %F.ref: %F.type.633 = name_ref F, @SomeClass.%F.decl [concrete = constants.%F.e19]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc23: %SomeClass = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %SomeClass: type = class_type @SomeClass [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %SomeClass.elem: type = unbound_element_type %SomeClass, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [concrete]
// CHECK:STDOUT:   %SomeClassAdapter: type = class_type @SomeClassAdapter [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .SomeClass = %SomeClass.decl
// CHECK:STDOUT:     .SomeClassAdapter = %SomeClassAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %SomeClass.decl: type = class_decl @SomeClass [concrete = constants.%SomeClass] {} {}
// CHECK:STDOUT:   %SomeClassAdapter.decl: type = class_decl @SomeClassAdapter [concrete = constants.%SomeClassAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %SomeClassAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %SomeClassAdapter = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %SomeClassAdapter = value_param call_param0
// CHECK:STDOUT:     %SomeClassAdapter.ref: type = name_ref SomeClassAdapter, file.%SomeClassAdapter.decl [concrete = constants.%SomeClassAdapter]
// CHECK:STDOUT:     %a: %SomeClassAdapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClass {
// CHECK:STDOUT:   %.loc5_8: %SomeClass.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %SomeClass.elem = var_pattern %.loc5_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc5: ref %SomeClass.elem = var <none>
// CHECK:STDOUT:   %.loc6_8: %SomeClass.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %SomeClass.elem = var_pattern %.loc6_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %SomeClass.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClass
// CHECK:STDOUT:   .a = %.loc5_8
// CHECK:STDOUT:   .b = %.loc6_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @SomeClassAdapter {
// CHECK:STDOUT:   %SomeClass.ref: type = name_ref SomeClass, file.%SomeClass.decl [concrete = constants.%SomeClass]
// CHECK:STDOUT:   adapt_decl %SomeClass.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%SomeClassAdapter
// CHECK:STDOUT:   .SomeClass = <poisoned>
// CHECK:STDOUT:   .b = <poisoned>
// CHECK:STDOUT:   extend %SomeClass.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %SomeClassAdapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %SomeClassAdapter = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %SomeClass.elem = name_ref b, @SomeClass.%.loc6_8 [concrete = @SomeClass.%.loc6_8]
// CHECK:STDOUT:   %.loc21_11.1: %SomeClass = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc21_11.2: %i32 = class_element_access <error>, element1 [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adapt_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %StructAdapter: type = class_type @StructAdapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .StructAdapter = %StructAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %StructAdapter.decl: type = class_decl @StructAdapter [concrete = constants.%StructAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %StructAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %StructAdapter = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %StructAdapter = value_param call_param0
// CHECK:STDOUT:     %StructAdapter.ref: type = name_ref StructAdapter, file.%StructAdapter.decl [concrete = constants.%StructAdapter]
// CHECK:STDOUT:     %a: %StructAdapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @StructAdapter {
// CHECK:STDOUT:   %int_32.loc5_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc5_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   adapt_decl %struct_type.a.b [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%StructAdapter
// CHECK:STDOUT:   .b = <poisoned>
// CHECK:STDOUT:   extend %struct_type.a.b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %StructAdapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %StructAdapter = name_ref a, %a
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_adapt_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TupleAdapter: type = class_type @TupleAdapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %complete_type.65d: <witness> = complete_type_witness %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .TupleAdapter = %TupleAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %TupleAdapter.decl: type = class_decl @TupleAdapter [concrete = constants.%TupleAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %TupleAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %TupleAdapter = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %TupleAdapter = value_param call_param0
// CHECK:STDOUT:     %TupleAdapter.ref: type = name_ref TupleAdapter, file.%TupleAdapter.decl [concrete = constants.%TupleAdapter]
// CHECK:STDOUT:     %a: %TupleAdapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @TupleAdapter {
// CHECK:STDOUT:   %int_32.loc5_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc5_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5_25: %tuple.type.24b = tuple_literal (%i32.loc5_17, %i32.loc5_22)
// CHECK:STDOUT:   %.loc5_26: type = converted %.loc5_25, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   adapt_decl %.loc5_26 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %tuple.type.d07 [concrete = constants.%complete_type.65d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%TupleAdapter
// CHECK:STDOUT:   extend %.loc5_26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %TupleAdapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %TupleAdapter = name_ref a, %a
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_builtin.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeInt.type: type = fn_type @MakeInt [concrete]
// CHECK:STDOUT:   %MakeInt: %MakeInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %IntAdapter: type = class_type @IntAdapter [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .MakeInt = %MakeInt.decl
// CHECK:STDOUT:     .IntAdapter = %IntAdapter.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %MakeInt.decl: %MakeInt.type = fn_decl @MakeInt [concrete = constants.%MakeInt] {
// CHECK:STDOUT:     %N.patt: Core.IntLiteral = binding_pattern N
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt, call_param0
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc4_31.1: type = splice_block %.loc4_31.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_31.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_31.3: type = converted %int_literal.make_type, %.loc4_31.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = bind_name N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntAdapter.decl: type = class_decl @IntAdapter [concrete = constants.%IntAdapter] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %IntAdapter = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %IntAdapter = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %IntAdapter = value_param call_param0
// CHECK:STDOUT:     %IntAdapter.ref: type = name_ref IntAdapter, file.%IntAdapter.decl [concrete = constants.%IntAdapter]
// CHECK:STDOUT:     %a: %IntAdapter = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @IntAdapter {
// CHECK:STDOUT:   %MakeInt.ref: %MakeInt.type = name_ref MakeInt, file.%MakeInt.decl [concrete = constants.%MakeInt]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call %MakeInt.ref(%int_32) [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc7_27.1: type = value_of_initializer %int.make_type_signed [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %.loc7_27.2: type = converted %int.make_type_signed, %.loc7_27.1 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   adapt_decl %.loc7_27.2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %i32.builtin [concrete = constants.%complete_type.f8a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%IntAdapter
// CHECK:STDOUT:   .MakeInt = <poisoned>
// CHECK:STDOUT:   .foo = <poisoned>
// CHECK:STDOUT:   extend %.loc7_27.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeInt(%N.param_patt: Core.IntLiteral) -> type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %IntAdapter) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %IntAdapter = name_ref a, %a
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
