// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/return.carbon

// ============================================================================
// Multiple function with a return value
// ============================================================================

// --- multiple.h

auto foo1() -> short;
auto foo2() -> int;

// --- import_multiple.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple.h";

fn IngestI16(x: i16);
fn IngestI32(x: i32);

fn F() {
  //@dump-sem-ir-begin
  IngestI16(Cpp.foo1());
  IngestI32(Cpp.foo2());
  //@dump-sem-ir-end
}

// ============================================================================
// Undeduced return type
// ============================================================================

// --- fail_return_undeduced_auto.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
auto UndeducedReturn();
''';

fn Call() {
  // CHECK:STDERR: fail_return_undeduced_auto.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: return type: auto` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.UndeducedReturn();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.UndeducedReturn();
}

// ============================================================================
// Initialize a variable from a thunk output parameter
// ============================================================================

// --- var_from_thunk_param.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
using nullptr_t = decltype(nullptr);
auto ReturnNullPtr() -> nullptr_t;
''';

fn InitializeFromReturn() {
  //@dump-sem-ir-begin
  var c: Cpp.nullptr_t = Cpp.ReturnNullPtr();
  //@dump-sem-ir-end
}

// ============================================================================
// Return from a thunk output parameter
// ============================================================================

// --- return_from_thunk_param.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

import Cpp inline '''
unsigned char ReturnU8();
''';

fn GetU8() -> u8 { return Cpp.ReturnU8(); }

// ============================================================================
// Initialize a variable from a thunk with no output parameter
// ============================================================================

// --- var_from_thunk_by_value.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

import Cpp inline '''
unsigned int ReturnU32();
''';

fn Var() {
  var my_u32: u32 = Cpp.ReturnU32();
}

// CHECK:STDOUT: --- import_multiple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %IngestI16.type: type = fn_type @IngestI16 [concrete]
// CHECK:STDOUT:   %IngestI16: %IngestI16.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %IngestI32.type: type = fn_type @IngestI32 [concrete]
// CHECK:STDOUT:   %IngestI32: %IngestI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo1.cpp_overload_set.type: type = cpp_overload_set_type @foo1.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo1.cpp_overload_set.value: %foo1.cpp_overload_set.type = cpp_overload_set_value @foo1.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo1__carbon_thunk.type: type = fn_type @foo1__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo1__carbon_thunk: %foo1__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo2.cpp_overload_set.type: type = cpp_overload_set_type @foo2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo2.cpp_overload_set.value: %foo2.cpp_overload_set.type = cpp_overload_set_value @foo2.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo2.type: type = fn_type @foo2 [concrete]
// CHECK:STDOUT:   %foo2: %foo2.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo1 = %foo1.cpp_overload_set.value
// CHECK:STDOUT:     .foo2 = %foo2.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo1.cpp_overload_set.value: %foo1.cpp_overload_set.type = cpp_overload_set_value @foo1.cpp_overload_set [concrete = constants.%foo1.cpp_overload_set.value]
// CHECK:STDOUT:   %foo1__carbon_thunk.decl: %foo1__carbon_thunk.type = fn_decl @foo1__carbon_thunk [concrete = constants.%foo1__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo2.cpp_overload_set.value: %foo2.cpp_overload_set.type = cpp_overload_set_value @foo2.cpp_overload_set [concrete = constants.%foo2.cpp_overload_set.value]
// CHECK:STDOUT:   %foo2.decl: %foo2.type = fn_decl @foo2 [concrete = constants.%foo2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %IngestI16.ref: %IngestI16.type = name_ref IngestI16, file.%IngestI16.decl [concrete = constants.%IngestI16]
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo1.ref: %foo1.cpp_overload_set.type = name_ref foo1, imports.%foo1.cpp_overload_set.value [concrete = constants.%foo1.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc11_22.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc11_22.1
// CHECK:STDOUT:   %foo1__carbon_thunk.call: init %empty_tuple.type = call imports.%foo1__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc11_22.2: init %i16 to %.loc11_22.1 = mark_in_place_init %foo1__carbon_thunk.call
// CHECK:STDOUT:   %.loc11_22.3: ref %i16 = temporary %.loc11_22.1, %.loc11_22.2
// CHECK:STDOUT:   %.loc11_22.4: %i16 = acquire_value %.loc11_22.3
// CHECK:STDOUT:   %IngestI16.call: init %empty_tuple.type = call %IngestI16.ref(%.loc11_22.4)
// CHECK:STDOUT:   %IngestI32.ref: %IngestI32.type = name_ref IngestI32, file.%IngestI32.decl [concrete = constants.%IngestI32]
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo2.ref: %foo2.cpp_overload_set.type = name_ref foo2, imports.%foo2.cpp_overload_set.value [concrete = constants.%foo2.cpp_overload_set.value]
// CHECK:STDOUT:   %foo2.call: init %i32 = call imports.%foo2.decl()
// CHECK:STDOUT:   %.loc12_22.1: %i32 = value_of_initializer %foo2.call
// CHECK:STDOUT:   %.loc12_22.2: %i32 = converted %foo2.call, %.loc12_22.1
// CHECK:STDOUT:   %IngestI32.call: init %empty_tuple.type = call %IngestI32.ref(%.loc12_22.2)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %.loc11_22.3, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%.loc11_22.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_from_thunk_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %pattern_type.f05: type = pattern_type %Cpp.nullptr_t [concrete]
// CHECK:STDOUT:   %ReturnNullPtr.cpp_overload_set.type: type = cpp_overload_set_type @ReturnNullPtr.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnNullPtr.cpp_overload_set.value: %ReturnNullPtr.cpp_overload_set.type = cpp_overload_set_value @ReturnNullPtr.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.142: type = ptr_type %Cpp.nullptr_t [concrete]
// CHECK:STDOUT:   %ReturnNullPtr__carbon_thunk.type: type = fn_type @ReturnNullPtr__carbon_thunk [concrete]
// CHECK:STDOUT:   %ReturnNullPtr__carbon_thunk: %ReturnNullPtr__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc11 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .nullptr_t = constants.%Cpp.nullptr_t
// CHECK:STDOUT:     .ReturnNullPtr = %ReturnNullPtr.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnNullPtr.cpp_overload_set.value: %ReturnNullPtr.cpp_overload_set.type = cpp_overload_set_value @ReturnNullPtr.cpp_overload_set [concrete = constants.%ReturnNullPtr.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnNullPtr__carbon_thunk.decl: %ReturnNullPtr__carbon_thunk.type = fn_decl @ReturnNullPtr__carbon_thunk [concrete = constants.%ReturnNullPtr__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitializeFromReturn() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f05 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.f05 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Cpp.nullptr_t = var %c.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc11_26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnNullPtr.ref: %ReturnNullPtr.cpp_overload_set.type = name_ref ReturnNullPtr, imports.%ReturnNullPtr.cpp_overload_set.value [concrete = constants.%ReturnNullPtr.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc11_3: ref %Cpp.nullptr_t = splice_block %c.var {}
// CHECK:STDOUT:   %addr: %ptr.142 = addr_of %.loc11_3
// CHECK:STDOUT:   %ReturnNullPtr__carbon_thunk.call: init %empty_tuple.type = call imports.%ReturnNullPtr__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc11_44: init %Cpp.nullptr_t to %.loc11_3 = mark_in_place_init %ReturnNullPtr__carbon_thunk.call
// CHECK:STDOUT:   assign %c.var, %.loc11_44
// CHECK:STDOUT:   %.loc11_13: type = splice_block %nullptr_t.ref [concrete = constants.%Cpp.nullptr_t] {
// CHECK:STDOUT:     %Cpp.ref.loc11_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %nullptr_t.ref: type = name_ref nullptr_t, constants.%Cpp.nullptr_t [concrete = constants.%Cpp.nullptr_t]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Cpp.nullptr_t = ref_binding c, %c.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %c.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%c.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc11(%self.param: %Cpp.nullptr_t) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_from_thunk_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %.98d: form = init_form %u8, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.8f3: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %GetU8.type: type = fn_type @GetU8 [concrete]
// CHECK:STDOUT:   %GetU8: %GetU8.type = struct_value () [concrete]
// CHECK:STDOUT:   %ReturnU8.cpp_overload_set.type: type = cpp_overload_set_type @ReturnU8.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnU8.cpp_overload_set.value: %ReturnU8.cpp_overload_set.type = cpp_overload_set_value @ReturnU8.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %pattern_type.a66: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %ReturnU8__carbon_thunk.type: type = fn_type @ReturnU8__carbon_thunk [concrete]
// CHECK:STDOUT:   %ReturnU8__carbon_thunk: %ReturnU8__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ReturnU8 = %ReturnU8.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %ReturnU8.cpp_overload_set.value: %ReturnU8.cpp_overload_set.type = cpp_overload_set_value @ReturnU8.cpp_overload_set [concrete = constants.%ReturnU8.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnU8__carbon_thunk.decl: %ReturnU8__carbon_thunk.type = fn_decl @ReturnU8__carbon_thunk [concrete = constants.%ReturnU8__carbon_thunk] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a66 = value_binding_pattern r#return [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a66 = value_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block constants.%ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:       %u8: type = class_type @UInt, @UInt(constants.%int_8) [concrete = constants.%u8]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %return: %ptr = value_binding r#return, %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .GetU8 = %GetU8.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GetU8.decl: %GetU8.type = fn_decl @GetU8 [concrete = constants.%GetU8] {
// CHECK:STDOUT:     %return.patt: %pattern_type.8f3 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8f3 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:     %u8: type = class_type @UInt, @UInt(constants.%int_8) [concrete = constants.%u8]
// CHECK:STDOUT:     %.loc8_15: form = init_form %u8, call_param0 [concrete = constants.%.98d]
// CHECK:STDOUT:     %return.param: ref %u8 = out_param call_param0
// CHECK:STDOUT:     %return: ref %u8 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetU8() -> out %return.param: %u8 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnU8.ref: %ReturnU8.cpp_overload_set.type = name_ref ReturnU8, imports.%ReturnU8.cpp_overload_set.value [concrete = constants.%ReturnU8.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_40.1: ref %u8 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_40.1
// CHECK:STDOUT:   %ReturnU8__carbon_thunk.call: init %empty_tuple.type = call imports.%ReturnU8__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_40.2: init %u8 to %.loc8_40.1 = mark_in_place_init %ReturnU8__carbon_thunk.call
// CHECK:STDOUT:   return %.loc8_40.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnU8() -> out %return.param: %u8;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnU8__carbon_thunk(%return.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_from_thunk_by_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Var.type: type = fn_type @Var [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Var: %Var.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %ReturnU32.cpp_overload_set.type: type = cpp_overload_set_type @ReturnU32.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnU32.cpp_overload_set.value: %ReturnU32.cpp_overload_set.type = cpp_overload_set_value @ReturnU32.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnU32.type: type = fn_type @ReturnU32 [concrete]
// CHECK:STDOUT:   %ReturnU32: %ReturnU32.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyOp.type: type = fn_type @DestroyOp [concrete]
// CHECK:STDOUT:   %DestroyOp: %DestroyOp.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ReturnU32 = %ReturnU32.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/types/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %ReturnU32.cpp_overload_set.value: %ReturnU32.cpp_overload_set.type = cpp_overload_set_value @ReturnU32.cpp_overload_set [concrete = constants.%ReturnU32.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnU32.decl: %ReturnU32.type = fn_decl @ReturnU32 [concrete = constants.%ReturnU32] {
// CHECK:STDOUT:     %return.patt: %pattern_type.4a9 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.4a9 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %return.param: ref %u32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %u32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .Var = %Var.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Var.decl: %Var.type = fn_decl @Var [concrete = constants.%Var] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %my_u32.patt: %pattern_type.4a9 = ref_binding_pattern my_u32 [concrete]
// CHECK:STDOUT:     %my_u32.var_patt: %pattern_type.4a9 = var_pattern %my_u32.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %my_u32.var: ref %u32 = var %my_u32.var_patt
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnU32.ref: %ReturnU32.cpp_overload_set.type = name_ref ReturnU32, imports.%ReturnU32.cpp_overload_set.value [concrete = constants.%ReturnU32.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnU32.call: init %u32 = call imports.%ReturnU32.decl()
// CHECK:STDOUT:   assign %my_u32.var, %ReturnU32.call
// CHECK:STDOUT:   %.loc9: type = splice_block %u32 [concrete = constants.%u32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %my_u32: ref %u32 = ref_binding my_u32, %my_u32.var
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %my_u32.var, constants.%DestroyOp
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%my_u32.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnU32() -> out %return.param: %u32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp(%self.param: %u32) = "no_op";
// CHECK:STDOUT:
