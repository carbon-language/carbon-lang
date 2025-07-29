// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init_nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init_nested.carbon

class Inner {
  var a: i32;
  var b: i32;
}

fn MakeInner() -> Inner;

class Outer {
  var c: Inner;
  var d: Inner;
}

fn MakeOuter() -> Outer {
  return {.c = MakeInner(), .d = MakeInner()};
}

// CHECK:STDOUT: --- init_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Inner: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.705: <witness> = complete_type_witness %struct_type.a.b [concrete]
// CHECK:STDOUT:   %pattern_type.a31: type = pattern_type %Inner [concrete]
// CHECK:STDOUT:   %MakeInner.type: type = fn_type @MakeInner [concrete]
// CHECK:STDOUT:   %MakeInner: %MakeInner.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer: type = class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.elem: type = unbound_element_type %Outer, %Inner [concrete]
// CHECK:STDOUT:   %struct_type.c.d.dce: type = struct_type {.c: %Inner, .d: %Inner} [concrete]
// CHECK:STDOUT:   %complete_type.8b6: <witness> = complete_type_witness %struct_type.c.d.dce [concrete]
// CHECK:STDOUT:   %pattern_type.e74: type = pattern_type %Outer [concrete]
// CHECK:STDOUT:   %MakeOuter.type: type = fn_type @MakeOuter [concrete]
// CHECK:STDOUT:   %MakeOuter: %MakeOuter.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.e24: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%Inner) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.7b1: %T.as.Destroy.impl.Op.type.e24 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.78b: type = ptr_type %Inner [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.7b1, @T.as.Destroy.impl.Op(%Inner) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .MakeInner = %MakeInner.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .MakeOuter = %MakeOuter.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner] {} {}
// CHECK:STDOUT:   %MakeInner.decl: %MakeInner.type = fn_decl @MakeInner [concrete = constants.%MakeInner] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a31 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a31 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, file.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     %return.param: ref %Inner = out_param call_param0
// CHECK:STDOUT:     %return: ref %Inner = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [concrete = constants.%Outer] {} {}
// CHECK:STDOUT:   %MakeOuter.decl: %MakeOuter.type = fn_decl @MakeOuter [concrete = constants.%MakeOuter] {
// CHECK:STDOUT:     %return.patt: %pattern_type.e74 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e74 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %return.param: ref %Outer = out_param call_param0
// CHECK:STDOUT:     %return: ref %Outer = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Inner.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc17: %Inner.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.b [concrete = constants.%complete_type.705]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .a = %.loc16
// CHECK:STDOUT:   .b = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.ref.loc23: type = name_ref Inner, file.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc23: %Outer.elem = field_decl c, element0 [concrete]
// CHECK:STDOUT:   %Inner.ref.loc24: type = name_ref Inner, file.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc24: %Outer.elem = field_decl d, element1 [concrete]
// CHECK:STDOUT:   %struct_type.c.d: type = struct_type {.c: %Inner, .d: %Inner} [concrete = constants.%struct_type.c.d.dce]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.c.d [concrete = constants.%complete_type.8b6]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .Inner = <poisoned>
// CHECK:STDOUT:   .c = %.loc23
// CHECK:STDOUT:   .d = %.loc24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeInner() -> %return.param: %Inner;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeOuter() -> %return.param: %Outer {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeInner.ref.loc28_16: %MakeInner.type = name_ref MakeInner, file.%MakeInner.decl [concrete = constants.%MakeInner]
// CHECK:STDOUT:   %.loc28_45.1: ref %Inner = class_element_access %return, element0
// CHECK:STDOUT:   %MakeInner.call.loc28_26: init %Inner = call %MakeInner.ref.loc28_16() to %.loc28_45.1
// CHECK:STDOUT:   %MakeInner.ref.loc28_34: %MakeInner.type = name_ref MakeInner, file.%MakeInner.decl [concrete = constants.%MakeInner]
// CHECK:STDOUT:   %.loc28_45.2: ref %Inner = class_element_access %return, element1
// CHECK:STDOUT:   %MakeInner.call.loc28_44: init %Inner = call %MakeInner.ref.loc28_34() to %.loc28_45.2
// CHECK:STDOUT:   %.loc28_45.3: %struct_type.c.d.dce = struct_literal (%MakeInner.call.loc28_26, %MakeInner.call.loc28_44)
// CHECK:STDOUT:   %.loc28_45.4: init %Outer = class_init (%MakeInner.call.loc28_26, %MakeInner.call.loc28_44), %return
// CHECK:STDOUT:   %.loc28_46: init %Outer = converted %.loc28_45.3, %.loc28_45.4
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc28_45.1: <bound method> = bound_method %.loc28_45.2, constants.%T.as.Destroy.impl.Op.7b1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.7b1, @T.as.Destroy.impl.Op(constants.%Inner) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_45.1: <bound method> = bound_method %.loc28_45.2, %T.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc28_45.1: %ptr.78b = addr_of %.loc28_45.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc28_45.1: init %empty_tuple.type = call %bound_method.loc28_45.1(%addr.loc28_45.1)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound.loc28_45.2: <bound method> = bound_method %.loc28_45.1, constants.%T.as.Destroy.impl.Op.7b1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.7b1, @T.as.Destroy.impl.Op(constants.%Inner) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_45.2: <bound method> = bound_method %.loc28_45.1, %T.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc28_45.2: %ptr.78b = addr_of %.loc28_45.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call.loc28_45.2: init %empty_tuple.type = call %bound_method.loc28_45.2(%addr.loc28_45.2)
// CHECK:STDOUT:   return %.loc28_46 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
