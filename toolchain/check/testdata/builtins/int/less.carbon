// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/less.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/less.carbon

// --- int_less.carbon

fn Less(a: i32, b: i32) -> bool = "int.less";
fn Negate(a: i32) -> i32 = "int.snegate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Less(1, 2) then True else False);
  false_ as (if Less(1, 1) then True else False);
  false_ as (if Less(1, 0) then True else False);
  true_ as (if Less(Negate(1), 0) then True else False);
  false_ as (if Less(0, Negate(1)) then True else False);
}

fn RuntimeCall(a: i32, b: i32) -> bool {
  return Less(a, b);
}

// CHECK:STDOUT: --- int_less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Less.type: type = fn_type @Less [template]
// CHECK:STDOUT:   %Less: %Less.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type: type = fn_type @Negate [template]
// CHECK:STDOUT:   %Negate: %Negate.type = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: bool = bool_literal true [template]
// CHECK:STDOUT:   %.7: bool = bool_literal false [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal -1 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Less = %Less.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Less.decl: %Less.type = fn_decl @Less [template = constants.%Less] {
// CHECK:STDOUT:     %int.make_type_32.loc2_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_12.1: type = value_of_initializer %int.make_type_32.loc2_12 [template = i32]
// CHECK:STDOUT:     %.loc2_12.2: type = converted %int.make_type_32.loc2_12, %.loc2_12.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_9.1: i32 = param a
// CHECK:STDOUT:     @Less.%a: i32 = bind_name a, %a.loc2_9.1
// CHECK:STDOUT:     %int.make_type_32.loc2_20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc2_20.1: type = value_of_initializer %int.make_type_32.loc2_20 [template = i32]
// CHECK:STDOUT:     %.loc2_20.2: type = converted %int.make_type_32.loc2_20, %.loc2_20.1 [template = i32]
// CHECK:STDOUT:     %b.loc2_17.1: i32 = param b
// CHECK:STDOUT:     @Less.%b: i32 = bind_name b, %b.loc2_17.1
// CHECK:STDOUT:     %bool.make_type.loc2: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc2_28.1: type = value_of_initializer %bool.make_type.loc2 [template = bool]
// CHECK:STDOUT:     %.loc2_28.2: type = converted %bool.make_type.loc2, %.loc2_28.1 [template = bool]
// CHECK:STDOUT:     @Less.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type = fn_decl @Negate [template = constants.%Negate] {
// CHECK:STDOUT:     %int.make_type_32.loc3_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc3_14.1: type = value_of_initializer %int.make_type_32.loc3_14 [template = i32]
// CHECK:STDOUT:     %.loc3_14.2: type = converted %int.make_type_32.loc3_14, %.loc3_14.1 [template = i32]
// CHECK:STDOUT:     %a.loc3_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc3_11.1
// CHECK:STDOUT:     %int.make_type_32.loc3_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc3_22.1: type = value_of_initializer %int.make_type_32.loc3_22 [template = i32]
// CHECK:STDOUT:     %.loc3_22.2: type = converted %int.make_type_32.loc3_22, %.loc3_22.1 [template = i32]
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [template = constants.%True] {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [template = constants.%False] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %True.ref: type = name_ref True, %True.decl [template = constants.%True]
// CHECK:STDOUT:     %true_.loc8_6.1: %True = param true_
// CHECK:STDOUT:     @F.%true_: %True = bind_name true_, %true_.loc8_6.1
// CHECK:STDOUT:     %False.ref: type = name_ref False, %False.decl [template = constants.%False]
// CHECK:STDOUT:     %false_.loc8_19.1: %False = param false_
// CHECK:STDOUT:     @F.%false_: %False = bind_name false_, %false_.loc8_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %int.make_type_32.loc16_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_19.1: type = value_of_initializer %int.make_type_32.loc16_19 [template = i32]
// CHECK:STDOUT:     %.loc16_19.2: type = converted %int.make_type_32.loc16_19, %.loc16_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc16_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc16_16.1
// CHECK:STDOUT:     %int.make_type_32.loc16_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_27.1: type = value_of_initializer %int.make_type_32.loc16_27 [template = i32]
// CHECK:STDOUT:     %.loc16_27.2: type = converted %int.make_type_32.loc16_27, %.loc16_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc16_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc16_24.1
// CHECK:STDOUT:     %bool.make_type.loc16: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc16_35.1: type = value_of_initializer %bool.make_type.loc16 [template = bool]
// CHECK:STDOUT:     %.loc16_35.2: type = converted %bool.make_type.loc16, %.loc16_35.1 [template = bool]
// CHECK:STDOUT:     @RuntimeCall.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @True {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%False
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Less(%a: i32, %b: i32) -> bool = "int.less";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_: %True, %false_: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref.loc9: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Less.ref.loc9: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %.loc9_21: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_24: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %int.less.loc9: init bool = call %Less.ref.loc9(%.loc9_21, %.loc9_24) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_13.1: bool = value_of_initializer %int.less.loc9 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_13.2: bool = converted %int.less.loc9, %.loc9_13.1 [template = constants.%.6]
// CHECK:STDOUT:   if %.loc9_13.2 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %True.ref.loc9: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc9(%True.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %False.ref.loc9: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc9(%False.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_13.3: type = block_arg !if.expr.result.loc9 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc10: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc10: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.less.loc10: init bool = call %Less.ref.loc10(%.loc10_22, %.loc10_25) [template = constants.%.7]
// CHECK:STDOUT:   %.loc10_14.1: bool = value_of_initializer %int.less.loc10 [template = constants.%.7]
// CHECK:STDOUT:   %.loc10_14.2: bool = converted %int.less.loc10, %.loc10_14.1 [template = constants.%.7]
// CHECK:STDOUT:   if %.loc10_14.2 br !if.expr.then.loc10 else br !if.expr.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc10:
// CHECK:STDOUT:   %True.ref.loc10: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc10(%True.ref.loc10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc10:
// CHECK:STDOUT:   %False.ref.loc10: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc10(%False.ref.loc10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc10:
// CHECK:STDOUT:   %.loc10_14.3: type = block_arg !if.expr.result.loc10 [template = constants.%False]
// CHECK:STDOUT:   %false_.ref.loc11: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc11: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %.loc11_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_25: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %int.less.loc11: init bool = call %Less.ref.loc11(%.loc11_22, %.loc11_25) [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %int.less.loc11 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %int.less.loc11, %.loc11_14.1 [template = constants.%.7]
// CHECK:STDOUT:   if %.loc11_14.2 br !if.expr.then.loc11 else br !if.expr.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc11:
// CHECK:STDOUT:   %True.ref.loc11: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc11(%True.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc11:
// CHECK:STDOUT:   %False.ref.loc11: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc11(%False.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc11:
// CHECK:STDOUT:   %.loc11_14.3: type = block_arg !if.expr.result.loc11 [template = constants.%False]
// CHECK:STDOUT:   %true_.ref.loc12: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Less.ref.loc12: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc12_28: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.snegate.loc12: init i32 = call %Negate.ref.loc12(%.loc12_28) [template = constants.%.9]
// CHECK:STDOUT:   %.loc12_32: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc12_20.1: i32 = value_of_initializer %int.snegate.loc12 [template = constants.%.9]
// CHECK:STDOUT:   %.loc12_20.2: i32 = converted %int.snegate.loc12, %.loc12_20.1 [template = constants.%.9]
// CHECK:STDOUT:   %int.less.loc12: init bool = call %Less.ref.loc12(%.loc12_20.2, %.loc12_32) [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_13.1: bool = value_of_initializer %int.less.loc12 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_13.2: bool = converted %int.less.loc12, %.loc12_13.1 [template = constants.%.6]
// CHECK:STDOUT:   if %.loc12_13.2 br !if.expr.then.loc12 else br !if.expr.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12:
// CHECK:STDOUT:   %True.ref.loc12: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc12(%True.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12:
// CHECK:STDOUT:   %False.ref.loc12: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc12(%False.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12:
// CHECK:STDOUT:   %.loc12_13.3: type = block_arg !if.expr.result.loc12 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc13: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc13: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %.loc13_22: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %Negate.ref.loc13: %Negate.type = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %.loc13_32: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.snegate.loc13: init i32 = call %Negate.ref.loc13(%.loc13_32) [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_21.1: i32 = value_of_initializer %int.snegate.loc13 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_21.2: i32 = converted %int.snegate.loc13, %.loc13_21.1 [template = constants.%.9]
// CHECK:STDOUT:   %int.less.loc13: init bool = call %Less.ref.loc13(%.loc13_22, %.loc13_21.2) [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_14.1: bool = value_of_initializer %int.less.loc13 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_14.2: bool = converted %int.less.loc13, %.loc13_14.1 [template = constants.%.7]
// CHECK:STDOUT:   if %.loc13_14.2 br !if.expr.then.loc13 else br !if.expr.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc13:
// CHECK:STDOUT:   %True.ref.loc13: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc13(%True.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc13:
// CHECK:STDOUT:   %False.ref.loc13: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc13(%False.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc13:
// CHECK:STDOUT:   %.loc13_14.3: type = block_arg !if.expr.result.loc13 [template = constants.%False]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref: %Less.type = name_ref Less, file.%Less.decl [template = constants.%Less]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.less: init bool = call %Less.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_20.1: bool = value_of_initializer %int.less
// CHECK:STDOUT:   %.loc17_20.2: bool = converted %int.less, %.loc17_20.1
// CHECK:STDOUT:   return %.loc17_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
