// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method.carbon

class Class {
  fn F[self: Self]() -> i32;
  fn G[addr self: Self*]() -> i32;

  alias A = F;

  var k: i32;
}

fn Class.F[self: Self]() -> i32 {
  return self.k;
}

fn Call(c: Class) -> i32 {
  // TODO: The sem-ir for this call doesn't distinguish the `self` argument from
  // the explicit arguments.
  return c.F();
}

fn CallAlias(c: Class) -> i32 {
  return c.A();
}

fn CallOnConstBoundMethod() -> i32 {
  return ({.k = 1} as Class).F();
}

fn CallWithAddr() -> i32 {
  var c: Class;
  return c.G();
}

fn CallFThroughPointer(p: Class*) -> i32 {
  return (*p).F();
}

fn CallGThroughPointer(p: Class*) -> i32 {
  return (*p).G();
}

fn Make() -> Class;

fn CallFOnInitializingExpr() -> i32 {
  return Make().F();
}

fn CallGOnInitializingExpr() -> i32 {
  return Make().G();
}

// CHECK:STDOUT: --- method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %Class [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Class, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %CallAlias.type: type = fn_type @CallAlias [template]
// CHECK:STDOUT:   %CallAlias: %CallAlias.type = struct_value () [template]
// CHECK:STDOUT:   %CallOnConstBoundMethod.type: type = fn_type @CallOnConstBoundMethod [template]
// CHECK:STDOUT:   %CallOnConstBoundMethod: %CallOnConstBoundMethod.type = struct_value () [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %struct: %Class = struct_value (%.6) [template]
// CHECK:STDOUT:   %CallWithAddr.type: type = fn_type @CallWithAddr [template]
// CHECK:STDOUT:   %CallWithAddr: %CallWithAddr.type = struct_value () [template]
// CHECK:STDOUT:   %CallFThroughPointer.type: type = fn_type @CallFThroughPointer [template]
// CHECK:STDOUT:   %CallFThroughPointer: %CallFThroughPointer.type = struct_value () [template]
// CHECK:STDOUT:   %CallGThroughPointer.type: type = fn_type @CallGThroughPointer [template]
// CHECK:STDOUT:   %CallGThroughPointer: %CallGThroughPointer.type = struct_value () [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %CallFOnInitializingExpr.type: type = fn_type @CallFOnInitializingExpr [template]
// CHECK:STDOUT:   %CallFOnInitializingExpr: %CallFOnInitializingExpr.type = struct_value () [template]
// CHECK:STDOUT:   %CallGOnInitializingExpr.type: type = fn_type @CallGOnInitializingExpr [template]
// CHECK:STDOUT:   %CallGOnInitializingExpr: %CallGOnInitializingExpr.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallAlias = %CallAlias.decl
// CHECK:STDOUT:     .CallOnConstBoundMethod = %CallOnConstBoundMethod.decl
// CHECK:STDOUT:     .CallWithAddr = %CallWithAddr.decl
// CHECK:STDOUT:     .CallFThroughPointer = %CallFThroughPointer.decl
// CHECK:STDOUT:     .CallGThroughPointer = %CallGThroughPointer.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .CallFOnInitializingExpr = %CallFOnInitializingExpr.decl
// CHECK:STDOUT:     .CallGOnInitializingExpr = %CallGOnInitializingExpr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc20_12.1: %Class = param self
// CHECK:STDOUT:     @F.%self: %Class = bind_name self, %self.loc20_12.1
// CHECK:STDOUT:     %int.make_type_32.loc20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc20_29.1: type = value_of_initializer %int.make_type_32.loc20 [template = i32]
// CHECK:STDOUT:     %.loc20_29.2: type = converted %int.make_type_32.loc20, %.loc20_29.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %Class.ref.loc24: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc24_9.1: %Class = param c
// CHECK:STDOUT:     @Call.%c: %Class = bind_name c, %c.loc24_9.1
// CHECK:STDOUT:     %int.make_type_32.loc24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc24_22.1: type = value_of_initializer %int.make_type_32.loc24 [template = i32]
// CHECK:STDOUT:     %.loc24_22.2: type = converted %int.make_type_32.loc24, %.loc24_22.1 [template = i32]
// CHECK:STDOUT:     @Call.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallAlias.decl: %CallAlias.type = fn_decl @CallAlias [template = constants.%CallAlias] {
// CHECK:STDOUT:     %Class.ref.loc30: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %c.loc30_14.1: %Class = param c
// CHECK:STDOUT:     @CallAlias.%c: %Class = bind_name c, %c.loc30_14.1
// CHECK:STDOUT:     %int.make_type_32.loc30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc30_27.1: type = value_of_initializer %int.make_type_32.loc30 [template = i32]
// CHECK:STDOUT:     %.loc30_27.2: type = converted %int.make_type_32.loc30, %.loc30_27.1 [template = i32]
// CHECK:STDOUT:     @CallAlias.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallOnConstBoundMethod.decl: %CallOnConstBoundMethod.type = fn_decl @CallOnConstBoundMethod [template = constants.%CallOnConstBoundMethod] {
// CHECK:STDOUT:     %int.make_type_32.loc34: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc34_32.1: type = value_of_initializer %int.make_type_32.loc34 [template = i32]
// CHECK:STDOUT:     %.loc34_32.2: type = converted %int.make_type_32.loc34, %.loc34_32.1 [template = i32]
// CHECK:STDOUT:     @CallOnConstBoundMethod.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallWithAddr.decl: %CallWithAddr.type = fn_decl @CallWithAddr [template = constants.%CallWithAddr] {
// CHECK:STDOUT:     %int.make_type_32.loc38: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc38_22.1: type = value_of_initializer %int.make_type_32.loc38 [template = i32]
// CHECK:STDOUT:     %.loc38_22.2: type = converted %int.make_type_32.loc38, %.loc38_22.1 [template = i32]
// CHECK:STDOUT:     @CallWithAddr.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFThroughPointer.decl: %CallFThroughPointer.type = fn_decl @CallFThroughPointer [template = constants.%CallFThroughPointer] {
// CHECK:STDOUT:     %Class.ref.loc43: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc43_32: type = ptr_type %Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc43_24.1: %.2 = param p
// CHECK:STDOUT:     @CallFThroughPointer.%p: %.2 = bind_name p, %p.loc43_24.1
// CHECK:STDOUT:     %int.make_type_32.loc43: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc43_38.1: type = value_of_initializer %int.make_type_32.loc43 [template = i32]
// CHECK:STDOUT:     %.loc43_38.2: type = converted %int.make_type_32.loc43, %.loc43_38.1 [template = i32]
// CHECK:STDOUT:     @CallFThroughPointer.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGThroughPointer.decl: %CallGThroughPointer.type = fn_decl @CallGThroughPointer [template = constants.%CallGThroughPointer] {
// CHECK:STDOUT:     %Class.ref.loc47: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %.loc47_32: type = ptr_type %Class [template = constants.%.2]
// CHECK:STDOUT:     %p.loc47_24.1: %.2 = param p
// CHECK:STDOUT:     @CallGThroughPointer.%p: %.2 = bind_name p, %p.loc47_24.1
// CHECK:STDOUT:     %int.make_type_32.loc47: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc47_38.1: type = value_of_initializer %int.make_type_32.loc47 [template = i32]
// CHECK:STDOUT:     %.loc47_38.2: type = converted %int.make_type_32.loc47, %.loc47_38.1 [template = i32]
// CHECK:STDOUT:     @CallGThroughPointer.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {
// CHECK:STDOUT:     %Class.ref.loc51: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:     @Make.%return: ref %Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallFOnInitializingExpr.decl: %CallFOnInitializingExpr.type = fn_decl @CallFOnInitializingExpr [template = constants.%CallFOnInitializingExpr] {
// CHECK:STDOUT:     %int.make_type_32.loc53: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc53_33.1: type = value_of_initializer %int.make_type_32.loc53 [template = i32]
// CHECK:STDOUT:     %.loc53_33.2: type = converted %int.make_type_32.loc53, %.loc53_33.1 [template = i32]
// CHECK:STDOUT:     @CallFOnInitializingExpr.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallGOnInitializingExpr.decl: %CallGOnInitializingExpr.type = fn_decl @CallGOnInitializingExpr [template = constants.%CallGOnInitializingExpr] {
// CHECK:STDOUT:     %int.make_type_32.loc57: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc57_33.1: type = value_of_initializer %int.make_type_32.loc57 [template = i32]
// CHECK:STDOUT:     %.loc57_33.2: type = converted %int.make_type_32.loc57, %.loc57_33.1 [template = i32]
// CHECK:STDOUT:     @CallGOnInitializingExpr.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc12_8.1: %Class = param self
// CHECK:STDOUT:     %self.loc12_8.2: %Class = bind_name self, %self.loc12_8.1
// CHECK:STDOUT:     %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_25.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:     %.loc12_25.2: type = converted %int.make_type_32.loc12, %.loc12_25.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc13_23: type = ptr_type %Class [template = constants.%.2]
// CHECK:STDOUT:     %self.loc13_13.1: %.2 = param self
// CHECK:STDOUT:     %self.loc13_13.3: %.2 = bind_name self, %self.loc13_13.1
// CHECK:STDOUT:     %.loc13_8: %.2 = addr_pattern %self.loc13_13.3
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_32.loc13, %.loc13_31.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc13: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, %F.decl [template = constants.%F]
// CHECK:STDOUT:   %A: %F.type = bind_alias A, %F.decl [template = constants.%F]
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_10.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_10.2: type = converted %int.make_type_32.loc17, %.loc17_10.1 [template = i32]
// CHECK:STDOUT:   %.loc17_8: %.3 = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .A = %A
// CHECK:STDOUT:   .k = %.loc17_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: %Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self
// CHECK:STDOUT:   %k.ref: %.3 = name_ref k, @Class.%.loc17_8 [template = @Class.%.loc17_8]
// CHECK:STDOUT:   %.loc21_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc21_14.2: i32 = bind_value %.loc21_14.1
// CHECK:STDOUT:   return %.loc21_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr @Class.%self.loc13_13.3: %.2]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%c: %Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc27_11: <bound method> = bound_method %c.ref, %F.ref
// CHECK:STDOUT:   %F.call: init i32 = call %.loc27_11(%c.ref)
// CHECK:STDOUT:   %.loc27_15.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc27_15.2: i32 = converted %F.call, %.loc27_15.1
// CHECK:STDOUT:   return %.loc27_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallAlias(%c: %Class) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class = name_ref c, %c
// CHECK:STDOUT:   %A.ref: %F.type = name_ref A, @Class.%A [template = constants.%F]
// CHECK:STDOUT:   %.loc31_11: <bound method> = bound_method %c.ref, %A.ref
// CHECK:STDOUT:   %F.call: init i32 = call %.loc31_11(%c.ref)
// CHECK:STDOUT:   %.loc31_15.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc31_15.2: i32 = converted %F.call, %.loc31_15.1
// CHECK:STDOUT:   return %.loc31_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallOnConstBoundMethod() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc35_17: i32 = int_literal 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc35_18.1: %.4 = struct_literal (%.loc35_17)
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %.loc35_18.2: ref %Class = temporary_storage
// CHECK:STDOUT:   %.loc35_18.3: ref i32 = class_element_access %.loc35_18.2, element0
// CHECK:STDOUT:   %.loc35_18.4: init i32 = initialize_from %.loc35_17 to %.loc35_18.3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc35_18.5: init %Class = class_init (%.loc35_18.4), %.loc35_18.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc35_18.6: ref %Class = temporary %.loc35_18.2, %.loc35_18.5
// CHECK:STDOUT:   %.loc35_20.1: ref %Class = converted %.loc35_18.1, %.loc35_18.6
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc35_29: <bound method> = bound_method %.loc35_20.1, %F.ref
// CHECK:STDOUT:   %.loc35_20.2: %Class = bind_value %.loc35_20.1
// CHECK:STDOUT:   %F.call: init i32 = call %.loc35_29(%.loc35_20.2)
// CHECK:STDOUT:   %.loc35_33.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc35_33.2: i32 = converted %F.call, %.loc35_33.1
// CHECK:STDOUT:   return %.loc35_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWithAddr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %c.var: ref %Class = var c
// CHECK:STDOUT:   %c: ref %Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc40_11: <bound method> = bound_method %c.ref, %G.ref
// CHECK:STDOUT:   %.loc40_10: %.2 = addr_of %c.ref
// CHECK:STDOUT:   %G.call: init i32 = call %.loc40_11(%.loc40_10)
// CHECK:STDOUT:   %.loc40_15.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc40_15.2: i32 = converted %G.call, %.loc40_15.1
// CHECK:STDOUT:   return %.loc40_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFThroughPointer(%p: %.2) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc44_11.1: ref %Class = deref %p.ref
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc44_14: <bound method> = bound_method %.loc44_11.1, %F.ref
// CHECK:STDOUT:   %.loc44_11.2: %Class = bind_value %.loc44_11.1
// CHECK:STDOUT:   %F.call: init i32 = call %.loc44_14(%.loc44_11.2)
// CHECK:STDOUT:   %.loc44_18.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc44_18.2: i32 = converted %F.call, %.loc44_18.1
// CHECK:STDOUT:   return %.loc44_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGThroughPointer(%p: %.2) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.2 = name_ref p, %p
// CHECK:STDOUT:   %.loc48_11.1: ref %Class = deref %p.ref
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc48_14: <bound method> = bound_method %.loc48_11.1, %G.ref
// CHECK:STDOUT:   %.loc48_11.2: %.2 = addr_of %.loc48_11.1
// CHECK:STDOUT:   %G.call: init i32 = call %.loc48_14(%.loc48_11.2)
// CHECK:STDOUT:   %.loc48_18.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc48_18.2: i32 = converted %G.call, %.loc48_18.1
// CHECK:STDOUT:   return %.loc48_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallFOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc54_14.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc54_14.1
// CHECK:STDOUT:   %.loc54_14.2: ref %Class = temporary %.loc54_14.1, %Make.call
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Class.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc54_16: <bound method> = bound_method %.loc54_14.2, %F.ref
// CHECK:STDOUT:   %.loc54_14.3: %Class = bind_value %.loc54_14.2
// CHECK:STDOUT:   %F.call: init i32 = call %.loc54_16(%.loc54_14.3)
// CHECK:STDOUT:   %.loc54_20.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc54_20.2: i32 = converted %F.call, %.loc54_20.1
// CHECK:STDOUT:   return %.loc54_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallGOnInitializingExpr() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [template = constants.%Make]
// CHECK:STDOUT:   %.loc58_14.1: ref %Class = temporary_storage
// CHECK:STDOUT:   %Make.call: init %Class = call %Make.ref() to %.loc58_14.1
// CHECK:STDOUT:   %.loc58_14.2: ref %Class = temporary %.loc58_14.1, %Make.call
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc58_16: <bound method> = bound_method %.loc58_14.2, %G.ref
// CHECK:STDOUT:   %.loc58_14.3: %.2 = addr_of %.loc58_14.2
// CHECK:STDOUT:   %G.call: init i32 = call %.loc58_16(%.loc58_14.3)
// CHECK:STDOUT:   %.loc58_20.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc58_20.2: i32 = converted %G.call, %.loc58_20.1
// CHECK:STDOUT:   return %.loc58_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
