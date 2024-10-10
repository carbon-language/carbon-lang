// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_abstract.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_abstract.carbon

abstract class Abstract {
  var a: i32;
}

class Contains {
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+7]]:10: error: field has abstract type `Abstract`
  // CHECK:STDERR:   var a: Abstract;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE-8]]:1: note: class was declared abstract here
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: Abstract;
}

class Adapter {
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+7]]:3: error: adapted type `Abstract` is an abstract type
  // CHECK:STDERR:   adapt Abstract;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE-19]]:1: note: class was declared abstract here
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  adapt Abstract;
}

// We could reject this outright if `Abstract`'s definition is available at
// this point, but for consistency, we aren't at the moment.
fn Param(a: Abstract);

fn Call(p: Abstract*) {
  Param(*p);
}

fn Param(a: Abstract) {
}

class Derived {
  extend base: Abstract;

  var d: i32;
}

fn Make() -> Derived {
  // TODO: This should be valid, and should construct an instance of `partial Abstract` as the base.
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+7]]:10: error: initialization of abstract type `Abstract`
  // CHECK:STDERR:   return {.base = {.a = 1}, .d = 7};
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE-48]]:1: note: class was declared abstract here
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return {.base = {.a = 1}, .d = 7};
}

fn Return() -> Abstract {
  var d: Derived;
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+7]]:3: error: initialization of abstract type `Abstract`
  // CHECK:STDERR:   return d.base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE-60]]:1: note: class was declared abstract here
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return d.base;
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.a);
}

fn Var() {
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE+6]]:10: error: variable has abstract type `Abstract`
  // CHECK:STDERR:   var v: Abstract;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_abstract.carbon:[[@LINE-75]]:1: note: class was declared abstract here
  // CHECK:STDERR: abstract class Abstract {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  var v: Abstract;
  // TODO: The initializer here should be invalid (though the `let` is itself valid)
  let l: Abstract = {};
}

// CHECK:STDOUT: --- fail_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Abstract, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %Contains: type = class_type @Contains [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %Adapter: type = class_type @Adapter [template]
// CHECK:STDOUT:   %Param.type: type = fn_type @Param [template]
// CHECK:STDOUT:   %Param: %Param.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type %Abstract [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Derived, %Abstract [template]
// CHECK:STDOUT:   %.8: type = unbound_element_type %Derived, i32 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.base: %Abstract, .d: i32} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [template]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [template]
// CHECK:STDOUT:   %.11: type = struct_type {.base: %.5, .d: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.11 [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.9 [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.16: type = struct_type {.base: %.3, .d: i32} [template]
// CHECK:STDOUT:   %Return.type: type = fn_type @Return [template]
// CHECK:STDOUT:   %Return: %Return.type = struct_value () [template]
// CHECK:STDOUT:   %.17: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.18: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [template]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [template]
// CHECK:STDOUT:   %.19: type = ptr_type %.18 [template]
// CHECK:STDOUT:   %Var.type: type = fn_type @Var [template]
// CHECK:STDOUT:   %Var: %Var.type = struct_value () [template]
// CHECK:STDOUT:   %.20: type = struct_type {} [template]
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
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:     .Contains = %Contains.decl
// CHECK:STDOUT:     .Adapter = %Adapter.decl
// CHECK:STDOUT:     .Param = %Param.decl.loc39
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Return = %Return.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:     .Var = %Var.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [template = constants.%Abstract] {} {}
// CHECK:STDOUT:   %Contains.decl: type = class_decl @Contains [template = constants.%Contains] {} {}
// CHECK:STDOUT:   %Adapter.decl: type = class_decl @Adapter [template = constants.%Adapter] {} {}
// CHECK:STDOUT:   %Param.decl.loc39: %Param.type = fn_decl @Param [template = constants.%Param] {
// CHECK:STDOUT:     %a.patt: %Abstract = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Abstract.ref.loc39: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:     %a.param.loc39: %Abstract = param a, runtime_param0
// CHECK:STDOUT:     %a.loc39: %Abstract = bind_name a, %a.param.loc39
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %p.patt: %.6 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:     %.loc41: type = ptr_type %Abstract [template = constants.%.6]
// CHECK:STDOUT:     %p.param: %.6 = param p, runtime_param0
// CHECK:STDOUT:     %p: %.6 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl.loc45: %Param.type = fn_decl @Param [template = constants.%Param] {
// CHECK:STDOUT:     %a.patt: %Abstract = binding_pattern a
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Abstract.ref.loc45: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:     %a.param.loc45: %Abstract = param a, runtime_param0
// CHECK:STDOUT:     %a.loc45: %Abstract = bind_name a, %a.param.loc45
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [template = constants.%Make] {} {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %return: ref %Derived = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Return.decl: %Return.type = fn_decl @Return [template = constants.%Return] {} {
// CHECK:STDOUT:     %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:     %return: ref %Abstract = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [template = constants.%Access] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d.param: %Derived = param d, runtime_param0
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %int.make_type_32.loc78_27: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc78_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc78_35.1: %.17 = tuple_literal (%int.make_type_32.loc78_27, %int.make_type_32.loc78_32)
// CHECK:STDOUT:     %.loc78_35.2: type = value_of_initializer %int.make_type_32.loc78_27 [template = i32]
// CHECK:STDOUT:     %.loc78_35.3: type = converted %int.make_type_32.loc78_27, %.loc78_35.2 [template = i32]
// CHECK:STDOUT:     %.loc78_35.4: type = value_of_initializer %int.make_type_32.loc78_32 [template = i32]
// CHECK:STDOUT:     %.loc78_35.5: type = converted %int.make_type_32.loc78_32, %.loc78_35.4 [template = i32]
// CHECK:STDOUT:     %.loc78_35.6: type = converted %.loc78_35.1, constants.%.18 [template = constants.%.18]
// CHECK:STDOUT:     %return: ref %.18 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Var.decl: %Var.type = fn_decl @Var [template = constants.%Var] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Contains {
// CHECK:STDOUT:   %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:   %.loc23: <error> = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc24: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Contains
// CHECK:STDOUT:   .a = %.loc23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Adapter {
// CHECK:STDOUT:   %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:   adapt_decl <error>
// CHECK:STDOUT:   %.loc35: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Adapter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Abstract.ref: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:   %.loc49: %.7 = base_decl %Abstract, element0 [template]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc51_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc51_10.2: type = converted %int.make_type_32, %.loc51_10.1 [template = i32]
// CHECK:STDOUT:   %.loc51_8: %.8 = field_decl d, element1 [template]
// CHECK:STDOUT:   %.loc52: <witness> = complete_type_witness %.9 [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc49
// CHECK:STDOUT:   .d = %.loc51_8
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Param(%a.loc45: %Abstract) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p: %.6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Param.ref: %Param.type = name_ref Param, file.%Param.decl.loc39 [template = constants.%Param]
// CHECK:STDOUT:   %p.ref: %.6 = name_ref p, %p
// CHECK:STDOUT:   %.loc42: ref %Abstract = deref %p.ref
// CHECK:STDOUT:   %Param.call: init %.1 = call %Param.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc63_25: i32 = int_literal 1 [template = constants.%.14]
// CHECK:STDOUT:   %.loc63_26: %.3 = struct_literal (%.loc63_25)
// CHECK:STDOUT:   %.loc63_34: i32 = int_literal 7 [template = constants.%.15]
// CHECK:STDOUT:   %.loc63_35: %.16 = struct_literal (%.loc63_26, %.loc63_34)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Return() -> %return: %Abstract {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:   %d.var: ref %Derived = var d
// CHECK:STDOUT:   %d: ref %Derived = bind_name d, %d.var
// CHECK:STDOUT:   %d.ref: ref %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %.7 = name_ref base, @Derived.%.loc49 [template = @Derived.%.loc49]
// CHECK:STDOUT:   %.loc75: ref %Abstract = class_element_access %d.ref, element0
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d: %Derived) -> %return: %.18 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc79_11: %Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc79_12: %.8 = name_ref d, @Derived.%.loc51_8 [template = @Derived.%.loc51_8]
// CHECK:STDOUT:   %.loc79_12.1: ref i32 = class_element_access %d.ref.loc79_11, element1
// CHECK:STDOUT:   %.loc79_12.2: i32 = bind_value %.loc79_12.1
// CHECK:STDOUT:   %d.ref.loc79_16: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %.7 = name_ref base, @Derived.%.loc49 [template = @Derived.%.loc49]
// CHECK:STDOUT:   %.loc79_17: ref %Abstract = class_element_access %d.ref.loc79_16, element0
// CHECK:STDOUT:   %.loc79_24: <error> = tuple_literal (%.loc79_12.2, <error>)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Var() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Abstract.ref.loc89: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   %Abstract.ref.loc91: type = name_ref Abstract, file.%Abstract.decl [template = constants.%Abstract]
// CHECK:STDOUT:   %.loc91: %.20 = struct_literal ()
// CHECK:STDOUT:   %l: %Abstract = bind_name l, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
