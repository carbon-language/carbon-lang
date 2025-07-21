// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_not_copyable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_not_copyable.carbon

class X {
}

fn F(x: X) {
  // TODO: Strings should eventually be copyable, once we decide how to
  // represent them.
  // CHECK:STDERR: fail_not_copyable.carbon:[[@LINE+4]]:19: error: cannot copy value of type `String` [CopyOfUncopyableType]
  // CHECK:STDERR:   var s: String = "hello";
  // CHECK:STDERR:                   ^~~~~~~
  // CHECK:STDERR:
  var s: String = "hello";

  // TODO: Decide on rules for when classes are copyable.
  // CHECK:STDERR: fail_not_copyable.carbon:[[@LINE+4]]:14: error: cannot copy value of type `X` [CopyOfUncopyableType]
  // CHECK:STDERR:   var y: X = x;
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  var y: X = x;
}

// CHECK:STDOUT: --- fail_not_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.019: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.a45: type = ptr_type String [concrete]
// CHECK:STDOUT:   %pattern_type.b05: type = pattern_type String [concrete]
// CHECK:STDOUT:   %str: String = string_literal "hello" [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.9e1: type = fn_type @Op.1, @Destroy.impl(%X) [concrete]
// CHECK:STDOUT:   %Op.33a: %Op.type.9e1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %Op.specific_fn.d1f: <specific function> = specific_function %Op.33a, @Op.1(%X) [concrete]
// CHECK:STDOUT:   %Op.type.051: type = fn_type @Op.1, @Destroy.impl(String) [concrete]
// CHECK:STDOUT:   %Op.5ef: %Op.type.051 = struct_value () [concrete]
// CHECK:STDOUT:   %Op.specific_fn.93f: <specific function> = specific_function %Op.5ef, @Op.1(String) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.patt: %pattern_type.019 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.019 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %X = value_param call_param0
// CHECK:STDOUT:     %X.ref.loc18: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:     %x: %X = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: %pattern_type.b05 = binding_pattern s [concrete]
// CHECK:STDOUT:     %s.var_patt: %pattern_type.b05 = var_pattern %s.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref String = var %s.var_patt
// CHECK:STDOUT:   %str: String = string_literal "hello" [concrete = constants.%str]
// CHECK:STDOUT:   assign %s.var, <error>
// CHECK:STDOUT:   %s: ref String = bind_name s, %s.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.019 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.019 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %X = var %y.var_patt
// CHECK:STDOUT:   %x.ref: %X = name_ref x, %x
// CHECK:STDOUT:   assign %y.var, <error>
// CHECK:STDOUT:   %X.ref.loc32: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %y: ref %X = bind_name y, %y.var
// CHECK:STDOUT:   %Op.bound.loc32: <bound method> = bound_method %y.var, constants.%Op.33a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.33a, @Op.1(constants.%X) [concrete = constants.%Op.specific_fn.d1f]
// CHECK:STDOUT:   %bound_method.loc32: <bound method> = bound_method %y.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc32: %ptr.d17 = addr_of %y.var
// CHECK:STDOUT:   %no_op.loc32: init %empty_tuple.type = call %bound_method.loc32(%addr.loc32)
// CHECK:STDOUT:   %Op.bound.loc25: <bound method> = bound_method %s.var, constants.%Op.5ef
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.5ef, @Op.1(String) [concrete = constants.%Op.specific_fn.93f]
// CHECK:STDOUT:   %bound_method.loc25: <bound method> = bound_method %s.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc25: %ptr.a45 = addr_of %s.var
// CHECK:STDOUT:   %no_op.loc25: init %empty_tuple.type = call %bound_method.loc25(%addr.loc25)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
