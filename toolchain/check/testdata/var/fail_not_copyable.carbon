// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.599: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(%X) [concrete]
// CHECK:STDOUT:   %Op.type.9e1: type = fn_type @Op.2, @impl(%X) [concrete]
// CHECK:STDOUT:   %Op.33a: %Op.type.9e1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.d17: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %Destroy.facet.177: %Destroy.type = facet_value %X, (%Destroy.impl_witness.599) [concrete]
// CHECK:STDOUT:   %.bfc: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.177 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.d1f: <specific function> = specific_function %Op.33a, @Op.2(%X) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.35b: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl(String) [concrete]
// CHECK:STDOUT:   %Op.type.051: type = fn_type @Op.2, @impl(String) [concrete]
// CHECK:STDOUT:   %Op.5ef: %Op.type.051 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.71a: %Destroy.type = facet_value String, (%Destroy.impl_witness.35b) [concrete]
// CHECK:STDOUT:   %.c94: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.71a [concrete]
// CHECK:STDOUT:   %Op.specific_fn.93f: <specific function> = specific_function %Op.5ef, @Op.2(String) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.%Op.type (%Op.type.bc9) = import_ref Core//prelude/destroy, loc15_29, loaded [symbolic = @impl.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl [concrete]
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
// CHECK:STDOUT:     %X.ref.loc17: type = name_ref X, file.%X.decl [concrete = constants.%X]
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
// CHECK:STDOUT:   %X.ref.loc31: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %y: ref %X = bind_name y, %y.var
// CHECK:STDOUT:   %impl.elem0.loc31: %.bfc = impl_witness_access constants.%Destroy.impl_witness.599, element0 [concrete = constants.%Op.33a]
// CHECK:STDOUT:   %bound_method.loc31_3.1: <bound method> = bound_method %y.var, %impl.elem0.loc31
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Op.2(constants.%X) [concrete = constants.%Op.specific_fn.d1f]
// CHECK:STDOUT:   %bound_method.loc31_3.2: <bound method> = bound_method %y.var, %specific_fn.loc31
// CHECK:STDOUT:   %addr.loc31: %ptr.d17 = addr_of %y.var
// CHECK:STDOUT:   %no_op.loc31: init %empty_tuple.type = call %bound_method.loc31_3.2(%addr.loc31)
// CHECK:STDOUT:   %impl.elem0.loc24: %.c94 = impl_witness_access constants.%Destroy.impl_witness.35b, element0 [concrete = constants.%Op.5ef]
// CHECK:STDOUT:   %bound_method.loc24_3.1: <bound method> = bound_method %s.var, %impl.elem0.loc24
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Op.2(String) [concrete = constants.%Op.specific_fn.93f]
// CHECK:STDOUT:   %bound_method.loc24_3.2: <bound method> = bound_method %s.var, %specific_fn.loc24
// CHECK:STDOUT:   %addr.loc24: %ptr.a45 = addr_of %s.var
// CHECK:STDOUT:   %no_op.loc24: init %empty_tuple.type = call %bound_method.loc24_3.2(%addr.loc24)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
