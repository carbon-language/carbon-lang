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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/shadow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/shadow.carbon

fn A();

namespace N;
fn N.A();

namespace N.M;

fn N.M.B() -> i32 {
  // This is N.A, not package.A.
  A();
  if (true) {
    var A: i32 = 0;
    // This is the local A.
    // TODO: Decide if we should warn or error on the shadowing of N.A here.
    return A;
  }
  return 0;
}

// CHECK:STDOUT: --- shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type.00d: type = fn_type @A.loc15 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.1db: %A.type.00d = struct_value () [concrete]
// CHECK:STDOUT:   %A.type.0e8: type = fn_type @A.loc18 [concrete]
// CHECK:STDOUT:   %A.281: %A.type.0e8 = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.2, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.1, @Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.e6a, @Op.1(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl.loc15
// CHECK:STDOUT:     .N = %N
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc15: %A.type.00d = fn_decl @A.loc15 [concrete = constants.%A.1db] {} {}
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl.loc18
// CHECK:STDOUT:     .M = %M
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc18: %A.type.0e8 = fn_decl @A.loc18 [concrete = constants.%A.281] {} {}
// CHECK:STDOUT:   %M: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.loc15();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.loc18();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc24: %A.type.0e8 = name_ref A, file.%A.decl.loc18 [concrete = constants.%A.281]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref.loc24()
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %A.patt: %pattern_type.7ce = binding_pattern A [concrete]
// CHECK:STDOUT:     %A.var_patt: %pattern_type.7ce = var_pattern %A.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.var: ref %i32 = var %A.var_patt
// CHECK:STDOUT:   %int_0.loc26: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc26: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc26_5.1: <bound method> = bound_method %int_0.loc26, %impl.elem0.loc26 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_5.2: <bound method> = bound_method %int_0.loc26, %specific_fn.loc26 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %i32 = call %bound_method.loc26_5.2(%int_0.loc26) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc26_5: init %i32 = converted %int_0.loc26, %int.convert_checked.loc26 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %A.var, %.loc26_5
// CHECK:STDOUT:   %.loc26_12: type = splice_block %i32.loc26 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: ref %i32 = bind_name A, %A.var
// CHECK:STDOUT:   %A.ref.loc29: ref %i32 = name_ref A, %A
// CHECK:STDOUT:   %.loc29: %i32 = bind_value %A.ref.loc29
// CHECK:STDOUT:   %Op.bound.loc26_5.1: <bound method> = bound_method %A.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.e6a, @Op.1(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_5.3: <bound method> = bound_method %A.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc26_5.1: %ptr.235 = addr_of %A.var
// CHECK:STDOUT:   %no_op.loc26_5.1: init %empty_tuple.type = call %bound_method.loc26_5.3(%addr.loc26_5.1)
// CHECK:STDOUT:   return %.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %int_0.loc31: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc31: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc31_11.1: <bound method> = bound_method %int_0.loc31, %impl.elem0.loc31 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_11.2: <bound method> = bound_method %int_0.loc31, %specific_fn.loc31 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc31: init %i32 = call %bound_method.loc31_11.2(%int_0.loc31) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31_11.1: %i32 = value_of_initializer %int.convert_checked.loc31 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31_11.2: %i32 = converted %int_0.loc31, %.loc31_11.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %Op.bound.loc26_5.2: <bound method> = bound_method %A.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.e6a, @Op.1(constants.%i32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_5.4: <bound method> = bound_method %A.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc26_5.2: %ptr.235 = addr_of %A.var
// CHECK:STDOUT:   %no_op.loc26_5.2: init %empty_tuple.type = call %bound_method.loc26_5.4(%addr.loc26_5.2)
// CHECK:STDOUT:   return %.loc31_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
