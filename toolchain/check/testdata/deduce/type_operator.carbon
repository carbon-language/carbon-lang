// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/type_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/type_operator.carbon

// --- pointer.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: T*) -> T { return F(p); }

fn G(p: C*) -> C {
  return F(p);
}

// --- const.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: const T*) -> T { return F(p); }

fn G(p: const C*) -> C {
  return F(p);
}

// --- nonconst_from_const.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: T*) -> T { return F(p); }

fn G(p: const C*) -> const C {
  return F(p);
}

// --- fail_const_from_nonconst.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: const T*) -> T { return F(p); }

fn G(p: C*) -> const C {
  // CHECK:STDERR: fail_const_from_nonconst.carbon:[[@LINE+6]]:10: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   return F(p);
  // CHECK:STDERR:          ^~
  // CHECK:STDERR: fail_const_from_nonconst.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! type](p: const T*) -> T { return F(p); }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return F(p);
}

// CHECK:STDOUT: --- pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.5: type = ptr_type %C [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_6.1, runtime_param<invalid> [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%.loc6_20.2 (%.3) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%.loc6_20.2 (%.3) = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_19: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_20.1: type = ptr_type %T [symbolic = %.loc6_20.2 (constants.%.3)]
// CHECK:STDOUT:     %T.ref.loc6_26: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%.loc6_20.2 (%.3) = value_param runtime_param0
// CHECK:STDOUT:     %p: @F.%.loc6_20.2 (%.3) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %.5 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.5 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_9: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc8_10: type = ptr_type %C [template = constants.%.5]
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %p.param: %.5 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.5 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc6_20.2: type = ptr_type @F.%T.loc6_6.2 (%T) [symbolic = %.loc6_20.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_37.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.2) [symbolic = %.loc6_37.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%p.param_patt: @F.%.loc6_20.2 (%.3)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%.loc6_20.2 (%.3) = name_ref p, %p
// CHECK:STDOUT:     %.loc6_37.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %.loc6_37.2 (constants.%.4)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.2 (%T) = call %.loc6_37.1(%p.ref)
// CHECK:STDOUT:     %.loc6_41.1: @F.%T.loc6_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_41.2: @F.%T.loc6_6.2 (%T) = converted %F.call, %.loc6_41.1
// CHECK:STDOUT:     return %.loc6_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %.5) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %.5 = name_ref p, %p
// CHECK:STDOUT:   %.loc9: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_13.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %.loc9(%p.ref) to %.loc8_13.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_20.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_37.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc6_6.2) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_20.2 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %.loc6_20.2 => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_37.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = const_type %T [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.6: type = const_type %C [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.6 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.9: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_6.1, runtime_param<invalid> [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%.loc6_26.2 (%.4) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%.loc6_26.2 (%.4) = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_25: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_19.1: type = const_type %T [symbolic = %.loc6_19.2 (constants.%.3)]
// CHECK:STDOUT:     %.loc6_26.1: type = ptr_type %.3 [symbolic = %.loc6_26.2 (constants.%.4)]
// CHECK:STDOUT:     %T.ref.loc6_32: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%.loc6_26.2 (%.4) = value_param runtime_param0
// CHECK:STDOUT:     %p: @F.%.loc6_26.2 (%.4) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %.7 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.7 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc8_9: type = const_type %C [template = constants.%.6]
// CHECK:STDOUT:     %.loc8_16: type = ptr_type %.6 [template = constants.%.7]
// CHECK:STDOUT:     %C.ref.loc8_22: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %p.param: %.7 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.7 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc6_19.2: type = const_type @F.%T.loc6_6.2 (%T) [symbolic = %.loc6_19.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc6_26.2: type = ptr_type @F.%.loc6_19.2 (%.3) [symbolic = %.loc6_26.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_43.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.2) [symbolic = %.loc6_43.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%p.param_patt: @F.%.loc6_26.2 (%.4)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%.loc6_26.2 (%.4) = name_ref p, %p
// CHECK:STDOUT:     %.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %.loc6_43.2 (constants.%.5)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.2 (%T) = call %.loc6_43.1(%p.ref)
// CHECK:STDOUT:     %.loc6_47.1: @F.%T.loc6_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_47.2: @F.%T.loc6_6.2 (%T) = converted %F.call, %.loc6_47.1
// CHECK:STDOUT:     return %.loc6_47.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %.7) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %.7 = name_ref p, %p
// CHECK:STDOUT:   %.loc9: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%.9]
// CHECK:STDOUT:   %.loc8_19.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %.loc9(%p.ref) to %.loc8_19.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_19.2 => constants.%.3
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_43.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc6_6.2) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_19.2 => constants.%.3
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %.loc6_19.2 => constants.%.6
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_43.2 => constants.%.9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nonconst_from_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.5: type = const_type %C [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.8: <specific function> = specific_function %F, @F(%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_6.1, runtime_param<invalid> [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%.loc6_20.2 (%.3) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%.loc6_20.2 (%.3) = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_19: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_20.1: type = ptr_type %T [symbolic = %.loc6_20.2 (constants.%.3)]
// CHECK:STDOUT:     %T.ref.loc6_26: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%.loc6_20.2 (%.3) = value_param runtime_param0
// CHECK:STDOUT:     %p: @F.%.loc6_20.2 (%.3) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %.6 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.6 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.5 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc8_9: type = const_type %C [template = constants.%.5]
// CHECK:STDOUT:     %.loc8_16: type = ptr_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     %C.ref.loc8_28: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc8_22: type = const_type %C [template = constants.%.5]
// CHECK:STDOUT:     %p.param: %.6 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.6 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.5 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc6_20.2: type = ptr_type @F.%T.loc6_6.2 (%T) [symbolic = %.loc6_20.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_37.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.2) [symbolic = %.loc6_37.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%p.param_patt: @F.%.loc6_20.2 (%.3)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%.loc6_20.2 (%.3) = name_ref p, %p
// CHECK:STDOUT:     %.loc6_37.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %.loc6_37.2 (constants.%.4)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.2 (%T) = call %.loc6_37.1(%p.ref)
// CHECK:STDOUT:     %.loc6_41.1: @F.%T.loc6_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_41.2: @F.%T.loc6_6.2 (%T) = converted %F.call, %.loc6_41.1
// CHECK:STDOUT:     return %.loc6_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %.6) -> %return: %.5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %.6 = name_ref p, %p
// CHECK:STDOUT:   %.loc9: <specific function> = specific_function %F.ref, @F(constants.%.5) [template = constants.%.8]
// CHECK:STDOUT:   %.loc8_19.2: ref %.5 = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %.5 = call %.loc9(%p.ref) to %.loc8_19.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_20.2 => constants.%.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_37.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc6_6.2) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_20.2 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.5) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%.5
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%.5
// CHECK:STDOUT:   %.loc6_20.2 => constants.%.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_37.2 => constants.%.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_const_from_nonconst.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %.3: type = const_type %T [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type %C [template]
// CHECK:STDOUT:   %.7: type = const_type %C [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc6_6.1, runtime_param<invalid> [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%.loc6_26.2 (%.4) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%.loc6_26.2 (%.4) = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_25: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_19.1: type = const_type %T [symbolic = %.loc6_19.2 (constants.%.3)]
// CHECK:STDOUT:     %.loc6_26.1: type = ptr_type %.3 [symbolic = %.loc6_26.2 (constants.%.4)]
// CHECK:STDOUT:     %T.ref.loc6_32: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%.loc6_26.2 (%.4) = value_param runtime_param0
// CHECK:STDOUT:     %p: @F.%.loc6_26.2 (%.4) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %.6 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.6 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %.7 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.7 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_9: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc8_10: type = ptr_type %C [template = constants.%.6]
// CHECK:STDOUT:     %C.ref.loc8_22: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc8_16: type = const_type %C [template = constants.%.7]
// CHECK:STDOUT:     %p.param: %.6 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.6 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.7 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %.7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %.loc6_19.2: type = const_type @F.%T.loc6_6.2 (%T) [symbolic = %.loc6_19.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc6_26.2: type = ptr_type @F.%.loc6_19.2 (%.3) [symbolic = %.loc6_26.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_43.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.2) [symbolic = %.loc6_43.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%p.param_patt: @F.%.loc6_26.2 (%.4)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%.loc6_26.2 (%.4) = name_ref p, %p
// CHECK:STDOUT:     %.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %.loc6_43.2 (constants.%.5)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.2 (%T) = call %.loc6_43.1(%p.ref)
// CHECK:STDOUT:     %.loc6_47.1: @F.%T.loc6_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_47.2: @F.%T.loc6_6.2 (%T) = converted %F.call, %.loc6_47.1
// CHECK:STDOUT:     return %.loc6_47.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %.6) -> %return: %.7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %.6 = name_ref p, %p
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_19.2 => constants.%.3
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc6_43.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc6_6.2) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %.loc6_19.2 => constants.%.3
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
