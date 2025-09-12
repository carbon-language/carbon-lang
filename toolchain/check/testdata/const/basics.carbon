// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/basics.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

class C;

//@dump-sem-ir-begin
fn A(p: const C**) -> const C** {
  return p;
}

fn B(p: const (C*)) -> const (C*) {
  return p;
}
//@dump-sem-ir-end

// --- collapse.carbon
library "[[@TEST_NAME]]";

class C;

// OK, `const (const C)` is the same type as `const C`.
//@dump-sem-ir-begin
// CHECK:STDERR: collapse.carbon:[[@LINE+4]]:23: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn F(p: const C**) -> const (const C)** {
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(p: const C**) -> const (const C)** {
  return p;
}
//@dump-sem-ir-end

// --- fail_collapse_in_error.carbon
library "[[@TEST_NAME]]";

class C;

// CHECK:STDERR: fail_collapse_in_error.carbon:[[@LINE+4]]:9: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn G(p: const (const C)**) -> C** {
// CHECK:STDERR:         ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(p: const (const C)**) -> C** {
  // CHECK:STDERR: fail_collapse_in_error.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `const C**` to `C**` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_collapse_in_error.carbon:[[@LINE+4]]:3: note: type `const C**` does not implement interface `Core.ImplicitAs(C**)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const.668: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const.668 [concrete]
// CHECK:STDOUT:   %ptr.564: type = ptr_type %ptr.801 [concrete]
// CHECK:STDOUT:   %pattern_type.28c: type = pattern_type %ptr.564 [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.578: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.906: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%T.578) [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.62b: %const.as.Copy.impl.Op.type.906 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f23: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.abf: %ptr.as.Copy.impl.Op.type.f23 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.2e1: <witness> = impl_witness imports.%Copy.impl_witness_table.a71, @ptr.as.Copy.impl(%ptr.801) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.032: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ptr.801) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.7ce: %ptr.as.Copy.impl.Op.type.032 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c61: %Copy.type = facet_value %ptr.564, (%Copy.impl_witness.2e1) [concrete]
// CHECK:STDOUT:   %.14c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c61 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.780: <specific function> = specific_function %ptr.as.Copy.impl.Op.7ce, @ptr.as.Copy.impl.Op(%ptr.801) [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %const.2b1: type = const_type %ptr.019 [concrete]
// CHECK:STDOUT:   %pattern_type.3a2: type = pattern_type %const.2b1 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a94: <witness> = impl_witness imports.%Copy.impl_witness_table.a71, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %Copy.facet.9a6: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.a94) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.7fd: <witness> = impl_witness imports.%Copy.impl_witness_table.91e, @const.as.Copy.impl(%Copy.facet.9a6) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.fbd: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%Copy.facet.9a6) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.9a4: %const.as.Copy.impl.Op.type.fbd = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b86: %Copy.type = facet_value %const.2b1, (%Copy.impl_witness.7fd) [concrete]
// CHECK:STDOUT:   %.e4f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.b86 [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %const.as.Copy.impl.Op.9a4, @const.as.Copy.impl.Op(%Copy.facet.9a6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7d6: @const.as.Copy.impl.%const.as.Copy.impl.Op.type (%const.as.Copy.impl.Op.type.906) = import_ref Core//prelude/parts/copy, loc16_31, loaded [symbolic = @const.as.Copy.impl.%const.as.Copy.impl.Op (constants.%const.as.Copy.impl.Op.62b)]
// CHECK:STDOUT:   %Copy.impl_witness_table.91e = impl_witness_table (%Core.import_ref.7d6), @const.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.de9: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.f23) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.abf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a71 = impl_witness_table (%Core.import_ref.de9), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.patt: %pattern_type.28c = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.28c = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.28c = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.28c = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc6_23: type = const_type %C.ref.loc6_29 [concrete = constants.%const.668]
// CHECK:STDOUT:     %ptr.loc6_30: type = ptr_type %const.loc6_23 [concrete = constants.%ptr.801]
// CHECK:STDOUT:     %ptr.loc6_31: type = ptr_type %ptr.loc6_30 [concrete = constants.%ptr.564]
// CHECK:STDOUT:     %p.param: %ptr.564 = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %ptr.loc6_17 [concrete = constants.%ptr.564] {
// CHECK:STDOUT:       %C.ref.loc6_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc6_9: type = const_type %C.ref.loc6_15 [concrete = constants.%const.668]
// CHECK:STDOUT:       %ptr.loc6_16: type = ptr_type %const.loc6_9 [concrete = constants.%ptr.801]
// CHECK:STDOUT:       %ptr.loc6_17: type = ptr_type %ptr.loc6_16 [concrete = constants.%ptr.564]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.564 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.564 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.564 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %p.patt: %pattern_type.3a2 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.3a2 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.3a2 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.3a2 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc10_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_32: type = ptr_type %C.ref.loc10_31 [concrete = constants.%ptr.019]
// CHECK:STDOUT:     %const.loc10_24: type = const_type %ptr.loc10_32 [concrete = constants.%const.2b1]
// CHECK:STDOUT:     %p.param: %const.2b1 = value_param call_param0
// CHECK:STDOUT:     %.loc10: type = splice_block %const.loc10_9 [concrete = constants.%const.2b1] {
// CHECK:STDOUT:       %C.ref.loc10_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc10_17: type = ptr_type %C.ref.loc10_16 [concrete = constants.%ptr.019]
// CHECK:STDOUT:       %const.loc10_9: type = const_type %ptr.loc10_17 [concrete = constants.%const.2b1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.2b1 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.2b1 = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.2b1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param: %ptr.564) -> %ptr.564 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.564 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.14c = impl_witness_access constants.%Copy.impl_witness.2e1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.7ce]
// CHECK:STDOUT:   %bound_method.loc7_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ptr.801) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.780]
// CHECK:STDOUT:   %bound_method.loc7_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.564 = call %bound_method.loc7_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p.param: %const.2b1) -> %const.2b1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.2b1 = name_ref p, %p
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%ptr.019, (constants.%Copy.impl_witness.a94) [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %.loc11: %Copy.type = converted constants.%ptr.019, %Copy.facet [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %impl.elem0: %.e4f = impl_witness_access constants.%Copy.impl_witness.7fd, element0 [concrete = constants.%const.as.Copy.impl.Op.9a4]
// CHECK:STDOUT:   %bound_method.loc11_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.Copy.impl.Op(constants.%Copy.facet.9a6) [concrete = constants.%const.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %const.as.Copy.impl.Op.call: init %const.2b1 = call %bound_method.loc11_10.2(%p.ref)
// CHECK:STDOUT:   return %const.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const.668: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const.668 [concrete]
// CHECK:STDOUT:   %ptr.564: type = ptr_type %ptr.801 [concrete]
// CHECK:STDOUT:   %pattern_type.28c: type = pattern_type %ptr.564 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f23: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.abf: %ptr.as.Copy.impl.Op.type.f23 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.2e1: <witness> = impl_witness imports.%Copy.impl_witness_table.a71, @ptr.as.Copy.impl(%ptr.801) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.032: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ptr.801) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.7ce: %ptr.as.Copy.impl.Op.type.032 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c61: %Copy.type = facet_value %ptr.564, (%Copy.impl_witness.2e1) [concrete]
// CHECK:STDOUT:   %.14c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c61 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.7ce, @ptr.as.Copy.impl.Op(%ptr.801) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.de9: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.f23) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.abf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a71 = impl_witness_table (%Core.import_ref.de9), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.patt: %pattern_type.28c = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.28c = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.28c = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.28c = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc11_36: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc11_30: type = const_type %C.ref.loc11_36 [concrete = constants.%const.668]
// CHECK:STDOUT:     %const.loc11_23: type = const_type %const.loc11_30 [concrete = constants.%const.668]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %const.loc11_23 [concrete = constants.%ptr.801]
// CHECK:STDOUT:     %ptr.loc11_39: type = ptr_type %ptr.loc11_38 [concrete = constants.%ptr.564]
// CHECK:STDOUT:     %p.param: %ptr.564 = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %ptr.loc11_17 [concrete = constants.%ptr.564] {
// CHECK:STDOUT:       %C.ref.loc11_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc11_9: type = const_type %C.ref.loc11_15 [concrete = constants.%const.668]
// CHECK:STDOUT:       %ptr.loc11_16: type = ptr_type %const.loc11_9 [concrete = constants.%ptr.801]
// CHECK:STDOUT:       %ptr.loc11_17: type = ptr_type %ptr.loc11_16 [concrete = constants.%ptr.564]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.564 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.564 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.564 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %ptr.564) -> %ptr.564 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.564 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.14c = impl_witness_access constants.%Copy.impl_witness.2e1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.7ce]
// CHECK:STDOUT:   %bound_method.loc12_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ptr.801) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.564 = call %bound_method.loc12_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
