// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
  // CHECK:STDERR: fail_const_from_nonconst.carbon:[[@LINE+7]]:10: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   return F(p);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_const_from_nonconst.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! type](p: const T*) -> T { return F(p); }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(p);
}

// CHECK:STDOUT: --- pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.d05: <witness> = complete_type_witness %ptr.019 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc6_16 (%pattern_type.afe) = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc6_16 (%pattern_type.afe) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_23 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_23 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_26: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_20.1 (%ptr.79f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_20: type = splice_block %ptr.loc6_20.2 [symbolic = %ptr.loc6_20.1 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc6_19: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_20.2: type = ptr_type %T.ref.loc6_19 [symbolic = %ptr.loc6_20.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_20.1 (%ptr.79f) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.44a = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.44a = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc8_10: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc8_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc8_9 [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc6_20.1: type = ptr_type %T.loc6_6.1 [symbolic = %ptr.loc6_20.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type.loc6_16: type = pattern_type %ptr.loc6_20.1 [symbolic = %pattern_type.loc6_16 (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %pattern_type.loc6_23: type = pattern_type %T.loc6_6.1 [symbolic = %pattern_type.loc6_23 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc6_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %ptr.loc6_20.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.1) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_20.1 (%ptr.79f)) -> @F.%T.loc6_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_20.1 (%ptr.79f) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_37.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.1 (%T) = call %F.specific_fn.loc6_37.1(%p.ref)
// CHECK:STDOUT:     %.loc6_41.1: @F.%T.loc6_6.1 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_41.2: @F.%T.loc6_6.1 (%T) = converted %F.call, %.loc6_41.1
// CHECK:STDOUT:     return %.loc6_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.019) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %.loc8_13: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%p.ref) to %.loc8_13
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_13, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc8_13
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc6_20.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.afe
// CHECK:STDOUT:   %pattern_type.loc6_23 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.6e5
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc6_20.1 => constants.%ptr.019
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.44a
// CHECK:STDOUT:   %pattern_type.loc6_23 => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.d05
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.04a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %const.a1a: type = const_type %T [symbolic]
// CHECK:STDOUT:   %ptr.6d4: type = ptr_type %const.a1a [symbolic]
// CHECK:STDOUT:   %pattern_type.fa5: type = pattern_type %ptr.6d4 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.20b: <witness> = require_complete_type %ptr.6d4 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %const.668: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const.668 [concrete]
// CHECK:STDOUT:   %pattern_type.c0d: type = pattern_type %ptr.801 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.247: <witness> = complete_type_witness %ptr.801 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc6_16 (%pattern_type.fa5) = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc6_16 (%pattern_type.fa5) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_29 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_29 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_32: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_26.1 (%ptr.6d4) = value_param call_param0
// CHECK:STDOUT:     %.loc6_26: type = splice_block %ptr.loc6_26.2 [symbolic = %ptr.loc6_26.1 (constants.%ptr.6d4)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %const.loc6_19.2: type = const_type %T.ref.loc6_25 [symbolic = %const.loc6_19.1 (constants.%const.a1a)]
// CHECK:STDOUT:       %ptr.loc6_26.2: type = ptr_type %const.loc6_19.2 [symbolic = %ptr.loc6_26.1 (constants.%ptr.6d4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_26.1 (%ptr.6d4) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.c0d = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.c0d = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %p.param: %ptr.801 = value_param call_param0
// CHECK:STDOUT:     %.loc8_16: type = splice_block %ptr [concrete = constants.%ptr.801] {
// CHECK:STDOUT:       %C.ref.loc8_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref.loc8_15 [concrete = constants.%const.668]
// CHECK:STDOUT:       %ptr: type = ptr_type %const [concrete = constants.%ptr.801]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.801 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %const.loc6_19.1: type = const_type %T.loc6_6.1 [symbolic = %const.loc6_19.1 (constants.%const.a1a)]
// CHECK:STDOUT:   %ptr.loc6_26.1: type = ptr_type %const.loc6_19.1 [symbolic = %ptr.loc6_26.1 (constants.%ptr.6d4)]
// CHECK:STDOUT:   %pattern_type.loc6_16: type = pattern_type %ptr.loc6_26.1 [symbolic = %pattern_type.loc6_16 (constants.%pattern_type.fa5)]
// CHECK:STDOUT:   %pattern_type.loc6_29: type = pattern_type %T.loc6_6.1 [symbolic = %pattern_type.loc6_29 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc6_29 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %ptr.loc6_26.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.20b)]
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.1) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_26.1 (%ptr.6d4)) -> @F.%T.loc6_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_26.1 (%ptr.6d4) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.1 (%T) = call %F.specific_fn.loc6_43.1(%p.ref)
// CHECK:STDOUT:     %.loc6_47.1: @F.%T.loc6_6.1 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_47.2: @F.%T.loc6_6.1 (%T) = converted %F.call, %.loc6_47.1
// CHECK:STDOUT:     return %.loc6_47.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.801) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.801 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %.loc8_19: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%p.ref) to %.loc8_19
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_19, constants.%C.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr: %ptr.019 = addr_of %.loc8_19
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %C.as.Destroy.impl.Op.bound(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %const.loc6_19.1 => constants.%const.a1a
// CHECK:STDOUT:   %ptr.loc6_26.1 => constants.%ptr.6d4
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.fa5
// CHECK:STDOUT:   %pattern_type.loc6_29 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.20b
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%C
// CHECK:STDOUT:   %const.loc6_19.1 => constants.%const.668
// CHECK:STDOUT:   %ptr.loc6_26.1 => constants.%ptr.801
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.c0d
// CHECK:STDOUT:   %pattern_type.loc6_29 => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.247
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn.04a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nonconst_from_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.c0d: type = pattern_type %ptr.801 [concrete]
// CHECK:STDOUT:   %pattern_type.6af: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.486: <specific function> = specific_function %F, @F(%const) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.dff: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%const) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.206: %T.as.Destroy.impl.Op.type.dff = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.247: <witness> = complete_type_witness %ptr.801 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.206, @T.as.Destroy.impl.Op(%const) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc6_16 (%pattern_type.afe) = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc6_16 (%pattern_type.afe) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_23 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_23 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_26: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_20.1 (%ptr.79f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_20: type = splice_block %ptr.loc6_20.2 [symbolic = %ptr.loc6_20.1 (constants.%ptr.79f)] {
// CHECK:STDOUT:       %T.ref.loc6_19: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_20.2: type = ptr_type %T.ref.loc6_19 [symbolic = %ptr.loc6_20.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_20.1 (%ptr.79f) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.c0d = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.c0d = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6af = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6af = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_28: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc8_22: type = const_type %C.ref.loc8_28 [concrete = constants.%const]
// CHECK:STDOUT:     %p.param: %ptr.801 = value_param call_param0
// CHECK:STDOUT:     %.loc8_16: type = splice_block %ptr [concrete = constants.%ptr.801] {
// CHECK:STDOUT:       %C.ref.loc8_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc8_9: type = const_type %C.ref.loc8_15 [concrete = constants.%const]
// CHECK:STDOUT:       %ptr: type = ptr_type %const.loc8_9 [concrete = constants.%ptr.801]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.801 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const = out_param call_param1
// CHECK:STDOUT:     %return: ref %const = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc6_20.1: type = ptr_type %T.loc6_6.1 [symbolic = %ptr.loc6_20.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type.loc6_16: type = pattern_type %ptr.loc6_20.1 [symbolic = %pattern_type.loc6_16 (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %pattern_type.loc6_23: type = pattern_type %T.loc6_6.1 [symbolic = %pattern_type.loc6_23 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc6_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %ptr.loc6_20.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.1) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_20.1 (%ptr.79f)) -> @F.%T.loc6_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_20.1 (%ptr.79f) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_37.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.1 (%T) = call %F.specific_fn.loc6_37.1(%p.ref)
// CHECK:STDOUT:     %.loc6_41.1: @F.%T.loc6_6.1 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_41.2: @F.%T.loc6_6.1 (%T) = converted %F.call, %.loc6_41.1
// CHECK:STDOUT:     return %.loc6_41.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.801) -> %return.param: %const {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.801 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%const) [concrete = constants.%F.specific_fn.486]
// CHECK:STDOUT:   %.loc8_19: ref %const = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %const = call %F.specific_fn(%p.ref) to %.loc8_19
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_19, constants.%T.as.Destroy.impl.Op.206
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.206, @T.as.Destroy.impl.Op(constants.%const) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc8_19, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.801 = addr_of %.loc8_19
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc6_20.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.afe
// CHECK:STDOUT:   %pattern_type.loc6_23 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.6e5
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%const) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%const
// CHECK:STDOUT:   %ptr.loc6_20.1 => constants.%ptr.801
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.c0d
// CHECK:STDOUT:   %pattern_type.loc6_23 => constants.%pattern_type.6af
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.247
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.486
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_const_from_nonconst.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %const.a1a: type = const_type %T [symbolic]
// CHECK:STDOUT:   %ptr.6d4: type = ptr_type %const.a1a [symbolic]
// CHECK:STDOUT:   %pattern_type.fa5: type = pattern_type %ptr.6d4 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.20b: <witness> = require_complete_type %ptr.6d4 [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %const.668: type = const_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.6af: type = pattern_type %const.668 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc6_16 (%pattern_type.fa5) = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc6_16 (%pattern_type.fa5) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_29 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_29 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_32: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_26.1 (%ptr.6d4) = value_param call_param0
// CHECK:STDOUT:     %.loc6_26: type = splice_block %ptr.loc6_26.2 [symbolic = %ptr.loc6_26.1 (constants.%ptr.6d4)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %const.loc6_19.2: type = const_type %T.ref.loc6_25 [symbolic = %const.loc6_19.1 (constants.%const.a1a)]
// CHECK:STDOUT:       %ptr.loc6_26.2: type = ptr_type %const.loc6_19.2 [symbolic = %ptr.loc6_26.1 (constants.%ptr.6d4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_26.1 (%ptr.6d4) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.44a = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.44a = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6af = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6af = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const: type = const_type %C.ref.loc8_22 [concrete = constants.%const.668]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc8: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref.loc8_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc8_9 [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.668 = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.668 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc4: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %const.loc6_19.1: type = const_type %T.loc6_6.1 [symbolic = %const.loc6_19.1 (constants.%const.a1a)]
// CHECK:STDOUT:   %ptr.loc6_26.1: type = ptr_type %const.loc6_19.1 [symbolic = %ptr.loc6_26.1 (constants.%ptr.6d4)]
// CHECK:STDOUT:   %pattern_type.loc6_16: type = pattern_type %ptr.loc6_26.1 [symbolic = %pattern_type.loc6_16 (constants.%pattern_type.fa5)]
// CHECK:STDOUT:   %pattern_type.loc6_29: type = pattern_type %T.loc6_6.1 [symbolic = %pattern_type.loc6_29 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc6_29 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %ptr.loc6_26.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.20b)]
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.1) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_26.1 (%ptr.6d4)) -> @F.%T.loc6_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_26.1 (%ptr.6d4) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.1 (%T) = call %F.specific_fn.loc6_43.1(%p.ref)
// CHECK:STDOUT:     %.loc6_47.1: @F.%T.loc6_6.1 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc6_47.2: @F.%T.loc6_6.1 (%T) = converted %F.call, %.loc6_47.1
// CHECK:STDOUT:     return %.loc6_47.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.019) -> %return.param: %const.668 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %const.loc6_19.1 => constants.%const.a1a
// CHECK:STDOUT:   %ptr.loc6_26.1 => constants.%ptr.6d4
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.fa5
// CHECK:STDOUT:   %pattern_type.loc6_29 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.20b
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
