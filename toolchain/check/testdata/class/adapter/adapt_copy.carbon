// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/adapt_copy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/adapt_copy.carbon

// TODO: Decide whether an adapter for a copyable type is copyable. As
// demonstrated in this test, our behavior is currently inconsistent.

// --- fail_adapt_copyable.carbon

library "[[@TEST_NAME]]";

class AdaptCopyable {
  adapt i32;
}

fn F(c: AdaptCopyable) -> AdaptCopyable {
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:26: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var d: AdaptCopyable = c;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:26: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var d: AdaptCopyable = c;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  var d: AdaptCopyable = c;
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return d;
}

fn InTuple(c: (AdaptCopyable, u32)) -> (AdaptCopyable, u32) {
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:33: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR:                                 ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:33: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR:                                 ^
  // CHECK:STDERR:
  var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return d;
}

// --- adapt_copyable_tuple.carbon

library "[[@TEST_NAME]]";

class AdaptTuple {
  adapt (i32, i32);
}

fn F(c: AdaptTuple) -> AdaptTuple {
  var d: AdaptTuple = c;
  return d;
}

fn InTuple(c: (AdaptTuple, u32)) -> (AdaptTuple, u32) {
  var d: (AdaptTuple, u32) = c;
  return d;
}

// --- fail_adapt_not_copyable.carbon

library "[[@TEST_NAME]]";

class Noncopyable {
  // TODO: Ensure this remains non-copyable once we have rules for class copyability.
}

class AdaptNoncopyable {
  adapt Noncopyable;
}

fn G(a: AdaptNoncopyable) -> AdaptNoncopyable {
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+7]]:29: error: cannot copy value of type `AdaptNoncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var b: AdaptNoncopyable = a;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+4]]:29: note: type `AdaptNoncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var b: AdaptNoncopyable = a;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR:
  var b: AdaptNoncopyable = a;
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptNoncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptNoncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return b;
}

// --- fail_adapt_not_copyable_indirect.carbon

library "[[@TEST_NAME]]";

class Noncopyable {
  // TODO: Ensure this remains non-copyable once we have rules for class copyability.
}

class AdaptNoncopyableIndirect {
  adapt (i32, Noncopyable, i32);
}

fn H(a: AdaptNoncopyableIndirect) -> AdaptNoncopyableIndirect {
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+10]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+7]]:3: note: type `Noncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+4]]:37: note: in copy of `AdaptNoncopyableIndirect` [InCopy]
  // CHECK:STDERR:   var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR:                                     ^
  // CHECK:STDERR:
  var b: AdaptNoncopyableIndirect = a;
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+10]]:3: error: cannot copy value of type `Noncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+7]]:3: note: type `Noncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+4]]:10: note: in copy of `AdaptNoncopyableIndirect` [InCopy]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return b;
}

// --- adapt_copyable_struct.carbon

library "[[@TEST_NAME]]";

class AdaptStruct {
  adapt {.e: i32, .f: i32};
}

fn I(g: AdaptStruct) -> AdaptStruct {
  var h: AdaptStruct = g;
  return h;
}

fn InTuple(c: (AdaptStruct, u32)) -> (AdaptStruct, u32) {
  var d: (AdaptStruct, u32) = c;
  return d;
}

// CHECK:STDOUT: --- fail_adapt_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptCopyable: type = class_type @AdaptCopyable [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.cdf: type = pattern_type %AdaptCopyable [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.567: %type_where = facet_value %AdaptCopyable, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.2be: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.567) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.1b8: %DestroyT.as_type.as.Destroy.impl.Op.type.2be = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.4c1: type = ptr_type %AdaptCopyable [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.6d3: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.1b8, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.567) [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.2a3: type = tuple_type (%AdaptCopyable, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.813: type = pattern_type %tuple.type.2a3 [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c30: type = ptr_type %tuple.type.2a3 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.cb3: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.ee7: %UInt.as.Copy.impl.Op.type.cb3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.61c: <witness> = impl_witness imports.%Copy.impl_witness_table.46a, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.31e: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.1bb: %UInt.as.Copy.impl.Op.type.31e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u32, (%Copy.impl_witness.61c) [concrete]
// CHECK:STDOUT:   %.fbf: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.1bb, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.6a4: %type_where = facet_value %tuple.type.2a3, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.285: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.6a4) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.8a01: %DestroyT.as_type.as.Destroy.impl.Op.type.285 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.610: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.8a01, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.6a4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.ba6: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.cb3) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.ee7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.46a = impl_witness_table (%Core.import_ref.ba6), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptCopyable = %AdaptCopyable.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptCopyable.decl: type = class_decl @AdaptCopyable [concrete = constants.%AdaptCopyable] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.cdf = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.cdf = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cdf = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cdf = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc8_27: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %c.param: %AdaptCopyable = value_param call_param0
// CHECK:STDOUT:     %AdaptCopyable.ref.loc8_9: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %c: %AdaptCopyable = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptCopyable = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptCopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.patt: %pattern_type.813 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.813 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.813 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.813 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc27_41: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %int_32.loc27_56: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc27_56: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc27_59.1: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc27_41, %u32.loc27_56)
// CHECK:STDOUT:     %.loc27_59.2: type = converted %.loc27_59.1, constants.%tuple.type.2a3 [concrete = constants.%tuple.type.2a3]
// CHECK:STDOUT:     %c.param: %tuple.type.2a3 = value_param call_param0
// CHECK:STDOUT:     %.loc27_34.1: type = splice_block %.loc27_34.3 [concrete = constants.%tuple.type.2a3] {
// CHECK:STDOUT:       %AdaptCopyable.ref.loc27_16: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:       %int_32.loc27_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %u32.loc27_31: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc27_34.2: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc27_16, %u32.loc27_31)
// CHECK:STDOUT:       %.loc27_34.3: type = converted %.loc27_34.2, constants.%tuple.type.2a3 [concrete = constants.%tuple.type.2a3]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.2a3 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.2a3 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.2a3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptCopyable {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%i32.builtin [concrete = constants.%complete_type.f8a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptCopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %AdaptCopyable) -> %AdaptCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cdf = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cdf = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptCopyable = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptCopyable = name_ref c, %c
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   %AdaptCopyable.ref.loc16: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:   %d: ref %AdaptCopyable = bind_name d, %d.var
// CHECK:STDOUT:   %d.ref: ref %AdaptCopyable = name_ref d, %d
// CHECK:STDOUT:   %.loc24: %AdaptCopyable = bind_value %d.ref
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AdaptCopyable, () [concrete = constants.%facet_value.567]
// CHECK:STDOUT:   %.loc16: %type_where = converted constants.%AdaptCopyable, %facet_value [concrete = constants.%facet_value.567]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %d.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.1b8
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.1b8, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value.567) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.6d3]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.4c1 = addr_of %d.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.2a3) -> %return.param: %tuple.type.2a3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.813 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.813 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.2a3 = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.2a3 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc35_33.1: %AdaptCopyable = tuple_access %c.ref, element0
// CHECK:STDOUT:   %tuple.elem0.loc35_33.2: ref %AdaptCopyable = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc35_33.1: init %AdaptCopyable = initialize_from <error> to %tuple.elem0.loc35_33.2 [concrete = <error>]
// CHECK:STDOUT:   %tuple.elem1.loc35_33.1: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc35: %.fbf = impl_witness_access constants.%Copy.impl_witness.61c, element0 [concrete = constants.%UInt.as.Copy.impl.Op.1bb]
// CHECK:STDOUT:   %bound_method.loc35_33.1: <bound method> = bound_method %tuple.elem1.loc35_33.1, %impl.elem0.loc35
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35_33.2: <bound method> = bound_method %tuple.elem1.loc35_33.1, %specific_fn.loc35
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc35: init %u32 = call %bound_method.loc35_33.2(%tuple.elem1.loc35_33.1)
// CHECK:STDOUT:   %tuple.elem1.loc35_33.2: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc35_33.2: init %u32 = initialize_from %UInt.as.Copy.impl.Op.call.loc35 to %tuple.elem1.loc35_33.2
// CHECK:STDOUT:   %.loc35_33.3: init %tuple.type.2a3 = tuple_init (%.loc35_33.1, %.loc35_33.2) to %d.var
// CHECK:STDOUT:   %.loc35_3.1: init %tuple.type.2a3 = converted %c.ref, %.loc35_33.3
// CHECK:STDOUT:   assign %d.var, %.loc35_3.1
// CHECK:STDOUT:   %.loc35_29.1: type = splice_block %.loc35_29.3 [concrete = constants.%tuple.type.2a3] {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc35: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %int_32.loc35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc35: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc35_29.2: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc35, %u32.loc35)
// CHECK:STDOUT:     %.loc35_29.3: type = converted %.loc35_29.2, constants.%tuple.type.2a3 [concrete = constants.%tuple.type.2a3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.2a3 = bind_name d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.2a3 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc43_10.1: ref %AdaptCopyable = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc43_10.1: %AdaptCopyable = bind_value %tuple.elem0.loc43_10.1
// CHECK:STDOUT:   %tuple.elem0.loc43_10.2: ref %AdaptCopyable = tuple_access %return, element0
// CHECK:STDOUT:   %.loc43_10.2: init %AdaptCopyable = initialize_from <error> to %tuple.elem0.loc43_10.2 [concrete = <error>]
// CHECK:STDOUT:   %tuple.elem1.loc43_10.1: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc43_10.3: %u32 = bind_value %tuple.elem1.loc43_10.1
// CHECK:STDOUT:   %impl.elem0.loc43: %.fbf = impl_witness_access constants.%Copy.impl_witness.61c, element0 [concrete = constants.%UInt.as.Copy.impl.Op.1bb]
// CHECK:STDOUT:   %bound_method.loc43_10.1: <bound method> = bound_method %.loc43_10.3, %impl.elem0.loc43
// CHECK:STDOUT:   %specific_fn.loc43: <specific function> = specific_function %impl.elem0.loc43, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc43_10.2: <bound method> = bound_method %.loc43_10.3, %specific_fn.loc43
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc43: init %u32 = call %bound_method.loc43_10.2(%.loc43_10.3)
// CHECK:STDOUT:   %tuple.elem1.loc43_10.2: ref %u32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc43_10.4: init %u32 = initialize_from %UInt.as.Copy.impl.Op.call.loc43 to %tuple.elem1.loc43_10.2
// CHECK:STDOUT:   %.loc43_10.5: init %tuple.type.2a3 = tuple_init (%.loc43_10.2, %.loc43_10.4) to %return
// CHECK:STDOUT:   %.loc43_11: init %tuple.type.2a3 = converted %d.ref, %.loc43_10.5
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%tuple.type.2a3, () [concrete = constants.%facet_value.6a4]
// CHECK:STDOUT:   %.loc35_3.2: %type_where = converted constants.%tuple.type.2a3, %facet_value [concrete = constants.%facet_value.6a4]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %d.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.8a01
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.8a01, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value.6a4) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.610]
// CHECK:STDOUT:   %bound_method.loc35_3: <bound method> = bound_method %d.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.c30 = addr_of %d.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc35_3(%addr)
// CHECK:STDOUT:   return %.loc43_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptTuple: type = class_type @AdaptTuple [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %complete_type.65d: <witness> = complete_type_witness %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %pattern_type.562: type = pattern_type %AdaptTuple [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c49: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c49 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.a7e: %type_where = facet_value %AdaptTuple, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.5a3: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.a7e) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.002: %DestroyT.as_type.as.Destroy.impl.Op.type.5a3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ca3: type = ptr_type %AdaptTuple [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.83b: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.002, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.a7e) [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.f69: type = tuple_type (%AdaptTuple, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.c9e: type = pattern_type %tuple.type.f69 [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.cb3: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.ee7: %UInt.as.Copy.impl.Op.type.cb3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.61c: <witness> = impl_witness imports.%Copy.impl_witness_table.46a, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.31e: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.1bb: %UInt.as.Copy.impl.Op.type.31e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.9a3: %Copy.type = facet_value %u32, (%Copy.impl_witness.61c) [concrete]
// CHECK:STDOUT:   %.fbf: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.9a3 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.1bb, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.262: %type_where = facet_value %tuple.type.f69, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.3a4: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.262) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.030: %DestroyT.as_type.as.Destroy.impl.Op.type.3a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ed5: type = ptr_type %tuple.type.f69 [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.554: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.030, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.262) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.ba6: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.cb3) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.ee7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.46a = impl_witness_table (%Core.import_ref.ba6), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptTuple = %AdaptTuple.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptTuple.decl: type = class_decl @AdaptTuple [concrete = constants.%AdaptTuple] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.patt: %pattern_type.562 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.562 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.562 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.562 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc8_24: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %c.param: %AdaptTuple = value_param call_param0
// CHECK:STDOUT:     %AdaptTuple.ref.loc8_9: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %c: %AdaptTuple = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptTuple = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptTuple = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c9e = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c9e = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c9e = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c9e = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc13_38: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %int_32.loc13_50: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc13_50: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc13_53.1: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc13_38, %u32.loc13_50)
// CHECK:STDOUT:     %.loc13_53.2: type = converted %.loc13_53.1, constants.%tuple.type.f69 [concrete = constants.%tuple.type.f69]
// CHECK:STDOUT:     %c.param: %tuple.type.f69 = value_param call_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %.loc13_31.3 [concrete = constants.%tuple.type.f69] {
// CHECK:STDOUT:       %AdaptTuple.ref.loc13_16: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:       %int_32.loc13_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %u32.loc13_28: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc13_31.2: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc13_16, %u32.loc13_28)
// CHECK:STDOUT:       %.loc13_31.3: type = converted %.loc13_31.2, constants.%tuple.type.f69 [concrete = constants.%tuple.type.f69]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.f69 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.f69 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.f69 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptTuple {
// CHECK:STDOUT:   %int_32.loc5_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc5_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5_18: %tuple.type.24b = tuple_literal (%i32.loc5_10, %i32.loc5_15)
// CHECK:STDOUT:   %.loc5_19: type = converted %.loc5_18, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   adapt_decl %.loc5_19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.d07 [concrete = constants.%complete_type.65d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptTuple
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %AdaptTuple) -> %return.param: %AdaptTuple {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.562 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.562 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptTuple = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptTuple = name_ref c, %c
// CHECK:STDOUT:   %.loc9_3.1: %tuple.type.d07 = as_compatible %c.ref
// CHECK:STDOUT:   %tuple.elem0.loc9_3.1: %i32 = tuple_access %.loc9_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc9_3.1: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc9_3.1: <bound method> = bound_method %tuple.elem0.loc9_3.1, %impl.elem0.loc9_3.1
// CHECK:STDOUT:   %specific_fn.loc9_3.1: <specific function> = specific_function %impl.elem0.loc9_3.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.2: <bound method> = bound_method %tuple.elem0.loc9_3.1, %specific_fn.loc9_3.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.1: init %i32 = call %bound_method.loc9_3.2(%tuple.elem0.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %tuple.type.d07 = as_compatible %d.var
// CHECK:STDOUT:   %tuple.elem0.loc9_3.2: ref %i32 = tuple_access %.loc9_3.2, element0
// CHECK:STDOUT:   %.loc9_3.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc9_3.1 to %tuple.elem0.loc9_3.2
// CHECK:STDOUT:   %tuple.elem1.loc9_3.1: %i32 = tuple_access %.loc9_3.1, element1
// CHECK:STDOUT:   %impl.elem0.loc9_3.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc9_3.3: <bound method> = bound_method %tuple.elem1.loc9_3.1, %impl.elem0.loc9_3.2
// CHECK:STDOUT:   %specific_fn.loc9_3.2: <specific function> = specific_function %impl.elem0.loc9_3.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.4: <bound method> = bound_method %tuple.elem1.loc9_3.1, %specific_fn.loc9_3.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.2: init %i32 = call %bound_method.loc9_3.4(%tuple.elem1.loc9_3.1)
// CHECK:STDOUT:   %tuple.elem1.loc9_3.2: ref %i32 = tuple_access %.loc9_3.2, element1
// CHECK:STDOUT:   %.loc9_3.4: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc9_3.2 to %tuple.elem1.loc9_3.2
// CHECK:STDOUT:   %.loc9_3.5: init %tuple.type.d07 = tuple_init (%.loc9_3.3, %.loc9_3.4) to %.loc9_3.2
// CHECK:STDOUT:   %.loc9_3.6: init %AdaptTuple = as_compatible %.loc9_3.5
// CHECK:STDOUT:   %.loc9_3.7: init %AdaptTuple = converted %c.ref, %.loc9_3.6
// CHECK:STDOUT:   assign %d.var, %.loc9_3.7
// CHECK:STDOUT:   %AdaptTuple.ref.loc9: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:   %d: ref %AdaptTuple = bind_name d, %d.var
// CHECK:STDOUT:   %d.ref: ref %AdaptTuple = name_ref d, %d
// CHECK:STDOUT:   %.loc10_11.1: ref %tuple.type.d07 = as_compatible %d.ref
// CHECK:STDOUT:   %tuple.elem0.loc10_11.1: ref %i32 = tuple_access %.loc10_11.1, element0
// CHECK:STDOUT:   %.loc10_11.2: %i32 = bind_value %tuple.elem0.loc10_11.1
// CHECK:STDOUT:   %impl.elem0.loc10_11.1: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.2, %impl.elem0.loc10_11.1
// CHECK:STDOUT:   %specific_fn.loc10_11.1: <specific function> = specific_function %impl.elem0.loc10_11.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.2, %specific_fn.loc10_11.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.1: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.2)
// CHECK:STDOUT:   %.loc10_11.3: ref %tuple.type.d07 = as_compatible %return
// CHECK:STDOUT:   %tuple.elem0.loc10_11.2: ref %i32 = tuple_access %.loc10_11.3, element0
// CHECK:STDOUT:   %.loc10_11.4: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc10_11.1 to %tuple.elem0.loc10_11.2
// CHECK:STDOUT:   %tuple.elem1.loc10_11.1: ref %i32 = tuple_access %.loc10_11.1, element1
// CHECK:STDOUT:   %.loc10_11.5: %i32 = bind_value %tuple.elem1.loc10_11.1
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.5, %impl.elem0.loc10_11.2
// CHECK:STDOUT:   %specific_fn.loc10_11.2: <specific function> = specific_function %impl.elem0.loc10_11.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.4: <bound method> = bound_method %.loc10_11.5, %specific_fn.loc10_11.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.2: init %i32 = call %bound_method.loc10_11.4(%.loc10_11.5)
// CHECK:STDOUT:   %tuple.elem1.loc10_11.2: ref %i32 = tuple_access %.loc10_11.3, element1
// CHECK:STDOUT:   %.loc10_11.6: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc10_11.2 to %tuple.elem1.loc10_11.2
// CHECK:STDOUT:   %.loc10_11.7: init %tuple.type.d07 = tuple_init (%.loc10_11.4, %.loc10_11.6) to %.loc10_11.3
// CHECK:STDOUT:   %.loc10_11.8: init %AdaptTuple = as_compatible %.loc10_11.7
// CHECK:STDOUT:   %.loc10_11.9: init %AdaptTuple = converted %d.ref, %.loc10_11.8
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AdaptTuple, () [concrete = constants.%facet_value.a7e]
// CHECK:STDOUT:   %.loc9_3.8: %type_where = converted constants.%AdaptTuple, %facet_value [concrete = constants.%facet_value.a7e]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %d.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.002
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.002, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value.a7e) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.83b]
// CHECK:STDOUT:   %bound_method.loc9_3.5: <bound method> = bound_method %d.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.ca3 = addr_of %d.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_3.5(%addr)
// CHECK:STDOUT:   return %.loc10_11.9 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.f69) -> %return.param: %tuple.type.f69 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.c9e = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.c9e = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.f69 = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.f69 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc14_30.1: %AdaptTuple = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_30.1: %tuple.type.d07 = as_compatible %tuple.elem0.loc14_30.1
// CHECK:STDOUT:   %tuple.elem0.loc14_30.2: %i32 = tuple_access %.loc14_30.1, element0
// CHECK:STDOUT:   %impl.elem0.loc14_30.1: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc14_30.1: <bound method> = bound_method %tuple.elem0.loc14_30.2, %impl.elem0.loc14_30.1
// CHECK:STDOUT:   %specific_fn.loc14_30.1: <specific function> = specific_function %impl.elem0.loc14_30.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.2: <bound method> = bound_method %tuple.elem0.loc14_30.2, %specific_fn.loc14_30.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_30.1: init %i32 = call %bound_method.loc14_30.2(%tuple.elem0.loc14_30.2)
// CHECK:STDOUT:   %tuple.elem0.loc14_30.3: ref %AdaptTuple = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc14_30.2: ref %tuple.type.d07 = as_compatible %tuple.elem0.loc14_30.3
// CHECK:STDOUT:   %tuple.elem0.loc14_30.4: ref %i32 = tuple_access %.loc14_30.2, element0
// CHECK:STDOUT:   %.loc14_30.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc14_30.1 to %tuple.elem0.loc14_30.4
// CHECK:STDOUT:   %tuple.elem1.loc14_30.1: %i32 = tuple_access %.loc14_30.1, element1
// CHECK:STDOUT:   %impl.elem0.loc14_30.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc14_30.3: <bound method> = bound_method %tuple.elem1.loc14_30.1, %impl.elem0.loc14_30.2
// CHECK:STDOUT:   %specific_fn.loc14_30.2: <specific function> = specific_function %impl.elem0.loc14_30.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.4: <bound method> = bound_method %tuple.elem1.loc14_30.1, %specific_fn.loc14_30.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_30.2: init %i32 = call %bound_method.loc14_30.4(%tuple.elem1.loc14_30.1)
// CHECK:STDOUT:   %tuple.elem1.loc14_30.2: ref %i32 = tuple_access %.loc14_30.2, element1
// CHECK:STDOUT:   %.loc14_30.4: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc14_30.2 to %tuple.elem1.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.5: init %tuple.type.d07 = tuple_init (%.loc14_30.3, %.loc14_30.4) to %.loc14_30.2
// CHECK:STDOUT:   %.loc14_30.6: init %AdaptTuple = as_compatible %.loc14_30.5
// CHECK:STDOUT:   %.loc14_30.7: init %AdaptTuple = converted %tuple.elem0.loc14_30.1, %.loc14_30.6
// CHECK:STDOUT:   %tuple.elem1.loc14_30.3: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc14_30.3: %.fbf = impl_witness_access constants.%Copy.impl_witness.61c, element0 [concrete = constants.%UInt.as.Copy.impl.Op.1bb]
// CHECK:STDOUT:   %bound_method.loc14_30.5: <bound method> = bound_method %tuple.elem1.loc14_30.3, %impl.elem0.loc14_30.3
// CHECK:STDOUT:   %specific_fn.loc14_30.3: <specific function> = specific_function %impl.elem0.loc14_30.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_30.6: <bound method> = bound_method %tuple.elem1.loc14_30.3, %specific_fn.loc14_30.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc14: init %u32 = call %bound_method.loc14_30.6(%tuple.elem1.loc14_30.3)
// CHECK:STDOUT:   %tuple.elem1.loc14_30.4: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc14_30.8: init %u32 = initialize_from %UInt.as.Copy.impl.Op.call.loc14 to %tuple.elem1.loc14_30.4
// CHECK:STDOUT:   %.loc14_30.9: init %tuple.type.f69 = tuple_init (%.loc14_30.7, %.loc14_30.8) to %d.var
// CHECK:STDOUT:   %.loc14_3.1: init %tuple.type.f69 = converted %c.ref, %.loc14_30.9
// CHECK:STDOUT:   assign %d.var, %.loc14_3.1
// CHECK:STDOUT:   %.loc14_26.1: type = splice_block %.loc14_26.3 [concrete = constants.%tuple.type.f69] {
// CHECK:STDOUT:     %AdaptTuple.ref.loc14: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc14: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc14_26.2: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc14, %u32.loc14)
// CHECK:STDOUT:     %.loc14_26.3: type = converted %.loc14_26.2, constants.%tuple.type.f69 [concrete = constants.%tuple.type.f69]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.f69 = bind_name d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.f69 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc15_10.1: ref %AdaptTuple = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc15_10.1: ref %tuple.type.d07 = as_compatible %tuple.elem0.loc15_10.1
// CHECK:STDOUT:   %tuple.elem0.loc15_10.2: ref %i32 = tuple_access %.loc15_10.1, element0
// CHECK:STDOUT:   %.loc15_10.2: %i32 = bind_value %tuple.elem0.loc15_10.2
// CHECK:STDOUT:   %impl.elem0.loc15_10.1: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc15_10.1: <bound method> = bound_method %.loc15_10.2, %impl.elem0.loc15_10.1
// CHECK:STDOUT:   %specific_fn.loc15_10.1: <specific function> = specific_function %impl.elem0.loc15_10.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.2: <bound method> = bound_method %.loc15_10.2, %specific_fn.loc15_10.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.1: init %i32 = call %bound_method.loc15_10.2(%.loc15_10.2)
// CHECK:STDOUT:   %tuple.elem0.loc15_10.3: ref %AdaptTuple = tuple_access %return, element0
// CHECK:STDOUT:   %.loc15_10.3: ref %tuple.type.d07 = as_compatible %tuple.elem0.loc15_10.3
// CHECK:STDOUT:   %tuple.elem0.loc15_10.4: ref %i32 = tuple_access %.loc15_10.3, element0
// CHECK:STDOUT:   %.loc15_10.4: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc15_10.1 to %tuple.elem0.loc15_10.4
// CHECK:STDOUT:   %tuple.elem1.loc15_10.1: ref %i32 = tuple_access %.loc15_10.1, element1
// CHECK:STDOUT:   %.loc15_10.5: %i32 = bind_value %tuple.elem1.loc15_10.1
// CHECK:STDOUT:   %impl.elem0.loc15_10.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc15_10.3: <bound method> = bound_method %.loc15_10.5, %impl.elem0.loc15_10.2
// CHECK:STDOUT:   %specific_fn.loc15_10.2: <specific function> = specific_function %impl.elem0.loc15_10.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.4: <bound method> = bound_method %.loc15_10.5, %specific_fn.loc15_10.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.2: init %i32 = call %bound_method.loc15_10.4(%.loc15_10.5)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.2: ref %i32 = tuple_access %.loc15_10.3, element1
// CHECK:STDOUT:   %.loc15_10.6: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc15_10.2 to %tuple.elem1.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.7: init %tuple.type.d07 = tuple_init (%.loc15_10.4, %.loc15_10.6) to %.loc15_10.3
// CHECK:STDOUT:   %.loc15_10.8: init %AdaptTuple = as_compatible %.loc15_10.7
// CHECK:STDOUT:   %.loc15_10.9: init %AdaptTuple = converted %tuple.elem0.loc15_10.1, %.loc15_10.8
// CHECK:STDOUT:   %tuple.elem1.loc15_10.3: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc15_10.10: %u32 = bind_value %tuple.elem1.loc15_10.3
// CHECK:STDOUT:   %impl.elem0.loc15_10.3: %.fbf = impl_witness_access constants.%Copy.impl_witness.61c, element0 [concrete = constants.%UInt.as.Copy.impl.Op.1bb]
// CHECK:STDOUT:   %bound_method.loc15_10.5: <bound method> = bound_method %.loc15_10.10, %impl.elem0.loc15_10.3
// CHECK:STDOUT:   %specific_fn.loc15_10.3: <specific function> = specific_function %impl.elem0.loc15_10.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.6: <bound method> = bound_method %.loc15_10.10, %specific_fn.loc15_10.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc15: init %u32 = call %bound_method.loc15_10.6(%.loc15_10.10)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.4: ref %u32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc15_10.11: init %u32 = initialize_from %UInt.as.Copy.impl.Op.call.loc15 to %tuple.elem1.loc15_10.4
// CHECK:STDOUT:   %.loc15_10.12: init %tuple.type.f69 = tuple_init (%.loc15_10.9, %.loc15_10.11) to %return
// CHECK:STDOUT:   %.loc15_11: init %tuple.type.f69 = converted %d.ref, %.loc15_10.12
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%tuple.type.f69, () [concrete = constants.%facet_value.262]
// CHECK:STDOUT:   %.loc14_3.2: %type_where = converted constants.%tuple.type.f69, %facet_value [concrete = constants.%facet_value.262]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %d.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.030
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.030, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value.262) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.554]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %d.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.ed5 = addr_of %d.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_3(%addr)
// CHECK:STDOUT:   return %.loc15_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptNoncopyable: type = class_type @AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.8f9: type = pattern_type %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %AdaptNoncopyable, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.3de: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.6f7: %DestroyT.as_type.as.Destroy.impl.Op.type.3de = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.ed9: type = ptr_type %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.6f7, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .AdaptNoncopyable = %AdaptNoncopyable.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [concrete = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %AdaptNoncopyable.decl: type = class_decl @AdaptNoncopyable [concrete = constants.%AdaptNoncopyable] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %a.patt: %pattern_type.8f9 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.8f9 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8f9 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8f9 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc12_30: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyable = value_param call_param0
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc12_9: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %a: %AdaptNoncopyable = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %AdaptNoncopyable = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptNoncopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNoncopyable {
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   adapt_decl %Noncopyable.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyable
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %AdaptNoncopyable) -> %return.param: %AdaptNoncopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.8f9 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.8f9 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyable = var %b.var_patt
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyable = name_ref a, %a
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %AdaptNoncopyable.ref.loc20: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:   %b: ref %AdaptNoncopyable = bind_name b, %b.var
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyable = name_ref b, %b
// CHECK:STDOUT:   %.loc28: %AdaptNoncopyable = bind_value %b.ref
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AdaptNoncopyable, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc20: %type_where = converted constants.%AdaptNoncopyable, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %b.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.6f7
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.6f7, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %b.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.ed9 = addr_of %b.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable_indirect.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptNoncopyableIndirect: type = class_type @AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.c9a: type = tuple_type (%i32, %Noncopyable, %i32) [concrete]
// CHECK:STDOUT:   %complete_type.201: <witness> = complete_type_witness %tuple.type.c9a [concrete]
// CHECK:STDOUT:   %pattern_type.7e5: type = pattern_type %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %AdaptNoncopyableIndirect, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.26a: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.02b: %DestroyT.as_type.as.Destroy.impl.Op.type.26a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.921: type = ptr_type %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.02b, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Noncopyable = %Noncopyable.decl
// CHECK:STDOUT:     .AdaptNoncopyableIndirect = %AdaptNoncopyableIndirect.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Noncopyable.decl: type = class_decl @Noncopyable [concrete = constants.%Noncopyable] {} {}
// CHECK:STDOUT:   %AdaptNoncopyableIndirect.decl: type = class_decl @AdaptNoncopyableIndirect [concrete = constants.%AdaptNoncopyableIndirect] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7e5 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7e5 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7e5 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7e5 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc12_38: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyableIndirect = value_param call_param0
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc12_9: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %a: %AdaptNoncopyableIndirect = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %AdaptNoncopyableIndirect = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptNoncopyableIndirect = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNoncopyableIndirect {
// CHECK:STDOUT:   %int_32.loc9_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %int_32.loc9_28: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9_28: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9_31: %tuple.type.ff9 = tuple_literal (%i32.loc9_10, %Noncopyable.ref, %i32.loc9_28)
// CHECK:STDOUT:   %.loc9_32: type = converted %.loc9_31, constants.%tuple.type.c9a [concrete = constants.%tuple.type.c9a]
// CHECK:STDOUT:   adapt_decl %.loc9_32 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.c9a [concrete = constants.%complete_type.201]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyableIndirect
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%a.param: %AdaptNoncopyableIndirect) -> %return.param: %AdaptNoncopyableIndirect {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7e5 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7e5 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyableIndirect = var %b.var_patt
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyableIndirect = name_ref a, %a
// CHECK:STDOUT:   %.loc23_3.1: %tuple.type.c9a = as_compatible %a.ref
// CHECK:STDOUT:   %tuple.elem0.loc23_3.1: %i32 = tuple_access %.loc23_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc23: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc23_3.1: <bound method> = bound_method %tuple.elem0.loc23_3.1, %impl.elem0.loc23
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_3.2: <bound method> = bound_method %tuple.elem0.loc23_3.1, %specific_fn.loc23
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc23: init %i32 = call %bound_method.loc23_3.2(%tuple.elem0.loc23_3.1)
// CHECK:STDOUT:   %.loc23_3.2: ref %tuple.type.c9a = as_compatible %b.var
// CHECK:STDOUT:   %tuple.elem0.loc23_3.2: ref %i32 = tuple_access %.loc23_3.2, element0
// CHECK:STDOUT:   %.loc23_3.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc23 to %tuple.elem0.loc23_3.2
// CHECK:STDOUT:   %tuple.elem1.loc23: %Noncopyable = tuple_access %.loc23_3.1, element1
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %AdaptNoncopyableIndirect.ref.loc23: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:   %b: ref %AdaptNoncopyableIndirect = bind_name b, %b.var
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyableIndirect = name_ref b, %b
// CHECK:STDOUT:   %.loc34_11.1: ref %tuple.type.c9a = as_compatible %b.ref
// CHECK:STDOUT:   %tuple.elem0.loc34_11.1: ref %i32 = tuple_access %.loc34_11.1, element0
// CHECK:STDOUT:   %.loc34_11.2: %i32 = bind_value %tuple.elem0.loc34_11.1
// CHECK:STDOUT:   %impl.elem0.loc34: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc34_11.1: <bound method> = bound_method %.loc34_11.2, %impl.elem0.loc34
// CHECK:STDOUT:   %specific_fn.loc34: <specific function> = specific_function %impl.elem0.loc34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_11.2: <bound method> = bound_method %.loc34_11.2, %specific_fn.loc34
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc34: init %i32 = call %bound_method.loc34_11.2(%.loc34_11.2)
// CHECK:STDOUT:   %.loc34_11.3: ref %tuple.type.c9a = as_compatible %return
// CHECK:STDOUT:   %tuple.elem0.loc34_11.2: ref %i32 = tuple_access %.loc34_11.3, element0
// CHECK:STDOUT:   %.loc34_11.4: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc34 to %tuple.elem0.loc34_11.2
// CHECK:STDOUT:   %tuple.elem1.loc34: ref %Noncopyable = tuple_access %.loc34_11.1, element1
// CHECK:STDOUT:   %.loc34_11.5: %Noncopyable = bind_value %tuple.elem1.loc34
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AdaptNoncopyableIndirect, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc23_3.4: %type_where = converted constants.%AdaptNoncopyableIndirect, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %b.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.02b
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.02b, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_3.3: <bound method> = bound_method %b.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.921 = addr_of %b.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc23_3.3(%addr)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptStruct: type = class_type @AdaptStruct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.511: <witness> = complete_type_witness %struct_type.e.f [concrete]
// CHECK:STDOUT:   %pattern_type.f45: type = pattern_type %AdaptStruct [concrete]
// CHECK:STDOUT:   %I.type: type = fn_type @I [concrete]
// CHECK:STDOUT:   %I: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c49: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c49 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.14c: %type_where = facet_value %AdaptStruct, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.b5b: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.14c) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.987: %DestroyT.as_type.as.Destroy.impl.Op.type.b5b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e10: type = ptr_type %AdaptStruct [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.4f8: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.987, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.14c) [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.80b: type = tuple_type (%AdaptStruct, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.31d: type = pattern_type %tuple.type.80b [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.cb3: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.ee7: %UInt.as.Copy.impl.Op.type.cb3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.61c: <witness> = impl_witness imports.%Copy.impl_witness_table.46a, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.31e: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.1bb: %UInt.as.Copy.impl.Op.type.31e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.9a3: %Copy.type = facet_value %u32, (%Copy.impl_witness.61c) [concrete]
// CHECK:STDOUT:   %.fbf: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.9a3 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.1bb, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %facet_value.03e: %type_where = facet_value %tuple.type.80b, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.a7b: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value.03e) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.4cc: %DestroyT.as_type.as.Destroy.impl.Op.type.a7b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b09: type = ptr_type %tuple.type.80b [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.0cd: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.4cc, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value.03e) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.ba6: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.cb3) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.ee7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.46a = impl_witness_table (%Core.import_ref.ba6), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AdaptStruct = %AdaptStruct.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .InTuple = %InTuple.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AdaptStruct.decl: type = class_decl @AdaptStruct [concrete = constants.%AdaptStruct] {} {}
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [concrete = constants.%I] {
// CHECK:STDOUT:     %g.patt: %pattern_type.f45 = binding_pattern g [concrete]
// CHECK:STDOUT:     %g.param_patt: %pattern_type.f45 = value_param_pattern %g.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f45 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f45 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc8_25: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %g.param: %AdaptStruct = value_param call_param0
// CHECK:STDOUT:     %AdaptStruct.ref.loc8_9: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %g: %AdaptStruct = bind_name g, %g.param
// CHECK:STDOUT:     %return.param: ref %AdaptStruct = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptStruct = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.patt: %pattern_type.31d = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.31d = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.31d = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.31d = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc13_39: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %int_32.loc13_52: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc13_52: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc13_55.1: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc13_39, %u32.loc13_52)
// CHECK:STDOUT:     %.loc13_55.2: type = converted %.loc13_55.1, constants.%tuple.type.80b [concrete = constants.%tuple.type.80b]
// CHECK:STDOUT:     %c.param: %tuple.type.80b = value_param call_param0
// CHECK:STDOUT:     %.loc13_32.1: type = splice_block %.loc13_32.3 [concrete = constants.%tuple.type.80b] {
// CHECK:STDOUT:       %AdaptStruct.ref.loc13_16: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:       %int_32.loc13_29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %u32.loc13_29: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc13_32.2: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc13_16, %u32.loc13_29)
// CHECK:STDOUT:       %.loc13_32.3: type = converted %.loc13_32.2, constants.%tuple.type.80b [concrete = constants.%tuple.type.80b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.80b = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.80b = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.80b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptStruct {
// CHECK:STDOUT:   %int_32.loc5_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc5_23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5_23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete = constants.%struct_type.e.f]
// CHECK:STDOUT:   adapt_decl %struct_type.e.f [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.e.f [concrete = constants.%complete_type.511]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptStruct
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I(%g.param: %AdaptStruct) -> %return.param: %AdaptStruct {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.f45 = binding_pattern h [concrete]
// CHECK:STDOUT:     %h.var_patt: %pattern_type.f45 = var_pattern %h.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %h.var: ref %AdaptStruct = var %h.var_patt
// CHECK:STDOUT:   %g.ref: %AdaptStruct = name_ref g, %g
// CHECK:STDOUT:   %.loc9_3.1: %struct_type.e.f = as_compatible %g.ref
// CHECK:STDOUT:   %.loc9_3.2: %i32 = struct_access %.loc9_3.1, element0
// CHECK:STDOUT:   %impl.elem0.loc9_3.1: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc9_3.1: <bound method> = bound_method %.loc9_3.2, %impl.elem0.loc9_3.1
// CHECK:STDOUT:   %specific_fn.loc9_3.1: <specific function> = specific_function %impl.elem0.loc9_3.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.2: <bound method> = bound_method %.loc9_3.2, %specific_fn.loc9_3.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.1: init %i32 = call %bound_method.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   %.loc9_3.3: ref %struct_type.e.f = as_compatible %h.var
// CHECK:STDOUT:   %.loc9_3.4: ref %i32 = struct_access %.loc9_3.3, element0
// CHECK:STDOUT:   %.loc9_3.5: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc9_3.1 to %.loc9_3.4
// CHECK:STDOUT:   %.loc9_3.6: %i32 = struct_access %.loc9_3.1, element1
// CHECK:STDOUT:   %impl.elem0.loc9_3.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc9_3.3: <bound method> = bound_method %.loc9_3.6, %impl.elem0.loc9_3.2
// CHECK:STDOUT:   %specific_fn.loc9_3.2: <specific function> = specific_function %impl.elem0.loc9_3.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_3.4: <bound method> = bound_method %.loc9_3.6, %specific_fn.loc9_3.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc9_3.2: init %i32 = call %bound_method.loc9_3.4(%.loc9_3.6)
// CHECK:STDOUT:   %.loc9_3.7: ref %i32 = struct_access %.loc9_3.3, element1
// CHECK:STDOUT:   %.loc9_3.8: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc9_3.2 to %.loc9_3.7
// CHECK:STDOUT:   %.loc9_3.9: init %struct_type.e.f = struct_init (%.loc9_3.5, %.loc9_3.8) to %.loc9_3.3
// CHECK:STDOUT:   %.loc9_3.10: init %AdaptStruct = as_compatible %.loc9_3.9
// CHECK:STDOUT:   %.loc9_3.11: init %AdaptStruct = converted %g.ref, %.loc9_3.10
// CHECK:STDOUT:   assign %h.var, %.loc9_3.11
// CHECK:STDOUT:   %AdaptStruct.ref.loc9: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:   %h: ref %AdaptStruct = bind_name h, %h.var
// CHECK:STDOUT:   %h.ref: ref %AdaptStruct = name_ref h, %h
// CHECK:STDOUT:   %.loc10_11.1: ref %struct_type.e.f = as_compatible %h.ref
// CHECK:STDOUT:   %.loc10_11.2: ref %i32 = struct_access %.loc10_11.1, element0
// CHECK:STDOUT:   %.loc10_11.3: %i32 = bind_value %.loc10_11.2
// CHECK:STDOUT:   %impl.elem0.loc10_11.1: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.3, %impl.elem0.loc10_11.1
// CHECK:STDOUT:   %specific_fn.loc10_11.1: <specific function> = specific_function %impl.elem0.loc10_11.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.3, %specific_fn.loc10_11.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.1: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.3)
// CHECK:STDOUT:   %.loc10_11.4: ref %struct_type.e.f = as_compatible %return
// CHECK:STDOUT:   %.loc10_11.5: ref %i32 = struct_access %.loc10_11.4, element0
// CHECK:STDOUT:   %.loc10_11.6: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc10_11.1 to %.loc10_11.5
// CHECK:STDOUT:   %.loc10_11.7: ref %i32 = struct_access %.loc10_11.1, element1
// CHECK:STDOUT:   %.loc10_11.8: %i32 = bind_value %.loc10_11.7
// CHECK:STDOUT:   %impl.elem0.loc10_11.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc10_11.3: <bound method> = bound_method %.loc10_11.8, %impl.elem0.loc10_11.2
// CHECK:STDOUT:   %specific_fn.loc10_11.2: <specific function> = specific_function %impl.elem0.loc10_11.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.4: <bound method> = bound_method %.loc10_11.8, %specific_fn.loc10_11.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc10_11.2: init %i32 = call %bound_method.loc10_11.4(%.loc10_11.8)
// CHECK:STDOUT:   %.loc10_11.9: ref %i32 = struct_access %.loc10_11.4, element1
// CHECK:STDOUT:   %.loc10_11.10: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc10_11.2 to %.loc10_11.9
// CHECK:STDOUT:   %.loc10_11.11: init %struct_type.e.f = struct_init (%.loc10_11.6, %.loc10_11.10) to %.loc10_11.4
// CHECK:STDOUT:   %.loc10_11.12: init %AdaptStruct = as_compatible %.loc10_11.11
// CHECK:STDOUT:   %.loc10_11.13: init %AdaptStruct = converted %h.ref, %.loc10_11.12
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%AdaptStruct, () [concrete = constants.%facet_value.14c]
// CHECK:STDOUT:   %.loc9_3.12: %type_where = converted constants.%AdaptStruct, %facet_value [concrete = constants.%facet_value.14c]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %h.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.987
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.987, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value.14c) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.4f8]
// CHECK:STDOUT:   %bound_method.loc9_3.5: <bound method> = bound_method %h.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.e10 = addr_of %h.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_3.5(%addr)
// CHECK:STDOUT:   return %.loc10_11.13 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.80b) -> %return.param: %tuple.type.80b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.31d = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.31d = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.80b = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.80b = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc14_31.1: %AdaptStruct = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_31.1: %struct_type.e.f = as_compatible %tuple.elem0.loc14_31.1
// CHECK:STDOUT:   %.loc14_31.2: %i32 = struct_access %.loc14_31.1, element0
// CHECK:STDOUT:   %impl.elem0.loc14_31.1: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc14_31.1: <bound method> = bound_method %.loc14_31.2, %impl.elem0.loc14_31.1
// CHECK:STDOUT:   %specific_fn.loc14_31.1: <specific function> = specific_function %impl.elem0.loc14_31.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_31.2: <bound method> = bound_method %.loc14_31.2, %specific_fn.loc14_31.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_31.1: init %i32 = call %bound_method.loc14_31.2(%.loc14_31.2)
// CHECK:STDOUT:   %tuple.elem0.loc14_31.2: ref %AdaptStruct = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc14_31.3: ref %struct_type.e.f = as_compatible %tuple.elem0.loc14_31.2
// CHECK:STDOUT:   %.loc14_31.4: ref %i32 = struct_access %.loc14_31.3, element0
// CHECK:STDOUT:   %.loc14_31.5: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc14_31.1 to %.loc14_31.4
// CHECK:STDOUT:   %.loc14_31.6: %i32 = struct_access %.loc14_31.1, element1
// CHECK:STDOUT:   %impl.elem0.loc14_31.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc14_31.3: <bound method> = bound_method %.loc14_31.6, %impl.elem0.loc14_31.2
// CHECK:STDOUT:   %specific_fn.loc14_31.2: <specific function> = specific_function %impl.elem0.loc14_31.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_31.4: <bound method> = bound_method %.loc14_31.6, %specific_fn.loc14_31.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_31.2: init %i32 = call %bound_method.loc14_31.4(%.loc14_31.6)
// CHECK:STDOUT:   %.loc14_31.7: ref %i32 = struct_access %.loc14_31.3, element1
// CHECK:STDOUT:   %.loc14_31.8: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc14_31.2 to %.loc14_31.7
// CHECK:STDOUT:   %.loc14_31.9: init %struct_type.e.f = struct_init (%.loc14_31.5, %.loc14_31.8) to %.loc14_31.3
// CHECK:STDOUT:   %.loc14_31.10: init %AdaptStruct = as_compatible %.loc14_31.9
// CHECK:STDOUT:   %.loc14_31.11: init %AdaptStruct = converted %tuple.elem0.loc14_31.1, %.loc14_31.10
// CHECK:STDOUT:   %tuple.elem1.loc14_31.1: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc14_31.3: %.fbf = impl_witness_access constants.%Copy.impl_witness.61c, element0 [concrete = constants.%UInt.as.Copy.impl.Op.1bb]
// CHECK:STDOUT:   %bound_method.loc14_31.5: <bound method> = bound_method %tuple.elem1.loc14_31.1, %impl.elem0.loc14_31.3
// CHECK:STDOUT:   %specific_fn.loc14_31.3: <specific function> = specific_function %impl.elem0.loc14_31.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_31.6: <bound method> = bound_method %tuple.elem1.loc14_31.1, %specific_fn.loc14_31.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc14: init %u32 = call %bound_method.loc14_31.6(%tuple.elem1.loc14_31.1)
// CHECK:STDOUT:   %tuple.elem1.loc14_31.2: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc14_31.12: init %u32 = initialize_from %UInt.as.Copy.impl.Op.call.loc14 to %tuple.elem1.loc14_31.2
// CHECK:STDOUT:   %.loc14_31.13: init %tuple.type.80b = tuple_init (%.loc14_31.11, %.loc14_31.12) to %d.var
// CHECK:STDOUT:   %.loc14_3.1: init %tuple.type.80b = converted %c.ref, %.loc14_31.13
// CHECK:STDOUT:   assign %d.var, %.loc14_3.1
// CHECK:STDOUT:   %.loc14_27.1: type = splice_block %.loc14_27.3 [concrete = constants.%tuple.type.80b] {
// CHECK:STDOUT:     %AdaptStruct.ref.loc14: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc14: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc14_27.2: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc14, %u32.loc14)
// CHECK:STDOUT:     %.loc14_27.3: type = converted %.loc14_27.2, constants.%tuple.type.80b [concrete = constants.%tuple.type.80b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.80b = bind_name d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.80b = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc15_10.1: ref %AdaptStruct = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc15_10.1: ref %struct_type.e.f = as_compatible %tuple.elem0.loc15_10.1
// CHECK:STDOUT:   %.loc15_10.2: ref %i32 = struct_access %.loc15_10.1, element0
// CHECK:STDOUT:   %.loc15_10.3: %i32 = bind_value %.loc15_10.2
// CHECK:STDOUT:   %impl.elem0.loc15_10.1: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc15_10.1: <bound method> = bound_method %.loc15_10.3, %impl.elem0.loc15_10.1
// CHECK:STDOUT:   %specific_fn.loc15_10.1: <specific function> = specific_function %impl.elem0.loc15_10.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.2: <bound method> = bound_method %.loc15_10.3, %specific_fn.loc15_10.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.1: init %i32 = call %bound_method.loc15_10.2(%.loc15_10.3)
// CHECK:STDOUT:   %tuple.elem0.loc15_10.2: ref %AdaptStruct = tuple_access %return, element0
// CHECK:STDOUT:   %.loc15_10.4: ref %struct_type.e.f = as_compatible %tuple.elem0.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.5: ref %i32 = struct_access %.loc15_10.4, element0
// CHECK:STDOUT:   %.loc15_10.6: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc15_10.1 to %.loc15_10.5
// CHECK:STDOUT:   %.loc15_10.7: ref %i32 = struct_access %.loc15_10.1, element1
// CHECK:STDOUT:   %.loc15_10.8: %i32 = bind_value %.loc15_10.7
// CHECK:STDOUT:   %impl.elem0.loc15_10.2: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc15_10.3: <bound method> = bound_method %.loc15_10.8, %impl.elem0.loc15_10.2
// CHECK:STDOUT:   %specific_fn.loc15_10.2: <specific function> = specific_function %impl.elem0.loc15_10.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.4: <bound method> = bound_method %.loc15_10.8, %specific_fn.loc15_10.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15_10.2: init %i32 = call %bound_method.loc15_10.4(%.loc15_10.8)
// CHECK:STDOUT:   %.loc15_10.9: ref %i32 = struct_access %.loc15_10.4, element1
// CHECK:STDOUT:   %.loc15_10.10: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc15_10.2 to %.loc15_10.9
// CHECK:STDOUT:   %.loc15_10.11: init %struct_type.e.f = struct_init (%.loc15_10.6, %.loc15_10.10) to %.loc15_10.4
// CHECK:STDOUT:   %.loc15_10.12: init %AdaptStruct = as_compatible %.loc15_10.11
// CHECK:STDOUT:   %.loc15_10.13: init %AdaptStruct = converted %tuple.elem0.loc15_10.1, %.loc15_10.12
// CHECK:STDOUT:   %tuple.elem1.loc15_10.1: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc15_10.14: %u32 = bind_value %tuple.elem1.loc15_10.1
// CHECK:STDOUT:   %impl.elem0.loc15_10.3: %.fbf = impl_witness_access constants.%Copy.impl_witness.61c, element0 [concrete = constants.%UInt.as.Copy.impl.Op.1bb]
// CHECK:STDOUT:   %bound_method.loc15_10.5: <bound method> = bound_method %.loc15_10.14, %impl.elem0.loc15_10.3
// CHECK:STDOUT:   %specific_fn.loc15_10.3: <specific function> = specific_function %impl.elem0.loc15_10.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_10.6: <bound method> = bound_method %.loc15_10.14, %specific_fn.loc15_10.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc15: init %u32 = call %bound_method.loc15_10.6(%.loc15_10.14)
// CHECK:STDOUT:   %tuple.elem1.loc15_10.2: ref %u32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc15_10.15: init %u32 = initialize_from %UInt.as.Copy.impl.Op.call.loc15 to %tuple.elem1.loc15_10.2
// CHECK:STDOUT:   %.loc15_10.16: init %tuple.type.80b = tuple_init (%.loc15_10.13, %.loc15_10.15) to %return
// CHECK:STDOUT:   %.loc15_11: init %tuple.type.80b = converted %d.ref, %.loc15_10.16
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%tuple.type.80b, () [concrete = constants.%facet_value.03e]
// CHECK:STDOUT:   %.loc14_3.2: %type_where = converted constants.%tuple.type.80b, %facet_value [concrete = constants.%facet_value.03e]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %d.var, constants.%DestroyT.as_type.as.Destroy.impl.Op.4cc
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.4cc, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value.03e) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn.0cd]
// CHECK:STDOUT:   %bound_method.loc14_3: <bound method> = bound_method %d.var, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.b09 = addr_of %d.var
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc14_3(%addr)
// CHECK:STDOUT:   return %.loc15_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
