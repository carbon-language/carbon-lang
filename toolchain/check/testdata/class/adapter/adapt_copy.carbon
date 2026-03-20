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
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:26: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var d: AdaptCopyable = c;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  var d: AdaptCopyable = c;
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  return d;
}

fn InTuple(c: (AdaptCopyable, u32)) -> (AdaptCopyable, u32) {
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:33: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR:                                 ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:33: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR:                                 ^
  // CHECK:STDERR:
  var d: (AdaptCopyable, u32) = c;
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptCopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return d;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptCopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
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
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+4]]:29: note: type `AdaptNoncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var b: AdaptNoncopyable = a;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR:
  var b: AdaptNoncopyable = a;
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+7]]:10: error: cannot copy value of type `AdaptNoncopyable` [CopyOfUncopyableType]
  // CHECK:STDERR:   return b;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_adapt_not_copyable.carbon:[[@LINE+4]]:10: note: type `AdaptNoncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
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
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+7]]:3: note: type `Noncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
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
  // CHECK:STDERR: fail_adapt_not_copyable_indirect.carbon:[[@LINE+7]]:3: note: type `Noncopyable` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.e78: type = pattern_type %AdaptCopyable [concrete]
// CHECK:STDOUT:   %.23e: Core.Form = init_form %AdaptCopyable [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc21 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.fd0: %tuple.type.24b = tuple_value (%AdaptCopyable, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.d78: type = tuple_type (%AdaptCopyable, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.87f: type = pattern_type %tuple.type.d78 [concrete]
// CHECK:STDOUT:   %.5d1: Core.Form = init_form %tuple.type.d78 [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc50 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %c.param_patt: %pattern_type.e78 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.e78 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e78 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.e78 = return_slot_pattern %return.param_patt, %AdaptCopyable.ref.loc19 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc19: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %.loc19: Core.Form = init_form %AdaptCopyable.ref.loc19 [concrete = constants.%.23e]
// CHECK:STDOUT:     %c.param: %AdaptCopyable = value_param call_param0
// CHECK:STDOUT:     %AdaptCopyable.ref.loc16: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %c: %AdaptCopyable = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptCopyable = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptCopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.87f = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.87f = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.87f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.87f = return_slot_pattern %return.param_patt, %.loc48.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc45: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %u32.loc47: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc48.1: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc45, %u32.loc47) [concrete = constants.%tuple.fd0]
// CHECK:STDOUT:     %.loc48.2: type = converted %.loc48.1, constants.%tuple.type.d78 [concrete = constants.%tuple.type.d78]
// CHECK:STDOUT:     %.loc48.3: Core.Form = init_form %.loc48.2 [concrete = constants.%.5d1]
// CHECK:STDOUT:     %c.param: %tuple.type.d78 = value_param call_param0
// CHECK:STDOUT:     %.loc41.1: type = splice_block %.loc41.3 [concrete = constants.%tuple.type.d78] {
// CHECK:STDOUT:       %AdaptCopyable.ref.loc38: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:       %u32.loc40: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc41.2: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc38, %u32.loc40) [concrete = constants.%tuple.fd0]
// CHECK:STDOUT:       %.loc41.3: type = converted %.loc41.2, constants.%tuple.type.d78 [concrete = constants.%tuple.type.d78]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.d78 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.d78 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.d78 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptCopyable {
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   adapt_decl %i32 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%i32.builtin [concrete = constants.%complete_type.f8a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptCopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %AdaptCopyable) -> out %return.param: %AdaptCopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.e78 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.e78 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptCopyable = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptCopyable = name_ref c, %c
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   %AdaptCopyable.ref.loc24: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:   %d: ref %AdaptCopyable = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %AdaptCopyable = name_ref d, %d
// CHECK:STDOUT:   %.loc29: %AdaptCopyable = acquire_value %d.ref
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21(%self.param: ref %AdaptCopyable) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.d78) -> out %return.param: %tuple.type.d78 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.87f = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.87f = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.d78 = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.d78 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc59: %AdaptCopyable = tuple_access %c.ref, element0
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   %.loc57.1: type = splice_block %.loc57.3 [concrete = constants.%tuple.type.d78] {
// CHECK:STDOUT:     %AdaptCopyable.ref.loc54: type = name_ref AdaptCopyable, file.%AdaptCopyable.decl [concrete = constants.%AdaptCopyable]
// CHECK:STDOUT:     %u32.loc56: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc57.2: %tuple.type.24b = tuple_literal (%AdaptCopyable.ref.loc54, %u32.loc56) [concrete = constants.%tuple.fd0]
// CHECK:STDOUT:     %.loc57.3: type = converted %.loc57.2, constants.%tuple.type.d78 [concrete = constants.%tuple.type.d78]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.d78 = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.d78 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc62: ref %AdaptCopyable = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc62: %AdaptCopyable = acquire_value %tuple.elem0.loc62
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc50(%self.param: ref %tuple.type.d78) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptTuple: type = class_type @AdaptTuple [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.95a: %tuple.type.24b = tuple_value (%i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %complete_type.65d: <witness> = complete_type_witness %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %pattern_type.6cd: type = pattern_type %AdaptTuple [concrete]
// CHECK:STDOUT:   %.52a: Core.Form = init_form %AdaptTuple [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.de4: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.de4) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet.de4 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc25 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.b75: %tuple.type.24b = tuple_value (%AdaptTuple, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.3c7: type = tuple_type (%AdaptTuple, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.6f4: type = pattern_type %tuple.type.3c7 [concrete]
// CHECK:STDOUT:   %.709: Core.Form = init_form %tuple.type.3c7 [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.68f: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.576: %UInt.as.Copy.impl.Op.type.68f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.514: <witness> = impl_witness imports.%Copy.impl_witness_table.bd0, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.2fc: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.c10: %UInt.as.Copy.impl.Op.type.2fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.10b: %Copy.type = facet_value %u32, (%Copy.impl_witness.514) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ad7: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.10b) [concrete]
// CHECK:STDOUT:   %.38c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ad7, %Copy.facet.10b [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.c10, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc54 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.c3c: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.68f) = import_ref Core//prelude/parts/uint, loc80, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.576)]
// CHECK:STDOUT:   %Copy.impl_witness_table.bd0 = impl_witness_table (%Core.import_ref.c3c), @UInt.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %c.param_patt: %pattern_type.6cd = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.6cd = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6cd = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6cd = return_slot_pattern %return.param_patt, %AdaptTuple.ref.loc23 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc23: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %AdaptTuple.ref.loc23 [concrete = constants.%.52a]
// CHECK:STDOUT:     %c.param: %AdaptTuple = value_param call_param0
// CHECK:STDOUT:     %AdaptTuple.ref.loc20: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %c: %AdaptTuple = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %AdaptTuple = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptTuple = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.6f4 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.6f4 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6f4 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6f4 = return_slot_pattern %return.param_patt, %.loc52.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptTuple.ref.loc49: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %u32.loc51: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc52.1: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc49, %u32.loc51) [concrete = constants.%tuple.b75]
// CHECK:STDOUT:     %.loc52.2: type = converted %.loc52.1, constants.%tuple.type.3c7 [concrete = constants.%tuple.type.3c7]
// CHECK:STDOUT:     %.loc52.3: Core.Form = init_form %.loc52.2 [concrete = constants.%.709]
// CHECK:STDOUT:     %c.param: %tuple.type.3c7 = value_param call_param0
// CHECK:STDOUT:     %.loc45.1: type = splice_block %.loc45.3 [concrete = constants.%tuple.type.3c7] {
// CHECK:STDOUT:       %AdaptTuple.ref.loc42: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:       %u32.loc44: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc45.2: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc42, %u32.loc44) [concrete = constants.%tuple.b75]
// CHECK:STDOUT:       %.loc45.3: type = converted %.loc45.2, constants.%tuple.type.3c7 [concrete = constants.%tuple.type.3c7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.3c7 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.3c7 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.3c7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptTuple {
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc12: %tuple.type.24b = tuple_literal (%i32.loc9, %i32.loc11) [concrete = constants.%tuple.95a]
// CHECK:STDOUT:   %.loc13: type = converted %.loc12, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   adapt_decl %.loc13 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.d07 [concrete = constants.%complete_type.65d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptTuple
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %AdaptTuple) -> out %return.param: %AdaptTuple {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.6cd = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.6cd = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %AdaptTuple = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %AdaptTuple = name_ref c, %c
// CHECK:STDOUT:   %.loc25.1: %tuple.type.d07 = as_compatible %c.ref
// CHECK:STDOUT:   %tuple.elem0.loc25.1: %i32 = tuple_access %.loc25.1, element0
// CHECK:STDOUT:   %impl.elem0.loc25.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc25.1: <bound method> = bound_method %tuple.elem0.loc25.1, %impl.elem0.loc25.1
// CHECK:STDOUT:   %specific_fn.loc25.1: <specific function> = specific_function %impl.elem0.loc25.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25.2: <bound method> = bound_method %tuple.elem0.loc25.1, %specific_fn.loc25.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc25.1: init %i32 = call %bound_method.loc25.2(%tuple.elem0.loc25.1)
// CHECK:STDOUT:   %.loc25.2: ref %tuple.type.d07 = as_compatible %d.var
// CHECK:STDOUT:   %tuple.elem0.loc25.2: ref %i32 = tuple_access %.loc25.2, element0
// CHECK:STDOUT:   %.loc25.3: init %i32 to %tuple.elem0.loc25.2 = in_place_init %Int.as.Copy.impl.Op.call.loc25.1
// CHECK:STDOUT:   %tuple.elem1.loc25.1: %i32 = tuple_access %.loc25.1, element1
// CHECK:STDOUT:   %impl.elem0.loc25.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc25.3: <bound method> = bound_method %tuple.elem1.loc25.1, %impl.elem0.loc25.2
// CHECK:STDOUT:   %specific_fn.loc25.2: <specific function> = specific_function %impl.elem0.loc25.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25.4: <bound method> = bound_method %tuple.elem1.loc25.1, %specific_fn.loc25.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc25.2: init %i32 = call %bound_method.loc25.4(%tuple.elem1.loc25.1)
// CHECK:STDOUT:   %tuple.elem1.loc25.2: ref %i32 = tuple_access %.loc25.2, element1
// CHECK:STDOUT:   %.loc25.4: init %i32 to %tuple.elem1.loc25.2 = in_place_init %Int.as.Copy.impl.Op.call.loc25.2
// CHECK:STDOUT:   %.loc25.5: init %tuple.type.d07 to %.loc25.2 = tuple_init (%.loc25.3, %.loc25.4)
// CHECK:STDOUT:   %.loc25.6: init %AdaptTuple = as_compatible %.loc25.5
// CHECK:STDOUT:   %.loc25.7: init %AdaptTuple = converted %c.ref, %.loc25.6
// CHECK:STDOUT:   assign %d.var, %.loc25.7
// CHECK:STDOUT:   %AdaptTuple.ref.loc28: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:   %d: ref %AdaptTuple = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %AdaptTuple = name_ref d, %d
// CHECK:STDOUT:   %.loc34.1: ref %tuple.type.d07 = as_compatible %d.ref
// CHECK:STDOUT:   %tuple.elem0.loc34.1: ref %i32 = tuple_access %.loc34.1, element0
// CHECK:STDOUT:   %.loc34.2: %i32 = acquire_value %tuple.elem0.loc34.1
// CHECK:STDOUT:   %impl.elem0.loc34.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc34.1: <bound method> = bound_method %.loc34.2, %impl.elem0.loc34.1
// CHECK:STDOUT:   %specific_fn.loc34.1: <specific function> = specific_function %impl.elem0.loc34.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34.2: <bound method> = bound_method %.loc34.2, %specific_fn.loc34.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc34.1: init %i32 = call %bound_method.loc34.2(%.loc34.2)
// CHECK:STDOUT:   %.loc34.3: ref %tuple.type.d07 = as_compatible %return.param
// CHECK:STDOUT:   %tuple.elem0.loc34.2: ref %i32 = tuple_access %.loc34.3, element0
// CHECK:STDOUT:   %.loc34.4: init %i32 to %tuple.elem0.loc34.2 = in_place_init %Int.as.Copy.impl.Op.call.loc34.1
// CHECK:STDOUT:   %tuple.elem1.loc34.1: ref %i32 = tuple_access %.loc34.1, element1
// CHECK:STDOUT:   %.loc34.5: %i32 = acquire_value %tuple.elem1.loc34.1
// CHECK:STDOUT:   %impl.elem0.loc34.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc34.3: <bound method> = bound_method %.loc34.5, %impl.elem0.loc34.2
// CHECK:STDOUT:   %specific_fn.loc34.2: <specific function> = specific_function %impl.elem0.loc34.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34.4: <bound method> = bound_method %.loc34.5, %specific_fn.loc34.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc34.2: init %i32 = call %bound_method.loc34.4(%.loc34.5)
// CHECK:STDOUT:   %tuple.elem1.loc34.2: ref %i32 = tuple_access %.loc34.3, element1
// CHECK:STDOUT:   %.loc34.6: init %i32 to %tuple.elem1.loc34.2 = in_place_init %Int.as.Copy.impl.Op.call.loc34.2
// CHECK:STDOUT:   %.loc34.7: init %tuple.type.d07 to %.loc34.3 = tuple_init (%.loc34.4, %.loc34.6)
// CHECK:STDOUT:   %.loc34.8: init %AdaptTuple = as_compatible %.loc34.7
// CHECK:STDOUT:   %.loc34.9: init %AdaptTuple = converted %d.ref, %.loc34.8
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return %.loc34.9 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc25(%self.param: ref %AdaptTuple) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.3c7) -> out %return.param: %tuple.type.3c7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.6f4 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.6f4 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.3c7 = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.3c7 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc63.1: %AdaptTuple = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc63.1: %tuple.type.d07 = as_compatible %tuple.elem0.loc63.1
// CHECK:STDOUT:   %tuple.elem0.loc63.2: %i32 = tuple_access %.loc63.1, element0
// CHECK:STDOUT:   %impl.elem0.loc63.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc63.1: <bound method> = bound_method %tuple.elem0.loc63.2, %impl.elem0.loc63.1
// CHECK:STDOUT:   %specific_fn.loc63.1: <specific function> = specific_function %impl.elem0.loc63.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc63.2: <bound method> = bound_method %tuple.elem0.loc63.2, %specific_fn.loc63.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc63.1: init %i32 = call %bound_method.loc63.2(%tuple.elem0.loc63.2)
// CHECK:STDOUT:   %tuple.elem0.loc63.3: ref %AdaptTuple = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc63.2: ref %tuple.type.d07 = as_compatible %tuple.elem0.loc63.3
// CHECK:STDOUT:   %tuple.elem0.loc63.4: ref %i32 = tuple_access %.loc63.2, element0
// CHECK:STDOUT:   %.loc63.3: init %i32 to %tuple.elem0.loc63.4 = in_place_init %Int.as.Copy.impl.Op.call.loc63.1
// CHECK:STDOUT:   %tuple.elem1.loc63.1: %i32 = tuple_access %.loc63.1, element1
// CHECK:STDOUT:   %impl.elem0.loc63.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc63.3: <bound method> = bound_method %tuple.elem1.loc63.1, %impl.elem0.loc63.2
// CHECK:STDOUT:   %specific_fn.loc63.2: <specific function> = specific_function %impl.elem0.loc63.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc63.4: <bound method> = bound_method %tuple.elem1.loc63.1, %specific_fn.loc63.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc63.2: init %i32 = call %bound_method.loc63.4(%tuple.elem1.loc63.1)
// CHECK:STDOUT:   %tuple.elem1.loc63.2: ref %i32 = tuple_access %.loc63.2, element1
// CHECK:STDOUT:   %.loc63.4: init %i32 to %tuple.elem1.loc63.2 = in_place_init %Int.as.Copy.impl.Op.call.loc63.2
// CHECK:STDOUT:   %.loc63.5: init %tuple.type.d07 to %.loc63.2 = tuple_init (%.loc63.3, %.loc63.4)
// CHECK:STDOUT:   %.loc63.6: init %AdaptTuple = as_compatible %.loc63.5
// CHECK:STDOUT:   %.loc63.7: init %AdaptTuple = converted %tuple.elem0.loc63.1, %.loc63.6
// CHECK:STDOUT:   %tuple.elem1.loc63.3: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc63.3: %.38c = impl_witness_access constants.%Copy.impl_witness.514, element0 [concrete = constants.%UInt.as.Copy.impl.Op.c10]
// CHECK:STDOUT:   %bound_method.loc63.5: <bound method> = bound_method %tuple.elem1.loc63.3, %impl.elem0.loc63.3
// CHECK:STDOUT:   %specific_fn.loc63.3: <specific function> = specific_function %impl.elem0.loc63.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc63.6: <bound method> = bound_method %tuple.elem1.loc63.3, %specific_fn.loc63.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc63: init %u32 = call %bound_method.loc63.6(%tuple.elem1.loc63.3)
// CHECK:STDOUT:   %tuple.elem1.loc63.4: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc63.8: init %u32 to %tuple.elem1.loc63.4 = in_place_init %UInt.as.Copy.impl.Op.call.loc63
// CHECK:STDOUT:   %.loc63.9: init %tuple.type.3c7 to %d.var = tuple_init (%.loc63.7, %.loc63.8)
// CHECK:STDOUT:   %.loc54: init %tuple.type.3c7 = converted %c.ref, %.loc63.9
// CHECK:STDOUT:   assign %d.var, %.loc54
// CHECK:STDOUT:   %.loc61.1: type = splice_block %.loc61.3 [concrete = constants.%tuple.type.3c7] {
// CHECK:STDOUT:     %AdaptTuple.ref.loc58: type = name_ref AdaptTuple, file.%AdaptTuple.decl [concrete = constants.%AdaptTuple]
// CHECK:STDOUT:     %u32.loc60: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc61.2: %tuple.type.24b = tuple_literal (%AdaptTuple.ref.loc58, %u32.loc60) [concrete = constants.%tuple.b75]
// CHECK:STDOUT:     %.loc61.3: type = converted %.loc61.2, constants.%tuple.type.3c7 [concrete = constants.%tuple.type.3c7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.3c7 = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.3c7 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc66.1: ref %AdaptTuple = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc66.1: ref %tuple.type.d07 = as_compatible %tuple.elem0.loc66.1
// CHECK:STDOUT:   %tuple.elem0.loc66.2: ref %i32 = tuple_access %.loc66.1, element0
// CHECK:STDOUT:   %.loc66.2: %i32 = acquire_value %tuple.elem0.loc66.2
// CHECK:STDOUT:   %impl.elem0.loc66.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc66.1: <bound method> = bound_method %.loc66.2, %impl.elem0.loc66.1
// CHECK:STDOUT:   %specific_fn.loc66.1: <specific function> = specific_function %impl.elem0.loc66.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc66.2: <bound method> = bound_method %.loc66.2, %specific_fn.loc66.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc66.1: init %i32 = call %bound_method.loc66.2(%.loc66.2)
// CHECK:STDOUT:   %tuple.elem0.loc66.3: ref %AdaptTuple = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc66.3: ref %tuple.type.d07 = as_compatible %tuple.elem0.loc66.3
// CHECK:STDOUT:   %tuple.elem0.loc66.4: ref %i32 = tuple_access %.loc66.3, element0
// CHECK:STDOUT:   %.loc66.4: init %i32 to %tuple.elem0.loc66.4 = in_place_init %Int.as.Copy.impl.Op.call.loc66.1
// CHECK:STDOUT:   %tuple.elem1.loc66.1: ref %i32 = tuple_access %.loc66.1, element1
// CHECK:STDOUT:   %.loc66.5: %i32 = acquire_value %tuple.elem1.loc66.1
// CHECK:STDOUT:   %impl.elem0.loc66.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc66.3: <bound method> = bound_method %.loc66.5, %impl.elem0.loc66.2
// CHECK:STDOUT:   %specific_fn.loc66.2: <specific function> = specific_function %impl.elem0.loc66.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc66.4: <bound method> = bound_method %.loc66.5, %specific_fn.loc66.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc66.2: init %i32 = call %bound_method.loc66.4(%.loc66.5)
// CHECK:STDOUT:   %tuple.elem1.loc66.2: ref %i32 = tuple_access %.loc66.3, element1
// CHECK:STDOUT:   %.loc66.6: init %i32 to %tuple.elem1.loc66.2 = in_place_init %Int.as.Copy.impl.Op.call.loc66.2
// CHECK:STDOUT:   %.loc66.7: init %tuple.type.d07 to %.loc66.3 = tuple_init (%.loc66.4, %.loc66.6)
// CHECK:STDOUT:   %.loc66.8: init %AdaptTuple = as_compatible %.loc66.7
// CHECK:STDOUT:   %.loc66.9: init %AdaptTuple = converted %tuple.elem0.loc66.1, %.loc66.8
// CHECK:STDOUT:   %tuple.elem1.loc66.3: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc66.10: %u32 = acquire_value %tuple.elem1.loc66.3
// CHECK:STDOUT:   %impl.elem0.loc66.3: %.38c = impl_witness_access constants.%Copy.impl_witness.514, element0 [concrete = constants.%UInt.as.Copy.impl.Op.c10]
// CHECK:STDOUT:   %bound_method.loc66.5: <bound method> = bound_method %.loc66.10, %impl.elem0.loc66.3
// CHECK:STDOUT:   %specific_fn.loc66.3: <specific function> = specific_function %impl.elem0.loc66.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc66.6: <bound method> = bound_method %.loc66.10, %specific_fn.loc66.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc66: init %u32 = call %bound_method.loc66.6(%.loc66.10)
// CHECK:STDOUT:   %tuple.elem1.loc66.4: ref %u32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc66.11: init %u32 to %tuple.elem1.loc66.4 = in_place_init %UInt.as.Copy.impl.Op.call.loc66
// CHECK:STDOUT:   %.loc66.12: init %tuple.type.3c7 to %return.param = tuple_init (%.loc66.9, %.loc66.11)
// CHECK:STDOUT:   %.loc67: init %tuple.type.3c7 = converted %d.ref, %.loc66.12
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return %.loc67 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc54(%self.param: ref %tuple.type.3c7) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_adapt_not_copyable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noncopyable: type = class_type @Noncopyable [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %AdaptNoncopyable: type = class_type @AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.92b: type = pattern_type %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %.e94: Core.Form = init_form %AdaptNoncopyable [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.92b = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.92b = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.92b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.92b = return_slot_pattern %return.param_patt, %AdaptNoncopyable.ref.loc23 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc23: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %AdaptNoncopyable.ref.loc23 [concrete = constants.%.e94]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyable = value_param call_param0
// CHECK:STDOUT:     %AdaptNoncopyable.ref.loc20: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:     %a: %AdaptNoncopyable = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %AdaptNoncopyable = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptNoncopyable = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Noncopyable {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Noncopyable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptNoncopyable {
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   adapt_decl %Noncopyable.ref [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyable
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %AdaptNoncopyable) -> out %return.param: %AdaptNoncopyable {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.92b = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.92b = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyable = var %b.var_patt
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyable = name_ref a, %a
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %AdaptNoncopyable.ref.loc28: type = name_ref AdaptNoncopyable, file.%AdaptNoncopyable.decl [concrete = constants.%AdaptNoncopyable]
// CHECK:STDOUT:   %b: ref %AdaptNoncopyable = ref_binding b, %b.var
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyable = name_ref b, %b
// CHECK:STDOUT:   %.loc33: %AdaptNoncopyable = acquire_value %b.ref
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%b.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %AdaptNoncopyable) = "no_op";
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
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.19a: %tuple.type.ff9 = tuple_value (%i32, %Noncopyable, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.7f9: type = tuple_type (%i32, %Noncopyable, %i32) [concrete]
// CHECK:STDOUT:   %complete_type.381: <witness> = complete_type_witness %tuple.type.7f9 [concrete]
// CHECK:STDOUT:   %pattern_type.ca6: type = pattern_type %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %.ae4: Core.Form = init_form %AdaptNoncopyableIndirect [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.ca6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.ca6 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ca6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.ca6 = return_slot_pattern %return.param_patt, %AdaptNoncopyableIndirect.ref.loc29 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc29: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %.loc29: Core.Form = init_form %AdaptNoncopyableIndirect.ref.loc29 [concrete = constants.%.ae4]
// CHECK:STDOUT:     %a.param: %AdaptNoncopyableIndirect = value_param call_param0
// CHECK:STDOUT:     %AdaptNoncopyableIndirect.ref.loc26: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:     %a: %AdaptNoncopyableIndirect = value_binding a, %a.param
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
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Noncopyable.ref: type = name_ref Noncopyable, file.%Noncopyable.decl [concrete = constants.%Noncopyable]
// CHECK:STDOUT:   %i32.loc17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc18: %tuple.type.ff9 = tuple_literal (%i32.loc13, %Noncopyable.ref, %i32.loc17) [concrete = constants.%tuple.19a]
// CHECK:STDOUT:   %.loc19: type = converted %.loc18, constants.%tuple.type.7f9 [concrete = constants.%tuple.type.7f9]
// CHECK:STDOUT:   adapt_decl %.loc19 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%tuple.type.7f9 [concrete = constants.%complete_type.381]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptNoncopyableIndirect
// CHECK:STDOUT:   .Noncopyable = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%a.param: %AdaptNoncopyableIndirect) -> out %return.param: %AdaptNoncopyableIndirect {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.ca6 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.ca6 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %AdaptNoncopyableIndirect = var %b.var_patt
// CHECK:STDOUT:   %a.ref: %AdaptNoncopyableIndirect = name_ref a, %a
// CHECK:STDOUT:   %.loc31.1: %tuple.type.7f9 = as_compatible %a.ref
// CHECK:STDOUT:   %tuple.elem0.loc31.1: %i32 = tuple_access %.loc31.1, element0
// CHECK:STDOUT:   %impl.elem0.loc31: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc31.1: <bound method> = bound_method %tuple.elem0.loc31.1, %impl.elem0.loc31
// CHECK:STDOUT:   %specific_fn.loc31: <specific function> = specific_function %impl.elem0.loc31, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31.2: <bound method> = bound_method %tuple.elem0.loc31.1, %specific_fn.loc31
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc31: init %i32 = call %bound_method.loc31.2(%tuple.elem0.loc31.1)
// CHECK:STDOUT:   %.loc31.2: ref %tuple.type.7f9 = as_compatible %b.var
// CHECK:STDOUT:   %tuple.elem0.loc31.2: ref %i32 = tuple_access %.loc31.2, element0
// CHECK:STDOUT:   %.loc31.3: init %i32 to %tuple.elem0.loc31.2 = in_place_init %Int.as.Copy.impl.Op.call.loc31
// CHECK:STDOUT:   %tuple.elem1.loc31: %Noncopyable = tuple_access %.loc31.1, element1
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   %AdaptNoncopyableIndirect.ref.loc34: type = name_ref AdaptNoncopyableIndirect, file.%AdaptNoncopyableIndirect.decl [concrete = constants.%AdaptNoncopyableIndirect]
// CHECK:STDOUT:   %b: ref %AdaptNoncopyableIndirect = ref_binding b, %b.var
// CHECK:STDOUT:   %b.ref: ref %AdaptNoncopyableIndirect = name_ref b, %b
// CHECK:STDOUT:   %.loc40.1: ref %tuple.type.7f9 = as_compatible %b.ref
// CHECK:STDOUT:   %tuple.elem0.loc40.1: ref %i32 = tuple_access %.loc40.1, element0
// CHECK:STDOUT:   %.loc40.2: %i32 = acquire_value %tuple.elem0.loc40.1
// CHECK:STDOUT:   %impl.elem0.loc40: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc40.1: <bound method> = bound_method %.loc40.2, %impl.elem0.loc40
// CHECK:STDOUT:   %specific_fn.loc40: <specific function> = specific_function %impl.elem0.loc40, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40.2: <bound method> = bound_method %.loc40.2, %specific_fn.loc40
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc40: init %i32 = call %bound_method.loc40.2(%.loc40.2)
// CHECK:STDOUT:   %.loc40.3: ref %tuple.type.7f9 = as_compatible %return.param
// CHECK:STDOUT:   %tuple.elem0.loc40.2: ref %i32 = tuple_access %.loc40.3, element0
// CHECK:STDOUT:   %.loc40.4: init %i32 to %tuple.elem0.loc40.2 = in_place_init %Int.as.Copy.impl.Op.call.loc40
// CHECK:STDOUT:   %tuple.elem1.loc40: ref %Noncopyable = tuple_access %.loc40.1, element1
// CHECK:STDOUT:   %.loc40.5: %Noncopyable = acquire_value %tuple.elem1.loc40
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %b.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%b.var)
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %AdaptNoncopyableIndirect) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt_copyable_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AdaptStruct: type = class_type @AdaptStruct [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.511: <witness> = complete_type_witness %struct_type.e.f [concrete]
// CHECK:STDOUT:   %pattern_type.341: type = pattern_type %AdaptStruct [concrete]
// CHECK:STDOUT:   %.8e3: Core.Form = init_form %AdaptStruct [concrete]
// CHECK:STDOUT:   %I.type: type = fn_type @I [concrete]
// CHECK:STDOUT:   %I: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.de4: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.de4) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet.de4 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc31 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.556: %tuple.type.24b = tuple_value (%AdaptStruct, %u32) [concrete]
// CHECK:STDOUT:   %tuple.type.691: type = tuple_type (%AdaptStruct, %u32) [concrete]
// CHECK:STDOUT:   %pattern_type.b13: type = pattern_type %tuple.type.691 [concrete]
// CHECK:STDOUT:   %.174: Core.Form = init_form %tuple.type.691 [concrete]
// CHECK:STDOUT:   %InTuple.type: type = fn_type @InTuple [concrete]
// CHECK:STDOUT:   %InTuple: %InTuple.type = struct_value () [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.68f: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.576: %UInt.as.Copy.impl.Op.type.68f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.514: <witness> = impl_witness imports.%Copy.impl_witness_table.bd0, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.2fc: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.c10: %UInt.as.Copy.impl.Op.type.2fc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.10b: %Copy.type = facet_value %u32, (%Copy.impl_witness.514) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ad7: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.10b) [concrete]
// CHECK:STDOUT:   %.38c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ad7, %Copy.facet.10b [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.c10, @UInt.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc60 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %Core.import_ref.c3c: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.68f) = import_ref Core//prelude/parts/uint, loc80, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.576)]
// CHECK:STDOUT:   %Copy.impl_witness_table.bd0 = impl_witness_table (%Core.import_ref.c3c), @UInt.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %g.param_patt: %pattern_type.341 = value_param_pattern [concrete]
// CHECK:STDOUT:     %g.patt: %pattern_type.341 = at_binding_pattern g, %g.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.341 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.341 = return_slot_pattern %return.param_patt, %AdaptStruct.ref.loc29 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc29: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %.loc29: Core.Form = init_form %AdaptStruct.ref.loc29 [concrete = constants.%.8e3]
// CHECK:STDOUT:     %g.param: %AdaptStruct = value_param call_param0
// CHECK:STDOUT:     %AdaptStruct.ref.loc26: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %g: %AdaptStruct = value_binding g, %g.param
// CHECK:STDOUT:     %return.param: ref %AdaptStruct = out_param call_param1
// CHECK:STDOUT:     %return: ref %AdaptStruct = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InTuple.decl: %InTuple.type = fn_decl @InTuple [concrete = constants.%InTuple] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.b13 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.b13 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b13 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.b13 = return_slot_pattern %return.param_patt, %.loc58.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %AdaptStruct.ref.loc55: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %u32.loc57: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc58.1: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc55, %u32.loc57) [concrete = constants.%tuple.556]
// CHECK:STDOUT:     %.loc58.2: type = converted %.loc58.1, constants.%tuple.type.691 [concrete = constants.%tuple.type.691]
// CHECK:STDOUT:     %.loc58.3: Core.Form = init_form %.loc58.2 [concrete = constants.%.174]
// CHECK:STDOUT:     %c.param: %tuple.type.691 = value_param call_param0
// CHECK:STDOUT:     %.loc51.1: type = splice_block %.loc51.3 [concrete = constants.%tuple.type.691] {
// CHECK:STDOUT:       %AdaptStruct.ref.loc48: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:       %u32.loc50: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:       %.loc51.2: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc48, %u32.loc50) [concrete = constants.%tuple.556]
// CHECK:STDOUT:       %.loc51.3: type = converted %.loc51.2, constants.%tuple.type.691 [concrete = constants.%tuple.type.691]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.691 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.691 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.691 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AdaptStruct {
// CHECK:STDOUT:   %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.loc17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %struct_type.e.f: type = struct_type {.e: %i32, .f: %i32} [concrete = constants.%struct_type.e.f]
// CHECK:STDOUT:   adapt_decl %struct_type.e.f [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.e.f [concrete = constants.%complete_type.511]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AdaptStruct
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I(%g.param: %AdaptStruct) -> out %return.param: %AdaptStruct {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.341 = ref_binding_pattern h [concrete]
// CHECK:STDOUT:     %h.var_patt: %pattern_type.341 = var_pattern %h.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %h.var: ref %AdaptStruct = var %h.var_patt
// CHECK:STDOUT:   %g.ref: %AdaptStruct = name_ref g, %g
// CHECK:STDOUT:   %.loc31.1: %struct_type.e.f = as_compatible %g.ref
// CHECK:STDOUT:   %.loc31.2: %i32 = struct_access %.loc31.1, element0
// CHECK:STDOUT:   %impl.elem0.loc31.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc31.1: <bound method> = bound_method %.loc31.2, %impl.elem0.loc31.1
// CHECK:STDOUT:   %specific_fn.loc31.1: <specific function> = specific_function %impl.elem0.loc31.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31.2: <bound method> = bound_method %.loc31.2, %specific_fn.loc31.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc31.1: init %i32 = call %bound_method.loc31.2(%.loc31.2)
// CHECK:STDOUT:   %.loc31.3: ref %struct_type.e.f = as_compatible %h.var
// CHECK:STDOUT:   %.loc31.4: ref %i32 = struct_access %.loc31.3, element0
// CHECK:STDOUT:   %.loc31.5: init %i32 to %.loc31.4 = in_place_init %Int.as.Copy.impl.Op.call.loc31.1
// CHECK:STDOUT:   %.loc31.6: %i32 = struct_access %.loc31.1, element1
// CHECK:STDOUT:   %impl.elem0.loc31.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc31.3: <bound method> = bound_method %.loc31.6, %impl.elem0.loc31.2
// CHECK:STDOUT:   %specific_fn.loc31.2: <specific function> = specific_function %impl.elem0.loc31.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31.4: <bound method> = bound_method %.loc31.6, %specific_fn.loc31.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc31.2: init %i32 = call %bound_method.loc31.4(%.loc31.6)
// CHECK:STDOUT:   %.loc31.7: ref %i32 = struct_access %.loc31.3, element1
// CHECK:STDOUT:   %.loc31.8: init %i32 to %.loc31.7 = in_place_init %Int.as.Copy.impl.Op.call.loc31.2
// CHECK:STDOUT:   %.loc31.9: init %struct_type.e.f to %.loc31.3 = struct_init (%.loc31.5, %.loc31.8)
// CHECK:STDOUT:   %.loc31.10: init %AdaptStruct = as_compatible %.loc31.9
// CHECK:STDOUT:   %.loc31.11: init %AdaptStruct = converted %g.ref, %.loc31.10
// CHECK:STDOUT:   assign %h.var, %.loc31.11
// CHECK:STDOUT:   %AdaptStruct.ref.loc34: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:   %h: ref %AdaptStruct = ref_binding h, %h.var
// CHECK:STDOUT:   %h.ref: ref %AdaptStruct = name_ref h, %h
// CHECK:STDOUT:   %.loc40.1: ref %struct_type.e.f = as_compatible %h.ref
// CHECK:STDOUT:   %.loc40.2: ref %i32 = struct_access %.loc40.1, element0
// CHECK:STDOUT:   %.loc40.3: %i32 = acquire_value %.loc40.2
// CHECK:STDOUT:   %impl.elem0.loc40.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc40.1: <bound method> = bound_method %.loc40.3, %impl.elem0.loc40.1
// CHECK:STDOUT:   %specific_fn.loc40.1: <specific function> = specific_function %impl.elem0.loc40.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40.2: <bound method> = bound_method %.loc40.3, %specific_fn.loc40.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc40.1: init %i32 = call %bound_method.loc40.2(%.loc40.3)
// CHECK:STDOUT:   %.loc40.4: ref %struct_type.e.f = as_compatible %return.param
// CHECK:STDOUT:   %.loc40.5: ref %i32 = struct_access %.loc40.4, element0
// CHECK:STDOUT:   %.loc40.6: init %i32 to %.loc40.5 = in_place_init %Int.as.Copy.impl.Op.call.loc40.1
// CHECK:STDOUT:   %.loc40.7: ref %i32 = struct_access %.loc40.1, element1
// CHECK:STDOUT:   %.loc40.8: %i32 = acquire_value %.loc40.7
// CHECK:STDOUT:   %impl.elem0.loc40.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc40.3: <bound method> = bound_method %.loc40.8, %impl.elem0.loc40.2
// CHECK:STDOUT:   %specific_fn.loc40.2: <specific function> = specific_function %impl.elem0.loc40.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc40.4: <bound method> = bound_method %.loc40.8, %specific_fn.loc40.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc40.2: init %i32 = call %bound_method.loc40.4(%.loc40.8)
// CHECK:STDOUT:   %.loc40.9: ref %i32 = struct_access %.loc40.4, element1
// CHECK:STDOUT:   %.loc40.10: init %i32 to %.loc40.9 = in_place_init %Int.as.Copy.impl.Op.call.loc40.2
// CHECK:STDOUT:   %.loc40.11: init %struct_type.e.f to %.loc40.4 = struct_init (%.loc40.6, %.loc40.10)
// CHECK:STDOUT:   %.loc40.12: init %AdaptStruct = as_compatible %.loc40.11
// CHECK:STDOUT:   %.loc40.13: init %AdaptStruct = converted %h.ref, %.loc40.12
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %h.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%h.var)
// CHECK:STDOUT:   return %.loc40.13 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc31(%self.param: ref %AdaptStruct) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InTuple(%c.param: %tuple.type.691) -> out %return.param: %tuple.type.691 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.b13 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.b13 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %tuple.type.691 = var %d.var_patt
// CHECK:STDOUT:   %c.ref: %tuple.type.691 = name_ref c, %c
// CHECK:STDOUT:   %tuple.elem0.loc69.1: %AdaptStruct = tuple_access %c.ref, element0
// CHECK:STDOUT:   %.loc69.1: %struct_type.e.f = as_compatible %tuple.elem0.loc69.1
// CHECK:STDOUT:   %.loc69.2: %i32 = struct_access %.loc69.1, element0
// CHECK:STDOUT:   %impl.elem0.loc69.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc69.1: <bound method> = bound_method %.loc69.2, %impl.elem0.loc69.1
// CHECK:STDOUT:   %specific_fn.loc69.1: <specific function> = specific_function %impl.elem0.loc69.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc69.2: <bound method> = bound_method %.loc69.2, %specific_fn.loc69.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc69.1: init %i32 = call %bound_method.loc69.2(%.loc69.2)
// CHECK:STDOUT:   %tuple.elem0.loc69.2: ref %AdaptStruct = tuple_access %d.var, element0
// CHECK:STDOUT:   %.loc69.3: ref %struct_type.e.f = as_compatible %tuple.elem0.loc69.2
// CHECK:STDOUT:   %.loc69.4: ref %i32 = struct_access %.loc69.3, element0
// CHECK:STDOUT:   %.loc69.5: init %i32 to %.loc69.4 = in_place_init %Int.as.Copy.impl.Op.call.loc69.1
// CHECK:STDOUT:   %.loc69.6: %i32 = struct_access %.loc69.1, element1
// CHECK:STDOUT:   %impl.elem0.loc69.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc69.3: <bound method> = bound_method %.loc69.6, %impl.elem0.loc69.2
// CHECK:STDOUT:   %specific_fn.loc69.2: <specific function> = specific_function %impl.elem0.loc69.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc69.4: <bound method> = bound_method %.loc69.6, %specific_fn.loc69.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc69.2: init %i32 = call %bound_method.loc69.4(%.loc69.6)
// CHECK:STDOUT:   %.loc69.7: ref %i32 = struct_access %.loc69.3, element1
// CHECK:STDOUT:   %.loc69.8: init %i32 to %.loc69.7 = in_place_init %Int.as.Copy.impl.Op.call.loc69.2
// CHECK:STDOUT:   %.loc69.9: init %struct_type.e.f to %.loc69.3 = struct_init (%.loc69.5, %.loc69.8)
// CHECK:STDOUT:   %.loc69.10: init %AdaptStruct = as_compatible %.loc69.9
// CHECK:STDOUT:   %.loc69.11: init %AdaptStruct = converted %tuple.elem0.loc69.1, %.loc69.10
// CHECK:STDOUT:   %tuple.elem1.loc69.1: %u32 = tuple_access %c.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc69.3: %.38c = impl_witness_access constants.%Copy.impl_witness.514, element0 [concrete = constants.%UInt.as.Copy.impl.Op.c10]
// CHECK:STDOUT:   %bound_method.loc69.5: <bound method> = bound_method %tuple.elem1.loc69.1, %impl.elem0.loc69.3
// CHECK:STDOUT:   %specific_fn.loc69.3: <specific function> = specific_function %impl.elem0.loc69.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc69.6: <bound method> = bound_method %tuple.elem1.loc69.1, %specific_fn.loc69.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc69: init %u32 = call %bound_method.loc69.6(%tuple.elem1.loc69.1)
// CHECK:STDOUT:   %tuple.elem1.loc69.2: ref %u32 = tuple_access %d.var, element1
// CHECK:STDOUT:   %.loc69.12: init %u32 to %tuple.elem1.loc69.2 = in_place_init %UInt.as.Copy.impl.Op.call.loc69
// CHECK:STDOUT:   %.loc69.13: init %tuple.type.691 to %d.var = tuple_init (%.loc69.11, %.loc69.12)
// CHECK:STDOUT:   %.loc60: init %tuple.type.691 = converted %c.ref, %.loc69.13
// CHECK:STDOUT:   assign %d.var, %.loc60
// CHECK:STDOUT:   %.loc67.1: type = splice_block %.loc67.3 [concrete = constants.%tuple.type.691] {
// CHECK:STDOUT:     %AdaptStruct.ref.loc64: type = name_ref AdaptStruct, file.%AdaptStruct.decl [concrete = constants.%AdaptStruct]
// CHECK:STDOUT:     %u32.loc66: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:     %.loc67.2: %tuple.type.24b = tuple_literal (%AdaptStruct.ref.loc64, %u32.loc66) [concrete = constants.%tuple.556]
// CHECK:STDOUT:     %.loc67.3: type = converted %.loc67.2, constants.%tuple.type.691 [concrete = constants.%tuple.type.691]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %tuple.type.691 = ref_binding d, %d.var
// CHECK:STDOUT:   %d.ref: ref %tuple.type.691 = name_ref d, %d
// CHECK:STDOUT:   %tuple.elem0.loc72.1: ref %AdaptStruct = tuple_access %d.ref, element0
// CHECK:STDOUT:   %.loc72.1: ref %struct_type.e.f = as_compatible %tuple.elem0.loc72.1
// CHECK:STDOUT:   %.loc72.2: ref %i32 = struct_access %.loc72.1, element0
// CHECK:STDOUT:   %.loc72.3: %i32 = acquire_value %.loc72.2
// CHECK:STDOUT:   %impl.elem0.loc72.1: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc72.1: <bound method> = bound_method %.loc72.3, %impl.elem0.loc72.1
// CHECK:STDOUT:   %specific_fn.loc72.1: <specific function> = specific_function %impl.elem0.loc72.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc72.2: <bound method> = bound_method %.loc72.3, %specific_fn.loc72.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc72.1: init %i32 = call %bound_method.loc72.2(%.loc72.3)
// CHECK:STDOUT:   %tuple.elem0.loc72.2: ref %AdaptStruct = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc72.4: ref %struct_type.e.f = as_compatible %tuple.elem0.loc72.2
// CHECK:STDOUT:   %.loc72.5: ref %i32 = struct_access %.loc72.4, element0
// CHECK:STDOUT:   %.loc72.6: init %i32 to %.loc72.5 = in_place_init %Int.as.Copy.impl.Op.call.loc72.1
// CHECK:STDOUT:   %.loc72.7: ref %i32 = struct_access %.loc72.1, element1
// CHECK:STDOUT:   %.loc72.8: %i32 = acquire_value %.loc72.7
// CHECK:STDOUT:   %impl.elem0.loc72.2: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc72.3: <bound method> = bound_method %.loc72.8, %impl.elem0.loc72.2
// CHECK:STDOUT:   %specific_fn.loc72.2: <specific function> = specific_function %impl.elem0.loc72.2, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc72.4: <bound method> = bound_method %.loc72.8, %specific_fn.loc72.2
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc72.2: init %i32 = call %bound_method.loc72.4(%.loc72.8)
// CHECK:STDOUT:   %.loc72.9: ref %i32 = struct_access %.loc72.4, element1
// CHECK:STDOUT:   %.loc72.10: init %i32 to %.loc72.9 = in_place_init %Int.as.Copy.impl.Op.call.loc72.2
// CHECK:STDOUT:   %.loc72.11: init %struct_type.e.f to %.loc72.4 = struct_init (%.loc72.6, %.loc72.10)
// CHECK:STDOUT:   %.loc72.12: init %AdaptStruct = as_compatible %.loc72.11
// CHECK:STDOUT:   %.loc72.13: init %AdaptStruct = converted %tuple.elem0.loc72.1, %.loc72.12
// CHECK:STDOUT:   %tuple.elem1.loc72.1: ref %u32 = tuple_access %d.ref, element1
// CHECK:STDOUT:   %.loc72.14: %u32 = acquire_value %tuple.elem1.loc72.1
// CHECK:STDOUT:   %impl.elem0.loc72.3: %.38c = impl_witness_access constants.%Copy.impl_witness.514, element0 [concrete = constants.%UInt.as.Copy.impl.Op.c10]
// CHECK:STDOUT:   %bound_method.loc72.5: <bound method> = bound_method %.loc72.14, %impl.elem0.loc72.3
// CHECK:STDOUT:   %specific_fn.loc72.3: <specific function> = specific_function %impl.elem0.loc72.3, @UInt.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc72.6: <bound method> = bound_method %.loc72.14, %specific_fn.loc72.3
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call.loc72: init %u32 = call %bound_method.loc72.6(%.loc72.14)
// CHECK:STDOUT:   %tuple.elem1.loc72.2: ref %u32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc72.15: init %u32 to %tuple.elem1.loc72.2 = in_place_init %UInt.as.Copy.impl.Op.call.loc72
// CHECK:STDOUT:   %.loc72.16: init %tuple.type.691 to %return.param = tuple_init (%.loc72.13, %.loc72.15)
// CHECK:STDOUT:   %.loc73: init %tuple.type.691 = converted %d.ref, %.loc72.16
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return %.loc73 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc60(%self.param: ref %tuple.type.691) = "no_op";
// CHECK:STDOUT:
