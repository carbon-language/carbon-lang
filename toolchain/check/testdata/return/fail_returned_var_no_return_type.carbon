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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_returned_var_no_return_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_returned_var_no_return_type.carbon

// --- fail_procedure_with_returned_var.carbon

library "[[@TEST_NAME]]";

fn Procedure() {
  // CHECK:STDERR: fail_procedure_with_returned_var.carbon:[[@LINE+3]]:3-10: error: cannot declare a `returned var` in this function [ReturnedVarWithNoReturnType]
  // CHECK:STDERR: fail_procedure_with_returned_var.carbon:[[@LINE-2]]:1-16: note: there was no return type provided [ReturnTypeOmittedNote]
  // CHECK:STDERR:
  returned var unused v: () = ();
  return;
}

// --- fail_forgot_return_type.carbon

library "[[@TEST_NAME]]";

fn ForgotReturnType() {
  // CHECK:STDERR: fail_forgot_return_type.carbon:[[@LINE+3]]:3-10: error: cannot declare a `returned var` in this function [ReturnedVarWithNoReturnType]
  // CHECK:STDERR: fail_forgot_return_type.carbon:[[@LINE-2]]:1-23: note: there was no return type provided [ReturnTypeOmittedNote]
  // CHECK:STDERR:
  returned var unused v: () = ();
  // CHECK:STDERR: fail_forgot_return_type.carbon:[[@LINE+2]]:3-13: error: `return var;` with no `returned var` in scope [ReturnVarWithNoReturnedVar]
  // CHECK:STDERR:
  return var;
}

// CHECK:STDOUT: --- fail_procedure_with_returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Procedure.type: type = fn_type @Procedure [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Procedure: %Procedure.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.cb1 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.cb1 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %empty_tuple.type, (%DestroyImpl.impl_witness.674, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Procedure = %Procedure.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Procedure.decl: %Procedure.type = fn_decl @Procedure [concrete = constants.%Procedure] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Procedure() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var_storage %v.var_patt
// CHECK:STDOUT:   %.loc8_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_32.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_12.1: init %empty_tuple.type = converted %.loc8_32.1, %.loc8_32.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %v.var, %.loc8_12.1
// CHECK:STDOUT:   %.loc8_27.1: type = splice_block %.loc8_27.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_27.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_27.3: type = converted %.loc8_27.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_tuple.type = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.cb1 = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.cb1 = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_12.1: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_12.2: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc8_12.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_12.2: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_12.3: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc8_12.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%v.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_forgot_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %ForgotReturnType.type: type = fn_type @ForgotReturnType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ForgotReturnType: %ForgotReturnType.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.cb1 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.cb1 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %empty_tuple.type, (%DestroyImpl.impl_witness.674, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ForgotReturnType = %ForgotReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ForgotReturnType.decl: %ForgotReturnType.type = fn_decl @ForgotReturnType [concrete = constants.%ForgotReturnType] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForgotReturnType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var_storage %v.var_patt
// CHECK:STDOUT:   %.loc8_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_32.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_12.1: init %empty_tuple.type = converted %.loc8_32.1, %.loc8_32.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %v.var, %.loc8_12.1
// CHECK:STDOUT:   %.loc8_27.1: type = splice_block %.loc8_27.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8_27.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_27.3: type = converted %.loc8_27.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_tuple.type = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.cb1 = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.cb1 = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc8_12.1: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_12.2: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc8_12.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc8_12.2: %Destroy.type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc8_12.3: %Destroy.type = converted constants.%empty_tuple.type, %Destroy.facet.loc8_12.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%v.var)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc8(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
