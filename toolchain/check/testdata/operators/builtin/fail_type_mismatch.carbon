// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_type_mismatch.carbon

fn Main() {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:24: error: cannot implicitly convert expression of type `Core.IntLiteral` to `bool` [ConversionFailure]
  // CHECK:STDERR:   var unused x: bool = not 12;
  // CHECK:STDERR:                        ^~~~~~
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+4]]:24: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(bool)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused x: bool = not 12;
  // CHECK:STDERR:                        ^~~~~~
  // CHECK:STDERR:
  var unused x: bool = not 12;
}

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.831 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.831 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %facet_value: %type = facet_value bool, () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.af6: type = fn_type @Destroy.WithSelf.SubobjectDestroy, @Destroy.WithSelf(%facet_value) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.1dc: %Destroy.WithSelf.SubobjectDestroy.type.af6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.286: type = fn_type @Destroy.WithSelf.SelfDestruct, @Destroy.WithSelf(%facet_value) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.a26: %Destroy.WithSelf.SelfDestruct.type.286 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.7b8: <witness> = custom_witness (%Destroy.Op, %Destroy.WithSelf.SubobjectDestroy.1dc, %Destroy.WithSelf.SelfDestruct.a26), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.737: %Destroy.type = facet_value bool, (%custom_witness.7b8) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn: <specific function> = specific_function %Destroy.WithSelf.SelfDestruct.a26, @Destroy.WithSelf.SelfDestruct(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref bool = var_storage %x.var_patt
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc23_24.1: bool = converted %int_12, <error> [concrete = <error>]
// CHECK:STDOUT:   %.loc23_24.2: <error> = not <error> [concrete = <error>]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   %.loc23_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %x: ref bool = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.831 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %x.var, constants.%Destroy.WithSelf.SelfDestruct.a26
// CHECK:STDOUT:   %Destroy.facet.loc23_3.1: %Destroy.type = facet_value bool, (constants.%custom_witness.7b8) [concrete = constants.%Destroy.facet.737]
// CHECK:STDOUT:   %.loc23_3.1: %Destroy.type = converted bool, %Destroy.facet.loc23_3.1 [concrete = constants.%Destroy.facet.737]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.2: %Destroy.type = facet_value bool, (constants.%custom_witness.7b8) [concrete = constants.%Destroy.facet.737]
// CHECK:STDOUT:   %.loc23_3.2: %Destroy.type = converted bool, %Destroy.facet.loc23_3.2 [concrete = constants.%Destroy.facet.737]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.specific_fn: <specific function> = specific_function constants.%Destroy.WithSelf.SelfDestruct.a26, @Destroy.WithSelf.SelfDestruct(constants.%facet_value) [concrete = constants.%Destroy.WithSelf.SelfDestruct.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %x.var, %Destroy.WithSelf.SelfDestruct.specific_fn
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %bound_method(%x.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
