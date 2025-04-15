// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/generic.carbon

fn F() {
  let T:! type = i32;
  var p: T*;
  var a: T = *p;
}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3af: <witness> = lookup_impl_witness %ptr, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7d0: %Destroy.type = facet_value %ptr, (%Destroy.lookup_impl_witness.3af) [symbolic]
// CHECK:STDOUT:   %.ea9: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.7d0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.63c: %.ea9 = impl_witness_access %Destroy.lookup_impl_witness.3af, element0 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.de0: <witness> = lookup_impl_witness %T, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.713: %Destroy.type = facet_value %T, (%Destroy.lookup_impl_witness.de0) [symbolic]
// CHECK:STDOUT:   %.a63: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.713 [symbolic]
// CHECK:STDOUT:   %impl.elem0.538: %.a63 = impl_witness_access %Destroy.lookup_impl_witness.de0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1af: <specific function> = specific_impl_function %impl.elem0.538, @Op.1(%Destroy.facet.713) [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e4a: <specific function> = specific_impl_function %impl.elem0.63c, @Op.1(%Destroy.facet.7d0) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, %i32 [symbolic = constants.%T]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %.loc13_3.1: %ptr = var_pattern %p.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr = var p
// CHECK:STDOUT:   %impl.elem0.loc13: %.ea9 = impl_witness_access constants.%Destroy.lookup_impl_witness.3af, element0 [symbolic = constants.%impl.elem0.63c]
// CHECK:STDOUT:   %bound_method.loc13: <bound method> = bound_method %p.var, %impl.elem0.loc13
// CHECK:STDOUT:   %.loc13_11: type = splice_block %ptr [symbolic = constants.%ptr] {
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:     %ptr: type = ptr_type %T.ref.loc13 [symbolic = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr = bind_name p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %T = binding_pattern a
// CHECK:STDOUT:     %.loc14_3.1: %T = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %T = var a
// CHECK:STDOUT:   %impl.elem0.loc14: %.a63 = impl_witness_access constants.%Destroy.lookup_impl_witness.de0, element0 [symbolic = constants.%impl.elem0.538]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %a.var, %impl.elem0.loc14
// CHECK:STDOUT:   %p.ref: ref %ptr = name_ref p, %p
// CHECK:STDOUT:   %.loc14_15: %ptr = bind_value %p.ref
// CHECK:STDOUT:   %.loc14_14.1: ref %T = deref %.loc14_15
// CHECK:STDOUT:   %.loc14_14.2: %T = bind_value %.loc14_14.1
// CHECK:STDOUT:   assign %a.var, %.loc14_14.2
// CHECK:STDOUT:   %T.ref.loc14: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %a: ref %T = bind_name a, %a.var
// CHECK:STDOUT:   %specific_impl_fn.loc14: <specific function> = specific_impl_function %impl.elem0.loc14, @Op.1(constants.%Destroy.facet.713) [symbolic = constants.%specific_impl_fn.1af]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %a.var, %specific_impl_fn.loc14
// CHECK:STDOUT:   %.loc14_3.2: %T = bind_value %a.var
// CHECK:STDOUT:   %.1: init %empty_tuple.type = call %bound_method.1(%.loc14_3.2)
// CHECK:STDOUT:   %specific_impl_fn.loc13: <specific function> = specific_impl_function %impl.elem0.loc13, @Op.1(constants.%Destroy.facet.7d0) [symbolic = constants.%specific_impl_fn.e4a]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %p.var, %specific_impl_fn.loc13
// CHECK:STDOUT:   %.loc13_3.2: %ptr = bind_value %p.var
// CHECK:STDOUT:   %.2: init %empty_tuple.type = call %bound_method.2(%.loc13_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
