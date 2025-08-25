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
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.de0: <witness> = lookup_impl_witness %T, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.713: %Destroy.type = facet_value %T, (%Destroy.lookup_impl_witness.de0) [symbolic]
// CHECK:STDOUT:   %.a63: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.713 [symbolic]
// CHECK:STDOUT:   %impl.elem0.538: %.a63 = impl_witness_access %Destroy.lookup_impl_witness.de0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.1af: <specific function> = specific_impl_function %impl.elem0.538, @Destroy.Op(%Destroy.facet.713) [symbolic]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.3af: <witness> = lookup_impl_witness %ptr.79f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.7d0: %Destroy.type = facet_value %ptr.79f, (%Destroy.lookup_impl_witness.3af) [symbolic]
// CHECK:STDOUT:   %.ea9: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.7d0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.63c: %.ea9 = impl_witness_access %Destroy.lookup_impl_witness.3af, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e4a: <specific function> = specific_impl_function %impl.elem0.63c, @Destroy.Op(%Destroy.facet.7d0) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, %i32 [symbolic = constants.%T]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.afe = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.var_patt: %pattern_type.afe = var_pattern %p.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr.79f = var %p.var_patt
// CHECK:STDOUT:   %.loc17_11: type = splice_block %ptr [symbolic = constants.%ptr.79f] {
// CHECK:STDOUT:     %T.ref.loc17: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:     %ptr: type = ptr_type %T.ref.loc17 [symbolic = constants.%ptr.79f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr.79f = bind_name p, %p.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7dc = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7dc = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %T = var %a.var_patt
// CHECK:STDOUT:   %p.ref: ref %ptr.79f = name_ref p, %p
// CHECK:STDOUT:   %.loc18_15: %ptr.79f = bind_value %p.ref
// CHECK:STDOUT:   %.loc18_14.1: ref %T = deref %.loc18_15
// CHECK:STDOUT:   %.loc18_14.2: %T = bind_value %.loc18_14.1
// CHECK:STDOUT:   assign %a.var, %.loc18_14.2
// CHECK:STDOUT:   %T.ref.loc18: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %a: ref %T = bind_name a, %a.var
// CHECK:STDOUT:   %impl.elem0.loc18: %.a63 = impl_witness_access constants.%Destroy.lookup_impl_witness.de0, element0 [symbolic = constants.%impl.elem0.538]
// CHECK:STDOUT:   %bound_method.loc18_3.1: <bound method> = bound_method %a.var, %impl.elem0.loc18
// CHECK:STDOUT:   %specific_impl_fn.loc18: <specific function> = specific_impl_function %impl.elem0.loc18, @Destroy.Op(constants.%Destroy.facet.713) [symbolic = constants.%specific_impl_fn.1af]
// CHECK:STDOUT:   %bound_method.loc18_3.2: <bound method> = bound_method %a.var, %specific_impl_fn.loc18
// CHECK:STDOUT:   %addr.loc18: %ptr.79f = addr_of %a.var
// CHECK:STDOUT:   %.loc18_3: init %empty_tuple.type = call %bound_method.loc18_3.2(%addr.loc18)
// CHECK:STDOUT:   %impl.elem0.loc17: %.ea9 = impl_witness_access constants.%Destroy.lookup_impl_witness.3af, element0 [symbolic = constants.%impl.elem0.63c]
// CHECK:STDOUT:   %bound_method.loc17_3.1: <bound method> = bound_method %p.var, %impl.elem0.loc17
// CHECK:STDOUT:   %specific_impl_fn.loc17: <specific function> = specific_impl_function %impl.elem0.loc17, @Destroy.Op(constants.%Destroy.facet.7d0) [symbolic = constants.%specific_impl_fn.e4a]
// CHECK:STDOUT:   %bound_method.loc17_3.2: <bound method> = bound_method %p.var, %specific_impl_fn.loc17
// CHECK:STDOUT:   %addr.loc17: %ptr.a13 = addr_of %p.var
// CHECK:STDOUT:   %.loc17_3: init %empty_tuple.type = call %bound_method.loc17_3.2(%addr.loc17)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
