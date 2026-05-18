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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/forward_declared.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/forward_declared.carbon

class Class;

fn F(p: Class*) -> Class* { return p; }

// CHECK:STDOUT: --- forward_declared.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %ptr.1fe: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.255: type = pattern_type %ptr.1fe [concrete]
// CHECK:STDOUT:   %.f87: Core.Form = init_form %ptr.1fe [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.8a8: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6bf: %ptr.as.Copy.impl.Op.type.8a8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ee2: <witness> = impl_witness imports.%Copy.impl_witness_table.3cb, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.1d6: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.172: %ptr.as.Copy.impl.Op.type.1d6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.1fe, (%Copy.impl_witness.ee2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a0f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.629: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a0f, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.172, @ptr.as.Copy.impl.Op(%Class) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.3ef: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.8a8) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.6bf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3cb = impl_witness_table (%Core.import_ref.3ef), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.255 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.255 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.255 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.255 = return_slot_pattern %return.param_patt, %ptr.loc17_25 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc17_20: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc17_25: type = ptr_type %Class.ref.loc17_20 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:     %.loc17_25: Core.Form = init_form %ptr.loc17_25 [concrete = constants.%.f87]
// CHECK:STDOUT:     %p.param: %ptr.1fe = value_param call_param0
// CHECK:STDOUT:     %.loc17_14: type = splice_block %ptr.loc17_14 [concrete = constants.%ptr.1fe] {
// CHECK:STDOUT:       %Class.ref.loc17_9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr.loc17_14: type = ptr_type %Class.ref.loc17_9 [concrete = constants.%ptr.1fe]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.1fe = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.1fe = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.1fe = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %ptr.1fe) -> out %return.param: %ptr.1fe {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1fe = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.629 = impl_witness_access constants.%Copy.impl_witness.ee2, element0 [concrete = constants.%ptr.as.Copy.impl.Op.172]
// CHECK:STDOUT:   %bound_method.loc17_36.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_36.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.1fe = call %bound_method.loc17_36.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
