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
// CHECK:STDOUT:   %ptr.bd7: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.9b1: type = pattern_type %ptr.bd7 [concrete]
// CHECK:STDOUT:   %.f1d: Core.Form = init_form %ptr.bd7 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.34c: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.eaf: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ef3: %ptr.as.Copy.impl.Op.type.eaf = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.bd7, (%Copy.impl_witness.34c) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e9c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.cc1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e9c, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.ef3, @ptr.as.Copy.impl.Op(%Class) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %p.param_patt: %pattern_type.9b1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.9b1 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9b1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.9b1 = return_slot_pattern %return.param_patt, %ptr.loc17_25 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc17_20: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %ptr.loc17_25: type = ptr_type %Class.ref.loc17_20 [concrete = constants.%ptr.bd7]
// CHECK:STDOUT:     %.loc17_25: Core.Form = init_form %ptr.loc17_25 [concrete = constants.%.f1d]
// CHECK:STDOUT:     %p.param: %ptr.bd7 = value_param call_param0
// CHECK:STDOUT:     %.loc17_14: type = splice_block %ptr.loc17_14 [concrete = constants.%ptr.bd7] {
// CHECK:STDOUT:       %Class.ref.loc17_9: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr.loc17_14: type = ptr_type %Class.ref.loc17_9 [concrete = constants.%ptr.bd7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.bd7 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.bd7 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.bd7 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %ptr.bd7) -> out %return.param: %ptr.bd7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.bd7 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.cc1 = impl_witness_access constants.%Copy.impl_witness.34c, element0 [concrete = constants.%ptr.as.Copy.impl.Op.ef3]
// CHECK:STDOUT:   %bound_method.loc17_36.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_36.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.bd7 = call %bound_method.loc17_36.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
