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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/base_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/base_field.carbon

base class Base {
  var a: i32;
  var b: i32;
  var c: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
  var e: i32;
}

fn Access(p: Derived*) -> i32* {
  return &(*p).c;
}

// CHECK:STDOUT: --- base_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.bae: <witness> = complete_type_witness %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem.93c: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.elem.577: type = unbound_element_type %Derived, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.e.26c: type = struct_type {.base: %Base, .d: %i32, .e: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.9b3: <witness> = complete_type_witness %struct_type.base.d.e.26c [concrete]
// CHECK:STDOUT:   %ptr.64c: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.56c: type = pattern_type %ptr.64c [concrete]
// CHECK:STDOUT:   %p.param_patt: %pattern_type.56c = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.56c = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.952: Core.Form = init_form %ptr.d08 [concrete]
// CHECK:STDOUT:   %pattern_type.f00: type = pattern_type %ptr.d08 [concrete]
// CHECK:STDOUT:   %return.param_patt.7dd: %pattern_type.f00 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.dbe: %pattern_type.f00 = return_slot_pattern %return.param_patt.7dd, %ptr.d08 [concrete]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [concrete]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ce1: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.c1c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.d1c: %ptr.as.Copy.impl.Op.type.c1c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.d08, (%Copy.impl_witness.ce1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f9d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f9d, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.d1c, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [concrete = constants.%Access] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.56c = value_param_pattern [concrete = constants.%p.param_patt]
// CHECK:STDOUT:     %p.patt: %pattern_type.56c = at_binding_pattern p, %p.param_patt [concrete = constants.%p.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f00 = out_param_pattern [concrete = constants.%return.param_patt.7dd]
// CHECK:STDOUT:     %return.patt: %pattern_type.f00 = return_slot_pattern %return.param_patt, %ptr.loc28_30 [concrete = constants.%return.patt.dbe]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc28_30: type = ptr_type %i32 [concrete = constants.%ptr.d08]
// CHECK:STDOUT:     %.loc28_30: Core.Form = init_form %ptr.loc28_30 [concrete = constants.%.952]
// CHECK:STDOUT:     %p.param: %ptr.64c = value_param call_param0
// CHECK:STDOUT:     %.loc28_21: type = splice_block %ptr.loc28_21 [concrete = constants.%ptr.64c] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc28_21: type = ptr_type %Derived.ref [concrete = constants.%ptr.64c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.64c = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.d08 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.d08 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc16: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.loc17: %Base.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %.loc18: %Base.elem = field_decl c, element2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.c [concrete = constants.%complete_type.bae]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc16
// CHECK:STDOUT:   .b = %.loc17
// CHECK:STDOUT:   .c = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc22: %Derived.elem.93c = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc24: %Derived.elem.577 = field_decl d, element1 [concrete]
// CHECK:STDOUT:   %.loc25: %Derived.elem.577 = field_decl e, element2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.d.e.26c [concrete = constants.%complete_type.9b3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc22
// CHECK:STDOUT:   .d = %.loc24
// CHECK:STDOUT:   .e = %.loc25
// CHECK:STDOUT:   .c = <poisoned>
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%p.param: %ptr.64c) -> out %return.param: %ptr.d08 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.64c = name_ref p, %p
// CHECK:STDOUT:   %.loc29_12: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %c.ref: %Base.elem = name_ref c, @Base.%.loc18 [concrete = @Base.%.loc18]
// CHECK:STDOUT:   %.loc29_15.1: ref %Base = as_compatible %.loc29_12
// CHECK:STDOUT:   %.loc29_15.2: ref %Base = class_element_access %.loc29_15.1, element0
// CHECK:STDOUT:   %.loc29_15.3: ref %Base = converted %.loc29_12, %.loc29_15.2
// CHECK:STDOUT:   %.loc29_15.4: ref %i32 = class_element_access %.loc29_15.3, element2
// CHECK:STDOUT:   %addr: %ptr.d08 = addr_of %.loc29_15.4
// CHECK:STDOUT:   %impl.elem0: %.ac9 = impl_witness_access constants.%Copy.impl_witness.ce1, element0 [concrete = constants.%ptr.as.Copy.impl.Op.d1c]
// CHECK:STDOUT:   %bound_method.loc29_10.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_10.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.d08 = call %bound_method.loc29_10.2(%addr)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
