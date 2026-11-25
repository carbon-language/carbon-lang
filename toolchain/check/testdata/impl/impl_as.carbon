// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_as.carbon

interface Simple {
  fn F();
}

class C {
  impl as Simple {
    fn F() {
      // C is a complete type here.
      var unused c: C = {};
    }
  }
}

// CHECK:STDOUT: --- impl_as.carbon = $impl_as
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Simple.type: type = facet_type <@Simple> [concrete]
// CHECK:STDOUT:   %Self.f79: %Simple.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.type.a46: type = fn_type @Simple.WithSelf.F, @Simple.WithSelf(%Self.f79) [symbolic]
// CHECK:STDOUT:   %Simple.WithSelf.F.0d0: %Simple.WithSelf.F.type.a46 = struct_value () [symbolic]
// CHECK:STDOUT:   %Simple.assoc_type: type = assoc_entity_type @Simple [concrete]
// CHECK:STDOUT:   %assoc0.2ae: %Simple.assoc_type = assoc_entity element0, @Simple.WithSelf.%Simple.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.dd3: <witness> = impl_self_witness %C, @Simple [symbolic_self]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness @C.as.Simple.impl.%Simple.impl_witness_table [concrete]
// CHECK:STDOUT:   %C.as.Simple.impl.F.type: type = fn_type @C.as.Simple.impl.F [concrete]
// CHECK:STDOUT:   %C.as.Simple.impl.F: %C.as.Simple.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Simple.facet: %Simple.type = facet_value %C, (%Simple.impl_witness) [concrete]
// CHECK:STDOUT:   %Simple.WithSelf.F.type.658: type = fn_type @Simple.WithSelf.F, @Simple.WithSelf(%Simple.facet) [concrete]
// CHECK:STDOUT:   %Simple.WithSelf.F.67c: %Simple.WithSelf.F.type.658 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %TrivialDestroyImpl.type: type = facet_type <@TrivialDestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .TrivialDestroyImpl = %Core.TrivialDestroyImpl
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.TrivialDestroyImpl: type = import_ref Core//prelude/parts/destroy, TrivialDestroyImpl, loaded [concrete = constants.%TrivialDestroyImpl.type]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Simple = %Simple.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Simple.decl: type = interface_decl @Simple [concrete = constants.%Simple.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Simple {
// CHECK:STDOUT:   %Self: %Simple.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f79]
// CHECK:STDOUT:   %Simple.WithSelf.decl = interface_with_self_decl @Simple [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Simple.WithSelf.F.decl: @Simple.WithSelf.%Simple.WithSelf.F.type (%Simple.WithSelf.F.type.a46) = fn_decl @Simple.WithSelf.F [symbolic = @Simple.WithSelf.%Simple.WithSelf.F (constants.%Simple.WithSelf.F.0d0)] {} {}
// CHECK:STDOUT:   %assoc0: %Simple.assoc_type = assoc_entity element0, %Simple.WithSelf.F.decl [concrete = constants.%assoc0.2ae]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @Simple.WithSelf.%assoc0
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   witness = (@Simple.WithSelf.%Simple.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Simple.impl: %Self.ref as %Simple.ref {
// CHECK:STDOUT:   %C.as.Simple.impl.F.decl: %C.as.Simple.impl.F.type = fn_decl @C.as.Simple.impl.F [concrete = constants.%C.as.Simple.impl.F] {} {}
// CHECK:STDOUT:   %Simple.impl_witness_table = impl_witness_table (%C.as.Simple.impl.F.decl), @C.as.Simple.impl [concrete]
// CHECK:STDOUT:   %Simple.impl_witness: <witness> = impl_witness %Simple.impl_witness_table [concrete = constants.%Simple.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %C.as.Simple.impl.F.decl
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   extend %Simple.ref
// CHECK:STDOUT:   witness = %Simple.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.Simple.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %Simple.ref: type = name_ref Simple, file.%Simple.decl [concrete = constants.%Simple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20: <witness> = impl_self_witness @C.as.Simple.impl.%Self.ref, @Simple [symbolic_self = constants.%.dd3]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Simple = <poisoned>
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Simple.WithSelf.F(@Simple.%Self: %Simple.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.Simple.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc23_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_26.2: init %C to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_7: init %C = converted %.loc23_26.1, %.loc23_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc23_7
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_7.2(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf(constants.%Self.f79) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self.f79
// CHECK:STDOUT:   %Simple.WithSelf.F.type => constants.%Simple.WithSelf.F.type.a46
// CHECK:STDOUT:   %Simple.WithSelf.F => constants.%Simple.WithSelf.F.0d0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf.F(constants.%Self.f79) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf(constants.%Simple.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Simple.facet
// CHECK:STDOUT:   %Simple.WithSelf.F.type => constants.%Simple.WithSelf.F.type.658
// CHECK:STDOUT:   %Simple.WithSelf.F => constants.%Simple.WithSelf.F.67c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Simple.WithSelf.F(constants.%Simple.facet) {}
// CHECK:STDOUT:
