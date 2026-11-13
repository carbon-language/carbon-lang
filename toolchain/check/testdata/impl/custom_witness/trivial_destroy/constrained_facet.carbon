// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/constrained_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/trivial_destroy/constrained_facet.carbon

// --- todo_constrained_facet.carbon
library "[[@TEST_NAME]]";

fn Make(generic T: Core.TrivialDestroy) -> typeof(T) {
  // CHECK:STDERR:        ╭─┤ todo_constrained_facet.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   returned var x: typeof(T);
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: value of type `<associated entity in Core.DefaultOrUnformed>` is
  // CHECK:STDERR:        ·      ┃        not callable
  // CHECK:STDERR:        ·      ┃        [CallToNonCallable]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  returned var x: typeof(T);
  return var;
}

fn Test(generic T: Core.TrivialDestroy) {
  typeof(T) as Core.SubobjectDestroy;
  typeof(T) as Core.Destroy;
  typeof(T) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  var unused x: typeof(T) = Make(T);
  //@dump-sem-ir-end
}

// --- todo_generic_class.carbon
library "[[@TEST_NAME]]";

class GenericClass(T: Core.TrivialDestroy) {
  var x: T;
}

fn Make(generic T: Core.TrivialDestroy) -> GenericClass(T);

fn Test(generic T: Core.TrivialDestroy) {
  typeof(GenericClass(T)) as Core.SubobjectDestroy;
  typeof(GenericClass(T)) as Core.Destroy;
  typeof(GenericClass(T)) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ todo_generic_class.carbon:[[@LINE+24]]:3
  // CHECK:STDERR: ->     │   var unused x: typeof(GenericClass(T)) = Make(T);
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `GenericClass(T)`
  // CHECK:STDERR:        ·   │  ┃        to `type`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `GenericClass(T)` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.ImplicitAs(type)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ todo_generic_class.carbon:[[@LINE+13]]:43
  // CHECK:STDERR:        │ fn Make(generic T: Core.TrivialDestroy) -> GenericClass(T);
  // CHECK:STDERR:        · ─────────────────────────────┬─────────────────────────────
  // CHECK:STDERR:        ·                              ╰──┤ generic function declared here
  // CHECK:STDERR:        ·                                 │ [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   var unused x: typeof(GenericClass(T)) = Make(T);
  // CHECK:STDERR:        ·                                           ┳━━━
  // CHECK:STDERR:        ·                                           ┗━━┫ error: use of undefined generic
  // CHECK:STDERR:        ·                                              ┃        function
  // CHECK:STDERR:        ·                                              ┃        [MissingGenericFunctionDefinition]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused x: typeof(GenericClass(T)) = Make(T);
  //@dump-sem-ir-end
}

// --- todo_struct.carbon
library "[[@TEST_NAME]]";

fn Make(generic T: Core.TrivialDestroy) -> {.a: T};

fn Test(generic T: Core.TrivialDestroy) {
  alias Struct = {.a: T};

  typeof(Struct) as Core.SubobjectDestroy;
  typeof(Struct) as Core.Destroy;
  typeof(Struct) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ todo_struct.carbon:[[@LINE+22]]:3
  // CHECK:STDERR: ->     │   var unused x: typeof(Struct) = Make(T);
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert non-type value of type `{.a: T}` to
  // CHECK:STDERR:        ·   │  ┃        `type`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:        ·   ╰────┤ type `{.a: T}` does not implement interface `Core.ImplicitAs(type)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ todo_struct.carbon:[[@LINE+12]]:34
  // CHECK:STDERR:        │ fn Make(generic T: Core.TrivialDestroy) -> {.a: T};
  // CHECK:STDERR:        · ─────────────────────────┬─────────────────────────
  // CHECK:STDERR:        ·                          ╰──┤ generic function declared here
  // CHECK:STDERR:        ·                             │ [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   var unused x: typeof(Struct) = Make(T);
  // CHECK:STDERR:        ·                                  ━━┳━
  // CHECK:STDERR:        ·                                    ┗━━┫ error: use of undefined generic function
  // CHECK:STDERR:        ·                                       ┃        [MissingGenericFunctionDefinition]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused x: typeof(Struct) = Make(T);
  //@dump-sem-ir-end
}

// --- todo_tuple.carbon
library "[[@TEST_NAME]]";

class HasTrivialDestruction {}

// CHECK:STDERR:        ╭─┤ todo_tuple.carbon:[[@LINE+7]]:48
// CHECK:STDERR: ->     │ fn Make(generic T: Core.TrivialDestroy) -> (T, HasNonTrivialDestruction);
// CHECK:STDERR:        ·                                                ━━━━━━━━━━━━┳━━━━━━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: name `HasNonTrivialDestruction` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Make(generic T: Core.TrivialDestroy) -> (T, HasNonTrivialDestruction);

fn Test(generic T: Core.TrivialDestroy) {
  alias Tuple = (T, HasTrivialDestruction);

  typeof(Tuple) as Core.SubobjectDestroy;
  typeof(Tuple) as Core.Destroy;
  typeof(Tuple) as Core.TrivialDestroy;

  //@dump-sem-ir-begin
  // CHECK:STDERR:        ╭─┤ todo_tuple.carbon:[[@LINE+12]]:33
  // CHECK:STDERR:        │ fn Make(generic T: Core.TrivialDestroy) -> (T, HasNonTrivialDestruction);
  // CHECK:STDERR:        · ────────────────────────────────────┬────────────────────────────────────
  // CHECK:STDERR:        ·                                     ╰──┤ generic function declared here
  // CHECK:STDERR:        ·                                        │ [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   var unused x: typeof(Tuple) = Make(T);
  // CHECK:STDERR:        ·                                 ━━┳━
  // CHECK:STDERR:        ·                                   ┗━━┫ error: use of undefined generic function
  // CHECK:STDERR:        ·                                      ┃        [MissingGenericFunctionDefinition]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused x: typeof(Tuple) = Make(T);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- todo_constrained_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9a587c.1 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9a587c.1 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc12 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value %Destroy.type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Destroy.type, (%DestroyImpl.impl_witness.a54, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%T.f45) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc16_18.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Make.specific_fn.loc22_29.2: <specific function> = specific_function constants.%Make, @Make(%T.loc16_18.1) [symbolic = %Make.specific_fn.loc22_29.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %x.var: ref %Destroy.type = var_storage %x.var_patt
// CHECK:STDOUT:     %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %T.ref.loc22_34: %Destroy.type = name_ref T, %T.loc16_18.2 [symbolic = %T.loc16_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %Make.specific_fn.loc22_29.1: <specific function> = specific_function %Make.ref, @Make(constants.%T.f45) [symbolic = %Make.specific_fn.loc22_29.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     %Make.call: init %Destroy.type = call %Make.specific_fn.loc22_29.1()
// CHECK:STDOUT:     assign %x.var, %Make.call
// CHECK:STDOUT:     %.loc22_25: type = type_of %T.ref.loc22_24 in [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %T.ref.loc22_24: %Destroy.type = name_ref T, %T.loc16_18.2 [symbolic = %T.loc16_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %Destroy.type = wrapper_binding x, %x.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.9a587c.1 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.9a587c.1 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.facet.loc22_3.1: %Destroy.type = facet_value constants.%Destroy.type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %.loc22_3.1: %Destroy.type = converted constants.%Destroy.type, %Destroy.facet.loc22_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %Destroy.facet.loc22_3.2: %Destroy.type = facet_value constants.%Destroy.type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %.loc22_3.2: %Destroy.type = converted constants.%Destroy.type, %Destroy.facet.loc22_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc16_18.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc16_18.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %GenericClass.type: type = parameterized_class_name_type @GenericClass [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GenericClass.generic: %GenericClass.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericClass: type = class_type @GenericClass, @GenericClass(%T.f45) [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type type [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value type, (%DestroyImpl.impl_witness.a54, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9a587c.2 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9a587c.2 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%T.f45) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc9_18.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.specific_fn.loc39_43.2: <specific function> = specific_function constants.%Make, @Make(%T.loc9_18.1) [symbolic = %Make.specific_fn.loc39_43.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %x.var: ref type = var_storage %x.var_patt
// CHECK:STDOUT:     %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %T.ref.loc39_48: %Destroy.type = name_ref T, %T.loc9_18.2 [symbolic = %T.loc9_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %Make.specific_fn.loc39_43.1: <specific function> = specific_function %Make.ref, @Make(constants.%T.f45) [symbolic = %Make.specific_fn.loc39_43.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     %.loc39_49: ref @Test.%GenericClass.loc10_24.2 (%GenericClass) = temporary_storage
// CHECK:STDOUT:     %Make.call: init @Test.%GenericClass.loc10_24.2 (%GenericClass) to %.loc39_49 = call %Make.specific_fn.loc39_43.1()
// CHECK:STDOUT:     %.loc39_3.1: type = converted %Make.call, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %x.var, <error>
// CHECK:STDOUT:     %.loc39_39: type = type_of %GenericClass.loc39 in [concrete = type] {
// CHECK:STDOUT:       %GenericClass.ref.loc39: %GenericClass.type = name_ref GenericClass, file.%GenericClass.decl [concrete = constants.%GenericClass.generic]
// CHECK:STDOUT:       %T.ref.loc39_37: %Destroy.type = name_ref T, %T.loc9_18.2 [symbolic = %T.loc9_18.1 (constants.%T.f45)]
// CHECK:STDOUT:       %GenericClass.loc39: type = class_type @GenericClass, @GenericClass(constants.%T.f45) [symbolic = %GenericClass.loc10_24.2 (constants.%GenericClass)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref type = wrapper_binding x, %x.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.9a587c.2 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.9a587c.2 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.facet.loc39_3.1: %Destroy.type = facet_value type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %.loc39_3.2: %Destroy.type = converted type, %Destroy.facet.loc39_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %Destroy.facet.loc39_3.2: %Destroy.type = facet_value type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %.loc39_3.3: %Destroy.type = converted type, %Destroy.facet.loc39_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc9_18.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc9_18.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.190: type = facet_access_type %T.f45 [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type.190} [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type type [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1f5: %SubobjectDestroy.type = facet_value type, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.a54: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1f5) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value type, (%DestroyImpl.impl_witness.a54, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.9a587c.2 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.9a587c.2 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%T.f45) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc5_18.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.specific_fn.loc35_34.2: <specific function> = specific_function constants.%Make, @Make(%T.loc5_18.1) [symbolic = %Make.specific_fn.loc35_34.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %x.var: ref type = var_storage %x.var_patt
// CHECK:STDOUT:     %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %T.ref.loc35: %Destroy.type = name_ref T, %T.loc5_18.2 [symbolic = %T.loc5_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %Make.specific_fn.loc35_34.1: <specific function> = specific_function %Make.ref, @Make(constants.%T.f45) [symbolic = %Make.specific_fn.loc35_34.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     %Make.call: init @Test.%struct_type.a.loc6_9 (%struct_type.a) = call %Make.specific_fn.loc35_34.1()
// CHECK:STDOUT:     %.loc35_3.1: type = converted %Make.call, <error> [concrete = <error>]
// CHECK:STDOUT:     assign %x.var, <error>
// CHECK:STDOUT:     %.loc35_30: type = type_of %Struct.ref.loc35 in [concrete = type] {
// CHECK:STDOUT:       %Struct.ref.loc35: type = name_ref Struct, %Struct [symbolic = %struct_type.a.loc6_9 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref type = wrapper_binding x, %x.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.9a587c.2 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.9a587c.2 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.facet.loc35_3.1: %Destroy.type = facet_value type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %.loc35_3.2: %Destroy.type = converted type, %Destroy.facet.loc35_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %Destroy.facet.loc35_3.2: %Destroy.type = facet_value type, (constants.%DestroyImpl.impl_witness.a54, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     %.loc35_3.3: %Destroy.type = converted type, %Destroy.facet.loc35_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc5_18.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc5_18.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %HasTrivialDestruction: type = class_type @HasTrivialDestruction [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f45: %Destroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%Destroy.type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%T.f45, %HasTrivialDestruction) [symbolic]
// CHECK:STDOUT:   %pattern_type.179: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc17_17.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.6f1: %SubobjectDestroy.type = facet_value %tuple.type, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.43c: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.6f1) [concrete]
// CHECK:STDOUT:   %Destroy.facet.42d: %Destroy.type = facet_value %tuple.type, (%DestroyImpl.impl_witness.43c, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.179 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.179 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%T.f45) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%T.loc14_18.2: %Destroy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Make.specific_fn.loc34_33.2: <specific function> = specific_function constants.%Make, @Make(%T.loc14_18.1) [symbolic = %Make.specific_fn.loc34_33.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %x.var: ref %tuple.type = var_storage %x.var_patt
// CHECK:STDOUT:     %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %T.ref.loc34: %Destroy.type = name_ref T, %T.loc14_18.2 [symbolic = %T.loc14_18.1 (constants.%T.f45)]
// CHECK:STDOUT:     %Make.specific_fn.loc34_33.1: <specific function> = specific_function %Make.ref, @Make(constants.%T.f45) [symbolic = %Make.specific_fn.loc34_33.2 (constants.%Make.specific_fn)]
// CHECK:STDOUT:     %Make.call: <error> = call %Make.specific_fn.loc34_33.1()
// CHECK:STDOUT:     assign %x.var, <error>
// CHECK:STDOUT:     %.loc34_29: type = type_of %Tuple.ref.loc34 in [concrete = constants.%tuple.type] {
// CHECK:STDOUT:       %Tuple.ref.loc34: %tuple.type = name_ref Tuple, %Tuple [symbolic = %tuple.loc15_42.2 (constants.%tuple)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %tuple.type = wrapper_binding x, %x.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.179 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:       %x.var_patt: %pattern_type.179 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.facet.loc34_3.1: %Destroy.type = facet_value constants.%tuple.type, (constants.%DestroyImpl.impl_witness.43c, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.42d]
// CHECK:STDOUT:     %.loc34_3.1: %Destroy.type = converted constants.%tuple.type, %Destroy.facet.loc34_3.1 [concrete = constants.%Destroy.facet.42d]
// CHECK:STDOUT:     %Destroy.facet.loc34_3.2: %Destroy.type = facet_value constants.%tuple.type, (constants.%DestroyImpl.impl_witness.43c, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.42d]
// CHECK:STDOUT:     %.loc34_3.2: %Destroy.type = converted constants.%tuple.type, %Destroy.facet.loc34_3.2 [concrete = constants.%Destroy.facet.42d]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%T.f45) {
// CHECK:STDOUT:   %T.patt.loc14_18.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc14_18.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
