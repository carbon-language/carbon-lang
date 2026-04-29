// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-raw-sem-ir --dump-sem-ir-ranges=if-present
//
// Check that we can combine textual IR and raw IR dumping in one compile.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/one_file_with_textual_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/one_file_with_textual_ir.carbon

fn Foo(n: ()) -> ((), ()) {
  return (n, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        one_file_with_textual_ir.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   names:
// CHECK:STDOUT:     name0:           Foo
// CHECK:STDOUT:     name1:           n
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst5000002A}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name50000000: {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function50000000: {name: name0, parent_scope: name_scope0, call_param_patterns_id: inst_block5000000B, call_params_id: inst_block5000000C, return_type_inst_id: inst50000020, return_form_inst_id: inst50000021, return_pattern_id: inst50000025, body: [inst_block5000000F]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:      {}
// CHECK:STDOUT:   associated_constants: {}
// CHECK:STDOUT:   impls:           {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   specific_interfaces: {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields_empty: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(FormType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(FormType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(InstType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(InstType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst50000010)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst50000010)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000001D)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst5000001D)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst5000001A)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst5000001D)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst5000002B)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst50000010)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:   facet_types:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instF:           {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst50000010:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst50000011:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000012:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000013:    {kind: Converted, arg0: inst50000011, arg1: inst50000010, type: type(TypeType)}
// CHECK:STDOUT:     inst50000014:    {kind: PatternType, arg0: inst50000010, type: type(TypeType)}
// CHECK:STDOUT:     inst50000015:    {kind: ValueParamPattern, arg0: name1, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000016:    {kind: ValueBinding, arg0: entity_name50000000, arg1: inst50000026, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000017:    {kind: WrapperBindingPattern, arg0: entity_name50000000, arg1: inst50000015, type: type(inst50000014)}
// CHECK:STDOUT:     inst50000018:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000019:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst50000010)}
// CHECK:STDOUT:     inst5000001A:    {kind: TupleType, arg0: inst_block50000008, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001B:    {kind: TupleLiteral, arg0: inst_block50000007, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000001C:    {kind: TupleValue, arg0: inst_block50000009, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000001D:    {kind: PointerType, arg0: inst5000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001E:    {kind: Converted, arg0: inst50000012, arg1: inst50000010, type: type(TypeType)}
// CHECK:STDOUT:     inst5000001F:    {kind: Converted, arg0: inst50000012, arg1: inst50000010, type: type(TypeType)}
// CHECK:STDOUT:     inst50000020:    {kind: Converted, arg0: inst5000001B, arg1: inst5000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst50000021:    {kind: InitForm, arg0: inst50000020, type: type(inst(FormType))}
// CHECK:STDOUT:     inst50000022:    {kind: InitForm, arg0: inst5000001A, type: type(inst(FormType))}
// CHECK:STDOUT:     inst50000023:    {kind: PatternType, arg0: inst5000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst50000024:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(inst50000023)}
// CHECK:STDOUT:     inst50000025:    {kind: ReturnSlotPattern, arg0: inst50000024, arg1: inst50000020, type: type(inst50000023)}
// CHECK:STDOUT:     inst50000026:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000027:    {kind: SpliceBlock, arg0: inst_block50000005, arg1: inst50000013, type: type(TypeType)}
// CHECK:STDOUT:     inst50000028:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000029:    {kind: ReturnSlot, arg0: inst5000001A, arg1: inst50000028, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000002A:    {kind: FunctionDecl, arg0: function50000000, arg1: inst_block5000000E, type: type(inst5000002B)}
// CHECK:STDOUT:     inst5000002B:    {kind: FunctionType, arg0: function50000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst5000002C:    {kind: StructValue, arg0: inst_block_empty, type: type(inst5000002B)}
// CHECK:STDOUT:     inst5000002D:    {kind: NameRef, arg0: name1, arg1: inst50000016, type: type(inst50000010)}
// CHECK:STDOUT:     inst5000002E:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst50000010)}
// CHECK:STDOUT:     inst5000002F:    {kind: TupleLiteral, arg0: inst_block50000010, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000030:    {kind: TupleAccess, arg0: inst50000028, arg1: element0, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000031:    {kind: TupleInit, arg0: inst_block50000011, arg1: inst50000030, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000032:    {kind: Converted, arg0: inst5000002D, arg1: inst50000031, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000033:    {kind: InPlaceInit, arg0: inst50000032, arg1: inst50000030, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000034:    {kind: TupleAccess, arg0: inst50000028, arg1: element1, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000035:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst50000034, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000036:    {kind: Converted, arg0: inst5000002E, arg1: inst50000035, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000037:    {kind: InPlaceInit, arg0: inst50000036, arg1: inst50000034, type: type(inst50000010)}
// CHECK:STDOUT:     inst50000038:    {kind: TupleInit, arg0: inst_block50000012, arg1: inst50000028, type: type(inst5000001A)}
// CHECK:STDOUT:     inst50000039:    {kind: Converted, arg0: inst5000002F, arg1: inst50000038, type: type(inst5000001A)}
// CHECK:STDOUT:     inst5000003A:    {kind: ReturnExpr, arg0: inst50000039, arg1: inst50000028}
// CHECK:STDOUT:   bundles:         {}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       instF:           concrete_constant(instF)
// CHECK:STDOUT:       inst50000010:    concrete_constant(inst50000010)
// CHECK:STDOUT:       inst50000011:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000012:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000013:    concrete_constant(inst50000010)
// CHECK:STDOUT:       inst50000014:    concrete_constant(inst50000014)
// CHECK:STDOUT:       inst50000015:    concrete_constant(inst50000015)
// CHECK:STDOUT:       inst50000017:    concrete_constant(inst50000017)
// CHECK:STDOUT:       inst50000018:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000019:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst5000001A:    concrete_constant(inst5000001A)
// CHECK:STDOUT:       inst5000001B:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst5000001C:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst5000001D:    concrete_constant(inst5000001D)
// CHECK:STDOUT:       inst5000001E:    concrete_constant(inst50000010)
// CHECK:STDOUT:       inst5000001F:    concrete_constant(inst50000010)
// CHECK:STDOUT:       inst50000020:    concrete_constant(inst5000001A)
// CHECK:STDOUT:       inst50000021:    concrete_constant(inst50000022)
// CHECK:STDOUT:       inst50000022:    concrete_constant(inst50000022)
// CHECK:STDOUT:       inst50000023:    concrete_constant(inst50000023)
// CHECK:STDOUT:       inst50000024:    concrete_constant(inst50000024)
// CHECK:STDOUT:       inst50000025:    concrete_constant(inst50000025)
// CHECK:STDOUT:       inst50000027:    concrete_constant(inst50000010)
// CHECK:STDOUT:       inst5000002A:    concrete_constant(inst5000002C)
// CHECK:STDOUT:       inst5000002B:    concrete_constant(inst5000002B)
// CHECK:STDOUT:       inst5000002C:    concrete_constant(inst5000002C)
// CHECK:STDOUT:       inst5000002E:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000031:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000032:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000033:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000035:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000036:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000037:    concrete_constant(inst50000012)
// CHECK:STDOUT:       inst50000038:    concrete_constant(inst5000001C)
// CHECK:STDOUT:       inst50000039:    concrete_constant(inst5000001C)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst5000002A
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block50000005:
// CHECK:STDOUT:       0:               inst50000011
// CHECK:STDOUT:       1:               inst50000013
// CHECK:STDOUT:     inst_block50000006:
// CHECK:STDOUT:       0:               inst50000017
// CHECK:STDOUT:     inst_block50000007:
// CHECK:STDOUT:       0:               inst50000018
// CHECK:STDOUT:       1:               inst50000019
// CHECK:STDOUT:     inst_block50000008:
// CHECK:STDOUT:       0:               inst50000010
// CHECK:STDOUT:       1:               inst50000010
// CHECK:STDOUT:     inst_block50000009:
// CHECK:STDOUT:       0:               inst50000012
// CHECK:STDOUT:       1:               inst50000012
// CHECK:STDOUT:     inst_block5000000A:
// CHECK:STDOUT:       0:               inst5000001E
// CHECK:STDOUT:       1:               inst5000001F
// CHECK:STDOUT:     inst_block5000000B:
// CHECK:STDOUT:       0:               inst50000015
// CHECK:STDOUT:       1:               inst50000024
// CHECK:STDOUT:     inst_block5000000C:
// CHECK:STDOUT:       0:               inst50000026
// CHECK:STDOUT:       1:               inst50000028
// CHECK:STDOUT:     inst_block5000000D:
// CHECK:STDOUT:       0:               inst50000015
// CHECK:STDOUT:       1:               inst50000017
// CHECK:STDOUT:       2:               inst50000024
// CHECK:STDOUT:       3:               inst50000025
// CHECK:STDOUT:     inst_block5000000E:
// CHECK:STDOUT:       0:               inst50000018
// CHECK:STDOUT:       1:               inst50000019
// CHECK:STDOUT:       2:               inst5000001B
// CHECK:STDOUT:       3:               inst5000001E
// CHECK:STDOUT:       4:               inst5000001F
// CHECK:STDOUT:       5:               inst50000020
// CHECK:STDOUT:       6:               inst50000021
// CHECK:STDOUT:       7:               inst50000026
// CHECK:STDOUT:       8:               inst50000027
// CHECK:STDOUT:       9:               inst50000016
// CHECK:STDOUT:       10:              inst50000028
// CHECK:STDOUT:       11:              inst50000029
// CHECK:STDOUT:     inst_block5000000F:
// CHECK:STDOUT:       0:               inst5000002D
// CHECK:STDOUT:       1:               inst5000002E
// CHECK:STDOUT:       2:               inst5000002F
// CHECK:STDOUT:       3:               inst50000030
// CHECK:STDOUT:       4:               inst50000031
// CHECK:STDOUT:       5:               inst50000032
// CHECK:STDOUT:       6:               inst50000033
// CHECK:STDOUT:       7:               inst50000034
// CHECK:STDOUT:       8:               inst50000035
// CHECK:STDOUT:       9:               inst50000036
// CHECK:STDOUT:       10:              inst50000037
// CHECK:STDOUT:       11:              inst50000038
// CHECK:STDOUT:       12:              inst50000039
// CHECK:STDOUT:       13:              inst5000003A
// CHECK:STDOUT:     inst_block50000010:
// CHECK:STDOUT:       0:               inst5000002D
// CHECK:STDOUT:       1:               inst5000002E
// CHECK:STDOUT:     inst_block50000011: {}
// CHECK:STDOUT:     inst_block50000012:
// CHECK:STDOUT:       0:               inst50000033
// CHECK:STDOUT:       1:               inst50000037
// CHECK:STDOUT:     inst_block50000013:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst5000002A
// CHECK:STDOUT:   value_stores:
// CHECK:STDOUT:     shared_values:
// CHECK:STDOUT:       ints:            {}
// CHECK:STDOUT:       reals:           {}
// CHECK:STDOUT:       floats:          {}
// CHECK:STDOUT:       identifiers:
// CHECK:STDOUT:         identifier0:     Foo
// CHECK:STDOUT:         identifier1:     n
// CHECK:STDOUT:       strings:         {}
// CHECK:STDOUT: ...
// CHECK:STDOUT: --- one_file_with_textual_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %.e3a: Core.Form = init_form %tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.cb1 = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5b8 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5b8 = return_slot_pattern %return.param_patt, %.loc16_25.4 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_20: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_24: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_25.1: %tuple.type = tuple_literal (%.loc16_20, %.loc16_24) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc16_25.2: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_25.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_25.4: type = converted %.loc16_25.1, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     %.loc16_25.5: Core.Form = init_form %.loc16_25.4 [concrete = constants.%.e3a]
// CHECK:STDOUT:     %n.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc16_12.1: type = splice_block %.loc16_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc16_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc16_12.3: type = converted %.loc16_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %empty_tuple.type = value_binding n, %n.param
// CHECK:STDOUT:     %return.param: ref %tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n.param: %empty_tuple.type) -> out %return.param: %tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   %.loc17_15.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.1: %tuple.type = tuple_literal (%n.ref, %.loc17_15.1)
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc17_11: init %empty_tuple.type to %tuple.elem0 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.2: init %empty_tuple.type = converted %n.ref, %.loc17_11 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.3: init %empty_tuple.type to %tuple.elem0 = in_place_init %.loc17_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc17_15.2: init %empty_tuple.type to %tuple.elem1 = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.4: init %empty_tuple.type = converted %.loc17_15.1, %.loc17_15.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.5: init %empty_tuple.type to %tuple.elem1 = in_place_init %.loc17_16.4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.6: init %tuple.type to %return.param = tuple_init (%.loc17_16.3, %.loc17_16.5) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc17_17: init %tuple.type = converted %.loc17_16.1, %.loc17_16.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   return %.loc17_17 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
