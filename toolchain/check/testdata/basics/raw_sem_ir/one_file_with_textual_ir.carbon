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
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst6000002A}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name60000000: {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function60000000: {name: name0, parent_scope: name_scope0, call_param_patterns_id: inst_block6000000B, call_params_id: inst_block6000000C, return_type_inst_id: inst60000020, return_form_inst_id: inst60000021, return_patterns_id: inst_block6000000A, body: [inst_block6000000F]}
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
// CHECK:STDOUT:     'type(inst(FormType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(FormType))}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst60000010)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000010)}
// CHECK:STDOUT:     'type(inst6000001D)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst6000001D)}
// CHECK:STDOUT:     'type(inst6000001A)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst6000001D)}
// CHECK:STDOUT:     'type(inst6000002B)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000010)}
// CHECK:STDOUT:   facet_types:     {}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instF:           {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000010:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000011:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000012:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000013:    {kind: Converted, arg0: inst60000011, arg1: inst60000010, type: type(TypeType)}
// CHECK:STDOUT:     inst60000014:    {kind: ValueBinding, arg0: entity_name60000000, arg1: inst60000026, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000015:    {kind: PatternType, arg0: inst60000010, type: type(TypeType)}
// CHECK:STDOUT:     inst60000016:    {kind: ValueBindingPattern, arg0: entity_name60000000, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000017:    {kind: ValueParamPattern, arg0: inst60000016, arg1: call_param0, type: type(inst60000015)}
// CHECK:STDOUT:     inst60000018:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000019:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000010)}
// CHECK:STDOUT:     inst6000001A:    {kind: TupleType, arg0: inst_block60000007, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001B:    {kind: TupleLiteral, arg0: inst_block60000006, type: type(inst6000001A)}
// CHECK:STDOUT:     inst6000001C:    {kind: TupleValue, arg0: inst_block60000008, type: type(inst6000001A)}
// CHECK:STDOUT:     inst6000001D:    {kind: PointerType, arg0: inst6000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001E:    {kind: Converted, arg0: inst60000012, arg1: inst60000010, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001F:    {kind: Converted, arg0: inst60000012, arg1: inst60000010, type: type(TypeType)}
// CHECK:STDOUT:     inst60000020:    {kind: Converted, arg0: inst6000001B, arg1: inst6000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000021:    {kind: InitForm, arg0: inst60000020, arg1: call_param1, type: type(inst(FormType))}
// CHECK:STDOUT:     inst60000022:    {kind: InitForm, arg0: inst6000001A, arg1: call_param1, type: type(inst(FormType))}
// CHECK:STDOUT:     inst60000023:    {kind: PatternType, arg0: inst6000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000024:    {kind: ReturnSlotPattern, arg0: inst60000020, type: type(inst60000023)}
// CHECK:STDOUT:     inst60000025:    {kind: OutParamPattern, arg0: inst60000024, arg1: call_param1, type: type(inst60000023)}
// CHECK:STDOUT:     inst60000026:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000027:    {kind: SpliceBlock, arg0: inst_block60000004, arg1: inst60000013, type: type(TypeType)}
// CHECK:STDOUT:     inst60000028:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(inst6000001A)}
// CHECK:STDOUT:     inst60000029:    {kind: ReturnSlot, arg0: inst6000001A, arg1: inst60000028, type: type(inst6000001A)}
// CHECK:STDOUT:     inst6000002A:    {kind: FunctionDecl, arg0: function60000000, arg1: inst_block6000000E, type: type(inst6000002B)}
// CHECK:STDOUT:     inst6000002B:    {kind: FunctionType, arg0: function60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002C:    {kind: StructValue, arg0: inst_block_empty, type: type(inst6000002B)}
// CHECK:STDOUT:     inst6000002D:    {kind: NameRef, arg0: name1, arg1: inst60000014, type: type(inst60000010)}
// CHECK:STDOUT:     inst6000002E:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000010)}
// CHECK:STDOUT:     inst6000002F:    {kind: TupleLiteral, arg0: inst_block60000010, type: type(inst6000001A)}
// CHECK:STDOUT:     inst60000030:    {kind: TupleAccess, arg0: inst60000028, arg1: element0, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000031:    {kind: TupleInit, arg0: inst_block60000011, arg1: inst<none>, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000032:    {kind: Converted, arg0: inst6000002D, arg1: inst60000031, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000033:    {kind: TupleAccess, arg0: inst60000028, arg1: element1, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000034:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst<none>, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000035:    {kind: Converted, arg0: inst6000002E, arg1: inst60000034, type: type(inst60000010)}
// CHECK:STDOUT:     inst60000036:    {kind: TupleInit, arg0: inst_block60000012, arg1: inst60000028, type: type(inst6000001A)}
// CHECK:STDOUT:     inst60000037:    {kind: Converted, arg0: inst6000002F, arg1: inst60000036, type: type(inst6000001A)}
// CHECK:STDOUT:     inst60000038:    {kind: ReturnExpr, arg0: inst60000037, arg1: inst60000028}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       instF:           concrete_constant(instF)
// CHECK:STDOUT:       inst60000010:    concrete_constant(inst60000010)
// CHECK:STDOUT:       inst60000011:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000012:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000013:    concrete_constant(inst60000010)
// CHECK:STDOUT:       inst60000015:    concrete_constant(inst60000015)
// CHECK:STDOUT:       inst60000016:    concrete_constant(inst60000016)
// CHECK:STDOUT:       inst60000017:    concrete_constant(inst60000017)
// CHECK:STDOUT:       inst60000018:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000019:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst6000001A:    concrete_constant(inst6000001A)
// CHECK:STDOUT:       inst6000001B:    concrete_constant(inst6000001C)
// CHECK:STDOUT:       inst6000001C:    concrete_constant(inst6000001C)
// CHECK:STDOUT:       inst6000001D:    concrete_constant(inst6000001D)
// CHECK:STDOUT:       inst6000001E:    concrete_constant(inst60000010)
// CHECK:STDOUT:       inst6000001F:    concrete_constant(inst60000010)
// CHECK:STDOUT:       inst60000020:    concrete_constant(inst6000001A)
// CHECK:STDOUT:       inst60000021:    concrete_constant(inst60000022)
// CHECK:STDOUT:       inst60000022:    concrete_constant(inst60000022)
// CHECK:STDOUT:       inst60000023:    concrete_constant(inst60000023)
// CHECK:STDOUT:       inst60000024:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst60000025:    concrete_constant(inst60000025)
// CHECK:STDOUT:       inst60000027:    concrete_constant(inst60000010)
// CHECK:STDOUT:       inst6000002A:    concrete_constant(inst6000002C)
// CHECK:STDOUT:       inst6000002B:    concrete_constant(inst6000002B)
// CHECK:STDOUT:       inst6000002C:    concrete_constant(inst6000002C)
// CHECK:STDOUT:       inst6000002E:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000031:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000032:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000034:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000035:    concrete_constant(inst60000012)
// CHECK:STDOUT:       inst60000036:    concrete_constant(inst6000001C)
// CHECK:STDOUT:       inst60000037:    concrete_constant(inst6000001C)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst6000002A
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block60000004:
// CHECK:STDOUT:       0:               inst60000011
// CHECK:STDOUT:       1:               inst60000013
// CHECK:STDOUT:     inst_block60000005:
// CHECK:STDOUT:       0:               inst60000017
// CHECK:STDOUT:     inst_block60000006:
// CHECK:STDOUT:       0:               inst60000018
// CHECK:STDOUT:       1:               inst60000019
// CHECK:STDOUT:     inst_block60000007:
// CHECK:STDOUT:       0:               inst60000010
// CHECK:STDOUT:       1:               inst60000010
// CHECK:STDOUT:     inst_block60000008:
// CHECK:STDOUT:       0:               inst60000012
// CHECK:STDOUT:       1:               inst60000012
// CHECK:STDOUT:     inst_block60000009:
// CHECK:STDOUT:       0:               inst6000001E
// CHECK:STDOUT:       1:               inst6000001F
// CHECK:STDOUT:     inst_block6000000A:
// CHECK:STDOUT:       0:               inst60000025
// CHECK:STDOUT:     inst_block6000000B:
// CHECK:STDOUT:       0:               inst60000017
// CHECK:STDOUT:       1:               inst60000025
// CHECK:STDOUT:     inst_block6000000C:
// CHECK:STDOUT:       0:               inst60000026
// CHECK:STDOUT:       1:               inst60000028
// CHECK:STDOUT:     inst_block6000000D:
// CHECK:STDOUT:       0:               inst60000016
// CHECK:STDOUT:       1:               inst60000017
// CHECK:STDOUT:       2:               inst60000024
// CHECK:STDOUT:       3:               inst60000025
// CHECK:STDOUT:     inst_block6000000E:
// CHECK:STDOUT:       0:               inst60000018
// CHECK:STDOUT:       1:               inst60000019
// CHECK:STDOUT:       2:               inst6000001B
// CHECK:STDOUT:       3:               inst6000001E
// CHECK:STDOUT:       4:               inst6000001F
// CHECK:STDOUT:       5:               inst60000020
// CHECK:STDOUT:       6:               inst60000021
// CHECK:STDOUT:       7:               inst60000026
// CHECK:STDOUT:       8:               inst60000027
// CHECK:STDOUT:       9:               inst60000014
// CHECK:STDOUT:       10:              inst60000028
// CHECK:STDOUT:       11:              inst60000029
// CHECK:STDOUT:     inst_block6000000F:
// CHECK:STDOUT:       0:               inst6000002D
// CHECK:STDOUT:       1:               inst6000002E
// CHECK:STDOUT:       2:               inst6000002F
// CHECK:STDOUT:       3:               inst60000030
// CHECK:STDOUT:       4:               inst60000031
// CHECK:STDOUT:       5:               inst60000032
// CHECK:STDOUT:       6:               inst60000033
// CHECK:STDOUT:       7:               inst60000034
// CHECK:STDOUT:       8:               inst60000035
// CHECK:STDOUT:       9:               inst60000036
// CHECK:STDOUT:       10:              inst60000037
// CHECK:STDOUT:       11:              inst60000038
// CHECK:STDOUT:     inst_block60000010:
// CHECK:STDOUT:       0:               inst6000002D
// CHECK:STDOUT:       1:               inst6000002E
// CHECK:STDOUT:     inst_block60000011: {}
// CHECK:STDOUT:     inst_block60000012:
// CHECK:STDOUT:       0:               inst60000032
// CHECK:STDOUT:       1:               inst60000035
// CHECK:STDOUT:     inst_block60000013:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst6000002A
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
// CHECK:STDOUT:   %.93a: Core.Form = init_form %tuple.type, call_param1 [concrete]
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
// CHECK:STDOUT:     %n.patt: %pattern_type.cb1 = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.cb1 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5b8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5b8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_20: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_24: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16_25.1: %tuple.type = tuple_literal (%.loc16_20, %.loc16_24) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc16_25.2: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_25.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_25.4: type = converted %.loc16_25.1, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     %.loc16_25.5: Core.Form = init_form %.loc16_25.4, call_param1 [concrete = constants.%.93a]
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
// CHECK:STDOUT:   %.loc17_11: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.2: init %empty_tuple.type = converted %n.ref, %.loc17_11 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc17_15.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.3: init %empty_tuple.type = converted %.loc17_15.1, %.loc17_15.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.4: init %tuple.type to %return.param = tuple_init (%.loc17_16.2, %.loc17_16.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc17_17: init %tuple.type = converted %.loc17_16.1, %.loc17_16.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   return %.loc17_17 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
