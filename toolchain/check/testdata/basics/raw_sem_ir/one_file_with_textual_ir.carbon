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
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst0000000E, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst60000025}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function60000000: {name: name0, parent_scope: name_scope0, call_params_id: inst_block9, return_slot_pattern: inst60000020, body: [inst_block12]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:        {}
// CHECK:STDOUT:   specifics:       {}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst6000000F)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst6000000F)}
// CHECK:STDOUT:     'type(inst60000018)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst6000001A)}
// CHECK:STDOUT:     'type(inst6000001A)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst6000001A)}
// CHECK:STDOUT:     'type(inst60000026)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst6000000F)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst0000000E:    {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst6000000F:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000010:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000011:    {kind: Converted, arg0: inst60000010, arg1: inst6000000F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000012:    {kind: BindName, arg0: entity_name0, arg1: inst60000021, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000013:    {kind: PatternType, arg0: inst6000000F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000014:    {kind: BindingPattern, arg0: entity_name0, type: type(inst60000013)}
// CHECK:STDOUT:     inst60000015:    {kind: ValueParamPattern, arg0: inst60000014, arg1: call_param0, type: type(inst60000013)}
// CHECK:STDOUT:     inst60000016:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000017:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000018:    {kind: TupleType, arg0: inst_block7, type: type(TypeType)}
// CHECK:STDOUT:     inst60000019:    {kind: TupleLiteral, arg0: inst_block6, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000001A:    {kind: PointerType, arg0: inst60000018, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001B:    {kind: Converted, arg0: inst60000016, arg1: inst6000000F, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001C:    {kind: Converted, arg0: inst60000017, arg1: inst6000000F, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001D:    {kind: Converted, arg0: inst60000019, arg1: inst60000018, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001E:    {kind: PatternType, arg0: inst60000018, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001F:    {kind: ReturnSlotPattern, arg0: inst6000001D, type: type(inst6000001E)}
// CHECK:STDOUT:     inst60000020:    {kind: OutParamPattern, arg0: inst6000001F, arg1: call_param1, type: type(inst6000001E)}
// CHECK:STDOUT:     inst60000021:    {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000022:    {kind: SpliceBlock, arg0: inst_block4, arg1: inst60000011, type: type(TypeType)}
// CHECK:STDOUT:     inst60000023:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(inst60000018)}
// CHECK:STDOUT:     inst60000024:    {kind: ReturnSlot, arg0: inst60000018, arg1: inst60000023, type: type(inst60000018)}
// CHECK:STDOUT:     inst60000025:    {kind: FunctionDecl, arg0: function60000000, arg1: inst_block11, type: type(inst60000026)}
// CHECK:STDOUT:     inst60000026:    {kind: FunctionType, arg0: function60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000027:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000026)}
// CHECK:STDOUT:     inst60000028:    {kind: NameRef, arg0: name1, arg1: inst60000012, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000029:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst6000000F)}
// CHECK:STDOUT:     inst6000002A:    {kind: TupleLiteral, arg0: inst_block13, type: type(inst60000018)}
// CHECK:STDOUT:     inst6000002B:    {kind: TupleAccess, arg0: inst60000024, arg1: element0, type: type(inst6000000F)}
// CHECK:STDOUT:     inst6000002C:    {kind: TupleInit, arg0: inst_block14, arg1: inst6000002B, type: type(inst6000000F)}
// CHECK:STDOUT:     inst6000002D:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst6000000F)}
// CHECK:STDOUT:     inst6000002E:    {kind: Converted, arg0: inst60000028, arg1: inst6000002C, type: type(inst6000000F)}
// CHECK:STDOUT:     inst6000002F:    {kind: TupleAccess, arg0: inst60000024, arg1: element1, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000030:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst6000002F, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000031:    {kind: Converted, arg0: inst60000029, arg1: inst60000030, type: type(inst6000000F)}
// CHECK:STDOUT:     inst60000032:    {kind: TupleInit, arg0: inst_block15, arg1: inst60000024, type: type(inst60000018)}
// CHECK:STDOUT:     inst60000033:    {kind: TupleValue, arg0: inst_block16, type: type(inst60000018)}
// CHECK:STDOUT:     inst60000034:    {kind: Converted, arg0: inst6000002A, arg1: inst60000032, type: type(inst60000018)}
// CHECK:STDOUT:     inst60000035:    {kind: ReturnExpr, arg0: inst60000034, arg1: inst60000024}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst0000000E:    concrete_constant(inst0000000E)
// CHECK:STDOUT:       inst6000000F:    concrete_constant(inst6000000F)
// CHECK:STDOUT:       inst60000011:    concrete_constant(inst6000000F)
// CHECK:STDOUT:       inst60000013:    concrete_constant(inst60000013)
// CHECK:STDOUT:       inst60000014:    concrete_constant(inst60000014)
// CHECK:STDOUT:       inst60000015:    concrete_constant(inst60000015)
// CHECK:STDOUT:       inst60000018:    concrete_constant(inst60000018)
// CHECK:STDOUT:       inst6000001A:    concrete_constant(inst6000001A)
// CHECK:STDOUT:       inst6000001B:    concrete_constant(inst6000000F)
// CHECK:STDOUT:       inst6000001C:    concrete_constant(inst6000000F)
// CHECK:STDOUT:       inst6000001D:    concrete_constant(inst60000018)
// CHECK:STDOUT:       inst6000001E:    concrete_constant(inst6000001E)
// CHECK:STDOUT:       inst6000001F:    concrete_constant(inst6000001F)
// CHECK:STDOUT:       inst60000020:    concrete_constant(inst60000020)
// CHECK:STDOUT:       inst60000022:    concrete_constant(inst6000000F)
// CHECK:STDOUT:       inst60000025:    concrete_constant(inst60000027)
// CHECK:STDOUT:       inst60000026:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000027:    concrete_constant(inst60000027)
// CHECK:STDOUT:       inst6000002C:    concrete_constant(inst6000002D)
// CHECK:STDOUT:       inst6000002D:    concrete_constant(inst6000002D)
// CHECK:STDOUT:       inst6000002E:    concrete_constant(inst6000002D)
// CHECK:STDOUT:       inst60000030:    concrete_constant(inst6000002D)
// CHECK:STDOUT:       inst60000031:    concrete_constant(inst6000002D)
// CHECK:STDOUT:       inst60000032:    concrete_constant(inst60000033)
// CHECK:STDOUT:       inst60000033:    concrete_constant(inst60000033)
// CHECK:STDOUT:       inst60000034:    concrete_constant(inst60000033)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst60000025
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst60000010
// CHECK:STDOUT:       1:               inst60000011
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst60000016
// CHECK:STDOUT:       1:               inst60000017
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst6000000F
// CHECK:STDOUT:       1:               inst6000000F
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst60000021
// CHECK:STDOUT:       1:               inst60000023
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst60000014
// CHECK:STDOUT:       1:               inst60000015
// CHECK:STDOUT:       2:               inst6000001F
// CHECK:STDOUT:       3:               inst60000020
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst60000016
// CHECK:STDOUT:       1:               inst60000017
// CHECK:STDOUT:       2:               inst60000019
// CHECK:STDOUT:       3:               inst6000001B
// CHECK:STDOUT:       4:               inst6000001C
// CHECK:STDOUT:       5:               inst6000001D
// CHECK:STDOUT:       6:               inst60000021
// CHECK:STDOUT:       7:               inst60000022
// CHECK:STDOUT:       8:               inst60000012
// CHECK:STDOUT:       9:               inst60000023
// CHECK:STDOUT:       10:              inst60000024
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst60000028
// CHECK:STDOUT:       1:               inst60000029
// CHECK:STDOUT:       2:               inst6000002A
// CHECK:STDOUT:       3:               inst6000002B
// CHECK:STDOUT:       4:               inst6000002C
// CHECK:STDOUT:       5:               inst6000002E
// CHECK:STDOUT:       6:               inst6000002F
// CHECK:STDOUT:       7:               inst60000030
// CHECK:STDOUT:       8:               inst60000031
// CHECK:STDOUT:       9:               inst60000032
// CHECK:STDOUT:       10:              inst60000034
// CHECK:STDOUT:       11:              inst60000035
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst60000028
// CHECK:STDOUT:       1:               inst60000029
// CHECK:STDOUT:     inst_block14:    {}
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst6000002E
// CHECK:STDOUT:       1:               inst60000031
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst6000002D
// CHECK:STDOUT:       1:               inst6000002D
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst0000000E
// CHECK:STDOUT:       1:               inst60000025
// CHECK:STDOUT: ...
// CHECK:STDOUT: --- one_file_with_textual_ir.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %Foo.type: type = fn_type @Foo [concrete]
// CHECK:STDOUT:   %Foo: %Foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Foo = %Foo.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Foo.decl: %Foo.type = fn_decl @Foo [concrete = constants.%Foo] {
// CHECK:STDOUT:     %n.patt: %pattern_type.cb1 = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.cb1 = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5b8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5b8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_20: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_24: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_25.1: %tuple.type = tuple_literal (%.loc16_20, %.loc16_24)
// CHECK:STDOUT:     %.loc16_25.2: type = converted %.loc16_20, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_25.3: type = converted %.loc16_24, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc16_25.4: type = converted %.loc16_25.1, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:     %n.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc16_12.1: type = splice_block %.loc16_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc16_12.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc16_12.3: type = converted %.loc16_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %empty_tuple.type = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref %tuple.type = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Foo(%n.param: %empty_tuple.type) -> %return.param: %tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %empty_tuple.type = name_ref n, %n
// CHECK:STDOUT:   %.loc17_15.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc17_16.1: %tuple.type = tuple_literal (%n.ref, %.loc17_15.1)
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %return, element0
// CHECK:STDOUT:   %.loc17_11: init %empty_tuple.type = tuple_init () to %tuple.elem0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.2: init %empty_tuple.type = converted %n.ref, %.loc17_11 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %return, element1
// CHECK:STDOUT:   %.loc17_15.2: init %empty_tuple.type = tuple_init () to %tuple.elem1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.3: init %empty_tuple.type = converted %.loc17_15.1, %.loc17_15.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17_16.4: init %tuple.type = tuple_init (%.loc17_16.2, %.loc17_16.3) to %return [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc17_17: init %tuple.type = converted %.loc17_16.1, %.loc17_16.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   return %.loc17_17 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
