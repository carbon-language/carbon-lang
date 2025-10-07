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
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: ir0.inst37}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block9, return_slot_pattern: ir0.inst32, body: [inst_block12]}
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
// CHECK:STDOUT:     'type(ir0.inst15)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst15)}
// CHECK:STDOUT:     'type(ir0.inst24)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(ir0.inst26)}
// CHECK:STDOUT:     'type(ir0.inst26)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(ir0.inst26)}
// CHECK:STDOUT:     'type(ir0.inst38)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst15)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst14:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     ir0.inst15:      {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst16:      {kind: TupleLiteral, arg0: inst_block_empty, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst17:      {kind: Converted, arg0: ir0.inst16, arg1: ir0.inst15, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst18:      {kind: BindName, arg0: entity_name0, arg1: ir0.inst33, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst19:      {kind: PatternType, arg0: ir0.inst15, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst20:      {kind: BindingPattern, arg0: entity_name0, type: type(ir0.inst19)}
// CHECK:STDOUT:     ir0.inst21:      {kind: ValueParamPattern, arg0: ir0.inst20, arg1: call_param0, type: type(ir0.inst19)}
// CHECK:STDOUT:     ir0.inst22:      {kind: TupleLiteral, arg0: inst_block_empty, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst23:      {kind: TupleLiteral, arg0: inst_block_empty, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst24:      {kind: TupleType, arg0: inst_block7, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst25:      {kind: TupleLiteral, arg0: inst_block6, type: type(ir0.inst24)}
// CHECK:STDOUT:     ir0.inst26:      {kind: PointerType, arg0: ir0.inst24, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst27:      {kind: Converted, arg0: ir0.inst22, arg1: ir0.inst15, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst28:      {kind: Converted, arg0: ir0.inst23, arg1: ir0.inst15, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst29:      {kind: Converted, arg0: ir0.inst25, arg1: ir0.inst24, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst30:      {kind: PatternType, arg0: ir0.inst24, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst31:      {kind: ReturnSlotPattern, arg0: ir0.inst29, type: type(ir0.inst30)}
// CHECK:STDOUT:     ir0.inst32:      {kind: OutParamPattern, arg0: ir0.inst31, arg1: call_param1, type: type(ir0.inst30)}
// CHECK:STDOUT:     ir0.inst33:      {kind: ValueParam, arg0: call_param0, arg1: name1, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst34:      {kind: SpliceBlock, arg0: inst_block4, arg1: ir0.inst17, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst35:      {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(ir0.inst24)}
// CHECK:STDOUT:     ir0.inst36:      {kind: ReturnSlot, arg0: ir0.inst24, arg1: ir0.inst35, type: type(ir0.inst24)}
// CHECK:STDOUT:     ir0.inst37:      {kind: FunctionDecl, arg0: function0, arg1: inst_block11, type: type(ir0.inst38)}
// CHECK:STDOUT:     ir0.inst38:      {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst39:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst38)}
// CHECK:STDOUT:     ir0.inst40:      {kind: NameRef, arg0: name1, arg1: ir0.inst18, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst41:      {kind: TupleLiteral, arg0: inst_block_empty, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst42:      {kind: TupleLiteral, arg0: inst_block13, type: type(ir0.inst24)}
// CHECK:STDOUT:     ir0.inst43:      {kind: TupleAccess, arg0: ir0.inst36, arg1: element0, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst44:      {kind: TupleInit, arg0: inst_block14, arg1: ir0.inst43, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst45:      {kind: TupleValue, arg0: inst_block_empty, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst46:      {kind: Converted, arg0: ir0.inst40, arg1: ir0.inst44, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst47:      {kind: TupleAccess, arg0: ir0.inst36, arg1: element1, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst48:      {kind: TupleInit, arg0: inst_block_empty, arg1: ir0.inst47, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst49:      {kind: Converted, arg0: ir0.inst41, arg1: ir0.inst48, type: type(ir0.inst15)}
// CHECK:STDOUT:     ir0.inst50:      {kind: TupleInit, arg0: inst_block15, arg1: ir0.inst36, type: type(ir0.inst24)}
// CHECK:STDOUT:     ir0.inst51:      {kind: TupleValue, arg0: inst_block16, type: type(ir0.inst24)}
// CHECK:STDOUT:     ir0.inst52:      {kind: Converted, arg0: ir0.inst42, arg1: ir0.inst50, type: type(ir0.inst24)}
// CHECK:STDOUT:     ir0.inst53:      {kind: ReturnExpr, arg0: ir0.inst52, arg1: ir0.inst36}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       ir0.inst15:      concrete_constant(ir0.inst15)
// CHECK:STDOUT:       ir0.inst17:      concrete_constant(ir0.inst15)
// CHECK:STDOUT:       ir0.inst19:      concrete_constant(ir0.inst19)
// CHECK:STDOUT:       ir0.inst20:      concrete_constant(ir0.inst20)
// CHECK:STDOUT:       ir0.inst21:      concrete_constant(ir0.inst21)
// CHECK:STDOUT:       ir0.inst24:      concrete_constant(ir0.inst24)
// CHECK:STDOUT:       ir0.inst26:      concrete_constant(ir0.inst26)
// CHECK:STDOUT:       ir0.inst27:      concrete_constant(ir0.inst15)
// CHECK:STDOUT:       ir0.inst28:      concrete_constant(ir0.inst15)
// CHECK:STDOUT:       ir0.inst29:      concrete_constant(ir0.inst24)
// CHECK:STDOUT:       ir0.inst30:      concrete_constant(ir0.inst30)
// CHECK:STDOUT:       ir0.inst31:      concrete_constant(ir0.inst31)
// CHECK:STDOUT:       ir0.inst32:      concrete_constant(ir0.inst32)
// CHECK:STDOUT:       ir0.inst34:      concrete_constant(ir0.inst15)
// CHECK:STDOUT:       ir0.inst37:      concrete_constant(ir0.inst39)
// CHECK:STDOUT:       ir0.inst38:      concrete_constant(ir0.inst38)
// CHECK:STDOUT:       ir0.inst39:      concrete_constant(ir0.inst39)
// CHECK:STDOUT:       ir0.inst44:      concrete_constant(ir0.inst45)
// CHECK:STDOUT:       ir0.inst45:      concrete_constant(ir0.inst45)
// CHECK:STDOUT:       ir0.inst46:      concrete_constant(ir0.inst45)
// CHECK:STDOUT:       ir0.inst48:      concrete_constant(ir0.inst45)
// CHECK:STDOUT:       ir0.inst49:      concrete_constant(ir0.inst45)
// CHECK:STDOUT:       ir0.inst50:      concrete_constant(ir0.inst51)
// CHECK:STDOUT:       ir0.inst51:      concrete_constant(ir0.inst51)
// CHECK:STDOUT:       ir0.inst52:      concrete_constant(ir0.inst51)
// CHECK:STDOUT:     symbolic_constants: {}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               ir0.inst37
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               ir0.inst16
// CHECK:STDOUT:       1:               ir0.inst17
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               ir0.inst21
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               ir0.inst22
// CHECK:STDOUT:       1:               ir0.inst23
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               ir0.inst15
// CHECK:STDOUT:       1:               ir0.inst15
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               ir0.inst27
// CHECK:STDOUT:       1:               ir0.inst28
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               ir0.inst33
// CHECK:STDOUT:       1:               ir0.inst35
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               ir0.inst20
// CHECK:STDOUT:       1:               ir0.inst21
// CHECK:STDOUT:       2:               ir0.inst31
// CHECK:STDOUT:       3:               ir0.inst32
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               ir0.inst22
// CHECK:STDOUT:       1:               ir0.inst23
// CHECK:STDOUT:       2:               ir0.inst25
// CHECK:STDOUT:       3:               ir0.inst27
// CHECK:STDOUT:       4:               ir0.inst28
// CHECK:STDOUT:       5:               ir0.inst29
// CHECK:STDOUT:       6:               ir0.inst33
// CHECK:STDOUT:       7:               ir0.inst34
// CHECK:STDOUT:       8:               ir0.inst18
// CHECK:STDOUT:       9:               ir0.inst35
// CHECK:STDOUT:       10:              ir0.inst36
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               ir0.inst40
// CHECK:STDOUT:       1:               ir0.inst41
// CHECK:STDOUT:       2:               ir0.inst42
// CHECK:STDOUT:       3:               ir0.inst43
// CHECK:STDOUT:       4:               ir0.inst44
// CHECK:STDOUT:       5:               ir0.inst46
// CHECK:STDOUT:       6:               ir0.inst47
// CHECK:STDOUT:       7:               ir0.inst48
// CHECK:STDOUT:       8:               ir0.inst49
// CHECK:STDOUT:       9:               ir0.inst50
// CHECK:STDOUT:       10:              ir0.inst52
// CHECK:STDOUT:       11:              ir0.inst53
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               ir0.inst40
// CHECK:STDOUT:       1:               ir0.inst41
// CHECK:STDOUT:     inst_block14:    {}
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               ir0.inst46
// CHECK:STDOUT:       1:               ir0.inst49
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               ir0.inst45
// CHECK:STDOUT:       1:               ir0.inst45
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               ir0.inst37
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
