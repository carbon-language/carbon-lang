// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-raw-sem-ir --no-dump-sem-ir
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/one_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/one_file.carbon

fn Foo[T:! type](n: T) -> (T, ()) {
  return (n, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        one_file.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst15, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst43}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name1:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block13, return_slot_pattern: inst39, body: [inst_block20]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst43, bindings: inst_block16}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: inst_block17}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst44)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst28)}
// CHECK:STDOUT:     'type(inst28)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst28)}
// CHECK:STDOUT:     'type(symbolic_constant0)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant0)}
// CHECK:STDOUT:     'type(symbolic_constant4)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant8)}
// CHECK:STDOUT:     'type(symbolic_constant8)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant8)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(symbolic_constant1)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant1)}
// CHECK:STDOUT:     'type(symbolic_constant5)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant8)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst15:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst16:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst17:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst18:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst19:          {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst20:          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(inst19)}
// CHECK:STDOUT:     inst21:          {kind: NameRef, arg0: name1, arg1: inst16, type: type(TypeType)}
// CHECK:STDOUT:     inst22:          {kind: BindName, arg0: entity_name1, arg1: inst40, type: type(symbolic_constant1)}
// CHECK:STDOUT:     inst23:          {kind: PatternType, arg0: inst17, type: type(TypeType)}
// CHECK:STDOUT:     inst24:          {kind: BindingPattern, arg0: entity_name1, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst25:          {kind: PatternType, arg0: inst18, type: type(TypeType)}
// CHECK:STDOUT:     inst26:          {kind: ValueParamPattern, arg0: inst24, arg1: call_param0, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst27:          {kind: NameRef, arg0: name1, arg1: inst16, type: type(TypeType)}
// CHECK:STDOUT:     inst28:          {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst29:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst28)}
// CHECK:STDOUT:     inst30:          {kind: TupleType, arg0: inst_block9, type: type(TypeType)}
// CHECK:STDOUT:     inst31:          {kind: TupleLiteral, arg0: inst_block8, type: type(inst30)}
// CHECK:STDOUT:     inst32:          {kind: Converted, arg0: inst29, arg1: inst28, type: type(TypeType)}
// CHECK:STDOUT:     inst33:          {kind: TupleType, arg0: inst_block11, type: type(TypeType)}
// CHECK:STDOUT:     inst34:          {kind: Converted, arg0: inst31, arg1: inst33, type: type(TypeType)}
// CHECK:STDOUT:     inst35:          {kind: TupleType, arg0: inst_block12, type: type(TypeType)}
// CHECK:STDOUT:     inst36:          {kind: PatternType, arg0: inst33, type: type(TypeType)}
// CHECK:STDOUT:     inst37:          {kind: ReturnSlotPattern, arg0: inst34, type: type(symbolic_constant7)}
// CHECK:STDOUT:     inst38:          {kind: PatternType, arg0: inst35, type: type(TypeType)}
// CHECK:STDOUT:     inst39:          {kind: OutParamPattern, arg0: inst37, arg1: call_param1, type: type(symbolic_constant7)}
// CHECK:STDOUT:     inst40:          {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant1)}
// CHECK:STDOUT:     inst41:          {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst42:          {kind: ReturnSlot, arg0: inst33, arg1: inst41, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst43:          {kind: FunctionDecl, arg0: function0, arg1: inst_block15, type: type(inst44)}
// CHECK:STDOUT:     inst44:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst45:          {kind: StructValue, arg0: inst_block_empty, type: type(inst44)}
// CHECK:STDOUT:     inst46:          {kind: PointerType, arg0: inst33, type: type(TypeType)}
// CHECK:STDOUT:     inst47:          {kind: RequireCompleteType, arg0: inst33, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst48:          {kind: RequireCompleteType, arg0: inst33, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst49:          {kind: RequireCompleteType, arg0: inst35, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst50:          {kind: RequireCompleteType, arg0: inst17, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst51:          {kind: RequireCompleteType, arg0: inst17, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst52:          {kind: RequireCompleteType, arg0: inst18, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst53:          {kind: NameRef, arg0: name2, arg1: inst22, type: type(symbolic_constant1)}
// CHECK:STDOUT:     inst54:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst28)}
// CHECK:STDOUT:     inst55:          {kind: TupleLiteral, arg0: inst_block21, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst56:          {kind: RequireCompleteType, arg0: inst33, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst57:          {kind: TupleAccess, arg0: inst42, arg1: element0, type: type(symbolic_constant1)}
// CHECK:STDOUT:     inst58:          {kind: RequireCompleteType, arg0: inst17, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst59:          {kind: InitializeFrom, arg0: inst53, arg1: inst57, type: type(symbolic_constant1)}
// CHECK:STDOUT:     inst60:          {kind: TupleAccess, arg0: inst42, arg1: element1, type: type(inst28)}
// CHECK:STDOUT:     inst61:          {kind: TupleInit, arg0: inst_block_empty, arg1: inst60, type: type(inst28)}
// CHECK:STDOUT:     inst62:          {kind: TupleValue, arg0: inst_block_empty, type: type(inst28)}
// CHECK:STDOUT:     inst63:          {kind: Converted, arg0: inst54, arg1: inst61, type: type(inst28)}
// CHECK:STDOUT:     inst64:          {kind: TupleInit, arg0: inst_block22, arg1: inst42, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst65:          {kind: Converted, arg0: inst55, arg1: inst64, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst66:          {kind: ReturnExpr, arg0: inst65, arg1: inst42}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst15:          concrete_constant(inst15)
// CHECK:STDOUT:       inst16:          symbolic_constant1
// CHECK:STDOUT:       inst17:          symbolic_constant0
// CHECK:STDOUT:       inst18:          symbolic_constant1
// CHECK:STDOUT:       inst19:          concrete_constant(inst19)
// CHECK:STDOUT:       inst20:          concrete_constant(inst20)
// CHECK:STDOUT:       inst21:          symbolic_constant1
// CHECK:STDOUT:       inst23:          symbolic_constant2
// CHECK:STDOUT:       inst24:          concrete_constant(inst24)
// CHECK:STDOUT:       inst25:          symbolic_constant3
// CHECK:STDOUT:       inst26:          concrete_constant(inst26)
// CHECK:STDOUT:       inst27:          symbolic_constant1
// CHECK:STDOUT:       inst28:          concrete_constant(inst28)
// CHECK:STDOUT:       inst30:          concrete_constant(inst30)
// CHECK:STDOUT:       inst32:          concrete_constant(inst28)
// CHECK:STDOUT:       inst33:          symbolic_constant4
// CHECK:STDOUT:       inst34:          symbolic_constant5
// CHECK:STDOUT:       inst35:          symbolic_constant5
// CHECK:STDOUT:       inst36:          symbolic_constant6
// CHECK:STDOUT:       inst37:          concrete_constant(inst37)
// CHECK:STDOUT:       inst38:          symbolic_constant7
// CHECK:STDOUT:       inst39:          concrete_constant(inst39)
// CHECK:STDOUT:       inst43:          concrete_constant(inst45)
// CHECK:STDOUT:       inst44:          concrete_constant(inst44)
// CHECK:STDOUT:       inst45:          concrete_constant(inst45)
// CHECK:STDOUT:       inst46:          symbolic_constant8
// CHECK:STDOUT:       inst47:          symbolic_constant10
// CHECK:STDOUT:       inst48:          symbolic_constant9
// CHECK:STDOUT:       inst49:          symbolic_constant10
// CHECK:STDOUT:       inst50:          symbolic_constant12
// CHECK:STDOUT:       inst51:          symbolic_constant11
// CHECK:STDOUT:       inst52:          symbolic_constant12
// CHECK:STDOUT:       inst56:          symbolic_constant10
// CHECK:STDOUT:       inst58:          symbolic_constant12
// CHECK:STDOUT:       inst61:          concrete_constant(inst62)
// CHECK:STDOUT:       inst62:          concrete_constant(inst62)
// CHECK:STDOUT:       inst63:          concrete_constant(inst62)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: inst17, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant1: {inst: inst17, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: inst23, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: inst23, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: inst33, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: inst33, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: inst36, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: inst36, generic: generic0, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: inst46, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: inst48, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant10: {inst: inst48, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant11: {inst: inst51, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant12: {inst: inst51, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst43
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst20
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst26
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst27
// CHECK:STDOUT:       1:               inst29
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst28
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst27
// CHECK:STDOUT:       1:               inst32
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst17
// CHECK:STDOUT:       1:               inst28
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst18
// CHECK:STDOUT:       1:               inst28
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst40
// CHECK:STDOUT:       1:               inst41
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst20
// CHECK:STDOUT:       1:               inst24
// CHECK:STDOUT:       2:               inst26
// CHECK:STDOUT:       3:               inst37
// CHECK:STDOUT:       4:               inst39
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst27
// CHECK:STDOUT:       1:               inst29
// CHECK:STDOUT:       2:               inst31
// CHECK:STDOUT:       3:               inst32
// CHECK:STDOUT:       4:               inst34
// CHECK:STDOUT:       5:               inst16
// CHECK:STDOUT:       6:               inst40
// CHECK:STDOUT:       7:               inst21
// CHECK:STDOUT:       8:               inst22
// CHECK:STDOUT:       9:               inst41
// CHECK:STDOUT:       10:              inst42
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst16
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst17
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               inst18
// CHECK:STDOUT:       1:               inst25
// CHECK:STDOUT:       2:               inst35
// CHECK:STDOUT:       3:               inst38
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               inst17
// CHECK:STDOUT:       1:               inst23
// CHECK:STDOUT:       2:               inst33
// CHECK:STDOUT:       3:               inst36
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               inst53
// CHECK:STDOUT:       1:               inst54
// CHECK:STDOUT:       2:               inst55
// CHECK:STDOUT:       3:               inst57
// CHECK:STDOUT:       4:               inst59
// CHECK:STDOUT:       5:               inst60
// CHECK:STDOUT:       6:               inst61
// CHECK:STDOUT:       7:               inst63
// CHECK:STDOUT:       8:               inst64
// CHECK:STDOUT:       9:               inst65
// CHECK:STDOUT:       10:              inst66
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               inst53
// CHECK:STDOUT:       1:               inst54
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               inst59
// CHECK:STDOUT:       1:               inst63
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               inst49
// CHECK:STDOUT:       1:               inst52
// CHECK:STDOUT:     inst_block24:
// CHECK:STDOUT:       0:               inst15
// CHECK:STDOUT:       1:               inst43
// CHECK:STDOUT: ...
