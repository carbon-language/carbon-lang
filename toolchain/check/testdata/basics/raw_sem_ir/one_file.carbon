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
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: ir0.inst45}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name1:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name2:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block13, return_slot_pattern: ir0.inst41, body: [inst_block20]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: ir0.inst45, bindings: inst_block16}
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
// CHECK:STDOUT:     'type(ir0.inst29)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(ir0.inst31)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(ir0.inst33)}
// CHECK:STDOUT:     'type(ir0.inst33)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(ir0.inst33)}
// CHECK:STDOUT:     'type(ir0.inst46)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst29)}
// CHECK:STDOUT:     'type(symbolic_constant1)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant1)}
// CHECK:STDOUT:     'type(symbolic_constant5)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant9)}
// CHECK:STDOUT:     'type(symbolic_constant9)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant9)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(symbolic_constant2)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant2)}
// CHECK:STDOUT:     'type(symbolic_constant6)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant9)}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst14:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     ir0.inst15:      {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst16:      {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst17:      {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst18:      {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst19:      {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst20:      {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst21:      {kind: SymbolicBindingPattern, arg0: entity_name1, type: type(ir0.inst20)}
// CHECK:STDOUT:     ir0.inst22:      {kind: NameRef, arg0: name1, arg1: ir0.inst17, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst23:      {kind: BindName, arg0: entity_name2, arg1: ir0.inst42, type: type(symbolic_constant2)}
// CHECK:STDOUT:     ir0.inst24:      {kind: PatternType, arg0: ir0.inst18, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst25:      {kind: BindingPattern, arg0: entity_name2, type: type(symbolic_constant4)}
// CHECK:STDOUT:     ir0.inst26:      {kind: PatternType, arg0: ir0.inst19, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst27:      {kind: ValueParamPattern, arg0: ir0.inst25, arg1: call_param0, type: type(symbolic_constant4)}
// CHECK:STDOUT:     ir0.inst28:      {kind: NameRef, arg0: name1, arg1: ir0.inst17, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst29:      {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst30:      {kind: TupleLiteral, arg0: inst_block_empty, type: type(ir0.inst29)}
// CHECK:STDOUT:     ir0.inst31:      {kind: TupleType, arg0: inst_block9, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst32:      {kind: TupleLiteral, arg0: inst_block8, type: type(ir0.inst31)}
// CHECK:STDOUT:     ir0.inst33:      {kind: PointerType, arg0: ir0.inst31, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst34:      {kind: Converted, arg0: ir0.inst30, arg1: ir0.inst29, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst35:      {kind: TupleType, arg0: inst_block11, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst36:      {kind: Converted, arg0: ir0.inst32, arg1: ir0.inst35, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst37:      {kind: TupleType, arg0: inst_block12, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst38:      {kind: PatternType, arg0: ir0.inst35, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst39:      {kind: ReturnSlotPattern, arg0: ir0.inst36, type: type(symbolic_constant8)}
// CHECK:STDOUT:     ir0.inst40:      {kind: PatternType, arg0: ir0.inst37, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst41:      {kind: OutParamPattern, arg0: ir0.inst39, arg1: call_param1, type: type(symbolic_constant8)}
// CHECK:STDOUT:     ir0.inst42:      {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant2)}
// CHECK:STDOUT:     ir0.inst43:      {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant6)}
// CHECK:STDOUT:     ir0.inst44:      {kind: ReturnSlot, arg0: ir0.inst35, arg1: ir0.inst43, type: type(symbolic_constant6)}
// CHECK:STDOUT:     ir0.inst45:      {kind: FunctionDecl, arg0: function0, arg1: inst_block15, type: type(ir0.inst46)}
// CHECK:STDOUT:     ir0.inst46:      {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst47:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst46)}
// CHECK:STDOUT:     ir0.inst48:      {kind: PointerType, arg0: ir0.inst35, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst49:      {kind: RequireCompleteType, arg0: ir0.inst35, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst50:      {kind: RequireCompleteType, arg0: ir0.inst35, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst51:      {kind: RequireCompleteType, arg0: ir0.inst37, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst52:      {kind: RequireCompleteType, arg0: ir0.inst18, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst53:      {kind: RequireCompleteType, arg0: ir0.inst18, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst54:      {kind: RequireCompleteType, arg0: ir0.inst19, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst55:      {kind: NameRef, arg0: name2, arg1: ir0.inst23, type: type(symbolic_constant2)}
// CHECK:STDOUT:     ir0.inst56:      {kind: TupleLiteral, arg0: inst_block_empty, type: type(ir0.inst29)}
// CHECK:STDOUT:     ir0.inst57:      {kind: TupleLiteral, arg0: inst_block21, type: type(symbolic_constant6)}
// CHECK:STDOUT:     ir0.inst58:      {kind: RequireCompleteType, arg0: ir0.inst35, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst59:      {kind: TupleAccess, arg0: ir0.inst44, arg1: element0, type: type(symbolic_constant2)}
// CHECK:STDOUT:     ir0.inst60:      {kind: RequireCompleteType, arg0: ir0.inst18, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst61:      {kind: InitializeFrom, arg0: ir0.inst55, arg1: ir0.inst59, type: type(symbolic_constant2)}
// CHECK:STDOUT:     ir0.inst62:      {kind: TupleAccess, arg0: ir0.inst44, arg1: element1, type: type(ir0.inst29)}
// CHECK:STDOUT:     ir0.inst63:      {kind: TupleInit, arg0: inst_block_empty, arg1: ir0.inst62, type: type(ir0.inst29)}
// CHECK:STDOUT:     ir0.inst64:      {kind: TupleValue, arg0: inst_block_empty, type: type(ir0.inst29)}
// CHECK:STDOUT:     ir0.inst65:      {kind: Converted, arg0: ir0.inst56, arg1: ir0.inst63, type: type(ir0.inst29)}
// CHECK:STDOUT:     ir0.inst66:      {kind: TupleInit, arg0: inst_block22, arg1: ir0.inst44, type: type(symbolic_constant6)}
// CHECK:STDOUT:     ir0.inst67:      {kind: Converted, arg0: ir0.inst57, arg1: ir0.inst66, type: type(symbolic_constant6)}
// CHECK:STDOUT:     ir0.inst68:      {kind: ReturnExpr, arg0: ir0.inst67, arg1: ir0.inst44}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       ir0.inst15:      symbolic_constant0
// CHECK:STDOUT:       ir0.inst16:      symbolic_constant0
// CHECK:STDOUT:       ir0.inst17:      symbolic_constant2
// CHECK:STDOUT:       ir0.inst18:      symbolic_constant1
// CHECK:STDOUT:       ir0.inst19:      symbolic_constant2
// CHECK:STDOUT:       ir0.inst20:      concrete_constant(ir0.inst20)
// CHECK:STDOUT:       ir0.inst21:      concrete_constant(ir0.inst21)
// CHECK:STDOUT:       ir0.inst22:      symbolic_constant2
// CHECK:STDOUT:       ir0.inst24:      symbolic_constant3
// CHECK:STDOUT:       ir0.inst25:      concrete_constant(ir0.inst25)
// CHECK:STDOUT:       ir0.inst26:      symbolic_constant4
// CHECK:STDOUT:       ir0.inst27:      concrete_constant(ir0.inst27)
// CHECK:STDOUT:       ir0.inst28:      symbolic_constant2
// CHECK:STDOUT:       ir0.inst29:      concrete_constant(ir0.inst29)
// CHECK:STDOUT:       ir0.inst31:      concrete_constant(ir0.inst31)
// CHECK:STDOUT:       ir0.inst33:      concrete_constant(ir0.inst33)
// CHECK:STDOUT:       ir0.inst34:      concrete_constant(ir0.inst29)
// CHECK:STDOUT:       ir0.inst35:      symbolic_constant5
// CHECK:STDOUT:       ir0.inst36:      symbolic_constant6
// CHECK:STDOUT:       ir0.inst37:      symbolic_constant6
// CHECK:STDOUT:       ir0.inst38:      symbolic_constant7
// CHECK:STDOUT:       ir0.inst39:      concrete_constant(ir0.inst39)
// CHECK:STDOUT:       ir0.inst40:      symbolic_constant8
// CHECK:STDOUT:       ir0.inst41:      concrete_constant(ir0.inst41)
// CHECK:STDOUT:       ir0.inst45:      concrete_constant(ir0.inst47)
// CHECK:STDOUT:       ir0.inst46:      concrete_constant(ir0.inst46)
// CHECK:STDOUT:       ir0.inst47:      concrete_constant(ir0.inst47)
// CHECK:STDOUT:       ir0.inst48:      symbolic_constant9
// CHECK:STDOUT:       ir0.inst49:      symbolic_constant11
// CHECK:STDOUT:       ir0.inst50:      symbolic_constant10
// CHECK:STDOUT:       ir0.inst51:      symbolic_constant11
// CHECK:STDOUT:       ir0.inst52:      symbolic_constant13
// CHECK:STDOUT:       ir0.inst53:      symbolic_constant12
// CHECK:STDOUT:       ir0.inst54:      symbolic_constant13
// CHECK:STDOUT:       ir0.inst58:      symbolic_constant11
// CHECK:STDOUT:       ir0.inst60:      symbolic_constant13
// CHECK:STDOUT:       ir0.inst63:      concrete_constant(ir0.inst64)
// CHECK:STDOUT:       ir0.inst64:      concrete_constant(ir0.inst64)
// CHECK:STDOUT:       ir0.inst65:      concrete_constant(ir0.inst64)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: ir0.inst16, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant1: {inst: ir0.inst18, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: ir0.inst18, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: ir0.inst24, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: ir0.inst24, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: ir0.inst35, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: ir0.inst35, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: ir0.inst38, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: ir0.inst38, generic: generic0, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: ir0.inst48, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant10: {inst: ir0.inst50, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant11: {inst: ir0.inst50, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant12: {inst: ir0.inst53, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant13: {inst: ir0.inst53, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               ir0.inst45
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               ir0.inst15
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               ir0.inst21
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               ir0.inst22
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               ir0.inst27
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               ir0.inst28
// CHECK:STDOUT:       1:               ir0.inst30
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               ir0.inst29
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               ir0.inst28
// CHECK:STDOUT:       1:               ir0.inst34
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               ir0.inst18
// CHECK:STDOUT:       1:               ir0.inst29
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               ir0.inst19
// CHECK:STDOUT:       1:               ir0.inst29
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               ir0.inst42
// CHECK:STDOUT:       1:               ir0.inst43
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               ir0.inst21
// CHECK:STDOUT:       1:               ir0.inst25
// CHECK:STDOUT:       2:               ir0.inst27
// CHECK:STDOUT:       3:               ir0.inst39
// CHECK:STDOUT:       4:               ir0.inst41
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               ir0.inst28
// CHECK:STDOUT:       1:               ir0.inst30
// CHECK:STDOUT:       2:               ir0.inst32
// CHECK:STDOUT:       3:               ir0.inst34
// CHECK:STDOUT:       4:               ir0.inst36
// CHECK:STDOUT:       5:               ir0.inst15
// CHECK:STDOUT:       6:               ir0.inst17
// CHECK:STDOUT:       7:               ir0.inst42
// CHECK:STDOUT:       8:               ir0.inst22
// CHECK:STDOUT:       9:               ir0.inst23
// CHECK:STDOUT:       10:              ir0.inst43
// CHECK:STDOUT:       11:              ir0.inst44
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               ir0.inst17
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               ir0.inst18
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               ir0.inst19
// CHECK:STDOUT:       1:               ir0.inst26
// CHECK:STDOUT:       2:               ir0.inst37
// CHECK:STDOUT:       3:               ir0.inst40
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               ir0.inst18
// CHECK:STDOUT:       1:               ir0.inst24
// CHECK:STDOUT:       2:               ir0.inst35
// CHECK:STDOUT:       3:               ir0.inst38
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               ir0.inst55
// CHECK:STDOUT:       1:               ir0.inst56
// CHECK:STDOUT:       2:               ir0.inst57
// CHECK:STDOUT:       3:               ir0.inst59
// CHECK:STDOUT:       4:               ir0.inst61
// CHECK:STDOUT:       5:               ir0.inst62
// CHECK:STDOUT:       6:               ir0.inst63
// CHECK:STDOUT:       7:               ir0.inst65
// CHECK:STDOUT:       8:               ir0.inst66
// CHECK:STDOUT:       9:               ir0.inst67
// CHECK:STDOUT:       10:              ir0.inst68
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               ir0.inst55
// CHECK:STDOUT:       1:               ir0.inst56
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               ir0.inst61
// CHECK:STDOUT:       1:               ir0.inst65
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               ir0.inst51
// CHECK:STDOUT:       1:               ir0.inst54
// CHECK:STDOUT:     inst_block24:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               ir0.inst45
// CHECK:STDOUT: ...
