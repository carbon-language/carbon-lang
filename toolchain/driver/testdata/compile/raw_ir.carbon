// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --no-prelude-import --phase=check --dump-raw-sem-ir %s
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/raw_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/raw_ir.carbon

fn Foo[T:! type](n: T) -> (T, ()) {
  return (n, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        raw_ir.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst12, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst34}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name1:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block9, return_slot_pattern: inst30, body: [inst_block16]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst34, bindings: inst_block12}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: inst_block13}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst38)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst22)}
// CHECK:STDOUT:     'type(inst22)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst22)}
// CHECK:STDOUT:     'type(symbolic_constant0)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant0)}
// CHECK:STDOUT:     'type(symbolic_constant2)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant6)}
// CHECK:STDOUT:     'type(symbolic_constant6)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant6)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(symbolic_constant3)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'type(symbolic_constant5)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant6)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:     type_block1:
// CHECK:STDOUT:       0:               type(TypeType)
// CHECK:STDOUT:       1:               type(inst22)
// CHECK:STDOUT:     type_block2:
// CHECK:STDOUT:       0:               type(symbolic_constant0)
// CHECK:STDOUT:       1:               type(inst22)
// CHECK:STDOUT:     type_block3:
// CHECK:STDOUT:       0:               type(symbolic_constant3)
// CHECK:STDOUT:       1:               type(inst22)
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst12:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst13:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst14:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst15:          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     inst16:          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     inst17:          {kind: NameRef, arg0: name1, arg1: inst13, type: type(TypeType)}
// CHECK:STDOUT:     inst18:          {kind: BindName, arg0: entity_name1, arg1: inst31, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst19:          {kind: BindingPattern, arg0: entity_name1, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst20:          {kind: ValueParamPattern, arg0: inst19, arg1: call_param0, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst21:          {kind: NameRef, arg0: name1, arg1: inst13, type: type(TypeType)}
// CHECK:STDOUT:     inst22:          {kind: TupleType, arg0: type_block0, type: type(TypeType)}
// CHECK:STDOUT:     inst23:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst22)}
// CHECK:STDOUT:     inst24:          {kind: TupleType, arg0: type_block1, type: type(TypeType)}
// CHECK:STDOUT:     inst25:          {kind: TupleLiteral, arg0: inst_block8, type: type(inst24)}
// CHECK:STDOUT:     inst26:          {kind: Converted, arg0: inst23, arg1: inst22, type: type(TypeType)}
// CHECK:STDOUT:     inst27:          {kind: TupleType, arg0: type_block2, type: type(TypeType)}
// CHECK:STDOUT:     inst28:          {kind: Converted, arg0: inst25, arg1: inst27, type: type(TypeType)}
// CHECK:STDOUT:     inst29:          {kind: ReturnSlotPattern, arg0: inst25, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst30:          {kind: OutParamPattern, arg0: inst29, arg1: call_param1, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst31:          {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst32:          {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst33:          {kind: ReturnSlot, arg0: inst25, arg1: inst32, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst34:          {kind: FunctionDecl, arg0: function0, arg1: inst_block11, type: type(inst38)}
// CHECK:STDOUT:     inst35:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst36:          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     inst37:          {kind: TupleType, arg0: type_block3, type: type(TypeType)}
// CHECK:STDOUT:     inst38:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst39:          {kind: StructValue, arg0: inst_block_empty, type: type(inst38)}
// CHECK:STDOUT:     inst40:          {kind: PointerType, arg0: type(symbolic_constant2), type: type(TypeType)}
// CHECK:STDOUT:     inst41:          {kind: RequireCompleteType, arg0: type(symbolic_constant2), type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst42:          {kind: RequireCompleteType, arg0: type(symbolic_constant2), type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst43:          {kind: RequireCompleteType, arg0: type(symbolic_constant3), type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst44:          {kind: RequireCompleteType, arg0: type(symbolic_constant0), type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst45:          {kind: NameRef, arg0: name2, arg1: inst18, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst46:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst22)}
// CHECK:STDOUT:     inst47:          {kind: TupleLiteral, arg0: inst_block17, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst48:          {kind: RequireCompleteType, arg0: type(symbolic_constant5), type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst49:          {kind: TupleAccess, arg0: inst33, arg1: element0, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst50:          {kind: RequireCompleteType, arg0: type(symbolic_constant0), type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst51:          {kind: InitializeFrom, arg0: inst45, arg1: inst49, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst52:          {kind: TupleAccess, arg0: inst33, arg1: element1, type: type(inst22)}
// CHECK:STDOUT:     inst53:          {kind: TupleInit, arg0: inst_block_empty, arg1: inst52, type: type(inst22)}
// CHECK:STDOUT:     inst54:          {kind: TupleValue, arg0: inst_block_empty, type: type(inst22)}
// CHECK:STDOUT:     inst55:          {kind: Converted, arg0: inst46, arg1: inst53, type: type(inst22)}
// CHECK:STDOUT:     inst56:          {kind: TupleInit, arg0: inst_block18, arg1: inst33, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst57:          {kind: Converted, arg0: inst47, arg1: inst56, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst58:          {kind: ReturnExpr, arg0: inst57, arg1: inst33}
// CHECK:STDOUT:     inst59:          {kind: RequireCompleteType, arg0: type(symbolic_constant5), type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60:          {kind: RequireCompleteType, arg0: type(symbolic_constant3), type: type(inst(WitnessType))}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst12:          concrete_constant(inst12)
// CHECK:STDOUT:     inst13:          symbolic_constant3
// CHECK:STDOUT:     inst14:          symbolic_constant0
// CHECK:STDOUT:     inst15:          symbolic_constant4
// CHECK:STDOUT:     inst16:          symbolic_constant1
// CHECK:STDOUT:     inst17:          symbolic_constant3
// CHECK:STDOUT:     inst21:          symbolic_constant3
// CHECK:STDOUT:     inst22:          concrete_constant(inst22)
// CHECK:STDOUT:     inst24:          concrete_constant(inst24)
// CHECK:STDOUT:     inst26:          concrete_constant(inst22)
// CHECK:STDOUT:     inst27:          symbolic_constant2
// CHECK:STDOUT:     inst28:          symbolic_constant5
// CHECK:STDOUT:     inst34:          concrete_constant(inst39)
// CHECK:STDOUT:     inst35:          symbolic_constant3
// CHECK:STDOUT:     inst36:          symbolic_constant4
// CHECK:STDOUT:     inst37:          symbolic_constant5
// CHECK:STDOUT:     inst38:          concrete_constant(inst38)
// CHECK:STDOUT:     inst39:          concrete_constant(inst39)
// CHECK:STDOUT:     inst40:          symbolic_constant6
// CHECK:STDOUT:     inst41:          symbolic_constant9
// CHECK:STDOUT:     inst42:          symbolic_constant7
// CHECK:STDOUT:     inst43:          symbolic_constant10
// CHECK:STDOUT:     inst44:          symbolic_constant8
// CHECK:STDOUT:     inst48:          symbolic_constant9
// CHECK:STDOUT:     inst50:          symbolic_constant10
// CHECK:STDOUT:     inst53:          concrete_constant(inst54)
// CHECK:STDOUT:     inst54:          concrete_constant(inst54)
// CHECK:STDOUT:     inst55:          concrete_constant(inst54)
// CHECK:STDOUT:     inst59:          symbolic_constant9
// CHECK:STDOUT:     inst60:          symbolic_constant10
// CHECK:STDOUT:   symbolic_constants:
// CHECK:STDOUT:     symbolic_constant0: {inst: inst14, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:     symbolic_constant1: {inst: inst16, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:     symbolic_constant2: {inst: inst27, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:     symbolic_constant3: {inst: inst14, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:     symbolic_constant4: {inst: inst16, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:     symbolic_constant5: {inst: inst27, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:     symbolic_constant6: {inst: inst40, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:     symbolic_constant7: {inst: inst42, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:     symbolic_constant8: {inst: inst44, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:     symbolic_constant9: {inst: inst42, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:     symbolic_constant10: {inst: inst44, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst34
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:     {}
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst15
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst17
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst20
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:       1:               inst23
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst31
// CHECK:STDOUT:       1:               inst32
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst15
// CHECK:STDOUT:       1:               inst19
// CHECK:STDOUT:       2:               inst20
// CHECK:STDOUT:       3:               inst29
// CHECK:STDOUT:       4:               inst30
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:       1:               inst23
// CHECK:STDOUT:       2:               inst25
// CHECK:STDOUT:       3:               inst26
// CHECK:STDOUT:       4:               inst28
// CHECK:STDOUT:       5:               inst13
// CHECK:STDOUT:       6:               inst31
// CHECK:STDOUT:       7:               inst17
// CHECK:STDOUT:       8:               inst18
// CHECK:STDOUT:       9:               inst32
// CHECK:STDOUT:       10:              inst33
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst13
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst35
// CHECK:STDOUT:       1:               inst36
// CHECK:STDOUT:       2:               inst37
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst14
// CHECK:STDOUT:       2:               inst27
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst45
// CHECK:STDOUT:       1:               inst46
// CHECK:STDOUT:       2:               inst47
// CHECK:STDOUT:       3:               inst49
// CHECK:STDOUT:       4:               inst51
// CHECK:STDOUT:       5:               inst52
// CHECK:STDOUT:       6:               inst53
// CHECK:STDOUT:       7:               inst55
// CHECK:STDOUT:       8:               inst56
// CHECK:STDOUT:       9:               inst57
// CHECK:STDOUT:       10:              inst58
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst45
// CHECK:STDOUT:       1:               inst46
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               inst51
// CHECK:STDOUT:       1:               inst55
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               inst59
// CHECK:STDOUT:       1:               inst60
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               inst12
// CHECK:STDOUT:       1:               inst34
// CHECK:STDOUT: ...
