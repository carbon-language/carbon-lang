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
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst.0.45}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name1:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name2:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block13, return_slot_pattern: inst.0.41, body: [inst_block20]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst.0.45, bindings: inst_block16}
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
// CHECK:STDOUT:     'type(inst.0.29)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst.0.29)}
// CHECK:STDOUT:     'type(inst.0.31)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst.0.33)}
// CHECK:STDOUT:     'type(inst.0.33)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst.0.33)}
// CHECK:STDOUT:     'type(inst.0.46)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst.0.29)}
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
// CHECK:STDOUT:     inst.0.15:       {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.16:       {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.17:       {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.18:       {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.19:       {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.20:       {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst.0.21:       {kind: SymbolicBindingPattern, arg0: entity_name1, type: type(inst.0.20)}
// CHECK:STDOUT:     inst.0.22:       {kind: NameRef, arg0: name1, arg1: inst.0.17, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.23:       {kind: BindName, arg0: entity_name2, arg1: inst.0.42, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst.0.24:       {kind: PatternType, arg0: inst.0.18, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.25:       {kind: BindingPattern, arg0: entity_name2, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst.0.26:       {kind: PatternType, arg0: inst.0.19, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.27:       {kind: ValueParamPattern, arg0: inst.0.25, arg1: call_param0, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst.0.28:       {kind: NameRef, arg0: name1, arg1: inst.0.17, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.29:       {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.30:       {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst.0.29)}
// CHECK:STDOUT:     inst.0.31:       {kind: TupleType, arg0: inst_block9, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.32:       {kind: TupleLiteral, arg0: inst_block8, type: type(inst.0.31)}
// CHECK:STDOUT:     inst.0.33:       {kind: PointerType, arg0: inst.0.31, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.34:       {kind: Converted, arg0: inst.0.30, arg1: inst.0.29, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.35:       {kind: TupleType, arg0: inst_block11, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.36:       {kind: Converted, arg0: inst.0.32, arg1: inst.0.35, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.37:       {kind: TupleType, arg0: inst_block12, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.38:       {kind: PatternType, arg0: inst.0.35, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.39:       {kind: ReturnSlotPattern, arg0: inst.0.36, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst.0.40:       {kind: PatternType, arg0: inst.0.37, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.41:       {kind: OutParamPattern, arg0: inst.0.39, arg1: call_param1, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst.0.42:       {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst.0.43:       {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst.0.44:       {kind: ReturnSlot, arg0: inst.0.35, arg1: inst.0.43, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst.0.45:       {kind: FunctionDecl, arg0: function0, arg1: inst_block15, type: type(inst.0.46)}
// CHECK:STDOUT:     inst.0.46:       {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.47:       {kind: StructValue, arg0: inst_block_empty, type: type(inst.0.46)}
// CHECK:STDOUT:     inst.0.48:       {kind: PointerType, arg0: inst.0.35, type: type(TypeType)}
// CHECK:STDOUT:     inst.0.49:       {kind: RequireCompleteType, arg0: inst.0.35, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst.0.50:       {kind: RequireCompleteType, arg0: inst.0.35, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst.0.51:       {kind: RequireCompleteType, arg0: inst.0.37, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst.0.52:       {kind: RequireCompleteType, arg0: inst.0.18, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst.0.53:       {kind: RequireCompleteType, arg0: inst.0.18, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst.0.54:       {kind: RequireCompleteType, arg0: inst.0.19, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst.0.55:       {kind: NameRef, arg0: name2, arg1: inst.0.23, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst.0.56:       {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst.0.29)}
// CHECK:STDOUT:     inst.0.57:       {kind: TupleLiteral, arg0: inst_block21, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst.0.58:       {kind: RequireCompleteType, arg0: inst.0.35, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst.0.59:       {kind: TupleAccess, arg0: inst.0.44, arg1: element0, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst.0.60:       {kind: RequireCompleteType, arg0: inst.0.18, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst.0.61:       {kind: InitializeFrom, arg0: inst.0.55, arg1: inst.0.59, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst.0.62:       {kind: TupleAccess, arg0: inst.0.44, arg1: element1, type: type(inst.0.29)}
// CHECK:STDOUT:     inst.0.63:       {kind: TupleInit, arg0: inst_block_empty, arg1: inst.0.62, type: type(inst.0.29)}
// CHECK:STDOUT:     inst.0.64:       {kind: TupleValue, arg0: inst_block_empty, type: type(inst.0.29)}
// CHECK:STDOUT:     inst.0.65:       {kind: Converted, arg0: inst.0.56, arg1: inst.0.63, type: type(inst.0.29)}
// CHECK:STDOUT:     inst.0.66:       {kind: TupleInit, arg0: inst_block22, arg1: inst.0.44, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst.0.67:       {kind: Converted, arg0: inst.0.57, arg1: inst.0.66, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst.0.68:       {kind: ReturnExpr, arg0: inst.0.67, arg1: inst.0.44}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       inst.0.15:       symbolic_constant0
// CHECK:STDOUT:       inst.0.16:       symbolic_constant0
// CHECK:STDOUT:       inst.0.17:       symbolic_constant2
// CHECK:STDOUT:       inst.0.18:       symbolic_constant1
// CHECK:STDOUT:       inst.0.19:       symbolic_constant2
// CHECK:STDOUT:       inst.0.20:       concrete_constant(inst.0.20)
// CHECK:STDOUT:       inst.0.21:       concrete_constant(inst.0.21)
// CHECK:STDOUT:       inst.0.22:       symbolic_constant2
// CHECK:STDOUT:       inst.0.24:       symbolic_constant3
// CHECK:STDOUT:       inst.0.25:       concrete_constant(inst.0.25)
// CHECK:STDOUT:       inst.0.26:       symbolic_constant4
// CHECK:STDOUT:       inst.0.27:       concrete_constant(inst.0.27)
// CHECK:STDOUT:       inst.0.28:       symbolic_constant2
// CHECK:STDOUT:       inst.0.29:       concrete_constant(inst.0.29)
// CHECK:STDOUT:       inst.0.31:       concrete_constant(inst.0.31)
// CHECK:STDOUT:       inst.0.33:       concrete_constant(inst.0.33)
// CHECK:STDOUT:       inst.0.34:       concrete_constant(inst.0.29)
// CHECK:STDOUT:       inst.0.35:       symbolic_constant5
// CHECK:STDOUT:       inst.0.36:       symbolic_constant6
// CHECK:STDOUT:       inst.0.37:       symbolic_constant6
// CHECK:STDOUT:       inst.0.38:       symbolic_constant7
// CHECK:STDOUT:       inst.0.39:       concrete_constant(inst.0.39)
// CHECK:STDOUT:       inst.0.40:       symbolic_constant8
// CHECK:STDOUT:       inst.0.41:       concrete_constant(inst.0.41)
// CHECK:STDOUT:       inst.0.45:       concrete_constant(inst.0.47)
// CHECK:STDOUT:       inst.0.46:       concrete_constant(inst.0.46)
// CHECK:STDOUT:       inst.0.47:       concrete_constant(inst.0.47)
// CHECK:STDOUT:       inst.0.48:       symbolic_constant9
// CHECK:STDOUT:       inst.0.49:       symbolic_constant11
// CHECK:STDOUT:       inst.0.50:       symbolic_constant10
// CHECK:STDOUT:       inst.0.51:       symbolic_constant11
// CHECK:STDOUT:       inst.0.52:       symbolic_constant13
// CHECK:STDOUT:       inst.0.53:       symbolic_constant12
// CHECK:STDOUT:       inst.0.54:       symbolic_constant13
// CHECK:STDOUT:       inst.0.58:       symbolic_constant11
// CHECK:STDOUT:       inst.0.60:       symbolic_constant13
// CHECK:STDOUT:       inst.0.63:       concrete_constant(inst.0.64)
// CHECK:STDOUT:       inst.0.64:       concrete_constant(inst.0.64)
// CHECK:STDOUT:       inst.0.65:       concrete_constant(inst.0.64)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: inst.0.16, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant1: {inst: inst.0.18, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: inst.0.18, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: inst.0.24, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: inst.0.24, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: inst.0.35, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: inst.0.35, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: inst.0.38, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: inst.0.38, generic: generic0, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: inst.0.48, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant10: {inst: inst.0.50, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant11: {inst: inst.0.50, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant12: {inst: inst.0.53, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant13: {inst: inst.0.53, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst.0.45
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst.0.15
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst.0.21
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst.0.22
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst.0.27
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst.0.28
// CHECK:STDOUT:       1:               inst.0.30
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst.0.29
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst.0.28
// CHECK:STDOUT:       1:               inst.0.34
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst.0.18
// CHECK:STDOUT:       1:               inst.0.29
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst.0.19
// CHECK:STDOUT:       1:               inst.0.29
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst.0.42
// CHECK:STDOUT:       1:               inst.0.43
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst.0.21
// CHECK:STDOUT:       1:               inst.0.25
// CHECK:STDOUT:       2:               inst.0.27
// CHECK:STDOUT:       3:               inst.0.39
// CHECK:STDOUT:       4:               inst.0.41
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst.0.28
// CHECK:STDOUT:       1:               inst.0.30
// CHECK:STDOUT:       2:               inst.0.32
// CHECK:STDOUT:       3:               inst.0.34
// CHECK:STDOUT:       4:               inst.0.36
// CHECK:STDOUT:       5:               inst.0.15
// CHECK:STDOUT:       6:               inst.0.17
// CHECK:STDOUT:       7:               inst.0.42
// CHECK:STDOUT:       8:               inst.0.22
// CHECK:STDOUT:       9:               inst.0.23
// CHECK:STDOUT:       10:              inst.0.43
// CHECK:STDOUT:       11:              inst.0.44
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst.0.17
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst.0.18
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               inst.0.19
// CHECK:STDOUT:       1:               inst.0.26
// CHECK:STDOUT:       2:               inst.0.37
// CHECK:STDOUT:       3:               inst.0.40
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               inst.0.18
// CHECK:STDOUT:       1:               inst.0.24
// CHECK:STDOUT:       2:               inst.0.35
// CHECK:STDOUT:       3:               inst.0.38
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               inst.0.55
// CHECK:STDOUT:       1:               inst.0.56
// CHECK:STDOUT:       2:               inst.0.57
// CHECK:STDOUT:       3:               inst.0.59
// CHECK:STDOUT:       4:               inst.0.61
// CHECK:STDOUT:       5:               inst.0.62
// CHECK:STDOUT:       6:               inst.0.63
// CHECK:STDOUT:       7:               inst.0.65
// CHECK:STDOUT:       8:               inst.0.66
// CHECK:STDOUT:       9:               inst.0.67
// CHECK:STDOUT:       10:              inst.0.68
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               inst.0.55
// CHECK:STDOUT:       1:               inst.0.56
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               inst.0.61
// CHECK:STDOUT:       1:               inst.0.65
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               inst.0.51
// CHECK:STDOUT:       1:               inst.0.54
// CHECK:STDOUT:     inst_block24:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst.0.45
// CHECK:STDOUT: ...
