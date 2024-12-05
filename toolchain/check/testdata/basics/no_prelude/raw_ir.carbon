// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --no-prelude-import --phase=check --dump-raw-sem-ir %s
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/no_prelude/raw_ir.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/no_prelude/raw_ir.carbon

fn Foo[T:! type](n: T) -> (T, ()) {
  return (n, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        raw_ir.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<invalid>, is_export: false}
// CHECK:STDOUT:   import_ir_insts: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst12, parent_scope: name_scope<invalid>, has_error: false, extended_scopes: [], names: {name0: inst36}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name1, parent_scope: name_scope<invalid>, index: comp_time_bind0}
// CHECK:STDOUT:     entity_name1:    {name: name2, parent_scope: name_scope<invalid>, index: comp_time_bind<invalid>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, return_slot_pattern: inst31, body: [inst_block14]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst36, bindings: inst_block10}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: inst_block12}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':  {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':     {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))': {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst40)':    {kind: none, type: type(inst23)}
// CHECK:STDOUT:     'type(inst23)':    {kind: none, type: type(inst23)}
// CHECK:STDOUT:     'type(symbolic_constant0)': {kind: copy, type: type(symbolic_constant0)}
// CHECK:STDOUT:     'type(symbolic_constant2)': {kind: pointer, type: type(symbolic_constant6)}
// CHECK:STDOUT:     'type(symbolic_constant6)': {kind: copy, type: type(symbolic_constant6)}
// CHECK:STDOUT:     'type(symbolic_constant3)': {kind: copy, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'type(symbolic_constant5)': {kind: pointer, type: type(symbolic_constant6)}
// CHECK:STDOUT:   type_blocks:
// CHECK:STDOUT:     type_block0:     {}
// CHECK:STDOUT:     type_block1:
// CHECK:STDOUT:       0:               type(TypeType)
// CHECK:STDOUT:       1:               type(inst23)
// CHECK:STDOUT:     type_block2:
// CHECK:STDOUT:       0:               type(symbolic_constant0)
// CHECK:STDOUT:       1:               type(inst23)
// CHECK:STDOUT:     type_block3:
// CHECK:STDOUT:       0:               type(symbolic_constant3)
// CHECK:STDOUT:       1:               type(inst23)
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst12:          {kind: Namespace, arg0: name_scope0, arg1: inst<invalid>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst13:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst32, type: type(TypeType)}
// CHECK:STDOUT:     inst14:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: type(TypeType)}
// CHECK:STDOUT:     inst15:          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     inst16:          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     inst17:          {kind: ValueParamPattern, arg0: inst15, arg1: runtime_param<invalid>, type: type(TypeType)}
// CHECK:STDOUT:     inst18:          {kind: NameRef, arg0: name1, arg1: inst13, type: type(TypeType)}
// CHECK:STDOUT:     inst19:          {kind: BindName, arg0: entity_name1, arg1: inst33, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst20:          {kind: BindingPattern, arg0: entity_name1, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst21:          {kind: ValueParamPattern, arg0: inst20, arg1: runtime_param0, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst22:          {kind: NameRef, arg0: name1, arg1: inst13, type: type(TypeType)}
// CHECK:STDOUT:     inst23:          {kind: TupleType, arg0: type_block0, type: type(TypeType)}
// CHECK:STDOUT:     inst24:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst23)}
// CHECK:STDOUT:     inst25:          {kind: TupleType, arg0: type_block1, type: type(TypeType)}
// CHECK:STDOUT:     inst26:          {kind: TupleLiteral, arg0: inst_block6, type: type(inst25)}
// CHECK:STDOUT:     inst27:          {kind: Converted, arg0: inst24, arg1: inst23, type: type(TypeType)}
// CHECK:STDOUT:     inst28:          {kind: TupleType, arg0: type_block2, type: type(TypeType)}
// CHECK:STDOUT:     inst29:          {kind: Converted, arg0: inst26, arg1: inst28, type: type(TypeType)}
// CHECK:STDOUT:     inst30:          {kind: ReturnSlotPattern, arg0: inst26, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst31:          {kind: OutParamPattern, arg0: inst30, arg1: runtime_param1, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst32:          {kind: ValueParam, arg0: runtime_param<invalid>, arg1: name1, type: type(TypeType)}
// CHECK:STDOUT:     inst33:          {kind: ValueParam, arg0: runtime_param0, arg1: name2, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst34:          {kind: OutParam, arg0: runtime_param1, arg1: name(ReturnSlot), type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst35:          {kind: ReturnSlot, arg0: inst26, arg1: inst34, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst36:          {kind: FunctionDecl, arg0: function0, arg1: inst_block9, type: type(inst40)}
// CHECK:STDOUT:     inst37:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<invalid>, type: type(TypeType)}
// CHECK:STDOUT:     inst38:          {kind: SymbolicBindingPattern, arg0: entity_name0, type: type(TypeType)}
// CHECK:STDOUT:     inst39:          {kind: TupleType, arg0: type_block3, type: type(TypeType)}
// CHECK:STDOUT:     inst40:          {kind: FunctionType, arg0: function0, arg1: specific<invalid>, type: type(TypeType)}
// CHECK:STDOUT:     inst41:          {kind: StructValue, arg0: inst_block_empty, type: type(inst40)}
// CHECK:STDOUT:     inst42:          {kind: PointerType, arg0: type(symbolic_constant2), type: type(TypeType)}
// CHECK:STDOUT:     inst43:          {kind: NameRef, arg0: name2, arg1: inst19, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst44:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst23)}
// CHECK:STDOUT:     inst45:          {kind: TupleLiteral, arg0: inst_block15, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst46:          {kind: TupleAccess, arg0: inst35, arg1: element0, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst47:          {kind: InitializeFrom, arg0: inst43, arg1: inst46, type: type(symbolic_constant3)}
// CHECK:STDOUT:     inst48:          {kind: TupleAccess, arg0: inst35, arg1: element1, type: type(inst23)}
// CHECK:STDOUT:     inst49:          {kind: TupleInit, arg0: inst_block_empty, arg1: inst48, type: type(inst23)}
// CHECK:STDOUT:     inst50:          {kind: TupleValue, arg0: inst_block_empty, type: type(inst23)}
// CHECK:STDOUT:     inst51:          {kind: Converted, arg0: inst44, arg1: inst49, type: type(inst23)}
// CHECK:STDOUT:     inst52:          {kind: TupleInit, arg0: inst_block16, arg1: inst35, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst53:          {kind: Converted, arg0: inst45, arg1: inst52, type: type(symbolic_constant5)}
// CHECK:STDOUT:     inst54:          {kind: ReturnExpr, arg0: inst53, arg1: inst35}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     inst12:          template_constant(inst12)
// CHECK:STDOUT:     inst13:          symbolic_constant3
// CHECK:STDOUT:     inst14:          symbolic_constant0
// CHECK:STDOUT:     inst15:          symbolic_constant4
// CHECK:STDOUT:     inst16:          symbolic_constant1
// CHECK:STDOUT:     inst17:          symbolic_constant4
// CHECK:STDOUT:     inst18:          symbolic_constant3
// CHECK:STDOUT:     inst22:          symbolic_constant3
// CHECK:STDOUT:     inst23:          template_constant(inst23)
// CHECK:STDOUT:     inst25:          template_constant(inst25)
// CHECK:STDOUT:     inst27:          template_constant(inst23)
// CHECK:STDOUT:     inst28:          symbolic_constant2
// CHECK:STDOUT:     inst29:          symbolic_constant5
// CHECK:STDOUT:     inst36:          template_constant(inst41)
// CHECK:STDOUT:     inst37:          symbolic_constant3
// CHECK:STDOUT:     inst38:          symbolic_constant4
// CHECK:STDOUT:     inst39:          symbolic_constant5
// CHECK:STDOUT:     inst40:          template_constant(inst40)
// CHECK:STDOUT:     inst41:          template_constant(inst41)
// CHECK:STDOUT:     inst42:          symbolic_constant6
// CHECK:STDOUT:     inst49:          template_constant(inst50)
// CHECK:STDOUT:     inst50:          template_constant(inst50)
// CHECK:STDOUT:     inst51:          template_constant(inst50)
// CHECK:STDOUT:   symbolic_constants:
// CHECK:STDOUT:     symbolic_constant0: {inst: inst14, generic: generic<invalid>, index: generic_inst<invalid>, .Self: false}
// CHECK:STDOUT:     symbolic_constant1: {inst: inst16, generic: generic<invalid>, index: generic_inst<invalid>, .Self: false}
// CHECK:STDOUT:     symbolic_constant2: {inst: inst28, generic: generic<invalid>, index: generic_inst<invalid>, .Self: false}
// CHECK:STDOUT:     symbolic_constant3: {inst: inst14, generic: generic0, index: generic_inst_in_decl0, .Self: false}
// CHECK:STDOUT:     symbolic_constant4: {inst: inst16, generic: generic0, index: generic_inst_in_decl1, .Self: false}
// CHECK:STDOUT:     symbolic_constant5: {inst: inst28, generic: generic0, index: generic_inst_in_decl2, .Self: false}
// CHECK:STDOUT:     symbolic_constant6: {inst: inst42, generic: generic<invalid>, index: generic_inst<invalid>, .Self: false}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst36
// CHECK:STDOUT:     import_refs:     {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst17
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst22
// CHECK:STDOUT:       1:               inst24
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst33
// CHECK:STDOUT:       1:               inst34
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst15
// CHECK:STDOUT:       1:               inst17
// CHECK:STDOUT:       2:               inst20
// CHECK:STDOUT:       3:               inst21
// CHECK:STDOUT:       4:               inst30
// CHECK:STDOUT:       5:               inst31
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst18
// CHECK:STDOUT:       1:               inst22
// CHECK:STDOUT:       2:               inst24
// CHECK:STDOUT:       3:               inst26
// CHECK:STDOUT:       4:               inst27
// CHECK:STDOUT:       5:               inst29
// CHECK:STDOUT:       6:               inst32
// CHECK:STDOUT:       7:               inst13
// CHECK:STDOUT:       8:               inst33
// CHECK:STDOUT:       9:               inst19
// CHECK:STDOUT:       10:              inst34
// CHECK:STDOUT:       11:              inst35
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst13
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst37
// CHECK:STDOUT:       1:               inst38
// CHECK:STDOUT:       2:               inst39
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst14
// CHECK:STDOUT:       2:               inst28
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst43
// CHECK:STDOUT:       1:               inst44
// CHECK:STDOUT:       2:               inst45
// CHECK:STDOUT:       3:               inst46
// CHECK:STDOUT:       4:               inst47
// CHECK:STDOUT:       5:               inst48
// CHECK:STDOUT:       6:               inst49
// CHECK:STDOUT:       7:               inst51
// CHECK:STDOUT:       8:               inst52
// CHECK:STDOUT:       9:               inst53
// CHECK:STDOUT:       10:              inst54
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst43
// CHECK:STDOUT:       1:               inst44
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst47
// CHECK:STDOUT:       1:               inst51
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst12
// CHECK:STDOUT:       1:               inst36
// CHECK:STDOUT: ...
