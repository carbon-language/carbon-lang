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
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name0: inst1610612781}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name1:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name2:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block13, return_slot_pattern: inst1610612777, body: [inst_block20]}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst1610612781, bindings: inst_block16}
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
// CHECK:STDOUT:     'type(inst1610612765)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst1610612765)}
// CHECK:STDOUT:     'type(inst1610612767)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst1610612769)}
// CHECK:STDOUT:     'type(inst1610612769)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst1610612769)}
// CHECK:STDOUT:     'type(inst1610612782)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst1610612765)}
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
// CHECK:STDOUT:     inst1610612751:  {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612752:  {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612753:  {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612754:  {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612755:  {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612756:  {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst1610612757:  {kind: SymbolicBindingPattern, arg0: entity_name1, type: type(inst1610612756)}
// CHECK:STDOUT:     inst1610612758:  {kind: NameRef, arg0: name1, arg1: inst1610612753, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612759:  {kind: BindName, arg0: entity_name2, arg1: inst1610612778, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst1610612760:  {kind: PatternType, arg0: inst1610612754, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612761:  {kind: BindingPattern, arg0: entity_name2, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst1610612762:  {kind: PatternType, arg0: inst1610612755, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612763:  {kind: ValueParamPattern, arg0: inst1610612761, arg1: call_param0, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst1610612764:  {kind: NameRef, arg0: name1, arg1: inst1610612753, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612765:  {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612766:  {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst1610612765)}
// CHECK:STDOUT:     inst1610612767:  {kind: TupleType, arg0: inst_block9, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612768:  {kind: TupleLiteral, arg0: inst_block8, type: type(inst1610612767)}
// CHECK:STDOUT:     inst1610612769:  {kind: PointerType, arg0: inst1610612767, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612770:  {kind: Converted, arg0: inst1610612766, arg1: inst1610612765, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612771:  {kind: TupleType, arg0: inst_block11, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612772:  {kind: Converted, arg0: inst1610612768, arg1: inst1610612771, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612773:  {kind: TupleType, arg0: inst_block12, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612774:  {kind: PatternType, arg0: inst1610612771, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612775:  {kind: ReturnSlotPattern, arg0: inst1610612772, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst1610612776:  {kind: PatternType, arg0: inst1610612773, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612777:  {kind: OutParamPattern, arg0: inst1610612775, arg1: call_param1, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst1610612778:  {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst1610612779:  {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst1610612780:  {kind: ReturnSlot, arg0: inst1610612771, arg1: inst1610612779, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst1610612781:  {kind: FunctionDecl, arg0: function0, arg1: inst_block15, type: type(inst1610612782)}
// CHECK:STDOUT:     inst1610612782:  {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612783:  {kind: StructValue, arg0: inst_block_empty, type: type(inst1610612782)}
// CHECK:STDOUT:     inst1610612784:  {kind: PointerType, arg0: inst1610612771, type: type(TypeType)}
// CHECK:STDOUT:     inst1610612785:  {kind: RequireCompleteType, arg0: inst1610612771, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst1610612786:  {kind: RequireCompleteType, arg0: inst1610612771, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst1610612787:  {kind: RequireCompleteType, arg0: inst1610612773, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst1610612788:  {kind: RequireCompleteType, arg0: inst1610612754, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst1610612789:  {kind: RequireCompleteType, arg0: inst1610612754, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst1610612790:  {kind: RequireCompleteType, arg0: inst1610612755, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst1610612791:  {kind: NameRef, arg0: name2, arg1: inst1610612759, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst1610612792:  {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst1610612765)}
// CHECK:STDOUT:     inst1610612793:  {kind: TupleLiteral, arg0: inst_block21, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst1610612794:  {kind: RequireCompleteType, arg0: inst1610612771, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst1610612795:  {kind: TupleAccess, arg0: inst1610612780, arg1: element0, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst1610612796:  {kind: RequireCompleteType, arg0: inst1610612754, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst1610612797:  {kind: InitializeFrom, arg0: inst1610612791, arg1: inst1610612795, type: type(symbolic_constant2)}
// CHECK:STDOUT:     inst1610612798:  {kind: TupleAccess, arg0: inst1610612780, arg1: element1, type: type(inst1610612765)}
// CHECK:STDOUT:     inst1610612799:  {kind: TupleInit, arg0: inst_block_empty, arg1: inst1610612798, type: type(inst1610612765)}
// CHECK:STDOUT:     inst1610612800:  {kind: TupleValue, arg0: inst_block_empty, type: type(inst1610612765)}
// CHECK:STDOUT:     inst1610612801:  {kind: Converted, arg0: inst1610612792, arg1: inst1610612799, type: type(inst1610612765)}
// CHECK:STDOUT:     inst1610612802:  {kind: TupleInit, arg0: inst_block22, arg1: inst1610612780, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst1610612803:  {kind: Converted, arg0: inst1610612793, arg1: inst1610612802, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst1610612804:  {kind: ReturnExpr, arg0: inst1610612803, arg1: inst1610612780}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       inst1610612751:  symbolic_constant0
// CHECK:STDOUT:       inst1610612752:  symbolic_constant0
// CHECK:STDOUT:       inst1610612753:  symbolic_constant2
// CHECK:STDOUT:       inst1610612754:  symbolic_constant1
// CHECK:STDOUT:       inst1610612755:  symbolic_constant2
// CHECK:STDOUT:       inst1610612756:  concrete_constant(inst1610612756)
// CHECK:STDOUT:       inst1610612757:  concrete_constant(inst1610612757)
// CHECK:STDOUT:       inst1610612758:  symbolic_constant2
// CHECK:STDOUT:       inst1610612760:  symbolic_constant3
// CHECK:STDOUT:       inst1610612761:  concrete_constant(inst1610612761)
// CHECK:STDOUT:       inst1610612762:  symbolic_constant4
// CHECK:STDOUT:       inst1610612763:  concrete_constant(inst1610612763)
// CHECK:STDOUT:       inst1610612764:  symbolic_constant2
// CHECK:STDOUT:       inst1610612765:  concrete_constant(inst1610612765)
// CHECK:STDOUT:       inst1610612767:  concrete_constant(inst1610612767)
// CHECK:STDOUT:       inst1610612769:  concrete_constant(inst1610612769)
// CHECK:STDOUT:       inst1610612770:  concrete_constant(inst1610612765)
// CHECK:STDOUT:       inst1610612771:  symbolic_constant5
// CHECK:STDOUT:       inst1610612772:  symbolic_constant6
// CHECK:STDOUT:       inst1610612773:  symbolic_constant6
// CHECK:STDOUT:       inst1610612774:  symbolic_constant7
// CHECK:STDOUT:       inst1610612775:  concrete_constant(inst1610612775)
// CHECK:STDOUT:       inst1610612776:  symbolic_constant8
// CHECK:STDOUT:       inst1610612777:  concrete_constant(inst1610612777)
// CHECK:STDOUT:       inst1610612781:  concrete_constant(inst1610612783)
// CHECK:STDOUT:       inst1610612782:  concrete_constant(inst1610612782)
// CHECK:STDOUT:       inst1610612783:  concrete_constant(inst1610612783)
// CHECK:STDOUT:       inst1610612784:  symbolic_constant9
// CHECK:STDOUT:       inst1610612785:  symbolic_constant11
// CHECK:STDOUT:       inst1610612786:  symbolic_constant10
// CHECK:STDOUT:       inst1610612787:  symbolic_constant11
// CHECK:STDOUT:       inst1610612788:  symbolic_constant13
// CHECK:STDOUT:       inst1610612789:  symbolic_constant12
// CHECK:STDOUT:       inst1610612790:  symbolic_constant13
// CHECK:STDOUT:       inst1610612794:  symbolic_constant11
// CHECK:STDOUT:       inst1610612796:  symbolic_constant13
// CHECK:STDOUT:       inst1610612799:  concrete_constant(inst1610612800)
// CHECK:STDOUT:       inst1610612800:  concrete_constant(inst1610612800)
// CHECK:STDOUT:       inst1610612801:  concrete_constant(inst1610612800)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: inst1610612752, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant1: {inst: inst1610612754, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: inst1610612754, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: inst1610612760, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: inst1610612760, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: inst1610612771, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: inst1610612771, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: inst1610612774, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: inst1610612774, generic: generic0, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: inst1610612784, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant10: {inst: inst1610612786, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant11: {inst: inst1610612786, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant12: {inst: inst1610612789, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant13: {inst: inst1610612789, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst1610612781
// CHECK:STDOUT:     imports:         {}
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst1610612751
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst1610612757
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst1610612758
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst1610612763
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst1610612764
// CHECK:STDOUT:       1:               inst1610612766
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst1610612765
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst1610612764
// CHECK:STDOUT:       1:               inst1610612770
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst1610612754
// CHECK:STDOUT:       1:               inst1610612765
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst1610612755
// CHECK:STDOUT:       1:               inst1610612765
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst1610612778
// CHECK:STDOUT:       1:               inst1610612779
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst1610612757
// CHECK:STDOUT:       1:               inst1610612761
// CHECK:STDOUT:       2:               inst1610612763
// CHECK:STDOUT:       3:               inst1610612775
// CHECK:STDOUT:       4:               inst1610612777
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst1610612764
// CHECK:STDOUT:       1:               inst1610612766
// CHECK:STDOUT:       2:               inst1610612768
// CHECK:STDOUT:       3:               inst1610612770
// CHECK:STDOUT:       4:               inst1610612772
// CHECK:STDOUT:       5:               inst1610612751
// CHECK:STDOUT:       6:               inst1610612753
// CHECK:STDOUT:       7:               inst1610612778
// CHECK:STDOUT:       8:               inst1610612758
// CHECK:STDOUT:       9:               inst1610612759
// CHECK:STDOUT:       10:              inst1610612779
// CHECK:STDOUT:       11:              inst1610612780
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst1610612753
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst1610612754
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               inst1610612755
// CHECK:STDOUT:       1:               inst1610612762
// CHECK:STDOUT:       2:               inst1610612773
// CHECK:STDOUT:       3:               inst1610612776
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               inst1610612754
// CHECK:STDOUT:       1:               inst1610612760
// CHECK:STDOUT:       2:               inst1610612771
// CHECK:STDOUT:       3:               inst1610612774
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               inst1610612791
// CHECK:STDOUT:       1:               inst1610612792
// CHECK:STDOUT:       2:               inst1610612793
// CHECK:STDOUT:       3:               inst1610612795
// CHECK:STDOUT:       4:               inst1610612797
// CHECK:STDOUT:       5:               inst1610612798
// CHECK:STDOUT:       6:               inst1610612799
// CHECK:STDOUT:       7:               inst1610612801
// CHECK:STDOUT:       8:               inst1610612802
// CHECK:STDOUT:       9:               inst1610612803
// CHECK:STDOUT:       10:              inst1610612804
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               inst1610612791
// CHECK:STDOUT:       1:               inst1610612792
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               inst1610612797
// CHECK:STDOUT:       1:               inst1610612801
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               inst1610612787
// CHECK:STDOUT:       1:               inst1610612790
// CHECK:STDOUT:     inst_block24:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst1610612781
// CHECK:STDOUT: ...
