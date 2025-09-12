// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --dump-raw-sem-ir --no-dump-sem-ir
//
// Check that raw IR dumping works as expected.
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/raw_sem_ir/one_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/raw_sem_ir/one_file.carbon

fn Foo[T:! type](p: T*) -> (T*, ()) {
  return (p, ());
}

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        one_file.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     ir0:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir1:             {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     ir2:             {decl_id: inst15, is_export: false}
// CHECK:STDOUT:     ir3:             {decl_id: inst15, is_export: false}
// CHECK:STDOUT:     ir4:             {decl_id: inst15, is_export: false}
// CHECK:STDOUT:     ir5:             {decl_id: inst15, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: ir4, inst_id: inst15}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: ir4, inst_id: inst15}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: ir4, inst_id: inst17}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: ir4, inst_id: inst45}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: ir4, inst_id: inst40}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: ir4, inst_id: inst40}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: ir4, inst_id: inst40}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: ir4, inst_id: inst34}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: ir4, inst_id: inst35}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: ir4, inst_id: inst28}
// CHECK:STDOUT:     import_ir_inst10: {ir_id: ir4, inst_id: inst30}
// CHECK:STDOUT:     import_ir_inst11: {ir_id: ir4, inst_id: inst17}
// CHECK:STDOUT:     import_ir_inst12: {ir_id: ir4, inst_id: inst21}
// CHECK:STDOUT:     import_ir_inst13: {ir_id: ir4, inst_id: inst24}
// CHECK:STDOUT:     import_ir_inst14: {ir_id: ir4, inst_id: inst29}
// CHECK:STDOUT:     import_ir_inst15: {ir_id: ir4, inst_id: inst97}
// CHECK:STDOUT:     import_ir_inst16: {ir_id: ir4, inst_id: inst95}
// CHECK:STDOUT:     import_ir_inst17: {ir_id: ir4, inst_id: inst84}
// CHECK:STDOUT:     import_ir_inst18: {ir_id: ir4, inst_id: inst80}
// CHECK:STDOUT:     import_ir_inst19: {ir_id: ir4, inst_id: inst90}
// CHECK:STDOUT:     import_ir_inst20: {ir_id: ir4, inst_id: inst93}
// CHECK:STDOUT:     import_ir_inst21: {ir_id: ir4, inst_id: inst116}
// CHECK:STDOUT:     import_ir_inst22: {ir_id: ir4, inst_id: inst96}
// CHECK:STDOUT:     import_ir_inst23: {ir_id: ir4, inst_id: inst82}
// CHECK:STDOUT:     import_ir_inst24: {ir_id: ir4, inst_id: inst88}
// CHECK:STDOUT:     import_ir_inst25: {ir_id: ir4, inst_id: inst92}
// CHECK:STDOUT:     import_ir_inst26: {ir_id: ir4, inst_id: inst99}
// CHECK:STDOUT:     import_ir_inst27: {ir_id: ir4, inst_id: inst116}
// CHECK:STDOUT:     import_ir_inst28: {ir_id: ir4, inst_id: inst111}
// CHECK:STDOUT:     import_ir_inst29: {ir_id: ir4, inst_id: inst112}
// CHECK:STDOUT:     import_ir_inst30: {ir_id: ir4, inst_id: inst107}
// CHECK:STDOUT:     import_ir_inst31: {ir_id: ir4, inst_id: inst109}
// CHECK:STDOUT:     import_ir_inst32: {ir_id: ir4, inst_id: inst80}
// CHECK:STDOUT:     import_ir_inst33: {ir_id: ir4, inst_id: inst119}
// CHECK:STDOUT:     import_ir_inst34: {ir_id: ir4, inst_id: inst120}
// CHECK:STDOUT:     import_ir_inst35: {ir_id: ir4, inst_id: inst102}
// CHECK:STDOUT:     import_ir_inst36: {ir_id: ir4, inst_id: inst103}
// CHECK:STDOUT:     import_ir_inst37: {ir_id: ir4, inst_id: inst104}
// CHECK:STDOUT:     import_ir_inst38: {ir_id: ir4, inst_id: inst108}
// CHECK:STDOUT:     import_ir_inst39: {ir_id: ir4, inst_id: inst124}
// CHECK:STDOUT:     import_ir_inst40: {ir_id: ir4, inst_id: inst132}
// CHECK:STDOUT:     import_ir_inst41: {ir_id: ir4, inst_id: inst139}
// CHECK:STDOUT:     import_ir_inst42: {ir_id: ir4, inst_id: inst144}
// CHECK:STDOUT:     import_ir_inst43: {ir_id: ir4, inst_id: inst145}
// CHECK:STDOUT:     import_ir_inst44: {ir_id: ir4, inst_id: inst146}
// CHECK:STDOUT:     import_ir_inst45: {ir_id: ir4, inst_id: inst151}
// CHECK:STDOUT:     import_ir_inst46: {ir_id: ir4, inst_id: inst171}
// CHECK:STDOUT:     import_ir_inst47: {ir_id: ir4, inst_id: inst169}
// CHECK:STDOUT:     import_ir_inst48: {ir_id: ir4, inst_id: inst167}
// CHECK:STDOUT:     import_ir_inst49: {ir_id: ir4, inst_id: inst168}
// CHECK:STDOUT:     import_ir_inst50: {ir_id: ir4, inst_id: inst195}
// CHECK:STDOUT:     import_ir_inst51: {ir_id: ir4, inst_id: inst193}
// CHECK:STDOUT:     import_ir_inst52: {ir_id: ir4, inst_id: inst191}
// CHECK:STDOUT:     import_ir_inst53: {ir_id: ir4, inst_id: inst192}
// CHECK:STDOUT:     import_ir_inst54: {ir_id: ir4, inst_id: inst219}
// CHECK:STDOUT:     import_ir_inst55: {ir_id: ir4, inst_id: inst217}
// CHECK:STDOUT:     import_ir_inst56: {ir_id: ir4, inst_id: inst215}
// CHECK:STDOUT:     import_ir_inst57: {ir_id: ir4, inst_id: inst216}
// CHECK:STDOUT:     import_ir_inst58: {ir_id: ir4, inst_id: inst243}
// CHECK:STDOUT:     import_ir_inst59: {ir_id: ir4, inst_id: inst241}
// CHECK:STDOUT:     import_ir_inst60: {ir_id: ir4, inst_id: inst239}
// CHECK:STDOUT:     import_ir_inst61: {ir_id: ir4, inst_id: inst240}
// CHECK:STDOUT:     import_ir_inst62: {ir_id: ir4, inst_id: inst272}
// CHECK:STDOUT:     import_ir_inst63: {ir_id: ir4, inst_id: inst270}
// CHECK:STDOUT:     import_ir_inst64: {ir_id: ir4, inst_id: inst264}
// CHECK:STDOUT:     import_ir_inst65: {ir_id: ir4, inst_id: inst261}
// CHECK:STDOUT:     import_ir_inst66: {ir_id: ir4, inst_id: inst266}
// CHECK:STDOUT:     import_ir_inst67: {ir_id: ir4, inst_id: inst269}
// CHECK:STDOUT:     import_ir_inst68: {ir_id: ir4, inst_id: inst289}
// CHECK:STDOUT:     import_ir_inst69: {ir_id: ir4, inst_id: inst271}
// CHECK:STDOUT:     import_ir_inst70: {ir_id: ir4, inst_id: inst263}
// CHECK:STDOUT:     import_ir_inst71: {ir_id: ir4, inst_id: inst268}
// CHECK:STDOUT:     import_ir_inst72: {ir_id: ir4, inst_id: inst274}
// CHECK:STDOUT:     import_ir_inst73: {ir_id: ir4, inst_id: inst289}
// CHECK:STDOUT:     import_ir_inst74: {ir_id: ir4, inst_id: inst284}
// CHECK:STDOUT:     import_ir_inst75: {ir_id: ir4, inst_id: inst285}
// CHECK:STDOUT:     import_ir_inst76: {ir_id: ir4, inst_id: inst280}
// CHECK:STDOUT:     import_ir_inst77: {ir_id: ir4, inst_id: inst282}
// CHECK:STDOUT:     import_ir_inst78: {ir_id: ir4, inst_id: inst261}
// CHECK:STDOUT:     import_ir_inst79: {ir_id: ir4, inst_id: inst292}
// CHECK:STDOUT:     import_ir_inst80: {ir_id: ir4, inst_id: inst293}
// CHECK:STDOUT:     import_ir_inst81: {ir_id: ir4, inst_id: inst296}
// CHECK:STDOUT:     import_ir_inst82: {ir_id: ir4, inst_id: inst276}
// CHECK:STDOUT:     import_ir_inst83: {ir_id: ir4, inst_id: inst277}
// CHECK:STDOUT:     import_ir_inst84: {ir_id: ir4, inst_id: inst281}
// CHECK:STDOUT:     import_ir_inst85: {ir_id: ir4, inst_id: inst302}
// CHECK:STDOUT:     import_ir_inst86: {ir_id: ir4, inst_id: inst300}
// CHECK:STDOUT:     import_ir_inst87: {ir_id: ir4, inst_id: inst(TypeType)}
// CHECK:STDOUT:     import_ir_inst88: {ir_id: ir4, inst_id: inst299}
// CHECK:STDOUT:     import_ir_inst89: {ir_id: ir4, inst_id: inst323}
// CHECK:STDOUT:     import_ir_inst90: {ir_id: ir4, inst_id: inst321}
// CHECK:STDOUT:     import_ir_inst91: {ir_id: ir4, inst_id: inst319}
// CHECK:STDOUT:     import_ir_inst92: {ir_id: ir4, inst_id: inst320}
// CHECK:STDOUT:     import_ir_inst93: {ir_id: ir4, inst_id: inst371}
// CHECK:STDOUT:     import_ir_inst94: {ir_id: ir4, inst_id: inst369}
// CHECK:STDOUT:     import_ir_inst95: {ir_id: ir4, inst_id: inst350}
// CHECK:STDOUT:     import_ir_inst96: {ir_id: ir4, inst_id: inst344}
// CHECK:STDOUT:     import_ir_inst97: {ir_id: ir4, inst_id: inst342}
// CHECK:STDOUT:     import_ir_inst98: {ir_id: ir4, inst_id: inst347}
// CHECK:STDOUT:     import_ir_inst99: {ir_id: ir4, inst_id: inst364}
// CHECK:STDOUT:     import_ir_inst100: {ir_id: ir4, inst_id: inst366}
// CHECK:STDOUT:     import_ir_inst101: {ir_id: ir4, inst_id: inst393}
// CHECK:STDOUT:     import_ir_inst102: {ir_id: ir4, inst_id: inst370}
// CHECK:STDOUT:     import_ir_inst103: {ir_id: ir4, inst_id: inst343}
// CHECK:STDOUT:     import_ir_inst104: {ir_id: ir4, inst_id: inst349}
// CHECK:STDOUT:     import_ir_inst105: {ir_id: ir4, inst_id: inst357}
// CHECK:STDOUT:     import_ir_inst106: {ir_id: ir4, inst_id: inst361}
// CHECK:STDOUT:     import_ir_inst107: {ir_id: ir4, inst_id: inst365}
// CHECK:STDOUT:     import_ir_inst108: {ir_id: ir4, inst_id: inst373}
// CHECK:STDOUT:     import_ir_inst109: {ir_id: ir4, inst_id: inst393}
// CHECK:STDOUT:     import_ir_inst110: {ir_id: ir4, inst_id: inst388}
// CHECK:STDOUT:     import_ir_inst111: {ir_id: ir4, inst_id: inst389}
// CHECK:STDOUT:     import_ir_inst112: {ir_id: ir4, inst_id: inst384}
// CHECK:STDOUT:     import_ir_inst113: {ir_id: ir4, inst_id: inst386}
// CHECK:STDOUT:     import_ir_inst114: {ir_id: ir4, inst_id: inst342}
// CHECK:STDOUT:     import_ir_inst115: {ir_id: ir4, inst_id: inst347}
// CHECK:STDOUT:     import_ir_inst116: {ir_id: ir4, inst_id: inst396}
// CHECK:STDOUT:     import_ir_inst117: {ir_id: ir4, inst_id: inst397}
// CHECK:STDOUT:     import_ir_inst118: {ir_id: ir4, inst_id: inst377}
// CHECK:STDOUT:     import_ir_inst119: {ir_id: ir4, inst_id: inst378}
// CHECK:STDOUT:     import_ir_inst120: {ir_id: ir4, inst_id: inst379}
// CHECK:STDOUT:     import_ir_inst121: {ir_id: ir4, inst_id: inst380}
// CHECK:STDOUT:     import_ir_inst122: {ir_id: ir4, inst_id: inst381}
// CHECK:STDOUT:     import_ir_inst123: {ir_id: ir4, inst_id: inst385}
// CHECK:STDOUT:     import_ir_inst124: {ir_id: ir4, inst_id: inst402}
// CHECK:STDOUT:     import_ir_inst125: {ir_id: ir4, inst_id: inst410}
// CHECK:STDOUT:     import_ir_inst126: {ir_id: ir4, inst_id: inst416}
// CHECK:STDOUT:     import_ir_inst127: {ir_id: ir4, inst_id: inst418}
// CHECK:STDOUT:     import_ir_inst128: {ir_id: ir4, inst_id: inst419}
// CHECK:STDOUT:     import_ir_inst129: {ir_id: ir4, inst_id: inst420}
// CHECK:STDOUT:     import_ir_inst130: {ir_id: ir4, inst_id: inst423}
// CHECK:STDOUT:     import_ir_inst131: {ir_id: ir4, inst_id: inst434}
// CHECK:STDOUT:     import_ir_inst132: {ir_id: ir4, inst_id: inst441}
// CHECK:STDOUT:     import_ir_inst133: {ir_id: ir4, inst_id: inst446}
// CHECK:STDOUT:     import_ir_inst134: {ir_id: ir4, inst_id: inst447}
// CHECK:STDOUT:     import_ir_inst135: {ir_id: ir4, inst_id: inst448}
// CHECK:STDOUT:     import_ir_inst136: {ir_id: ir4, inst_id: inst453}
// CHECK:STDOUT:     import_ir_inst137: {ir_id: ir4, inst_id: inst509}
// CHECK:STDOUT:     import_ir_inst138: {ir_id: ir4, inst_id: inst507}
// CHECK:STDOUT:     import_ir_inst139: {ir_id: ir4, inst_id: inst483}
// CHECK:STDOUT:     import_ir_inst140: {ir_id: ir4, inst_id: inst477}
// CHECK:STDOUT:     import_ir_inst141: {ir_id: ir4, inst_id: inst472}
// CHECK:STDOUT:     import_ir_inst142: {ir_id: ir4, inst_id: inst470}
// CHECK:STDOUT:     import_ir_inst143: {ir_id: ir4, inst_id: inst475}
// CHECK:STDOUT:     import_ir_inst144: {ir_id: ir4, inst_id: inst480}
// CHECK:STDOUT:     import_ir_inst145: {ir_id: ir4, inst_id: inst501}
// CHECK:STDOUT:     import_ir_inst146: {ir_id: ir4, inst_id: inst503}
// CHECK:STDOUT:     import_ir_inst147: {ir_id: ir4, inst_id: inst534}
// CHECK:STDOUT:     import_ir_inst148: {ir_id: ir4, inst_id: inst508}
// CHECK:STDOUT:     import_ir_inst149: {ir_id: ir4, inst_id: inst471}
// CHECK:STDOUT:     import_ir_inst150: {ir_id: ir4, inst_id: inst476}
// CHECK:STDOUT:     import_ir_inst151: {ir_id: ir4, inst_id: inst482}
// CHECK:STDOUT:     import_ir_inst152: {ir_id: ir4, inst_id: inst491}
// CHECK:STDOUT:     import_ir_inst153: {ir_id: ir4, inst_id: inst494}
// CHECK:STDOUT:     import_ir_inst154: {ir_id: ir4, inst_id: inst498}
// CHECK:STDOUT:     import_ir_inst155: {ir_id: ir4, inst_id: inst502}
// CHECK:STDOUT:     import_ir_inst156: {ir_id: ir4, inst_id: inst511}
// CHECK:STDOUT:     import_ir_inst157: {ir_id: ir4, inst_id: inst534}
// CHECK:STDOUT:     import_ir_inst158: {ir_id: ir4, inst_id: inst529}
// CHECK:STDOUT:     import_ir_inst159: {ir_id: ir4, inst_id: inst530}
// CHECK:STDOUT:     import_ir_inst160: {ir_id: ir4, inst_id: inst525}
// CHECK:STDOUT:     import_ir_inst161: {ir_id: ir4, inst_id: inst527}
// CHECK:STDOUT:     import_ir_inst162: {ir_id: ir4, inst_id: inst470}
// CHECK:STDOUT:     import_ir_inst163: {ir_id: ir4, inst_id: inst475}
// CHECK:STDOUT:     import_ir_inst164: {ir_id: ir4, inst_id: inst480}
// CHECK:STDOUT:     import_ir_inst165: {ir_id: ir4, inst_id: inst537}
// CHECK:STDOUT:     import_ir_inst166: {ir_id: ir4, inst_id: inst538}
// CHECK:STDOUT:     import_ir_inst167: {ir_id: ir4, inst_id: inst516}
// CHECK:STDOUT:     import_ir_inst168: {ir_id: ir4, inst_id: inst517}
// CHECK:STDOUT:     import_ir_inst169: {ir_id: ir4, inst_id: inst518}
// CHECK:STDOUT:     import_ir_inst170: {ir_id: ir4, inst_id: inst519}
// CHECK:STDOUT:     import_ir_inst171: {ir_id: ir4, inst_id: inst520}
// CHECK:STDOUT:     import_ir_inst172: {ir_id: ir4, inst_id: inst521}
// CHECK:STDOUT:     import_ir_inst173: {ir_id: ir4, inst_id: inst522}
// CHECK:STDOUT:     import_ir_inst174: {ir_id: ir4, inst_id: inst526}
// CHECK:STDOUT:     import_ir_inst175: {ir_id: ir4, inst_id: inst543}
// CHECK:STDOUT:     import_ir_inst176: {ir_id: ir4, inst_id: inst550}
// CHECK:STDOUT:     import_ir_inst177: {ir_id: ir4, inst_id: inst556}
// CHECK:STDOUT:     import_ir_inst178: {ir_id: ir4, inst_id: inst558}
// CHECK:STDOUT:     import_ir_inst179: {ir_id: ir4, inst_id: inst559}
// CHECK:STDOUT:     import_ir_inst180: {ir_id: ir4, inst_id: inst560}
// CHECK:STDOUT:     import_ir_inst181: {ir_id: ir4, inst_id: inst563}
// CHECK:STDOUT:     import_ir_inst182: {ir_id: ir4, inst_id: inst572}
// CHECK:STDOUT:     import_ir_inst183: {ir_id: ir4, inst_id: inst578}
// CHECK:STDOUT:     import_ir_inst184: {ir_id: ir4, inst_id: inst580}
// CHECK:STDOUT:     import_ir_inst185: {ir_id: ir4, inst_id: inst581}
// CHECK:STDOUT:     import_ir_inst186: {ir_id: ir4, inst_id: inst582}
// CHECK:STDOUT:     import_ir_inst187: {ir_id: ir4, inst_id: inst585}
// CHECK:STDOUT:     import_ir_inst188: {ir_id: ir4, inst_id: inst596}
// CHECK:STDOUT:     import_ir_inst189: {ir_id: ir4, inst_id: inst603}
// CHECK:STDOUT:     import_ir_inst190: {ir_id: ir4, inst_id: inst608}
// CHECK:STDOUT:     import_ir_inst191: {ir_id: ir4, inst_id: inst609}
// CHECK:STDOUT:     import_ir_inst192: {ir_id: ir4, inst_id: inst610}
// CHECK:STDOUT:     import_ir_inst193: {ir_id: ir4, inst_id: inst615}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst16, name0: inst53}}
// CHECK:STDOUT:     name_scope1:     {inst: inst16, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: inst69}}
// CHECK:STDOUT:     name_scope2:     {inst: inst70, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {name(SelfType): inst73, name4: inst74}}
// CHECK:STDOUT:     name_scope3:     {inst: inst94, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope4:     {inst: inst141, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope5:     {inst: inst145, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6:     {inst: inst149, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7:     {inst: inst153, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope8:     {inst: inst157, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope9:     {inst: inst183, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope10:    {inst: inst187, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope11:    {inst: inst191, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope12:    {inst: inst250, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name1:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name2:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name3:    {name: name3, parent_scope: name_scope1, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name4:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name5:    {name: name4, parent_scope: name_scope2, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name6:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name7:    {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name8:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name9:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name10:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name11:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name12:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name13:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name14:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name15:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name16:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name17:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name18:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name19:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name20:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name21:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name22:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name23:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name24:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name25:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name26:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name27:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name28:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name29:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name30:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name31:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name32:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name33:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name34:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name35:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name36:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name37:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name38:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name39:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name40:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name41:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name42:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name43:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name44:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block13, return_slot_pattern: inst48, body: [inst_block20]}
// CHECK:STDOUT:     function1:       {name: name4, parent_scope: name_scope2, return_slot_pattern: inst85}
// CHECK:STDOUT:     function2:       {name: name4, parent_scope: name_scope3, return_slot_pattern: inst115}
// CHECK:STDOUT:     function3:       {name: name4, parent_scope: name_scope8, return_slot_pattern: inst172}
// CHECK:STDOUT:     function4:       {name: name4, parent_scope: name_scope11, return_slot_pattern: inst215}
// CHECK:STDOUT:     function5:       {name: name4, parent_scope: name_scope12, return_slot_pattern: inst278}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst53, bindings: inst_block16}
// CHECK:STDOUT:     generic1:        {decl: inst79, bindings: inst_block24}
// CHECK:STDOUT:     generic2:        {decl: inst94, bindings: inst_block29}
// CHECK:STDOUT:     generic3:        {decl: inst110, bindings: inst_block36}
// CHECK:STDOUT:     generic4:        {decl: inst157, bindings: inst_block49}
// CHECK:STDOUT:     generic5:        {decl: inst168, bindings: inst_block55}
// CHECK:STDOUT:     generic6:        {decl: inst191, bindings: inst_block62}
// CHECK:STDOUT:     generic7:        {decl: inst210, bindings: inst_block70}
// CHECK:STDOUT:     generic8:        {decl: inst250, bindings: inst_block87}
// CHECK:STDOUT:     generic9:        {decl: inst273, bindings: inst_block95}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: inst_block17}
// CHECK:STDOUT:     specific1:       {generic: generic1, args: inst_block26}
// CHECK:STDOUT:     specific2:       {generic: generic2, args: inst_block31}
// CHECK:STDOUT:     specific3:       {generic: generic2, args: inst_block32}
// CHECK:STDOUT:     specific4:       {generic: generic3, args: inst_block31}
// CHECK:STDOUT:     specific5:       {generic: generic1, args: inst_block42}
// CHECK:STDOUT:     specific6:       {generic: generic1, args: inst_block44}
// CHECK:STDOUT:     specific7:       {generic: generic4, args: inst_block17}
// CHECK:STDOUT:     specific8:       {generic: generic4, args: inst_block51}
// CHECK:STDOUT:     specific9:       {generic: generic5, args: inst_block17}
// CHECK:STDOUT:     specific10:      {generic: generic6, args: inst_block64}
// CHECK:STDOUT:     specific11:      {generic: generic6, args: inst_block66}
// CHECK:STDOUT:     specific12:      {generic: generic7, args: inst_block64}
// CHECK:STDOUT:     specific13:      {generic: generic1, args: inst_block77}
// CHECK:STDOUT:     specific14:      {generic: generic1, args: inst_block79}
// CHECK:STDOUT:     specific15:      {generic: generic1, args: inst_block81}
// CHECK:STDOUT:     specific16:      {generic: generic8, args: inst_block89}
// CHECK:STDOUT:     specific17:      {generic: generic8, args: inst_block91}
// CHECK:STDOUT:     specific18:      {generic: generic9, args: inst_block89}
// CHECK:STDOUT:     specific19:      {generic: generic1, args: inst_block102}
// CHECK:STDOUT:     specific20:      {generic: generic1, args: inst_block104}
// CHECK:STDOUT:     specific21:      {generic: generic1, args: inst_block106}
// CHECK:STDOUT:     specific22:      {generic: generic1, args: inst_block108}
// CHECK:STDOUT:     specific23:      {generic: generic1, args: inst_block115}
// CHECK:STDOUT:     specific24:      {generic: generic1, args: inst_block117}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst36)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst36)}
// CHECK:STDOUT:     'type(inst38)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst40)}
// CHECK:STDOUT:     'type(inst40)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst40)}
// CHECK:STDOUT:     'type(inst54)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst36)}
// CHECK:STDOUT:     'type(symbolic_constant3)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'type(symbolic_constant7)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant11)}
// CHECK:STDOUT:     'type(symbolic_constant11)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant11)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(symbolic_constant4)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant4)}
// CHECK:STDOUT:     'type(symbolic_constant8)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant11)}
// CHECK:STDOUT:     'type(inst71)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst71)}
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     'type(symbolic_constant268)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst36)}
// CHECK:STDOUT:     'type(symbolic_constant271)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst36)}
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst14:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst15:          {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst16:          {kind: Namespace, arg0: name_scope1, arg1: inst15, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst17:          {kind: FacetType, arg0: facet_type0, type: type(TypeType)}
// CHECK:STDOUT:     inst18:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(inst17)}
// CHECK:STDOUT:     inst19:          {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(inst17)}
// CHECK:STDOUT:     inst20:          {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst21:          {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst22:          {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst23:          {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst24:          {kind: SymbolicBindingPattern, arg0: entity_name1, type: type(inst23)}
// CHECK:STDOUT:     inst25:          {kind: NameRef, arg0: name1, arg1: inst20, type: type(TypeType)}
// CHECK:STDOUT:     inst26:          {kind: PointerType, arg0: inst25, type: type(TypeType)}
// CHECK:STDOUT:     inst27:          {kind: PointerType, arg0: inst21, type: type(TypeType)}
// CHECK:STDOUT:     inst28:          {kind: PointerType, arg0: inst22, type: type(TypeType)}
// CHECK:STDOUT:     inst29:          {kind: BindName, arg0: entity_name2, arg1: inst49, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst30:          {kind: PatternType, arg0: inst27, type: type(TypeType)}
// CHECK:STDOUT:     inst31:          {kind: BindingPattern, arg0: entity_name2, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst32:          {kind: PatternType, arg0: inst28, type: type(TypeType)}
// CHECK:STDOUT:     inst33:          {kind: ValueParamPattern, arg0: inst31, arg1: call_param0, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst34:          {kind: NameRef, arg0: name1, arg1: inst20, type: type(TypeType)}
// CHECK:STDOUT:     inst35:          {kind: PointerType, arg0: inst34, type: type(TypeType)}
// CHECK:STDOUT:     inst36:          {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst37:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst36)}
// CHECK:STDOUT:     inst38:          {kind: TupleType, arg0: inst_block9, type: type(TypeType)}
// CHECK:STDOUT:     inst39:          {kind: TupleLiteral, arg0: inst_block8, type: type(inst38)}
// CHECK:STDOUT:     inst40:          {kind: PointerType, arg0: inst38, type: type(TypeType)}
// CHECK:STDOUT:     inst41:          {kind: Converted, arg0: inst37, arg1: inst36, type: type(TypeType)}
// CHECK:STDOUT:     inst42:          {kind: TupleType, arg0: inst_block11, type: type(TypeType)}
// CHECK:STDOUT:     inst43:          {kind: Converted, arg0: inst39, arg1: inst42, type: type(TypeType)}
// CHECK:STDOUT:     inst44:          {kind: TupleType, arg0: inst_block12, type: type(TypeType)}
// CHECK:STDOUT:     inst45:          {kind: PatternType, arg0: inst42, type: type(TypeType)}
// CHECK:STDOUT:     inst46:          {kind: ReturnSlotPattern, arg0: inst43, type: type(symbolic_constant10)}
// CHECK:STDOUT:     inst47:          {kind: PatternType, arg0: inst44, type: type(TypeType)}
// CHECK:STDOUT:     inst48:          {kind: OutParamPattern, arg0: inst46, arg1: call_param1, type: type(symbolic_constant10)}
// CHECK:STDOUT:     inst49:          {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst50:          {kind: SpliceBlock, arg0: inst_block6, arg1: inst26, type: type(TypeType)}
// CHECK:STDOUT:     inst51:          {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst52:          {kind: ReturnSlot, arg0: inst42, arg1: inst51, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst53:          {kind: FunctionDecl, arg0: function0, arg1: inst_block15, type: type(inst54)}
// CHECK:STDOUT:     inst54:          {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst55:          {kind: StructValue, arg0: inst_block_empty, type: type(inst54)}
// CHECK:STDOUT:     inst56:          {kind: PointerType, arg0: inst42, type: type(TypeType)}
// CHECK:STDOUT:     inst57:          {kind: RequireCompleteType, arg0: inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst58:          {kind: RequireCompleteType, arg0: inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst59:          {kind: RequireCompleteType, arg0: inst44, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60:          {kind: RequireCompleteType, arg0: inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst61:          {kind: RequireCompleteType, arg0: inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst62:          {kind: RequireCompleteType, arg0: inst28, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst63:          {kind: NameRef, arg0: name2, arg1: inst29, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst64:          {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst36)}
// CHECK:STDOUT:     inst65:          {kind: TupleLiteral, arg0: inst_block21, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst66:          {kind: RequireCompleteType, arg0: inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst67:          {kind: TupleAccess, arg0: inst52, arg1: element0, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst68:          {kind: RequireCompleteType, arg0: inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst69:          {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name3, type: type(TypeType)}
// CHECK:STDOUT:     inst70:          {kind: InterfaceDecl, arg0: interface0, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst71:          {kind: FacetType, arg0: facet_type1, type: type(TypeType)}
// CHECK:STDOUT:     inst72:          {kind: BindSymbolicName, arg0: entity_name4, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst73:          {kind: ImportRefUnloaded, arg0: import_ir_inst2, arg1: entity_name<none>}
// CHECK:STDOUT:     inst74:          {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst76)}
// CHECK:STDOUT:     inst75:          {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name5}
// CHECK:STDOUT:     inst76:          {kind: AssociatedEntityType, arg0: interface0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst77:          {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(inst80)}
// CHECK:STDOUT:     inst78:          {kind: AssociatedEntity, arg0: element0, arg1: inst77, type: type(inst76)}
// CHECK:STDOUT:     inst79:          {kind: FunctionDecl, arg0: function1, arg1: inst_block_empty, type: type(inst80)}
// CHECK:STDOUT:     inst80:          {kind: FunctionType, arg0: function1, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst81:          {kind: StructValue, arg0: inst_block_empty, type: type(inst80)}
// CHECK:STDOUT:     inst82:          {kind: FacetAccessType, arg0: inst72, type: type(TypeType)}
// CHECK:STDOUT:     inst83:          {kind: PatternType, arg0: inst82, type: type(TypeType)}
// CHECK:STDOUT:     inst84:          {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst85:          {kind: OutParamPattern, arg0: inst84, arg1: call_param1, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst86:          {kind: BindingPattern, arg0: entity_name7, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst87:          {kind: ValueParamPattern, arg0: inst86, arg1: call_param0, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst88:          {kind: ImportRefLoaded, arg0: import_ir_inst11, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst89:          {kind: BindSymbolicName, arg0: entity_name4, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst90:          {kind: FacetAccessType, arg0: inst89, type: type(TypeType)}
// CHECK:STDOUT:     inst91:          {kind: PatternType, arg0: inst90, type: type(TypeType)}
// CHECK:STDOUT:     inst92:          {kind: LookupImplWitness, arg0: inst27, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst93:          {kind: ImportRefUnloaded, arg0: import_ir_inst15, arg1: entity_name<none>}
// CHECK:STDOUT:     inst94:          {kind: ImplDecl, arg0: impl0, arg1: inst_block_empty}
// CHECK:STDOUT:     inst95:          {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst96:          {kind: FacetAccessType, arg0: inst95, type: type(TypeType)}
// CHECK:STDOUT:     inst97:          {kind: ConstType, arg0: inst96, type: type(TypeType)}
// CHECK:STDOUT:     inst98:          {kind: PatternType, arg0: inst71, type: type(TypeType)}
// CHECK:STDOUT:     inst99:          {kind: SymbolicBindingPattern, arg0: entity_name10, type: type(inst98)}
// CHECK:STDOUT:     inst100:         {kind: ImportRefLoaded, arg0: import_ir_inst18, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst101:         {kind: ImportRefLoaded, arg0: import_ir_inst19, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst102:         {kind: ImportRefLoaded, arg0: import_ir_inst20, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst103:         {kind: ImportRefUnloaded, arg0: import_ir_inst21, arg1: entity_name<none>}
// CHECK:STDOUT:     inst104:         {kind: ImplWitnessTable, arg0: inst_block30, arg1: impl0}
// CHECK:STDOUT:     inst105:         {kind: ImplWitness, arg0: inst104, arg1: specific2, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst106:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst107:         {kind: FacetAccessType, arg0: inst106, type: type(TypeType)}
// CHECK:STDOUT:     inst108:         {kind: ConstType, arg0: inst107, type: type(TypeType)}
// CHECK:STDOUT:     inst109:         {kind: ImplWitness, arg0: inst104, arg1: specific3, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst110:         {kind: FunctionDecl, arg0: function2, arg1: inst_block_empty, type: type(symbolic_constant38)}
// CHECK:STDOUT:     inst111:         {kind: FunctionType, arg0: function2, arg1: specific2, type: type(TypeType)}
// CHECK:STDOUT:     inst112:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant38)}
// CHECK:STDOUT:     inst113:         {kind: PatternType, arg0: inst97, type: type(TypeType)}
// CHECK:STDOUT:     inst114:         {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant43)}
// CHECK:STDOUT:     inst115:         {kind: OutParamPattern, arg0: inst114, arg1: call_param1, type: type(symbolic_constant43)}
// CHECK:STDOUT:     inst116:         {kind: BindingPattern, arg0: entity_name13, type: type(symbolic_constant43)}
// CHECK:STDOUT:     inst117:         {kind: ValueParamPattern, arg0: inst116, arg1: call_param0, type: type(symbolic_constant43)}
// CHECK:STDOUT:     inst118:         {kind: ImportRefLoaded, arg0: import_ir_inst32, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst119:         {kind: FunctionType, arg0: function2, arg1: specific3, type: type(TypeType)}
// CHECK:STDOUT:     inst120:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant45)}
// CHECK:STDOUT:     inst121:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst122:         {kind: FacetAccessType, arg0: inst121, type: type(TypeType)}
// CHECK:STDOUT:     inst123:         {kind: ConstType, arg0: inst122, type: type(TypeType)}
// CHECK:STDOUT:     inst124:         {kind: PatternType, arg0: inst123, type: type(TypeType)}
// CHECK:STDOUT:     inst125:         {kind: RequireCompleteType, arg0: inst97, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst126:         {kind: RequireCompleteType, arg0: inst96, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst127:         {kind: LookupImplWitness, arg0: inst95, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst128:         {kind: FacetValue, arg0: inst96, arg1: inst_block41, type: type(inst71)}
// CHECK:STDOUT:     inst129:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst128, type: type(TypeType)}
// CHECK:STDOUT:     inst130:         {kind: ImplWitnessAccess, arg0: inst127, arg1: element0, type: type(symbolic_constant62)}
// CHECK:STDOUT:     inst131:         {kind: SpecificImplFunction, arg0: inst130, arg1: specific5, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst132:         {kind: RequireCompleteType, arg0: inst123, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst133:         {kind: RequireCompleteType, arg0: inst122, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst134:         {kind: LookupImplWitness, arg0: inst121, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst135:         {kind: FacetValue, arg0: inst122, arg1: inst_block43, type: type(inst71)}
// CHECK:STDOUT:     inst136:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst135, type: type(TypeType)}
// CHECK:STDOUT:     inst137:         {kind: ImplWitnessAccess, arg0: inst134, arg1: element0, type: type(symbolic_constant72)}
// CHECK:STDOUT:     inst138:         {kind: SpecificImplFunction, arg0: inst137, arg1: specific6, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst139:         {kind: PatternType, arg0: inst96, type: type(TypeType)}
// CHECK:STDOUT:     inst140:         {kind: ImportRefUnloaded, arg0: import_ir_inst46, arg1: entity_name<none>}
// CHECK:STDOUT:     inst141:         {kind: ImplDecl, arg0: impl1, arg1: inst_block_empty}
// CHECK:STDOUT:     inst142:         {kind: ImportRefLoaded, arg0: import_ir_inst48, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst143:         {kind: ImportRefLoaded, arg0: import_ir_inst49, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst144:         {kind: ImportRefUnloaded, arg0: import_ir_inst50, arg1: entity_name<none>}
// CHECK:STDOUT:     inst145:         {kind: ImplDecl, arg0: impl2, arg1: inst_block_empty}
// CHECK:STDOUT:     inst146:         {kind: ImportRefLoaded, arg0: import_ir_inst52, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst147:         {kind: ImportRefLoaded, arg0: import_ir_inst53, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst148:         {kind: ImportRefUnloaded, arg0: import_ir_inst54, arg1: entity_name<none>}
// CHECK:STDOUT:     inst149:         {kind: ImplDecl, arg0: impl3, arg1: inst_block_empty}
// CHECK:STDOUT:     inst150:         {kind: ImportRefLoaded, arg0: import_ir_inst56, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst151:         {kind: ImportRefLoaded, arg0: import_ir_inst57, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst152:         {kind: ImportRefUnloaded, arg0: import_ir_inst58, arg1: entity_name<none>}
// CHECK:STDOUT:     inst153:         {kind: ImplDecl, arg0: impl4, arg1: inst_block_empty}
// CHECK:STDOUT:     inst154:         {kind: ImportRefLoaded, arg0: import_ir_inst60, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst155:         {kind: ImportRefLoaded, arg0: import_ir_inst61, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst156:         {kind: ImportRefLoaded, arg0: import_ir_inst62, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst157:         {kind: ImplDecl, arg0: impl5, arg1: inst_block_empty}
// CHECK:STDOUT:     inst158:         {kind: SymbolicBindingPattern, arg0: entity_name16, type: type(inst23)}
// CHECK:STDOUT:     inst159:         {kind: ImportRefLoaded, arg0: import_ir_inst65, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst160:         {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst161:         {kind: ImportRefLoaded, arg0: import_ir_inst67, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst162:         {kind: ImportRefUnloaded, arg0: import_ir_inst68, arg1: entity_name<none>}
// CHECK:STDOUT:     inst163:         {kind: ImplWitnessTable, arg0: inst_block50, arg1: impl5}
// CHECK:STDOUT:     inst164:         {kind: ImplWitness, arg0: inst163, arg1: specific7, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst165:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst166:         {kind: PointerType, arg0: inst165, type: type(TypeType)}
// CHECK:STDOUT:     inst167:         {kind: ImplWitness, arg0: inst163, arg1: specific8, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst168:         {kind: FunctionDecl, arg0: function3, arg1: inst_block_empty, type: type(symbolic_constant84)}
// CHECK:STDOUT:     inst169:         {kind: FunctionType, arg0: function3, arg1: specific7, type: type(TypeType)}
// CHECK:STDOUT:     inst170:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant84)}
// CHECK:STDOUT:     inst171:         {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant88)}
// CHECK:STDOUT:     inst172:         {kind: OutParamPattern, arg0: inst171, arg1: call_param1, type: type(symbolic_constant88)}
// CHECK:STDOUT:     inst173:         {kind: BindingPattern, arg0: entity_name19, type: type(symbolic_constant88)}
// CHECK:STDOUT:     inst174:         {kind: ValueParamPattern, arg0: inst173, arg1: call_param0, type: type(symbolic_constant88)}
// CHECK:STDOUT:     inst175:         {kind: ImportRefLoaded, arg0: import_ir_inst78, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst176:         {kind: FunctionType, arg0: function3, arg1: specific8, type: type(TypeType)}
// CHECK:STDOUT:     inst177:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant91)}
// CHECK:STDOUT:     inst178:         {kind: RequireCompleteType, arg0: inst166, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst179:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst180:         {kind: PointerType, arg0: inst179, type: type(TypeType)}
// CHECK:STDOUT:     inst181:         {kind: PatternType, arg0: inst180, type: type(TypeType)}
// CHECK:STDOUT:     inst182:         {kind: ImportRefUnloaded, arg0: import_ir_inst85, arg1: entity_name<none>}
// CHECK:STDOUT:     inst183:         {kind: ImplDecl, arg0: impl6, arg1: inst_block_empty}
// CHECK:STDOUT:     inst184:         {kind: ImportRefLoaded, arg0: import_ir_inst87, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst185:         {kind: ImportRefLoaded, arg0: import_ir_inst88, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst186:         {kind: ImportRefUnloaded, arg0: import_ir_inst89, arg1: entity_name<none>}
// CHECK:STDOUT:     inst187:         {kind: ImplDecl, arg0: impl7, arg1: inst_block_empty}
// CHECK:STDOUT:     inst188:         {kind: ImportRefLoaded, arg0: import_ir_inst91, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst189:         {kind: ImportRefLoaded, arg0: import_ir_inst92, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst190:         {kind: ImportRefUnloaded, arg0: import_ir_inst93, arg1: entity_name<none>}
// CHECK:STDOUT:     inst191:         {kind: ImplDecl, arg0: impl8, arg1: inst_block_empty}
// CHECK:STDOUT:     inst192:         {kind: BindSymbolicName, arg0: entity_name21, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst193:         {kind: FacetAccessType, arg0: inst192, type: type(TypeType)}
// CHECK:STDOUT:     inst194:         {kind: TupleType, arg0: inst_block60, type: type(TypeType)}
// CHECK:STDOUT:     inst195:         {kind: SymbolicBindingPattern, arg0: entity_name22, type: type(inst98)}
// CHECK:STDOUT:     inst196:         {kind: SymbolicBindingPattern, arg0: entity_name23, type: type(inst98)}
// CHECK:STDOUT:     inst197:         {kind: ImportRefLoaded, arg0: import_ir_inst97, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst198:         {kind: ImportRefLoaded, arg0: import_ir_inst98, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst199:         {kind: ImportRefLoaded, arg0: import_ir_inst99, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst200:         {kind: ImportRefLoaded, arg0: import_ir_inst100, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst201:         {kind: ImportRefUnloaded, arg0: import_ir_inst101, arg1: entity_name<none>}
// CHECK:STDOUT:     inst202:         {kind: ImplWitnessTable, arg0: inst_block63, arg1: impl8}
// CHECK:STDOUT:     inst203:         {kind: ImplWitness, arg0: inst202, arg1: specific10, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst204:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst205:         {kind: BindSymbolicName, arg0: entity_name21, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst206:         {kind: FacetAccessType, arg0: inst204, type: type(TypeType)}
// CHECK:STDOUT:     inst207:         {kind: FacetAccessType, arg0: inst205, type: type(TypeType)}
// CHECK:STDOUT:     inst208:         {kind: TupleType, arg0: inst_block65, type: type(TypeType)}
// CHECK:STDOUT:     inst209:         {kind: ImplWitness, arg0: inst202, arg1: specific11, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst210:         {kind: FunctionDecl, arg0: function4, arg1: inst_block_empty, type: type(symbolic_constant117)}
// CHECK:STDOUT:     inst211:         {kind: FunctionType, arg0: function4, arg1: specific10, type: type(TypeType)}
// CHECK:STDOUT:     inst212:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant117)}
// CHECK:STDOUT:     inst213:         {kind: PatternType, arg0: inst194, type: type(TypeType)}
// CHECK:STDOUT:     inst214:         {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant122)}
// CHECK:STDOUT:     inst215:         {kind: OutParamPattern, arg0: inst214, arg1: call_param1, type: type(symbolic_constant122)}
// CHECK:STDOUT:     inst216:         {kind: BindingPattern, arg0: entity_name28, type: type(symbolic_constant122)}
// CHECK:STDOUT:     inst217:         {kind: ValueParamPattern, arg0: inst216, arg1: call_param0, type: type(symbolic_constant122)}
// CHECK:STDOUT:     inst218:         {kind: ImportRefLoaded, arg0: import_ir_inst114, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst219:         {kind: ImportRefLoaded, arg0: import_ir_inst115, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst220:         {kind: FunctionType, arg0: function4, arg1: specific11, type: type(TypeType)}
// CHECK:STDOUT:     inst221:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant124)}
// CHECK:STDOUT:     inst222:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst223:         {kind: FacetAccessType, arg0: inst222, type: type(TypeType)}
// CHECK:STDOUT:     inst224:         {kind: BindSymbolicName, arg0: entity_name21, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst225:         {kind: FacetAccessType, arg0: inst224, type: type(TypeType)}
// CHECK:STDOUT:     inst226:         {kind: TupleType, arg0: inst_block73, type: type(TypeType)}
// CHECK:STDOUT:     inst227:         {kind: PatternType, arg0: inst226, type: type(TypeType)}
// CHECK:STDOUT:     inst228:         {kind: RequireCompleteType, arg0: inst194, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst229:         {kind: RequireCompleteType, arg0: inst193, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst230:         {kind: LookupImplWitness, arg0: inst192, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst231:         {kind: FacetValue, arg0: inst193, arg1: inst_block76, type: type(inst71)}
// CHECK:STDOUT:     inst232:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst231, type: type(TypeType)}
// CHECK:STDOUT:     inst233:         {kind: ImplWitnessAccess, arg0: inst230, arg1: element0, type: type(symbolic_constant151)}
// CHECK:STDOUT:     inst234:         {kind: SpecificImplFunction, arg0: inst233, arg1: specific13, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst235:         {kind: RequireCompleteType, arg0: inst226, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst236:         {kind: RequireCompleteType, arg0: inst223, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst237:         {kind: LookupImplWitness, arg0: inst222, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst238:         {kind: FacetValue, arg0: inst223, arg1: inst_block78, type: type(inst71)}
// CHECK:STDOUT:     inst239:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst238, type: type(TypeType)}
// CHECK:STDOUT:     inst240:         {kind: ImplWitnessAccess, arg0: inst237, arg1: element0, type: type(symbolic_constant161)}
// CHECK:STDOUT:     inst241:         {kind: SpecificImplFunction, arg0: inst240, arg1: specific14, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst242:         {kind: RequireCompleteType, arg0: inst225, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst243:         {kind: LookupImplWitness, arg0: inst224, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst244:         {kind: FacetValue, arg0: inst225, arg1: inst_block80, type: type(inst71)}
// CHECK:STDOUT:     inst245:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst244, type: type(TypeType)}
// CHECK:STDOUT:     inst246:         {kind: ImplWitnessAccess, arg0: inst243, arg1: element0, type: type(symbolic_constant167)}
// CHECK:STDOUT:     inst247:         {kind: SpecificImplFunction, arg0: inst246, arg1: specific15, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst248:         {kind: PatternType, arg0: inst193, type: type(TypeType)}
// CHECK:STDOUT:     inst249:         {kind: ImportRefUnloaded, arg0: import_ir_inst137, arg1: entity_name<none>}
// CHECK:STDOUT:     inst250:         {kind: ImplDecl, arg0: impl9, arg1: inst_block_empty}
// CHECK:STDOUT:     inst251:         {kind: BindSymbolicName, arg0: entity_name31, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst252:         {kind: FacetAccessType, arg0: inst251, type: type(TypeType)}
// CHECK:STDOUT:     inst253:         {kind: TupleType, arg0: inst_block85, type: type(TypeType)}
// CHECK:STDOUT:     inst254:         {kind: SymbolicBindingPattern, arg0: entity_name32, type: type(inst98)}
// CHECK:STDOUT:     inst255:         {kind: SymbolicBindingPattern, arg0: entity_name33, type: type(inst98)}
// CHECK:STDOUT:     inst256:         {kind: SymbolicBindingPattern, arg0: entity_name34, type: type(inst98)}
// CHECK:STDOUT:     inst257:         {kind: ImportRefLoaded, arg0: import_ir_inst142, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst258:         {kind: ImportRefLoaded, arg0: import_ir_inst143, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst259:         {kind: ImportRefLoaded, arg0: import_ir_inst144, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst260:         {kind: ImportRefLoaded, arg0: import_ir_inst145, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst261:         {kind: ImportRefLoaded, arg0: import_ir_inst146, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst262:         {kind: ImportRefUnloaded, arg0: import_ir_inst147, arg1: entity_name<none>}
// CHECK:STDOUT:     inst263:         {kind: ImplWitnessTable, arg0: inst_block88, arg1: impl9}
// CHECK:STDOUT:     inst264:         {kind: ImplWitness, arg0: inst263, arg1: specific16, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst265:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst266:         {kind: BindSymbolicName, arg0: entity_name21, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst267:         {kind: BindSymbolicName, arg0: entity_name31, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst268:         {kind: FacetAccessType, arg0: inst265, type: type(TypeType)}
// CHECK:STDOUT:     inst269:         {kind: FacetAccessType, arg0: inst266, type: type(TypeType)}
// CHECK:STDOUT:     inst270:         {kind: FacetAccessType, arg0: inst267, type: type(TypeType)}
// CHECK:STDOUT:     inst271:         {kind: TupleType, arg0: inst_block90, type: type(TypeType)}
// CHECK:STDOUT:     inst272:         {kind: ImplWitness, arg0: inst263, arg1: specific17, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst273:         {kind: FunctionDecl, arg0: function5, arg1: inst_block_empty, type: type(symbolic_constant194)}
// CHECK:STDOUT:     inst274:         {kind: FunctionType, arg0: function5, arg1: specific16, type: type(TypeType)}
// CHECK:STDOUT:     inst275:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant194)}
// CHECK:STDOUT:     inst276:         {kind: PatternType, arg0: inst253, type: type(TypeType)}
// CHECK:STDOUT:     inst277:         {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant199)}
// CHECK:STDOUT:     inst278:         {kind: OutParamPattern, arg0: inst277, arg1: call_param1, type: type(symbolic_constant199)}
// CHECK:STDOUT:     inst279:         {kind: BindingPattern, arg0: entity_name41, type: type(symbolic_constant199)}
// CHECK:STDOUT:     inst280:         {kind: ValueParamPattern, arg0: inst279, arg1: call_param0, type: type(symbolic_constant199)}
// CHECK:STDOUT:     inst281:         {kind: ImportRefLoaded, arg0: import_ir_inst162, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst282:         {kind: ImportRefLoaded, arg0: import_ir_inst163, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst283:         {kind: ImportRefLoaded, arg0: import_ir_inst164, arg1: entity_name<none>, type: type(inst71)}
// CHECK:STDOUT:     inst284:         {kind: FunctionType, arg0: function5, arg1: specific17, type: type(TypeType)}
// CHECK:STDOUT:     inst285:         {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant201)}
// CHECK:STDOUT:     inst286:         {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst287:         {kind: FacetAccessType, arg0: inst286, type: type(TypeType)}
// CHECK:STDOUT:     inst288:         {kind: BindSymbolicName, arg0: entity_name21, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst289:         {kind: FacetAccessType, arg0: inst288, type: type(TypeType)}
// CHECK:STDOUT:     inst290:         {kind: BindSymbolicName, arg0: entity_name31, arg1: inst<none>, type: type(inst71)}
// CHECK:STDOUT:     inst291:         {kind: FacetAccessType, arg0: inst290, type: type(TypeType)}
// CHECK:STDOUT:     inst292:         {kind: TupleType, arg0: inst_block98, type: type(TypeType)}
// CHECK:STDOUT:     inst293:         {kind: PatternType, arg0: inst292, type: type(TypeType)}
// CHECK:STDOUT:     inst294:         {kind: RequireCompleteType, arg0: inst253, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst295:         {kind: RequireCompleteType, arg0: inst252, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst296:         {kind: LookupImplWitness, arg0: inst251, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst297:         {kind: FacetValue, arg0: inst252, arg1: inst_block101, type: type(inst71)}
// CHECK:STDOUT:     inst298:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst297, type: type(TypeType)}
// CHECK:STDOUT:     inst299:         {kind: ImplWitnessAccess, arg0: inst296, arg1: element0, type: type(symbolic_constant238)}
// CHECK:STDOUT:     inst300:         {kind: SpecificImplFunction, arg0: inst299, arg1: specific19, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst301:         {kind: RequireCompleteType, arg0: inst292, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst302:         {kind: RequireCompleteType, arg0: inst287, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst303:         {kind: LookupImplWitness, arg0: inst286, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst304:         {kind: FacetValue, arg0: inst287, arg1: inst_block103, type: type(inst71)}
// CHECK:STDOUT:     inst305:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst304, type: type(TypeType)}
// CHECK:STDOUT:     inst306:         {kind: ImplWitnessAccess, arg0: inst303, arg1: element0, type: type(symbolic_constant248)}
// CHECK:STDOUT:     inst307:         {kind: SpecificImplFunction, arg0: inst306, arg1: specific20, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst308:         {kind: RequireCompleteType, arg0: inst289, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst309:         {kind: LookupImplWitness, arg0: inst288, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst310:         {kind: FacetValue, arg0: inst289, arg1: inst_block105, type: type(inst71)}
// CHECK:STDOUT:     inst311:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst310, type: type(TypeType)}
// CHECK:STDOUT:     inst312:         {kind: ImplWitnessAccess, arg0: inst309, arg1: element0, type: type(symbolic_constant254)}
// CHECK:STDOUT:     inst313:         {kind: SpecificImplFunction, arg0: inst312, arg1: specific21, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst314:         {kind: RequireCompleteType, arg0: inst291, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst315:         {kind: LookupImplWitness, arg0: inst290, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst316:         {kind: FacetValue, arg0: inst291, arg1: inst_block107, type: type(inst71)}
// CHECK:STDOUT:     inst317:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst316, type: type(TypeType)}
// CHECK:STDOUT:     inst318:         {kind: ImplWitnessAccess, arg0: inst315, arg1: element0, type: type(symbolic_constant260)}
// CHECK:STDOUT:     inst319:         {kind: SpecificImplFunction, arg0: inst318, arg1: specific22, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst320:         {kind: PatternType, arg0: inst252, type: type(TypeType)}
// CHECK:STDOUT:     inst321:         {kind: LookupImplWitness, arg0: inst27, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst322:         {kind: LookupImplWitness, arg0: inst28, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst323:         {kind: FacetValue, arg0: inst27, arg1: inst_block113, type: type(inst71)}
// CHECK:STDOUT:     inst324:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst323, type: type(TypeType)}
// CHECK:STDOUT:     inst325:         {kind: ImplWitnessAccess, arg0: inst321, arg1: element0, type: type(symbolic_constant271)}
// CHECK:STDOUT:     inst326:         {kind: ImplWitnessAccess, arg0: inst321, arg1: element0, type: type(symbolic_constant268)}
// CHECK:STDOUT:     inst327:         {kind: FacetValue, arg0: inst28, arg1: inst_block114, type: type(inst71)}
// CHECK:STDOUT:     inst328:         {kind: FunctionTypeWithSelfType, arg0: inst80, arg1: inst327, type: type(TypeType)}
// CHECK:STDOUT:     inst329:         {kind: ImplWitnessAccess, arg0: inst322, arg1: element0, type: type(symbolic_constant271)}
// CHECK:STDOUT:     inst330:         {kind: BoundMethod, arg0: inst63, arg1: inst325, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst331:         {kind: SpecificImplFunction, arg0: inst325, arg1: specific23, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst332:         {kind: SpecificImplFunction, arg0: inst326, arg1: specific23, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst333:         {kind: SpecificImplFunction, arg0: inst329, arg1: specific24, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst334:         {kind: BoundMethod, arg0: inst63, arg1: inst331, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst335:         {kind: RequireCompleteType, arg0: inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst336:         {kind: Call, arg0: inst334, arg1: inst_block118, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst337:         {kind: InitializeFrom, arg0: inst336, arg1: inst67, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst338:         {kind: TupleAccess, arg0: inst52, arg1: element1, type: type(inst36)}
// CHECK:STDOUT:     inst339:         {kind: TupleInit, arg0: inst_block_empty, arg1: inst338, type: type(inst36)}
// CHECK:STDOUT:     inst340:         {kind: TupleValue, arg0: inst_block_empty, type: type(inst36)}
// CHECK:STDOUT:     inst341:         {kind: Converted, arg0: inst64, arg1: inst339, type: type(inst36)}
// CHECK:STDOUT:     inst342:         {kind: TupleInit, arg0: inst_block119, arg1: inst52, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst343:         {kind: Converted, arg0: inst65, arg1: inst342, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst344:         {kind: ReturnExpr, arg0: inst343, arg1: inst52}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       inst16:          concrete_constant(inst16)
// CHECK:STDOUT:       inst17:          concrete_constant(inst17)
// CHECK:STDOUT:       inst18:          symbolic_constant0
// CHECK:STDOUT:       inst19:          symbolic_constant0
// CHECK:STDOUT:       inst20:          symbolic_constant2
// CHECK:STDOUT:       inst21:          symbolic_constant1
// CHECK:STDOUT:       inst22:          symbolic_constant2
// CHECK:STDOUT:       inst23:          concrete_constant(inst23)
// CHECK:STDOUT:       inst24:          concrete_constant(inst24)
// CHECK:STDOUT:       inst25:          symbolic_constant2
// CHECK:STDOUT:       inst26:          symbolic_constant4
// CHECK:STDOUT:       inst27:          symbolic_constant3
// CHECK:STDOUT:       inst28:          symbolic_constant4
// CHECK:STDOUT:       inst30:          symbolic_constant5
// CHECK:STDOUT:       inst31:          concrete_constant(inst31)
// CHECK:STDOUT:       inst32:          symbolic_constant6
// CHECK:STDOUT:       inst33:          concrete_constant(inst33)
// CHECK:STDOUT:       inst34:          symbolic_constant2
// CHECK:STDOUT:       inst35:          symbolic_constant4
// CHECK:STDOUT:       inst36:          concrete_constant(inst36)
// CHECK:STDOUT:       inst38:          concrete_constant(inst38)
// CHECK:STDOUT:       inst40:          concrete_constant(inst40)
// CHECK:STDOUT:       inst41:          concrete_constant(inst36)
// CHECK:STDOUT:       inst42:          symbolic_constant7
// CHECK:STDOUT:       inst43:          symbolic_constant8
// CHECK:STDOUT:       inst44:          symbolic_constant8
// CHECK:STDOUT:       inst45:          symbolic_constant9
// CHECK:STDOUT:       inst46:          concrete_constant(inst46)
// CHECK:STDOUT:       inst47:          symbolic_constant10
// CHECK:STDOUT:       inst48:          concrete_constant(inst48)
// CHECK:STDOUT:       inst50:          symbolic_constant4
// CHECK:STDOUT:       inst53:          concrete_constant(inst55)
// CHECK:STDOUT:       inst54:          concrete_constant(inst54)
// CHECK:STDOUT:       inst55:          concrete_constant(inst55)
// CHECK:STDOUT:       inst56:          symbolic_constant11
// CHECK:STDOUT:       inst57:          symbolic_constant13
// CHECK:STDOUT:       inst58:          symbolic_constant12
// CHECK:STDOUT:       inst59:          symbolic_constant13
// CHECK:STDOUT:       inst60:          symbolic_constant15
// CHECK:STDOUT:       inst61:          symbolic_constant14
// CHECK:STDOUT:       inst62:          symbolic_constant15
// CHECK:STDOUT:       inst66:          symbolic_constant13
// CHECK:STDOUT:       inst68:          symbolic_constant15
// CHECK:STDOUT:       inst69:          concrete_constant(inst71)
// CHECK:STDOUT:       inst70:          concrete_constant(inst71)
// CHECK:STDOUT:       inst71:          concrete_constant(inst71)
// CHECK:STDOUT:       inst72:          symbolic_constant16
// CHECK:STDOUT:       inst73:          constant<none>
// CHECK:STDOUT:       inst74:          concrete_constant(inst78)
// CHECK:STDOUT:       inst75:          constant<none>
// CHECK:STDOUT:       inst76:          concrete_constant(inst76)
// CHECK:STDOUT:       inst77:          concrete_constant(inst81)
// CHECK:STDOUT:       inst78:          concrete_constant(inst78)
// CHECK:STDOUT:       inst79:          concrete_constant(inst81)
// CHECK:STDOUT:       inst80:          concrete_constant(inst80)
// CHECK:STDOUT:       inst81:          concrete_constant(inst81)
// CHECK:STDOUT:       inst82:          symbolic_constant17
// CHECK:STDOUT:       inst83:          symbolic_constant18
// CHECK:STDOUT:       inst84:          concrete_constant(inst84)
// CHECK:STDOUT:       inst85:          concrete_constant(inst85)
// CHECK:STDOUT:       inst86:          concrete_constant(inst86)
// CHECK:STDOUT:       inst87:          concrete_constant(inst87)
// CHECK:STDOUT:       inst88:          symbolic_constant16
// CHECK:STDOUT:       inst89:          symbolic_constant22
// CHECK:STDOUT:       inst90:          symbolic_constant23
// CHECK:STDOUT:       inst91:          symbolic_constant24
// CHECK:STDOUT:       inst92:          symbolic_constant266
// CHECK:STDOUT:       inst93:          constant<none>
// CHECK:STDOUT:       inst94:          concrete_constant(inst94)
// CHECK:STDOUT:       inst95:          symbolic_constant25
// CHECK:STDOUT:       inst96:          symbolic_constant26
// CHECK:STDOUT:       inst97:          symbolic_constant27
// CHECK:STDOUT:       inst98:          concrete_constant(inst98)
// CHECK:STDOUT:       inst99:          concrete_constant(inst99)
// CHECK:STDOUT:       inst100:         symbolic_constant29
// CHECK:STDOUT:       inst101:         symbolic_constant28
// CHECK:STDOUT:       inst102:         concrete_constant(inst71)
// CHECK:STDOUT:       inst103:         constant<none>
// CHECK:STDOUT:       inst104:         concrete_constant(inst104)
// CHECK:STDOUT:       inst105:         symbolic_constant32
// CHECK:STDOUT:       inst106:         symbolic_constant34
// CHECK:STDOUT:       inst107:         symbolic_constant35
// CHECK:STDOUT:       inst108:         symbolic_constant36
// CHECK:STDOUT:       inst109:         symbolic_constant37
// CHECK:STDOUT:       inst110:         symbolic_constant40
// CHECK:STDOUT:       inst111:         symbolic_constant38
// CHECK:STDOUT:       inst112:         symbolic_constant39
// CHECK:STDOUT:       inst113:         symbolic_constant42
// CHECK:STDOUT:       inst114:         concrete_constant(inst114)
// CHECK:STDOUT:       inst115:         concrete_constant(inst115)
// CHECK:STDOUT:       inst116:         concrete_constant(inst116)
// CHECK:STDOUT:       inst117:         concrete_constant(inst117)
// CHECK:STDOUT:       inst118:         symbolic_constant29
// CHECK:STDOUT:       inst119:         symbolic_constant45
// CHECK:STDOUT:       inst120:         symbolic_constant46
// CHECK:STDOUT:       inst121:         symbolic_constant50
// CHECK:STDOUT:       inst122:         symbolic_constant51
// CHECK:STDOUT:       inst123:         symbolic_constant52
// CHECK:STDOUT:       inst124:         symbolic_constant53
// CHECK:STDOUT:       inst125:         symbolic_constant54
// CHECK:STDOUT:       inst126:         symbolic_constant56
// CHECK:STDOUT:       inst127:         symbolic_constant58
// CHECK:STDOUT:       inst128:         symbolic_constant60
// CHECK:STDOUT:       inst129:         symbolic_constant62
// CHECK:STDOUT:       inst130:         symbolic_constant64
// CHECK:STDOUT:       inst131:         symbolic_constant66
// CHECK:STDOUT:       inst132:         symbolic_constant68
// CHECK:STDOUT:       inst133:         symbolic_constant69
// CHECK:STDOUT:       inst134:         symbolic_constant70
// CHECK:STDOUT:       inst135:         symbolic_constant71
// CHECK:STDOUT:       inst136:         symbolic_constant72
// CHECK:STDOUT:       inst137:         symbolic_constant73
// CHECK:STDOUT:       inst138:         symbolic_constant74
// CHECK:STDOUT:       inst139:         symbolic_constant75
// CHECK:STDOUT:       inst140:         constant<none>
// CHECK:STDOUT:       inst141:         concrete_constant(inst141)
// CHECK:STDOUT:       inst142:         concrete_constant(inst(BoolType))
// CHECK:STDOUT:       inst143:         concrete_constant(inst71)
// CHECK:STDOUT:       inst144:         constant<none>
// CHECK:STDOUT:       inst145:         concrete_constant(inst145)
// CHECK:STDOUT:       inst146:         concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       inst147:         concrete_constant(inst71)
// CHECK:STDOUT:       inst148:         constant<none>
// CHECK:STDOUT:       inst149:         concrete_constant(inst149)
// CHECK:STDOUT:       inst150:         concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       inst151:         concrete_constant(inst71)
// CHECK:STDOUT:       inst152:         constant<none>
// CHECK:STDOUT:       inst153:         concrete_constant(inst153)
// CHECK:STDOUT:       inst154:         concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       inst155:         concrete_constant(inst71)
// CHECK:STDOUT:       inst156:         symbolic_constant264
// CHECK:STDOUT:       inst157:         concrete_constant(inst157)
// CHECK:STDOUT:       inst158:         concrete_constant(inst158)
// CHECK:STDOUT:       inst159:         symbolic_constant77
// CHECK:STDOUT:       inst160:         symbolic_constant76
// CHECK:STDOUT:       inst161:         concrete_constant(inst71)
// CHECK:STDOUT:       inst162:         constant<none>
// CHECK:STDOUT:       inst163:         concrete_constant(inst163)
// CHECK:STDOUT:       inst164:         symbolic_constant79
// CHECK:STDOUT:       inst165:         symbolic_constant81
// CHECK:STDOUT:       inst166:         symbolic_constant82
// CHECK:STDOUT:       inst167:         symbolic_constant83
// CHECK:STDOUT:       inst168:         symbolic_constant86
// CHECK:STDOUT:       inst169:         symbolic_constant84
// CHECK:STDOUT:       inst170:         symbolic_constant85
// CHECK:STDOUT:       inst171:         concrete_constant(inst171)
// CHECK:STDOUT:       inst172:         concrete_constant(inst172)
// CHECK:STDOUT:       inst173:         concrete_constant(inst173)
// CHECK:STDOUT:       inst174:         concrete_constant(inst174)
// CHECK:STDOUT:       inst175:         symbolic_constant77
// CHECK:STDOUT:       inst176:         symbolic_constant91
// CHECK:STDOUT:       inst177:         symbolic_constant92
// CHECK:STDOUT:       inst178:         symbolic_constant93
// CHECK:STDOUT:       inst179:         symbolic_constant96
// CHECK:STDOUT:       inst180:         symbolic_constant97
// CHECK:STDOUT:       inst181:         symbolic_constant98
// CHECK:STDOUT:       inst182:         constant<none>
// CHECK:STDOUT:       inst183:         concrete_constant(inst183)
// CHECK:STDOUT:       inst184:         concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst185:         concrete_constant(inst71)
// CHECK:STDOUT:       inst186:         constant<none>
// CHECK:STDOUT:       inst187:         concrete_constant(inst187)
// CHECK:STDOUT:       inst188:         concrete_constant(inst36)
// CHECK:STDOUT:       inst189:         concrete_constant(inst71)
// CHECK:STDOUT:       inst190:         constant<none>
// CHECK:STDOUT:       inst191:         concrete_constant(inst191)
// CHECK:STDOUT:       inst192:         symbolic_constant99
// CHECK:STDOUT:       inst193:         symbolic_constant100
// CHECK:STDOUT:       inst194:         symbolic_constant101
// CHECK:STDOUT:       inst195:         concrete_constant(inst195)
// CHECK:STDOUT:       inst196:         concrete_constant(inst196)
// CHECK:STDOUT:       inst197:         symbolic_constant103
// CHECK:STDOUT:       inst198:         symbolic_constant104
// CHECK:STDOUT:       inst199:         symbolic_constant102
// CHECK:STDOUT:       inst200:         concrete_constant(inst71)
// CHECK:STDOUT:       inst201:         constant<none>
// CHECK:STDOUT:       inst202:         concrete_constant(inst202)
// CHECK:STDOUT:       inst203:         symbolic_constant109
// CHECK:STDOUT:       inst204:         symbolic_constant111
// CHECK:STDOUT:       inst205:         symbolic_constant112
// CHECK:STDOUT:       inst206:         symbolic_constant113
// CHECK:STDOUT:       inst207:         symbolic_constant114
// CHECK:STDOUT:       inst208:         symbolic_constant115
// CHECK:STDOUT:       inst209:         symbolic_constant116
// CHECK:STDOUT:       inst210:         symbolic_constant119
// CHECK:STDOUT:       inst211:         symbolic_constant117
// CHECK:STDOUT:       inst212:         symbolic_constant118
// CHECK:STDOUT:       inst213:         symbolic_constant121
// CHECK:STDOUT:       inst214:         concrete_constant(inst214)
// CHECK:STDOUT:       inst215:         concrete_constant(inst215)
// CHECK:STDOUT:       inst216:         concrete_constant(inst216)
// CHECK:STDOUT:       inst217:         concrete_constant(inst217)
// CHECK:STDOUT:       inst218:         symbolic_constant103
// CHECK:STDOUT:       inst219:         symbolic_constant104
// CHECK:STDOUT:       inst220:         symbolic_constant124
// CHECK:STDOUT:       inst221:         symbolic_constant125
// CHECK:STDOUT:       inst222:         symbolic_constant131
// CHECK:STDOUT:       inst223:         symbolic_constant132
// CHECK:STDOUT:       inst224:         symbolic_constant133
// CHECK:STDOUT:       inst225:         symbolic_constant134
// CHECK:STDOUT:       inst226:         symbolic_constant135
// CHECK:STDOUT:       inst227:         symbolic_constant136
// CHECK:STDOUT:       inst228:         symbolic_constant137
// CHECK:STDOUT:       inst229:         symbolic_constant145
// CHECK:STDOUT:       inst230:         symbolic_constant147
// CHECK:STDOUT:       inst231:         symbolic_constant149
// CHECK:STDOUT:       inst232:         symbolic_constant151
// CHECK:STDOUT:       inst233:         symbolic_constant153
// CHECK:STDOUT:       inst234:         symbolic_constant155
// CHECK:STDOUT:       inst235:         symbolic_constant157
// CHECK:STDOUT:       inst236:         symbolic_constant158
// CHECK:STDOUT:       inst237:         symbolic_constant159
// CHECK:STDOUT:       inst238:         symbolic_constant160
// CHECK:STDOUT:       inst239:         symbolic_constant161
// CHECK:STDOUT:       inst240:         symbolic_constant162
// CHECK:STDOUT:       inst241:         symbolic_constant163
// CHECK:STDOUT:       inst242:         symbolic_constant164
// CHECK:STDOUT:       inst243:         symbolic_constant165
// CHECK:STDOUT:       inst244:         symbolic_constant166
// CHECK:STDOUT:       inst245:         symbolic_constant167
// CHECK:STDOUT:       inst246:         symbolic_constant168
// CHECK:STDOUT:       inst247:         symbolic_constant169
// CHECK:STDOUT:       inst248:         symbolic_constant170
// CHECK:STDOUT:       inst249:         constant<none>
// CHECK:STDOUT:       inst250:         concrete_constant(inst250)
// CHECK:STDOUT:       inst251:         symbolic_constant171
// CHECK:STDOUT:       inst252:         symbolic_constant172
// CHECK:STDOUT:       inst253:         symbolic_constant173
// CHECK:STDOUT:       inst254:         concrete_constant(inst254)
// CHECK:STDOUT:       inst255:         concrete_constant(inst255)
// CHECK:STDOUT:       inst256:         concrete_constant(inst256)
// CHECK:STDOUT:       inst257:         symbolic_constant175
// CHECK:STDOUT:       inst258:         symbolic_constant176
// CHECK:STDOUT:       inst259:         symbolic_constant177
// CHECK:STDOUT:       inst260:         symbolic_constant174
// CHECK:STDOUT:       inst261:         concrete_constant(inst71)
// CHECK:STDOUT:       inst262:         constant<none>
// CHECK:STDOUT:       inst263:         concrete_constant(inst263)
// CHECK:STDOUT:       inst264:         symbolic_constant184
// CHECK:STDOUT:       inst265:         symbolic_constant186
// CHECK:STDOUT:       inst266:         symbolic_constant187
// CHECK:STDOUT:       inst267:         symbolic_constant188
// CHECK:STDOUT:       inst268:         symbolic_constant189
// CHECK:STDOUT:       inst269:         symbolic_constant190
// CHECK:STDOUT:       inst270:         symbolic_constant191
// CHECK:STDOUT:       inst271:         symbolic_constant192
// CHECK:STDOUT:       inst272:         symbolic_constant193
// CHECK:STDOUT:       inst273:         symbolic_constant196
// CHECK:STDOUT:       inst274:         symbolic_constant194
// CHECK:STDOUT:       inst275:         symbolic_constant195
// CHECK:STDOUT:       inst276:         symbolic_constant198
// CHECK:STDOUT:       inst277:         concrete_constant(inst277)
// CHECK:STDOUT:       inst278:         concrete_constant(inst278)
// CHECK:STDOUT:       inst279:         concrete_constant(inst279)
// CHECK:STDOUT:       inst280:         concrete_constant(inst280)
// CHECK:STDOUT:       inst281:         symbolic_constant175
// CHECK:STDOUT:       inst282:         symbolic_constant176
// CHECK:STDOUT:       inst283:         symbolic_constant177
// CHECK:STDOUT:       inst284:         symbolic_constant201
// CHECK:STDOUT:       inst285:         symbolic_constant202
// CHECK:STDOUT:       inst286:         symbolic_constant210
// CHECK:STDOUT:       inst287:         symbolic_constant211
// CHECK:STDOUT:       inst288:         symbolic_constant212
// CHECK:STDOUT:       inst289:         symbolic_constant213
// CHECK:STDOUT:       inst290:         symbolic_constant214
// CHECK:STDOUT:       inst291:         symbolic_constant215
// CHECK:STDOUT:       inst292:         symbolic_constant216
// CHECK:STDOUT:       inst293:         symbolic_constant217
// CHECK:STDOUT:       inst294:         symbolic_constant218
// CHECK:STDOUT:       inst295:         symbolic_constant232
// CHECK:STDOUT:       inst296:         symbolic_constant234
// CHECK:STDOUT:       inst297:         symbolic_constant236
// CHECK:STDOUT:       inst298:         symbolic_constant238
// CHECK:STDOUT:       inst299:         symbolic_constant240
// CHECK:STDOUT:       inst300:         symbolic_constant242
// CHECK:STDOUT:       inst301:         symbolic_constant244
// CHECK:STDOUT:       inst302:         symbolic_constant245
// CHECK:STDOUT:       inst303:         symbolic_constant246
// CHECK:STDOUT:       inst304:         symbolic_constant247
// CHECK:STDOUT:       inst305:         symbolic_constant248
// CHECK:STDOUT:       inst306:         symbolic_constant249
// CHECK:STDOUT:       inst307:         symbolic_constant250
// CHECK:STDOUT:       inst308:         symbolic_constant251
// CHECK:STDOUT:       inst309:         symbolic_constant252
// CHECK:STDOUT:       inst310:         symbolic_constant253
// CHECK:STDOUT:       inst311:         symbolic_constant254
// CHECK:STDOUT:       inst312:         symbolic_constant255
// CHECK:STDOUT:       inst313:         symbolic_constant256
// CHECK:STDOUT:       inst314:         symbolic_constant257
// CHECK:STDOUT:       inst315:         symbolic_constant258
// CHECK:STDOUT:       inst316:         symbolic_constant259
// CHECK:STDOUT:       inst317:         symbolic_constant260
// CHECK:STDOUT:       inst318:         symbolic_constant261
// CHECK:STDOUT:       inst319:         symbolic_constant262
// CHECK:STDOUT:       inst320:         symbolic_constant263
// CHECK:STDOUT:       inst321:         symbolic_constant265
// CHECK:STDOUT:       inst322:         symbolic_constant266
// CHECK:STDOUT:       inst323:         symbolic_constant267
// CHECK:STDOUT:       inst324:         symbolic_constant268
// CHECK:STDOUT:       inst325:         symbolic_constant272
// CHECK:STDOUT:       inst326:         symbolic_constant269
// CHECK:STDOUT:       inst327:         symbolic_constant270
// CHECK:STDOUT:       inst328:         symbolic_constant271
// CHECK:STDOUT:       inst329:         symbolic_constant272
// CHECK:STDOUT:       inst331:         symbolic_constant274
// CHECK:STDOUT:       inst332:         symbolic_constant273
// CHECK:STDOUT:       inst333:         symbolic_constant274
// CHECK:STDOUT:       inst335:         symbolic_constant15
// CHECK:STDOUT:       inst339:         concrete_constant(inst340)
// CHECK:STDOUT:       inst340:         concrete_constant(inst340)
// CHECK:STDOUT:       inst341:         concrete_constant(inst340)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: inst19, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant1: {inst: inst21, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: inst21, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: inst27, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: inst27, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: inst30, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: inst30, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: inst42, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: inst42, generic: generic0, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: inst45, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant10: {inst: inst45, generic: generic0, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant11: {inst: inst56, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant12: {inst: inst58, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant13: {inst: inst58, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant14: {inst: inst61, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant15: {inst: inst61, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant16: {inst: inst72, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant17: {inst: inst82, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant18: {inst: inst83, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant19: {inst: inst83, generic: generic1, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant20: {inst: inst72, generic: generic1, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant21: {inst: inst82, generic: generic1, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant22: {inst: inst72, generic: generic1, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant23: {inst: inst82, generic: generic1, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant24: {inst: inst83, generic: generic1, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant25: {inst: inst95, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant26: {inst: inst96, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant27: {inst: inst97, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant28: {inst: inst97, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant29: {inst: inst95, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant30: {inst: inst95, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant31: {inst: inst96, generic: generic2, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant32: {inst: inst105, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant33: {inst: inst105, generic: generic2, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant34: {inst: inst95, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant35: {inst: inst96, generic: generic2, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant36: {inst: inst97, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant37: {inst: inst105, generic: generic2, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant38: {inst: inst111, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant39: {inst: inst112, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant40: {inst: inst112, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant41: {inst: inst111, generic: generic2, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant42: {inst: inst113, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant43: {inst: inst113, generic: generic3, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant44: {inst: inst112, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant45: {inst: inst111, generic: generic2, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant46: {inst: inst112, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant47: {inst: inst95, generic: generic3, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant48: {inst: inst96, generic: generic3, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant49: {inst: inst97, generic: generic3, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant50: {inst: inst95, generic: generic3, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant51: {inst: inst96, generic: generic3, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant52: {inst: inst97, generic: generic3, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant53: {inst: inst113, generic: generic3, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant54: {inst: inst125, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant55: {inst: inst125, generic: generic3, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant56: {inst: inst126, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant57: {inst: inst126, generic: generic3, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant58: {inst: inst127, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant59: {inst: inst127, generic: generic3, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60: {inst: inst128, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant61: {inst: inst128, generic: generic3, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant62: {inst: inst129, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant63: {inst: inst129, generic: generic3, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant64: {inst: inst130, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant65: {inst: inst130, generic: generic3, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant66: {inst: inst131, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant67: {inst: inst131, generic: generic3, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant68: {inst: inst125, generic: generic3, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant69: {inst: inst126, generic: generic3, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant70: {inst: inst127, generic: generic3, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant71: {inst: inst128, generic: generic3, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant72: {inst: inst129, generic: generic3, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant73: {inst: inst130, generic: generic3, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant74: {inst: inst131, generic: generic3, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant75: {inst: inst139, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant76: {inst: inst27, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant77: {inst: inst21, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant78: {inst: inst21, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant79: {inst: inst164, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant80: {inst: inst164, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant81: {inst: inst21, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant82: {inst: inst27, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant83: {inst: inst164, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant84: {inst: inst169, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant85: {inst: inst170, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant86: {inst: inst170, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant87: {inst: inst169, generic: generic4, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant88: {inst: inst30, generic: generic5, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant89: {inst: inst170, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant90: {inst: inst61, generic: generic4, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant91: {inst: inst169, generic: generic4, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant92: {inst: inst170, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant93: {inst: inst61, generic: generic4, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant94: {inst: inst21, generic: generic5, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant95: {inst: inst27, generic: generic5, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant96: {inst: inst21, generic: generic5, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant97: {inst: inst27, generic: generic5, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant98: {inst: inst30, generic: generic5, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant99: {inst: inst192, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant100: {inst: inst193, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant101: {inst: inst194, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant102: {inst: inst194, generic: generic6, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant103: {inst: inst95, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant104: {inst: inst192, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant105: {inst: inst95, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant106: {inst: inst192, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant107: {inst: inst96, generic: generic6, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant108: {inst: inst193, generic: generic6, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant109: {inst: inst203, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant110: {inst: inst203, generic: generic6, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant111: {inst: inst95, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant112: {inst: inst192, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant113: {inst: inst96, generic: generic6, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant114: {inst: inst193, generic: generic6, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant115: {inst: inst194, generic: generic6, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant116: {inst: inst203, generic: generic6, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant117: {inst: inst211, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant118: {inst: inst212, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant119: {inst: inst212, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant120: {inst: inst211, generic: generic6, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant121: {inst: inst213, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant122: {inst: inst213, generic: generic7, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant123: {inst: inst212, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant124: {inst: inst211, generic: generic6, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant125: {inst: inst212, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant126: {inst: inst95, generic: generic7, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant127: {inst: inst96, generic: generic7, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant128: {inst: inst192, generic: generic7, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant129: {inst: inst193, generic: generic7, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant130: {inst: inst194, generic: generic7, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant131: {inst: inst95, generic: generic7, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant132: {inst: inst96, generic: generic7, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant133: {inst: inst192, generic: generic7, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant134: {inst: inst193, generic: generic7, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant135: {inst: inst194, generic: generic7, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant136: {inst: inst213, generic: generic7, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant137: {inst: inst228, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant138: {inst: inst228, generic: generic7, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant139: {inst: inst126, generic: generic7, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant140: {inst: inst127, generic: generic7, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant141: {inst: inst128, generic: generic7, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant142: {inst: inst129, generic: generic7, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant143: {inst: inst130, generic: generic7, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant144: {inst: inst131, generic: generic7, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant145: {inst: inst229, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant146: {inst: inst229, generic: generic7, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant147: {inst: inst230, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant148: {inst: inst230, generic: generic7, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant149: {inst: inst231, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant150: {inst: inst231, generic: generic7, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant151: {inst: inst232, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant152: {inst: inst232, generic: generic7, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant153: {inst: inst233, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant154: {inst: inst233, generic: generic7, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant155: {inst: inst234, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant156: {inst: inst234, generic: generic7, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant157: {inst: inst228, generic: generic7, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant158: {inst: inst126, generic: generic7, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant159: {inst: inst127, generic: generic7, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant160: {inst: inst128, generic: generic7, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant161: {inst: inst129, generic: generic7, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant162: {inst: inst130, generic: generic7, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant163: {inst: inst131, generic: generic7, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant164: {inst: inst229, generic: generic7, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant165: {inst: inst230, generic: generic7, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant166: {inst: inst231, generic: generic7, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant167: {inst: inst232, generic: generic7, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant168: {inst: inst233, generic: generic7, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant169: {inst: inst234, generic: generic7, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant170: {inst: inst248, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant171: {inst: inst251, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant172: {inst: inst252, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant173: {inst: inst253, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant174: {inst: inst253, generic: generic8, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant175: {inst: inst95, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant176: {inst: inst192, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant177: {inst: inst251, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant178: {inst: inst95, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant179: {inst: inst192, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant180: {inst: inst251, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant181: {inst: inst96, generic: generic8, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant182: {inst: inst193, generic: generic8, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant183: {inst: inst252, generic: generic8, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant184: {inst: inst264, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant185: {inst: inst264, generic: generic8, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant186: {inst: inst95, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant187: {inst: inst192, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant188: {inst: inst251, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant189: {inst: inst96, generic: generic8, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant190: {inst: inst193, generic: generic8, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant191: {inst: inst252, generic: generic8, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant192: {inst: inst253, generic: generic8, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant193: {inst: inst264, generic: generic8, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant194: {inst: inst274, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant195: {inst: inst275, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant196: {inst: inst275, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant197: {inst: inst274, generic: generic8, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant198: {inst: inst276, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant199: {inst: inst276, generic: generic9, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant200: {inst: inst275, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant201: {inst: inst274, generic: generic8, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant202: {inst: inst275, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant203: {inst: inst95, generic: generic9, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant204: {inst: inst96, generic: generic9, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant205: {inst: inst192, generic: generic9, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant206: {inst: inst193, generic: generic9, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant207: {inst: inst251, generic: generic9, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant208: {inst: inst252, generic: generic9, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant209: {inst: inst253, generic: generic9, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant210: {inst: inst95, generic: generic9, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant211: {inst: inst96, generic: generic9, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant212: {inst: inst192, generic: generic9, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant213: {inst: inst193, generic: generic9, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant214: {inst: inst251, generic: generic9, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant215: {inst: inst252, generic: generic9, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant216: {inst: inst253, generic: generic9, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant217: {inst: inst276, generic: generic9, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant218: {inst: inst294, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant219: {inst: inst294, generic: generic9, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant220: {inst: inst126, generic: generic9, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant221: {inst: inst127, generic: generic9, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant222: {inst: inst128, generic: generic9, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant223: {inst: inst129, generic: generic9, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant224: {inst: inst130, generic: generic9, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant225: {inst: inst131, generic: generic9, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant226: {inst: inst229, generic: generic9, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant227: {inst: inst230, generic: generic9, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant228: {inst: inst231, generic: generic9, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant229: {inst: inst232, generic: generic9, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant230: {inst: inst233, generic: generic9, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant231: {inst: inst234, generic: generic9, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant232: {inst: inst295, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant233: {inst: inst295, generic: generic9, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant234: {inst: inst296, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant235: {inst: inst296, generic: generic9, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant236: {inst: inst297, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant237: {inst: inst297, generic: generic9, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant238: {inst: inst298, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant239: {inst: inst298, generic: generic9, index: generic_inst_in_def16, kind: checked}
// CHECK:STDOUT:       symbolic_constant240: {inst: inst299, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant241: {inst: inst299, generic: generic9, index: generic_inst_in_def17, kind: checked}
// CHECK:STDOUT:       symbolic_constant242: {inst: inst300, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant243: {inst: inst300, generic: generic9, index: generic_inst_in_def18, kind: checked}
// CHECK:STDOUT:       symbolic_constant244: {inst: inst294, generic: generic9, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant245: {inst: inst126, generic: generic9, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant246: {inst: inst127, generic: generic9, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant247: {inst: inst128, generic: generic9, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant248: {inst: inst129, generic: generic9, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant249: {inst: inst130, generic: generic9, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant250: {inst: inst131, generic: generic9, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant251: {inst: inst229, generic: generic9, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant252: {inst: inst230, generic: generic9, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant253: {inst: inst231, generic: generic9, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant254: {inst: inst232, generic: generic9, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant255: {inst: inst233, generic: generic9, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant256: {inst: inst234, generic: generic9, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant257: {inst: inst295, generic: generic9, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant258: {inst: inst296, generic: generic9, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant259: {inst: inst297, generic: generic9, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant260: {inst: inst298, generic: generic9, index: generic_inst_in_def16, kind: checked}
// CHECK:STDOUT:       symbolic_constant261: {inst: inst299, generic: generic9, index: generic_inst_in_def17, kind: checked}
// CHECK:STDOUT:       symbolic_constant262: {inst: inst300, generic: generic9, index: generic_inst_in_def18, kind: checked}
// CHECK:STDOUT:       symbolic_constant263: {inst: inst320, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant264: {inst: inst164, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant265: {inst: inst321, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant266: {inst: inst321, generic: generic0, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant267: {inst: inst323, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant268: {inst: inst324, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant269: {inst: inst326, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant270: {inst: inst323, generic: generic0, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant271: {inst: inst324, generic: generic0, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant272: {inst: inst326, generic: generic0, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant273: {inst: inst332, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant274: {inst: inst332, generic: generic0, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst53
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst16
// CHECK:STDOUT:       1:               inst69
// CHECK:STDOUT:       2:               inst70
// CHECK:STDOUT:       3:               inst73
// CHECK:STDOUT:       4:               inst74
// CHECK:STDOUT:       5:               inst75
// CHECK:STDOUT:       6:               inst77
// CHECK:STDOUT:       7:               inst79
// CHECK:STDOUT:       8:               inst84
// CHECK:STDOUT:       9:               inst85
// CHECK:STDOUT:       10:              inst86
// CHECK:STDOUT:       11:              inst87
// CHECK:STDOUT:       12:              inst88
// CHECK:STDOUT:       13:              inst93
// CHECK:STDOUT:       14:              inst94
// CHECK:STDOUT:       15:              inst99
// CHECK:STDOUT:       16:              inst100
// CHECK:STDOUT:       17:              inst101
// CHECK:STDOUT:       18:              inst102
// CHECK:STDOUT:       19:              inst103
// CHECK:STDOUT:       20:              inst104
// CHECK:STDOUT:       21:              inst110
// CHECK:STDOUT:       22:              inst114
// CHECK:STDOUT:       23:              inst115
// CHECK:STDOUT:       24:              inst116
// CHECK:STDOUT:       25:              inst117
// CHECK:STDOUT:       26:              inst118
// CHECK:STDOUT:       27:              inst140
// CHECK:STDOUT:       28:              inst141
// CHECK:STDOUT:       29:              inst142
// CHECK:STDOUT:       30:              inst143
// CHECK:STDOUT:       31:              inst144
// CHECK:STDOUT:       32:              inst145
// CHECK:STDOUT:       33:              inst146
// CHECK:STDOUT:       34:              inst147
// CHECK:STDOUT:       35:              inst148
// CHECK:STDOUT:       36:              inst149
// CHECK:STDOUT:       37:              inst150
// CHECK:STDOUT:       38:              inst151
// CHECK:STDOUT:       39:              inst152
// CHECK:STDOUT:       40:              inst153
// CHECK:STDOUT:       41:              inst154
// CHECK:STDOUT:       42:              inst155
// CHECK:STDOUT:       43:              inst156
// CHECK:STDOUT:       44:              inst157
// CHECK:STDOUT:       45:              inst158
// CHECK:STDOUT:       46:              inst159
// CHECK:STDOUT:       47:              inst160
// CHECK:STDOUT:       48:              inst161
// CHECK:STDOUT:       49:              inst162
// CHECK:STDOUT:       50:              inst163
// CHECK:STDOUT:       51:              inst168
// CHECK:STDOUT:       52:              inst171
// CHECK:STDOUT:       53:              inst172
// CHECK:STDOUT:       54:              inst173
// CHECK:STDOUT:       55:              inst174
// CHECK:STDOUT:       56:              inst175
// CHECK:STDOUT:       57:              inst182
// CHECK:STDOUT:       58:              inst183
// CHECK:STDOUT:       59:              inst184
// CHECK:STDOUT:       60:              inst185
// CHECK:STDOUT:       61:              inst186
// CHECK:STDOUT:       62:              inst187
// CHECK:STDOUT:       63:              inst188
// CHECK:STDOUT:       64:              inst189
// CHECK:STDOUT:       65:              inst190
// CHECK:STDOUT:       66:              inst191
// CHECK:STDOUT:       67:              inst195
// CHECK:STDOUT:       68:              inst196
// CHECK:STDOUT:       69:              inst197
// CHECK:STDOUT:       70:              inst198
// CHECK:STDOUT:       71:              inst199
// CHECK:STDOUT:       72:              inst200
// CHECK:STDOUT:       73:              inst201
// CHECK:STDOUT:       74:              inst202
// CHECK:STDOUT:       75:              inst210
// CHECK:STDOUT:       76:              inst214
// CHECK:STDOUT:       77:              inst215
// CHECK:STDOUT:       78:              inst216
// CHECK:STDOUT:       79:              inst217
// CHECK:STDOUT:       80:              inst218
// CHECK:STDOUT:       81:              inst219
// CHECK:STDOUT:       82:              inst249
// CHECK:STDOUT:       83:              inst250
// CHECK:STDOUT:       84:              inst254
// CHECK:STDOUT:       85:              inst255
// CHECK:STDOUT:       86:              inst256
// CHECK:STDOUT:       87:              inst257
// CHECK:STDOUT:       88:              inst258
// CHECK:STDOUT:       89:              inst259
// CHECK:STDOUT:       90:              inst260
// CHECK:STDOUT:       91:              inst261
// CHECK:STDOUT:       92:              inst262
// CHECK:STDOUT:       93:              inst263
// CHECK:STDOUT:       94:              inst273
// CHECK:STDOUT:       95:              inst277
// CHECK:STDOUT:       96:              inst278
// CHECK:STDOUT:       97:              inst279
// CHECK:STDOUT:       98:              inst280
// CHECK:STDOUT:       99:              inst281
// CHECK:STDOUT:       100:             inst282
// CHECK:STDOUT:       101:             inst283
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst18
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst24
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst25
// CHECK:STDOUT:       1:               inst26
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst33
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst35
// CHECK:STDOUT:       1:               inst37
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst36
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst35
// CHECK:STDOUT:       1:               inst41
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst27
// CHECK:STDOUT:       1:               inst36
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst28
// CHECK:STDOUT:       1:               inst36
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst49
// CHECK:STDOUT:       1:               inst51
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst24
// CHECK:STDOUT:       1:               inst31
// CHECK:STDOUT:       2:               inst33
// CHECK:STDOUT:       3:               inst46
// CHECK:STDOUT:       4:               inst48
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst34
// CHECK:STDOUT:       1:               inst35
// CHECK:STDOUT:       2:               inst37
// CHECK:STDOUT:       3:               inst39
// CHECK:STDOUT:       4:               inst41
// CHECK:STDOUT:       5:               inst43
// CHECK:STDOUT:       6:               inst18
// CHECK:STDOUT:       7:               inst20
// CHECK:STDOUT:       8:               inst49
// CHECK:STDOUT:       9:               inst50
// CHECK:STDOUT:       10:              inst29
// CHECK:STDOUT:       11:              inst51
// CHECK:STDOUT:       12:              inst52
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst20
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               inst22
// CHECK:STDOUT:       1:               inst28
// CHECK:STDOUT:       2:               inst32
// CHECK:STDOUT:       3:               inst44
// CHECK:STDOUT:       4:               inst47
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:       1:               inst27
// CHECK:STDOUT:       2:               inst30
// CHECK:STDOUT:       3:               inst42
// CHECK:STDOUT:       4:               inst45
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               inst63
// CHECK:STDOUT:       1:               inst64
// CHECK:STDOUT:       2:               inst65
// CHECK:STDOUT:       3:               inst325
// CHECK:STDOUT:       4:               inst330
// CHECK:STDOUT:       5:               inst331
// CHECK:STDOUT:       6:               inst334
// CHECK:STDOUT:       7:               inst336
// CHECK:STDOUT:       8:               inst67
// CHECK:STDOUT:       9:               inst337
// CHECK:STDOUT:       10:              inst338
// CHECK:STDOUT:       11:              inst339
// CHECK:STDOUT:       12:              inst341
// CHECK:STDOUT:       13:              inst342
// CHECK:STDOUT:       14:              inst343
// CHECK:STDOUT:       15:              inst344
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               inst63
// CHECK:STDOUT:       1:               inst64
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               inst75
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               inst87
// CHECK:STDOUT:     inst_block24:
// CHECK:STDOUT:       0:               inst88
// CHECK:STDOUT:     inst_block25:
// CHECK:STDOUT:       0:               inst89
// CHECK:STDOUT:       1:               inst90
// CHECK:STDOUT:       2:               inst91
// CHECK:STDOUT:     inst_block26:
// CHECK:STDOUT:       0:               inst72
// CHECK:STDOUT:     inst_block27:
// CHECK:STDOUT:       0:               inst72
// CHECK:STDOUT:       1:               inst82
// CHECK:STDOUT:       2:               inst83
// CHECK:STDOUT:     inst_block28:
// CHECK:STDOUT:       0:               inst99
// CHECK:STDOUT:     inst_block29:
// CHECK:STDOUT:       0:               inst100
// CHECK:STDOUT:     inst_block30:
// CHECK:STDOUT:       0:               inst103
// CHECK:STDOUT:     inst_block31:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:     inst_block32:
// CHECK:STDOUT:       0:               inst106
// CHECK:STDOUT:     inst_block33:
// CHECK:STDOUT:       0:               inst106
// CHECK:STDOUT:       1:               inst107
// CHECK:STDOUT:       2:               inst108
// CHECK:STDOUT:       3:               inst109
// CHECK:STDOUT:     inst_block34:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:       1:               inst96
// CHECK:STDOUT:       2:               inst97
// CHECK:STDOUT:       3:               inst105
// CHECK:STDOUT:     inst_block35:
// CHECK:STDOUT:       0:               inst117
// CHECK:STDOUT:     inst_block36:
// CHECK:STDOUT:       0:               inst118
// CHECK:STDOUT:     inst_block37:
// CHECK:STDOUT:       0:               inst106
// CHECK:STDOUT:     inst_block38:
// CHECK:STDOUT:       0:               inst119
// CHECK:STDOUT:       1:               inst120
// CHECK:STDOUT:     inst_block39:
// CHECK:STDOUT:       0:               inst121
// CHECK:STDOUT:       1:               inst122
// CHECK:STDOUT:       2:               inst123
// CHECK:STDOUT:       3:               inst124
// CHECK:STDOUT:     inst_block40:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:       1:               inst96
// CHECK:STDOUT:       2:               inst97
// CHECK:STDOUT:       3:               inst113
// CHECK:STDOUT:     inst_block41:
// CHECK:STDOUT:       0:               inst127
// CHECK:STDOUT:     inst_block42:
// CHECK:STDOUT:       0:               inst128
// CHECK:STDOUT:     inst_block43:
// CHECK:STDOUT:       0:               inst134
// CHECK:STDOUT:     inst_block44:
// CHECK:STDOUT:       0:               inst135
// CHECK:STDOUT:     inst_block45:
// CHECK:STDOUT:       0:               inst132
// CHECK:STDOUT:       1:               inst133
// CHECK:STDOUT:       2:               inst134
// CHECK:STDOUT:       3:               inst135
// CHECK:STDOUT:       4:               inst136
// CHECK:STDOUT:       5:               inst137
// CHECK:STDOUT:       6:               inst138
// CHECK:STDOUT:     inst_block46:
// CHECK:STDOUT:       0:               inst128
// CHECK:STDOUT:       1:               inst96
// CHECK:STDOUT:       2:               inst139
// CHECK:STDOUT:     inst_block47:
// CHECK:STDOUT:       0:               inst111
// CHECK:STDOUT:       1:               inst112
// CHECK:STDOUT:     inst_block48:
// CHECK:STDOUT:       0:               inst158
// CHECK:STDOUT:     inst_block49:
// CHECK:STDOUT:       0:               inst159
// CHECK:STDOUT:     inst_block50:
// CHECK:STDOUT:       0:               inst162
// CHECK:STDOUT:     inst_block51:
// CHECK:STDOUT:       0:               inst165
// CHECK:STDOUT:     inst_block52:
// CHECK:STDOUT:       0:               inst165
// CHECK:STDOUT:       1:               inst166
// CHECK:STDOUT:       2:               inst167
// CHECK:STDOUT:     inst_block53:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:       1:               inst27
// CHECK:STDOUT:       2:               inst164
// CHECK:STDOUT:     inst_block54:
// CHECK:STDOUT:       0:               inst174
// CHECK:STDOUT:     inst_block55:
// CHECK:STDOUT:       0:               inst175
// CHECK:STDOUT:     inst_block56:
// CHECK:STDOUT:       0:               inst165
// CHECK:STDOUT:     inst_block57:
// CHECK:STDOUT:       0:               inst176
// CHECK:STDOUT:       1:               inst177
// CHECK:STDOUT:       2:               inst178
// CHECK:STDOUT:     inst_block58:
// CHECK:STDOUT:       0:               inst179
// CHECK:STDOUT:       1:               inst180
// CHECK:STDOUT:       2:               inst181
// CHECK:STDOUT:     inst_block59:
// CHECK:STDOUT:       0:               inst21
// CHECK:STDOUT:       1:               inst27
// CHECK:STDOUT:       2:               inst30
// CHECK:STDOUT:     inst_block60:
// CHECK:STDOUT:       0:               inst96
// CHECK:STDOUT:       1:               inst193
// CHECK:STDOUT:     inst_block61:
// CHECK:STDOUT:       0:               inst196
// CHECK:STDOUT:       1:               inst195
// CHECK:STDOUT:     inst_block62:
// CHECK:STDOUT:       0:               inst197
// CHECK:STDOUT:       1:               inst198
// CHECK:STDOUT:     inst_block63:
// CHECK:STDOUT:       0:               inst201
// CHECK:STDOUT:     inst_block64:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:       1:               inst192
// CHECK:STDOUT:     inst_block65:
// CHECK:STDOUT:       0:               inst206
// CHECK:STDOUT:       1:               inst207
// CHECK:STDOUT:     inst_block66:
// CHECK:STDOUT:       0:               inst204
// CHECK:STDOUT:       1:               inst205
// CHECK:STDOUT:     inst_block67:
// CHECK:STDOUT:       0:               inst204
// CHECK:STDOUT:       1:               inst205
// CHECK:STDOUT:       2:               inst206
// CHECK:STDOUT:       3:               inst207
// CHECK:STDOUT:       4:               inst208
// CHECK:STDOUT:       5:               inst209
// CHECK:STDOUT:     inst_block68:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:       1:               inst192
// CHECK:STDOUT:       2:               inst96
// CHECK:STDOUT:       3:               inst193
// CHECK:STDOUT:       4:               inst194
// CHECK:STDOUT:       5:               inst203
// CHECK:STDOUT:     inst_block69:
// CHECK:STDOUT:       0:               inst217
// CHECK:STDOUT:     inst_block70:
// CHECK:STDOUT:       0:               inst218
// CHECK:STDOUT:       1:               inst219
// CHECK:STDOUT:     inst_block71:
// CHECK:STDOUT:       0:               inst204
// CHECK:STDOUT:       1:               inst205
// CHECK:STDOUT:     inst_block72:
// CHECK:STDOUT:       0:               inst220
// CHECK:STDOUT:       1:               inst221
// CHECK:STDOUT:     inst_block73:
// CHECK:STDOUT:       0:               inst223
// CHECK:STDOUT:       1:               inst225
// CHECK:STDOUT:     inst_block74:
// CHECK:STDOUT:       0:               inst222
// CHECK:STDOUT:       1:               inst223
// CHECK:STDOUT:       2:               inst224
// CHECK:STDOUT:       3:               inst225
// CHECK:STDOUT:       4:               inst226
// CHECK:STDOUT:       5:               inst227
// CHECK:STDOUT:     inst_block75:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:       1:               inst96
// CHECK:STDOUT:       2:               inst192
// CHECK:STDOUT:       3:               inst193
// CHECK:STDOUT:       4:               inst194
// CHECK:STDOUT:       5:               inst213
// CHECK:STDOUT:     inst_block76:
// CHECK:STDOUT:       0:               inst230
// CHECK:STDOUT:     inst_block77:
// CHECK:STDOUT:       0:               inst231
// CHECK:STDOUT:     inst_block78:
// CHECK:STDOUT:       0:               inst237
// CHECK:STDOUT:     inst_block79:
// CHECK:STDOUT:       0:               inst238
// CHECK:STDOUT:     inst_block80:
// CHECK:STDOUT:       0:               inst243
// CHECK:STDOUT:     inst_block81:
// CHECK:STDOUT:       0:               inst244
// CHECK:STDOUT:     inst_block82:
// CHECK:STDOUT:       0:               inst235
// CHECK:STDOUT:       1:               inst236
// CHECK:STDOUT:       2:               inst237
// CHECK:STDOUT:       3:               inst238
// CHECK:STDOUT:       4:               inst239
// CHECK:STDOUT:       5:               inst240
// CHECK:STDOUT:       6:               inst241
// CHECK:STDOUT:       7:               inst242
// CHECK:STDOUT:       8:               inst243
// CHECK:STDOUT:       9:               inst244
// CHECK:STDOUT:       10:              inst245
// CHECK:STDOUT:       11:              inst246
// CHECK:STDOUT:       12:              inst247
// CHECK:STDOUT:     inst_block83:
// CHECK:STDOUT:       0:               inst231
// CHECK:STDOUT:       1:               inst193
// CHECK:STDOUT:       2:               inst248
// CHECK:STDOUT:     inst_block84:
// CHECK:STDOUT:       0:               inst211
// CHECK:STDOUT:       1:               inst212
// CHECK:STDOUT:     inst_block85:
// CHECK:STDOUT:       0:               inst96
// CHECK:STDOUT:       1:               inst193
// CHECK:STDOUT:       2:               inst252
// CHECK:STDOUT:     inst_block86:
// CHECK:STDOUT:       0:               inst256
// CHECK:STDOUT:       1:               inst255
// CHECK:STDOUT:       2:               inst254
// CHECK:STDOUT:     inst_block87:
// CHECK:STDOUT:       0:               inst257
// CHECK:STDOUT:       1:               inst258
// CHECK:STDOUT:       2:               inst259
// CHECK:STDOUT:     inst_block88:
// CHECK:STDOUT:       0:               inst262
// CHECK:STDOUT:     inst_block89:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:       1:               inst192
// CHECK:STDOUT:       2:               inst251
// CHECK:STDOUT:     inst_block90:
// CHECK:STDOUT:       0:               inst268
// CHECK:STDOUT:       1:               inst269
// CHECK:STDOUT:       2:               inst270
// CHECK:STDOUT:     inst_block91:
// CHECK:STDOUT:       0:               inst265
// CHECK:STDOUT:       1:               inst266
// CHECK:STDOUT:       2:               inst267
// CHECK:STDOUT:     inst_block92:
// CHECK:STDOUT:       0:               inst265
// CHECK:STDOUT:       1:               inst266
// CHECK:STDOUT:       2:               inst267
// CHECK:STDOUT:       3:               inst268
// CHECK:STDOUT:       4:               inst269
// CHECK:STDOUT:       5:               inst270
// CHECK:STDOUT:       6:               inst271
// CHECK:STDOUT:       7:               inst272
// CHECK:STDOUT:     inst_block93:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:       1:               inst192
// CHECK:STDOUT:       2:               inst251
// CHECK:STDOUT:       3:               inst96
// CHECK:STDOUT:       4:               inst193
// CHECK:STDOUT:       5:               inst252
// CHECK:STDOUT:       6:               inst253
// CHECK:STDOUT:       7:               inst264
// CHECK:STDOUT:     inst_block94:
// CHECK:STDOUT:       0:               inst280
// CHECK:STDOUT:     inst_block95:
// CHECK:STDOUT:       0:               inst281
// CHECK:STDOUT:       1:               inst282
// CHECK:STDOUT:       2:               inst283
// CHECK:STDOUT:     inst_block96:
// CHECK:STDOUT:       0:               inst265
// CHECK:STDOUT:       1:               inst266
// CHECK:STDOUT:       2:               inst267
// CHECK:STDOUT:     inst_block97:
// CHECK:STDOUT:       0:               inst284
// CHECK:STDOUT:       1:               inst285
// CHECK:STDOUT:     inst_block98:
// CHECK:STDOUT:       0:               inst287
// CHECK:STDOUT:       1:               inst289
// CHECK:STDOUT:       2:               inst291
// CHECK:STDOUT:     inst_block99:
// CHECK:STDOUT:       0:               inst286
// CHECK:STDOUT:       1:               inst287
// CHECK:STDOUT:       2:               inst288
// CHECK:STDOUT:       3:               inst289
// CHECK:STDOUT:       4:               inst290
// CHECK:STDOUT:       5:               inst291
// CHECK:STDOUT:       6:               inst292
// CHECK:STDOUT:       7:               inst293
// CHECK:STDOUT:     inst_block100:
// CHECK:STDOUT:       0:               inst95
// CHECK:STDOUT:       1:               inst96
// CHECK:STDOUT:       2:               inst192
// CHECK:STDOUT:       3:               inst193
// CHECK:STDOUT:       4:               inst251
// CHECK:STDOUT:       5:               inst252
// CHECK:STDOUT:       6:               inst253
// CHECK:STDOUT:       7:               inst276
// CHECK:STDOUT:     inst_block101:
// CHECK:STDOUT:       0:               inst296
// CHECK:STDOUT:     inst_block102:
// CHECK:STDOUT:       0:               inst297
// CHECK:STDOUT:     inst_block103:
// CHECK:STDOUT:       0:               inst303
// CHECK:STDOUT:     inst_block104:
// CHECK:STDOUT:       0:               inst304
// CHECK:STDOUT:     inst_block105:
// CHECK:STDOUT:       0:               inst309
// CHECK:STDOUT:     inst_block106:
// CHECK:STDOUT:       0:               inst310
// CHECK:STDOUT:     inst_block107:
// CHECK:STDOUT:       0:               inst315
// CHECK:STDOUT:     inst_block108:
// CHECK:STDOUT:       0:               inst316
// CHECK:STDOUT:     inst_block109:
// CHECK:STDOUT:       0:               inst301
// CHECK:STDOUT:       1:               inst302
// CHECK:STDOUT:       2:               inst303
// CHECK:STDOUT:       3:               inst304
// CHECK:STDOUT:       4:               inst305
// CHECK:STDOUT:       5:               inst306
// CHECK:STDOUT:       6:               inst307
// CHECK:STDOUT:       7:               inst308
// CHECK:STDOUT:       8:               inst309
// CHECK:STDOUT:       9:               inst310
// CHECK:STDOUT:       10:              inst311
// CHECK:STDOUT:       11:              inst312
// CHECK:STDOUT:       12:              inst313
// CHECK:STDOUT:       13:              inst314
// CHECK:STDOUT:       14:              inst315
// CHECK:STDOUT:       15:              inst316
// CHECK:STDOUT:       16:              inst317
// CHECK:STDOUT:       17:              inst318
// CHECK:STDOUT:       18:              inst319
// CHECK:STDOUT:     inst_block110:
// CHECK:STDOUT:       0:               inst297
// CHECK:STDOUT:       1:               inst252
// CHECK:STDOUT:       2:               inst320
// CHECK:STDOUT:     inst_block111:
// CHECK:STDOUT:       0:               inst274
// CHECK:STDOUT:       1:               inst275
// CHECK:STDOUT:     inst_block112:
// CHECK:STDOUT:       0:               inst169
// CHECK:STDOUT:       1:               inst170
// CHECK:STDOUT:       2:               inst61
// CHECK:STDOUT:     inst_block113:
// CHECK:STDOUT:       0:               inst321
// CHECK:STDOUT:     inst_block114:
// CHECK:STDOUT:       0:               inst322
// CHECK:STDOUT:     inst_block115:
// CHECK:STDOUT:       0:               inst323
// CHECK:STDOUT:     inst_block116:
// CHECK:STDOUT:       0:               inst323
// CHECK:STDOUT:       1:               inst27
// CHECK:STDOUT:       2:               inst30
// CHECK:STDOUT:     inst_block117:
// CHECK:STDOUT:       0:               inst327
// CHECK:STDOUT:     inst_block118:
// CHECK:STDOUT:       0:               inst63
// CHECK:STDOUT:     inst_block119:
// CHECK:STDOUT:       0:               inst337
// CHECK:STDOUT:       1:               inst341
// CHECK:STDOUT:     inst_block120:
// CHECK:STDOUT:       0:               inst59
// CHECK:STDOUT:       1:               inst62
// CHECK:STDOUT:       2:               inst322
// CHECK:STDOUT:       3:               inst327
// CHECK:STDOUT:       4:               inst328
// CHECK:STDOUT:       5:               inst329
// CHECK:STDOUT:       6:               inst333
// CHECK:STDOUT:     inst_block121:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               inst15
// CHECK:STDOUT:       2:               inst53
// CHECK:STDOUT: ...
