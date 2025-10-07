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
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     import_ir2:      {decl_id: ir0.inst15, is_export: false}
// CHECK:STDOUT:     import_ir3:      {decl_id: ir0.inst15, is_export: false}
// CHECK:STDOUT:     import_ir4:      {decl_id: ir0.inst15, is_export: false}
// CHECK:STDOUT:     import_ir5:      {decl_id: ir0.inst15, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir4, inst_id: ir3.inst15}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir4, inst_id: ir3.inst15}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir4, inst_id: ir3.inst17}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir4, inst_id: ir3.inst45}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir4, inst_id: ir3.inst40}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: import_ir4, inst_id: ir3.inst40}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: import_ir4, inst_id: ir3.inst40}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: import_ir4, inst_id: ir3.inst34}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: import_ir4, inst_id: ir3.inst35}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: import_ir4, inst_id: ir3.inst28}
// CHECK:STDOUT:     import_ir_inst10: {ir_id: import_ir4, inst_id: ir3.inst30}
// CHECK:STDOUT:     import_ir_inst11: {ir_id: import_ir4, inst_id: ir3.inst17}
// CHECK:STDOUT:     import_ir_inst12: {ir_id: import_ir4, inst_id: ir3.inst21}
// CHECK:STDOUT:     import_ir_inst13: {ir_id: import_ir4, inst_id: ir3.inst24}
// CHECK:STDOUT:     import_ir_inst14: {ir_id: import_ir4, inst_id: ir3.inst29}
// CHECK:STDOUT:     import_ir_inst15: {ir_id: import_ir4, inst_id: ir3.inst97}
// CHECK:STDOUT:     import_ir_inst16: {ir_id: import_ir4, inst_id: ir3.inst95}
// CHECK:STDOUT:     import_ir_inst17: {ir_id: import_ir4, inst_id: ir3.inst84}
// CHECK:STDOUT:     import_ir_inst18: {ir_id: import_ir4, inst_id: ir3.inst80}
// CHECK:STDOUT:     import_ir_inst19: {ir_id: import_ir4, inst_id: ir3.inst90}
// CHECK:STDOUT:     import_ir_inst20: {ir_id: import_ir4, inst_id: ir3.inst93}
// CHECK:STDOUT:     import_ir_inst21: {ir_id: import_ir4, inst_id: ir3.inst116}
// CHECK:STDOUT:     import_ir_inst22: {ir_id: import_ir4, inst_id: ir3.inst96}
// CHECK:STDOUT:     import_ir_inst23: {ir_id: import_ir4, inst_id: ir3.inst82}
// CHECK:STDOUT:     import_ir_inst24: {ir_id: import_ir4, inst_id: ir3.inst88}
// CHECK:STDOUT:     import_ir_inst25: {ir_id: import_ir4, inst_id: ir3.inst92}
// CHECK:STDOUT:     import_ir_inst26: {ir_id: import_ir4, inst_id: ir3.inst99}
// CHECK:STDOUT:     import_ir_inst27: {ir_id: import_ir4, inst_id: ir3.inst116}
// CHECK:STDOUT:     import_ir_inst28: {ir_id: import_ir4, inst_id: ir3.inst111}
// CHECK:STDOUT:     import_ir_inst29: {ir_id: import_ir4, inst_id: ir3.inst112}
// CHECK:STDOUT:     import_ir_inst30: {ir_id: import_ir4, inst_id: ir3.inst107}
// CHECK:STDOUT:     import_ir_inst31: {ir_id: import_ir4, inst_id: ir3.inst109}
// CHECK:STDOUT:     import_ir_inst32: {ir_id: import_ir4, inst_id: ir3.inst80}
// CHECK:STDOUT:     import_ir_inst33: {ir_id: import_ir4, inst_id: ir3.inst119}
// CHECK:STDOUT:     import_ir_inst34: {ir_id: import_ir4, inst_id: ir3.inst120}
// CHECK:STDOUT:     import_ir_inst35: {ir_id: import_ir4, inst_id: ir3.inst102}
// CHECK:STDOUT:     import_ir_inst36: {ir_id: import_ir4, inst_id: ir3.inst103}
// CHECK:STDOUT:     import_ir_inst37: {ir_id: import_ir4, inst_id: ir3.inst104}
// CHECK:STDOUT:     import_ir_inst38: {ir_id: import_ir4, inst_id: ir3.inst108}
// CHECK:STDOUT:     import_ir_inst39: {ir_id: import_ir4, inst_id: ir3.inst124}
// CHECK:STDOUT:     import_ir_inst40: {ir_id: import_ir4, inst_id: ir3.inst132}
// CHECK:STDOUT:     import_ir_inst41: {ir_id: import_ir4, inst_id: ir3.inst139}
// CHECK:STDOUT:     import_ir_inst42: {ir_id: import_ir4, inst_id: ir3.inst143}
// CHECK:STDOUT:     import_ir_inst43: {ir_id: import_ir4, inst_id: ir3.inst144}
// CHECK:STDOUT:     import_ir_inst44: {ir_id: import_ir4, inst_id: ir3.inst149}
// CHECK:STDOUT:     import_ir_inst45: {ir_id: import_ir4, inst_id: ir3.inst170}
// CHECK:STDOUT:     import_ir_inst46: {ir_id: import_ir4, inst_id: ir3.inst168}
// CHECK:STDOUT:     import_ir_inst47: {ir_id: import_ir4, inst_id: ir3.inst166}
// CHECK:STDOUT:     import_ir_inst48: {ir_id: import_ir4, inst_id: ir3.inst167}
// CHECK:STDOUT:     import_ir_inst49: {ir_id: import_ir4, inst_id: ir3.inst194}
// CHECK:STDOUT:     import_ir_inst50: {ir_id: import_ir4, inst_id: ir3.inst192}
// CHECK:STDOUT:     import_ir_inst51: {ir_id: import_ir4, inst_id: ir3.inst190}
// CHECK:STDOUT:     import_ir_inst52: {ir_id: import_ir4, inst_id: ir3.inst191}
// CHECK:STDOUT:     import_ir_inst53: {ir_id: import_ir4, inst_id: ir3.inst218}
// CHECK:STDOUT:     import_ir_inst54: {ir_id: import_ir4, inst_id: ir3.inst216}
// CHECK:STDOUT:     import_ir_inst55: {ir_id: import_ir4, inst_id: ir3.inst214}
// CHECK:STDOUT:     import_ir_inst56: {ir_id: import_ir4, inst_id: ir3.inst215}
// CHECK:STDOUT:     import_ir_inst57: {ir_id: import_ir4, inst_id: ir3.inst242}
// CHECK:STDOUT:     import_ir_inst58: {ir_id: import_ir4, inst_id: ir3.inst240}
// CHECK:STDOUT:     import_ir_inst59: {ir_id: import_ir4, inst_id: ir3.inst238}
// CHECK:STDOUT:     import_ir_inst60: {ir_id: import_ir4, inst_id: ir3.inst239}
// CHECK:STDOUT:     import_ir_inst61: {ir_id: import_ir4, inst_id: ir3.inst271}
// CHECK:STDOUT:     import_ir_inst62: {ir_id: import_ir4, inst_id: ir3.inst269}
// CHECK:STDOUT:     import_ir_inst63: {ir_id: import_ir4, inst_id: ir3.inst263}
// CHECK:STDOUT:     import_ir_inst64: {ir_id: import_ir4, inst_id: ir3.inst260}
// CHECK:STDOUT:     import_ir_inst65: {ir_id: import_ir4, inst_id: ir3.inst265}
// CHECK:STDOUT:     import_ir_inst66: {ir_id: import_ir4, inst_id: ir3.inst268}
// CHECK:STDOUT:     import_ir_inst67: {ir_id: import_ir4, inst_id: ir3.inst288}
// CHECK:STDOUT:     import_ir_inst68: {ir_id: import_ir4, inst_id: ir3.inst270}
// CHECK:STDOUT:     import_ir_inst69: {ir_id: import_ir4, inst_id: ir3.inst262}
// CHECK:STDOUT:     import_ir_inst70: {ir_id: import_ir4, inst_id: ir3.inst267}
// CHECK:STDOUT:     import_ir_inst71: {ir_id: import_ir4, inst_id: ir3.inst273}
// CHECK:STDOUT:     import_ir_inst72: {ir_id: import_ir4, inst_id: ir3.inst288}
// CHECK:STDOUT:     import_ir_inst73: {ir_id: import_ir4, inst_id: ir3.inst283}
// CHECK:STDOUT:     import_ir_inst74: {ir_id: import_ir4, inst_id: ir3.inst284}
// CHECK:STDOUT:     import_ir_inst75: {ir_id: import_ir4, inst_id: ir3.inst279}
// CHECK:STDOUT:     import_ir_inst76: {ir_id: import_ir4, inst_id: ir3.inst281}
// CHECK:STDOUT:     import_ir_inst77: {ir_id: import_ir4, inst_id: ir3.inst260}
// CHECK:STDOUT:     import_ir_inst78: {ir_id: import_ir4, inst_id: ir3.inst291}
// CHECK:STDOUT:     import_ir_inst79: {ir_id: import_ir4, inst_id: ir3.inst292}
// CHECK:STDOUT:     import_ir_inst80: {ir_id: import_ir4, inst_id: ir3.inst295}
// CHECK:STDOUT:     import_ir_inst81: {ir_id: import_ir4, inst_id: ir3.inst275}
// CHECK:STDOUT:     import_ir_inst82: {ir_id: import_ir4, inst_id: ir3.inst276}
// CHECK:STDOUT:     import_ir_inst83: {ir_id: import_ir4, inst_id: ir3.inst280}
// CHECK:STDOUT:     import_ir_inst84: {ir_id: import_ir4, inst_id: ir3.inst301}
// CHECK:STDOUT:     import_ir_inst85: {ir_id: import_ir4, inst_id: ir3.inst299}
// CHECK:STDOUT:     import_ir_inst86: {ir_id: import_ir4, inst_id: inst(TypeType)}
// CHECK:STDOUT:     import_ir_inst87: {ir_id: import_ir4, inst_id: ir3.inst298}
// CHECK:STDOUT:     import_ir_inst88: {ir_id: import_ir4, inst_id: ir3.inst322}
// CHECK:STDOUT:     import_ir_inst89: {ir_id: import_ir4, inst_id: ir3.inst320}
// CHECK:STDOUT:     import_ir_inst90: {ir_id: import_ir4, inst_id: ir3.inst318}
// CHECK:STDOUT:     import_ir_inst91: {ir_id: import_ir4, inst_id: ir3.inst319}
// CHECK:STDOUT:     import_ir_inst92: {ir_id: import_ir4, inst_id: ir3.inst370}
// CHECK:STDOUT:     import_ir_inst93: {ir_id: import_ir4, inst_id: ir3.inst368}
// CHECK:STDOUT:     import_ir_inst94: {ir_id: import_ir4, inst_id: ir3.inst349}
// CHECK:STDOUT:     import_ir_inst95: {ir_id: import_ir4, inst_id: ir3.inst343}
// CHECK:STDOUT:     import_ir_inst96: {ir_id: import_ir4, inst_id: ir3.inst341}
// CHECK:STDOUT:     import_ir_inst97: {ir_id: import_ir4, inst_id: ir3.inst346}
// CHECK:STDOUT:     import_ir_inst98: {ir_id: import_ir4, inst_id: ir3.inst363}
// CHECK:STDOUT:     import_ir_inst99: {ir_id: import_ir4, inst_id: ir3.inst365}
// CHECK:STDOUT:     import_ir_inst100: {ir_id: import_ir4, inst_id: ir3.inst392}
// CHECK:STDOUT:     import_ir_inst101: {ir_id: import_ir4, inst_id: ir3.inst369}
// CHECK:STDOUT:     import_ir_inst102: {ir_id: import_ir4, inst_id: ir3.inst342}
// CHECK:STDOUT:     import_ir_inst103: {ir_id: import_ir4, inst_id: ir3.inst348}
// CHECK:STDOUT:     import_ir_inst104: {ir_id: import_ir4, inst_id: ir3.inst356}
// CHECK:STDOUT:     import_ir_inst105: {ir_id: import_ir4, inst_id: ir3.inst360}
// CHECK:STDOUT:     import_ir_inst106: {ir_id: import_ir4, inst_id: ir3.inst364}
// CHECK:STDOUT:     import_ir_inst107: {ir_id: import_ir4, inst_id: ir3.inst372}
// CHECK:STDOUT:     import_ir_inst108: {ir_id: import_ir4, inst_id: ir3.inst392}
// CHECK:STDOUT:     import_ir_inst109: {ir_id: import_ir4, inst_id: ir3.inst387}
// CHECK:STDOUT:     import_ir_inst110: {ir_id: import_ir4, inst_id: ir3.inst388}
// CHECK:STDOUT:     import_ir_inst111: {ir_id: import_ir4, inst_id: ir3.inst383}
// CHECK:STDOUT:     import_ir_inst112: {ir_id: import_ir4, inst_id: ir3.inst385}
// CHECK:STDOUT:     import_ir_inst113: {ir_id: import_ir4, inst_id: ir3.inst341}
// CHECK:STDOUT:     import_ir_inst114: {ir_id: import_ir4, inst_id: ir3.inst346}
// CHECK:STDOUT:     import_ir_inst115: {ir_id: import_ir4, inst_id: ir3.inst395}
// CHECK:STDOUT:     import_ir_inst116: {ir_id: import_ir4, inst_id: ir3.inst396}
// CHECK:STDOUT:     import_ir_inst117: {ir_id: import_ir4, inst_id: ir3.inst376}
// CHECK:STDOUT:     import_ir_inst118: {ir_id: import_ir4, inst_id: ir3.inst377}
// CHECK:STDOUT:     import_ir_inst119: {ir_id: import_ir4, inst_id: ir3.inst378}
// CHECK:STDOUT:     import_ir_inst120: {ir_id: import_ir4, inst_id: ir3.inst379}
// CHECK:STDOUT:     import_ir_inst121: {ir_id: import_ir4, inst_id: ir3.inst380}
// CHECK:STDOUT:     import_ir_inst122: {ir_id: import_ir4, inst_id: ir3.inst384}
// CHECK:STDOUT:     import_ir_inst123: {ir_id: import_ir4, inst_id: ir3.inst401}
// CHECK:STDOUT:     import_ir_inst124: {ir_id: import_ir4, inst_id: ir3.inst409}
// CHECK:STDOUT:     import_ir_inst125: {ir_id: import_ir4, inst_id: ir3.inst413}
// CHECK:STDOUT:     import_ir_inst126: {ir_id: import_ir4, inst_id: ir3.inst415}
// CHECK:STDOUT:     import_ir_inst127: {ir_id: import_ir4, inst_id: ir3.inst416}
// CHECK:STDOUT:     import_ir_inst128: {ir_id: import_ir4, inst_id: ir3.inst419}
// CHECK:STDOUT:     import_ir_inst129: {ir_id: import_ir4, inst_id: ir3.inst431}
// CHECK:STDOUT:     import_ir_inst130: {ir_id: import_ir4, inst_id: ir3.inst436}
// CHECK:STDOUT:     import_ir_inst131: {ir_id: import_ir4, inst_id: ir3.inst440}
// CHECK:STDOUT:     import_ir_inst132: {ir_id: import_ir4, inst_id: ir3.inst441}
// CHECK:STDOUT:     import_ir_inst133: {ir_id: import_ir4, inst_id: ir3.inst446}
// CHECK:STDOUT:     import_ir_inst134: {ir_id: import_ir4, inst_id: ir3.inst503}
// CHECK:STDOUT:     import_ir_inst135: {ir_id: import_ir4, inst_id: ir3.inst501}
// CHECK:STDOUT:     import_ir_inst136: {ir_id: import_ir4, inst_id: ir3.inst477}
// CHECK:STDOUT:     import_ir_inst137: {ir_id: import_ir4, inst_id: ir3.inst471}
// CHECK:STDOUT:     import_ir_inst138: {ir_id: import_ir4, inst_id: ir3.inst466}
// CHECK:STDOUT:     import_ir_inst139: {ir_id: import_ir4, inst_id: ir3.inst464}
// CHECK:STDOUT:     import_ir_inst140: {ir_id: import_ir4, inst_id: ir3.inst469}
// CHECK:STDOUT:     import_ir_inst141: {ir_id: import_ir4, inst_id: ir3.inst474}
// CHECK:STDOUT:     import_ir_inst142: {ir_id: import_ir4, inst_id: ir3.inst495}
// CHECK:STDOUT:     import_ir_inst143: {ir_id: import_ir4, inst_id: ir3.inst497}
// CHECK:STDOUT:     import_ir_inst144: {ir_id: import_ir4, inst_id: ir3.inst528}
// CHECK:STDOUT:     import_ir_inst145: {ir_id: import_ir4, inst_id: ir3.inst502}
// CHECK:STDOUT:     import_ir_inst146: {ir_id: import_ir4, inst_id: ir3.inst465}
// CHECK:STDOUT:     import_ir_inst147: {ir_id: import_ir4, inst_id: ir3.inst470}
// CHECK:STDOUT:     import_ir_inst148: {ir_id: import_ir4, inst_id: ir3.inst476}
// CHECK:STDOUT:     import_ir_inst149: {ir_id: import_ir4, inst_id: ir3.inst485}
// CHECK:STDOUT:     import_ir_inst150: {ir_id: import_ir4, inst_id: ir3.inst488}
// CHECK:STDOUT:     import_ir_inst151: {ir_id: import_ir4, inst_id: ir3.inst492}
// CHECK:STDOUT:     import_ir_inst152: {ir_id: import_ir4, inst_id: ir3.inst496}
// CHECK:STDOUT:     import_ir_inst153: {ir_id: import_ir4, inst_id: ir3.inst505}
// CHECK:STDOUT:     import_ir_inst154: {ir_id: import_ir4, inst_id: ir3.inst528}
// CHECK:STDOUT:     import_ir_inst155: {ir_id: import_ir4, inst_id: ir3.inst523}
// CHECK:STDOUT:     import_ir_inst156: {ir_id: import_ir4, inst_id: ir3.inst524}
// CHECK:STDOUT:     import_ir_inst157: {ir_id: import_ir4, inst_id: ir3.inst519}
// CHECK:STDOUT:     import_ir_inst158: {ir_id: import_ir4, inst_id: ir3.inst521}
// CHECK:STDOUT:     import_ir_inst159: {ir_id: import_ir4, inst_id: ir3.inst464}
// CHECK:STDOUT:     import_ir_inst160: {ir_id: import_ir4, inst_id: ir3.inst469}
// CHECK:STDOUT:     import_ir_inst161: {ir_id: import_ir4, inst_id: ir3.inst474}
// CHECK:STDOUT:     import_ir_inst162: {ir_id: import_ir4, inst_id: ir3.inst531}
// CHECK:STDOUT:     import_ir_inst163: {ir_id: import_ir4, inst_id: ir3.inst532}
// CHECK:STDOUT:     import_ir_inst164: {ir_id: import_ir4, inst_id: ir3.inst510}
// CHECK:STDOUT:     import_ir_inst165: {ir_id: import_ir4, inst_id: ir3.inst511}
// CHECK:STDOUT:     import_ir_inst166: {ir_id: import_ir4, inst_id: ir3.inst512}
// CHECK:STDOUT:     import_ir_inst167: {ir_id: import_ir4, inst_id: ir3.inst513}
// CHECK:STDOUT:     import_ir_inst168: {ir_id: import_ir4, inst_id: ir3.inst514}
// CHECK:STDOUT:     import_ir_inst169: {ir_id: import_ir4, inst_id: ir3.inst515}
// CHECK:STDOUT:     import_ir_inst170: {ir_id: import_ir4, inst_id: ir3.inst516}
// CHECK:STDOUT:     import_ir_inst171: {ir_id: import_ir4, inst_id: ir3.inst520}
// CHECK:STDOUT:     import_ir_inst172: {ir_id: import_ir4, inst_id: ir3.inst537}
// CHECK:STDOUT:     import_ir_inst173: {ir_id: import_ir4, inst_id: ir3.inst544}
// CHECK:STDOUT:     import_ir_inst174: {ir_id: import_ir4, inst_id: ir3.inst548}
// CHECK:STDOUT:     import_ir_inst175: {ir_id: import_ir4, inst_id: ir3.inst550}
// CHECK:STDOUT:     import_ir_inst176: {ir_id: import_ir4, inst_id: ir3.inst551}
// CHECK:STDOUT:     import_ir_inst177: {ir_id: import_ir4, inst_id: ir3.inst554}
// CHECK:STDOUT:     import_ir_inst178: {ir_id: import_ir4, inst_id: ir3.inst564}
// CHECK:STDOUT:     import_ir_inst179: {ir_id: import_ir4, inst_id: ir3.inst568}
// CHECK:STDOUT:     import_ir_inst180: {ir_id: import_ir4, inst_id: ir3.inst570}
// CHECK:STDOUT:     import_ir_inst181: {ir_id: import_ir4, inst_id: ir3.inst571}
// CHECK:STDOUT:     import_ir_inst182: {ir_id: import_ir4, inst_id: ir3.inst574}
// CHECK:STDOUT:     import_ir_inst183: {ir_id: import_ir4, inst_id: ir3.inst586}
// CHECK:STDOUT:     import_ir_inst184: {ir_id: import_ir4, inst_id: ir3.inst591}
// CHECK:STDOUT:     import_ir_inst185: {ir_id: import_ir4, inst_id: ir3.inst595}
// CHECK:STDOUT:     import_ir_inst186: {ir_id: import_ir4, inst_id: ir3.inst596}
// CHECK:STDOUT:     import_ir_inst187: {ir_id: import_ir4, inst_id: ir3.inst601}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst14, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): ir0.inst16, name0: ir0.inst53}}
// CHECK:STDOUT:     name_scope1:     {inst: ir0.inst16, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: ir0.inst69}}
// CHECK:STDOUT:     name_scope2:     {inst: ir0.inst70, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {name(SelfType): ir0.inst73, name4: ir0.inst74}}
// CHECK:STDOUT:     name_scope3:     {inst: ir0.inst94, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope4:     {inst: ir0.inst139, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope5:     {inst: ir0.inst143, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6:     {inst: ir0.inst147, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7:     {inst: ir0.inst151, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope8:     {inst: ir0.inst155, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope9:     {inst: ir0.inst181, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope10:    {inst: ir0.inst185, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope11:    {inst: ir0.inst189, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope12:    {inst: ir0.inst245, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name0:    {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name1:    {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name2:    {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name3:    {name: name3, parent_scope: name_scope1, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name4:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name5:    {name: name4, parent_scope: name_scope2, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name6:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name7:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name8:    {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name9:    {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name10:   {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name11:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name12:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name13:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name14:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name15:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name16:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name17:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name18:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name19:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name20:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name21:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name22:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name23:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name24:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name25:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name26:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name27:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name28:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name29:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name30:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name31:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name32:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name33:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name34:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name35:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name36:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name37:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name38:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name39:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name40:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name41:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name42:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name43:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name44:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name45:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name46:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name47:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name48:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name49:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name50:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name51:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name52:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name53:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name54:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name55:   {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name56:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name57:   {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name58:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name59:   {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name60:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:     entity_name61:   {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, clang_decl_id: clang_decl_id<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function0:       {name: name0, parent_scope: name_scope0, call_params_id: inst_block13, return_slot_pattern: ir0.inst48, body: [inst_block20]}
// CHECK:STDOUT:     function1:       {name: name4, parent_scope: name_scope2, return_slot_pattern: ir0.inst85}
// CHECK:STDOUT:     function2:       {name: name4, parent_scope: name_scope3, return_slot_pattern: ir0.inst115}
// CHECK:STDOUT:     function3:       {name: name4, parent_scope: name_scope8, return_slot_pattern: ir0.inst170}
// CHECK:STDOUT:     function4:       {name: name4, parent_scope: name_scope11, return_slot_pattern: ir0.inst213}
// CHECK:STDOUT:     function5:       {name: name4, parent_scope: name_scope12, return_slot_pattern: ir0.inst273}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: ir0.inst53, bindings: inst_block16}
// CHECK:STDOUT:     generic1:        {decl: ir0.inst79, bindings: inst_block24}
// CHECK:STDOUT:     generic2:        {decl: ir0.inst94, bindings: inst_block29}
// CHECK:STDOUT:     generic3:        {decl: ir0.inst110, bindings: inst_block36}
// CHECK:STDOUT:     generic4:        {decl: ir0.inst155, bindings: inst_block46}
// CHECK:STDOUT:     generic5:        {decl: ir0.inst166, bindings: inst_block52}
// CHECK:STDOUT:     generic6:        {decl: ir0.inst189, bindings: inst_block59}
// CHECK:STDOUT:     generic7:        {decl: ir0.inst208, bindings: inst_block67}
// CHECK:STDOUT:     generic8:        {decl: ir0.inst245, bindings: inst_block81}
// CHECK:STDOUT:     generic9:        {decl: ir0.inst268, bindings: inst_block89}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific0:       {generic: generic0, args: inst_block17}
// CHECK:STDOUT:     specific1:       {generic: generic1, args: inst_block26}
// CHECK:STDOUT:     specific2:       {generic: generic2, args: inst_block31}
// CHECK:STDOUT:     specific3:       {generic: generic2, args: inst_block32}
// CHECK:STDOUT:     specific4:       {generic: generic3, args: inst_block31}
// CHECK:STDOUT:     specific5:       {generic: generic1, args: inst_block31}
// CHECK:STDOUT:     specific6:       {generic: generic1, args: inst_block41}
// CHECK:STDOUT:     specific7:       {generic: generic4, args: inst_block17}
// CHECK:STDOUT:     specific8:       {generic: generic4, args: inst_block48}
// CHECK:STDOUT:     specific9:       {generic: generic5, args: inst_block17}
// CHECK:STDOUT:     specific10:      {generic: generic6, args: inst_block61}
// CHECK:STDOUT:     specific11:      {generic: generic6, args: inst_block63}
// CHECK:STDOUT:     specific12:      {generic: generic7, args: inst_block61}
// CHECK:STDOUT:     specific13:      {generic: generic1, args: inst_block73}
// CHECK:STDOUT:     specific14:      {generic: generic1, args: inst_block74}
// CHECK:STDOUT:     specific15:      {generic: generic1, args: inst_block75}
// CHECK:STDOUT:     specific16:      {generic: generic8, args: inst_block83}
// CHECK:STDOUT:     specific17:      {generic: generic8, args: inst_block85}
// CHECK:STDOUT:     specific18:      {generic: generic9, args: inst_block83}
// CHECK:STDOUT:     specific19:      {generic: generic1, args: inst_block95}
// CHECK:STDOUT:     specific20:      {generic: generic1, args: inst_block96}
// CHECK:STDOUT:     specific21:      {generic: generic1, args: inst_block97}
// CHECK:STDOUT:     specific22:      {generic: generic1, args: inst_block98}
// CHECK:STDOUT:     specific23:      {generic: generic1, args: inst_block105}
// CHECK:STDOUT:     specific24:      {generic: generic1, args: inst_block107}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields0: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(ir0.inst36)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst36)}
// CHECK:STDOUT:     'type(ir0.inst38)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(ir0.inst40)}
// CHECK:STDOUT:     'type(ir0.inst40)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(ir0.inst40)}
// CHECK:STDOUT:     'type(ir0.inst54)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst36)}
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
// CHECK:STDOUT:     'type(ir0.inst71)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(ir0.inst71)}
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     'type(symbolic_constant253)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst36)}
// CHECK:STDOUT:     'type(symbolic_constant256)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(ir0.inst36)}
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst14:          {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     ir0.inst15:      {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     ir0.inst16:      {kind: Namespace, arg0: name_scope1, arg1: ir0.inst15, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     ir0.inst17:      {kind: FacetType, arg0: facet_type0, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst18:      {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(ir0.inst17)}
// CHECK:STDOUT:     ir0.inst19:      {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(ir0.inst17)}
// CHECK:STDOUT:     ir0.inst20:      {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst21:      {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst22:      {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst23:      {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst24:      {kind: SymbolicBindingPattern, arg0: entity_name1, type: type(ir0.inst23)}
// CHECK:STDOUT:     ir0.inst25:      {kind: NameRef, arg0: name1, arg1: ir0.inst20, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst26:      {kind: PointerType, arg0: ir0.inst25, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst27:      {kind: PointerType, arg0: ir0.inst21, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst28:      {kind: PointerType, arg0: ir0.inst22, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst29:      {kind: BindName, arg0: entity_name2, arg1: ir0.inst49, type: type(symbolic_constant4)}
// CHECK:STDOUT:     ir0.inst30:      {kind: PatternType, arg0: ir0.inst27, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst31:      {kind: BindingPattern, arg0: entity_name2, type: type(symbolic_constant6)}
// CHECK:STDOUT:     ir0.inst32:      {kind: PatternType, arg0: ir0.inst28, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst33:      {kind: ValueParamPattern, arg0: ir0.inst31, arg1: call_param0, type: type(symbolic_constant6)}
// CHECK:STDOUT:     ir0.inst34:      {kind: NameRef, arg0: name1, arg1: ir0.inst20, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst35:      {kind: PointerType, arg0: ir0.inst34, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst36:      {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst37:      {kind: TupleLiteral, arg0: inst_block_empty, type: type(ir0.inst36)}
// CHECK:STDOUT:     ir0.inst38:      {kind: TupleType, arg0: inst_block9, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst39:      {kind: TupleLiteral, arg0: inst_block8, type: type(ir0.inst38)}
// CHECK:STDOUT:     ir0.inst40:      {kind: PointerType, arg0: ir0.inst38, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst41:      {kind: Converted, arg0: ir0.inst37, arg1: ir0.inst36, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst42:      {kind: TupleType, arg0: inst_block11, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst43:      {kind: Converted, arg0: ir0.inst39, arg1: ir0.inst42, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst44:      {kind: TupleType, arg0: inst_block12, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst45:      {kind: PatternType, arg0: ir0.inst42, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst46:      {kind: ReturnSlotPattern, arg0: ir0.inst43, type: type(symbolic_constant10)}
// CHECK:STDOUT:     ir0.inst47:      {kind: PatternType, arg0: ir0.inst44, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst48:      {kind: OutParamPattern, arg0: ir0.inst46, arg1: call_param1, type: type(symbolic_constant10)}
// CHECK:STDOUT:     ir0.inst49:      {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant4)}
// CHECK:STDOUT:     ir0.inst50:      {kind: SpliceBlock, arg0: inst_block6, arg1: ir0.inst26, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst51:      {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant8)}
// CHECK:STDOUT:     ir0.inst52:      {kind: ReturnSlot, arg0: ir0.inst42, arg1: ir0.inst51, type: type(symbolic_constant8)}
// CHECK:STDOUT:     ir0.inst53:      {kind: FunctionDecl, arg0: function0, arg1: inst_block15, type: type(ir0.inst54)}
// CHECK:STDOUT:     ir0.inst54:      {kind: FunctionType, arg0: function0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst55:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst54)}
// CHECK:STDOUT:     ir0.inst56:      {kind: PointerType, arg0: ir0.inst42, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst57:      {kind: RequireCompleteType, arg0: ir0.inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst58:      {kind: RequireCompleteType, arg0: ir0.inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst59:      {kind: RequireCompleteType, arg0: ir0.inst44, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst60:      {kind: RequireCompleteType, arg0: ir0.inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst61:      {kind: RequireCompleteType, arg0: ir0.inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst62:      {kind: RequireCompleteType, arg0: ir0.inst28, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst63:      {kind: NameRef, arg0: name2, arg1: ir0.inst29, type: type(symbolic_constant4)}
// CHECK:STDOUT:     ir0.inst64:      {kind: TupleLiteral, arg0: inst_block_empty, type: type(ir0.inst36)}
// CHECK:STDOUT:     ir0.inst65:      {kind: TupleLiteral, arg0: inst_block21, type: type(symbolic_constant8)}
// CHECK:STDOUT:     ir0.inst66:      {kind: RequireCompleteType, arg0: ir0.inst42, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst67:      {kind: TupleAccess, arg0: ir0.inst52, arg1: element0, type: type(symbolic_constant4)}
// CHECK:STDOUT:     ir0.inst68:      {kind: RequireCompleteType, arg0: ir0.inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst69:      {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name3, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst70:      {kind: InterfaceDecl, arg0: interface0, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst71:      {kind: FacetType, arg0: facet_type1, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst72:      {kind: BindSymbolicName, arg0: entity_name4, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst73:      {kind: ImportRefUnloaded, arg0: import_ir_inst2, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst74:      {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(ir0.inst76)}
// CHECK:STDOUT:     ir0.inst75:      {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name5}
// CHECK:STDOUT:     ir0.inst76:      {kind: AssociatedEntityType, arg0: interface0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst77:      {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(ir0.inst80)}
// CHECK:STDOUT:     ir0.inst78:      {kind: AssociatedEntity, arg0: element0, arg1: ir0.inst77, type: type(ir0.inst76)}
// CHECK:STDOUT:     ir0.inst79:      {kind: FunctionDecl, arg0: function1, arg1: inst_block_empty, type: type(ir0.inst80)}
// CHECK:STDOUT:     ir0.inst80:      {kind: FunctionType, arg0: function1, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst81:      {kind: StructValue, arg0: inst_block_empty, type: type(ir0.inst80)}
// CHECK:STDOUT:     ir0.inst82:      {kind: SymbolicBindingType, arg0: entity_name4, arg1: ir0.inst72, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst83:      {kind: PatternType, arg0: ir0.inst82, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst84:      {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant19)}
// CHECK:STDOUT:     ir0.inst85:      {kind: OutParamPattern, arg0: ir0.inst84, arg1: call_param1, type: type(symbolic_constant19)}
// CHECK:STDOUT:     ir0.inst86:      {kind: BindingPattern, arg0: entity_name8, type: type(symbolic_constant19)}
// CHECK:STDOUT:     ir0.inst87:      {kind: ValueParamPattern, arg0: ir0.inst86, arg1: call_param0, type: type(symbolic_constant19)}
// CHECK:STDOUT:     ir0.inst88:      {kind: ImportRefLoaded, arg0: import_ir_inst11, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst89:      {kind: BindSymbolicName, arg0: entity_name4, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst90:      {kind: SymbolicBindingType, arg0: entity_name4, arg1: ir0.inst89, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst91:      {kind: PatternType, arg0: ir0.inst90, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst92:      {kind: LookupImplWitness, arg0: ir0.inst27, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst93:      {kind: ImportRefUnloaded, arg0: import_ir_inst15, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst94:      {kind: ImplDecl, arg0: impl0, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst95:      {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst96:      {kind: SymbolicBindingType, arg0: entity_name1, arg1: ir0.inst95, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst97:      {kind: ConstType, arg0: ir0.inst96, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst98:      {kind: PatternType, arg0: ir0.inst71, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst99:      {kind: SymbolicBindingPattern, arg0: entity_name13, type: type(ir0.inst98)}
// CHECK:STDOUT:     ir0.inst100:     {kind: ImportRefLoaded, arg0: import_ir_inst18, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst101:     {kind: ImportRefLoaded, arg0: import_ir_inst19, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst102:     {kind: ImportRefLoaded, arg0: import_ir_inst20, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst103:     {kind: ImportRefUnloaded, arg0: import_ir_inst21, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst104:     {kind: ImplWitnessTable, arg0: inst_block30, arg1: impl0}
// CHECK:STDOUT:     ir0.inst105:     {kind: ImplWitness, arg0: ir0.inst104, arg1: specific2, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst106:     {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst107:     {kind: SymbolicBindingType, arg0: entity_name1, arg1: ir0.inst106, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst108:     {kind: ConstType, arg0: ir0.inst107, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst109:     {kind: ImplWitness, arg0: ir0.inst104, arg1: specific3, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst110:     {kind: FunctionDecl, arg0: function2, arg1: inst_block_empty, type: type(symbolic_constant38)}
// CHECK:STDOUT:     ir0.inst111:     {kind: FunctionType, arg0: function2, arg1: specific2, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst112:     {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant38)}
// CHECK:STDOUT:     ir0.inst113:     {kind: PatternType, arg0: ir0.inst97, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst114:     {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant43)}
// CHECK:STDOUT:     ir0.inst115:     {kind: OutParamPattern, arg0: ir0.inst114, arg1: call_param1, type: type(symbolic_constant43)}
// CHECK:STDOUT:     ir0.inst116:     {kind: BindingPattern, arg0: entity_name17, type: type(symbolic_constant43)}
// CHECK:STDOUT:     ir0.inst117:     {kind: ValueParamPattern, arg0: ir0.inst116, arg1: call_param0, type: type(symbolic_constant43)}
// CHECK:STDOUT:     ir0.inst118:     {kind: ImportRefLoaded, arg0: import_ir_inst32, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst119:     {kind: FunctionType, arg0: function2, arg1: specific3, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst120:     {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant45)}
// CHECK:STDOUT:     ir0.inst121:     {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst122:     {kind: SymbolicBindingType, arg0: entity_name1, arg1: ir0.inst121, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst123:     {kind: ConstType, arg0: ir0.inst122, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst124:     {kind: PatternType, arg0: ir0.inst123, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst125:     {kind: RequireCompleteType, arg0: ir0.inst97, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst126:     {kind: RequireCompleteType, arg0: ir0.inst96, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst127:     {kind: LookupImplWitness, arg0: ir0.inst95, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst128:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst95, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst129:     {kind: ImplWitnessAccess, arg0: ir0.inst127, arg1: element0, type: type(symbolic_constant60)}
// CHECK:STDOUT:     ir0.inst130:     {kind: SpecificImplFunction, arg0: ir0.inst129, arg1: specific5, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst131:     {kind: RequireCompleteType, arg0: ir0.inst123, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst132:     {kind: RequireCompleteType, arg0: ir0.inst122, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst133:     {kind: LookupImplWitness, arg0: ir0.inst121, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst134:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst121, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst135:     {kind: ImplWitnessAccess, arg0: ir0.inst133, arg1: element0, type: type(symbolic_constant69)}
// CHECK:STDOUT:     ir0.inst136:     {kind: SpecificImplFunction, arg0: ir0.inst135, arg1: specific6, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst137:     {kind: PatternType, arg0: ir0.inst96, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst138:     {kind: ImportRefUnloaded, arg0: import_ir_inst45, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst139:     {kind: ImplDecl, arg0: impl1, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst140:     {kind: ImportRefLoaded, arg0: import_ir_inst47, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst141:     {kind: ImportRefLoaded, arg0: import_ir_inst48, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst142:     {kind: ImportRefUnloaded, arg0: import_ir_inst49, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst143:     {kind: ImplDecl, arg0: impl2, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst144:     {kind: ImportRefLoaded, arg0: import_ir_inst51, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst145:     {kind: ImportRefLoaded, arg0: import_ir_inst52, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst146:     {kind: ImportRefUnloaded, arg0: import_ir_inst53, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst147:     {kind: ImplDecl, arg0: impl3, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst148:     {kind: ImportRefLoaded, arg0: import_ir_inst55, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst149:     {kind: ImportRefLoaded, arg0: import_ir_inst56, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst150:     {kind: ImportRefUnloaded, arg0: import_ir_inst57, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst151:     {kind: ImplDecl, arg0: impl4, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst152:     {kind: ImportRefLoaded, arg0: import_ir_inst59, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst153:     {kind: ImportRefLoaded, arg0: import_ir_inst60, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst154:     {kind: ImportRefLoaded, arg0: import_ir_inst61, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst155:     {kind: ImplDecl, arg0: impl5, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst156:     {kind: SymbolicBindingPattern, arg0: entity_name21, type: type(ir0.inst23)}
// CHECK:STDOUT:     ir0.inst157:     {kind: ImportRefLoaded, arg0: import_ir_inst64, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst158:     {kind: ImportRefLoaded, arg0: import_ir_inst65, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst159:     {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst160:     {kind: ImportRefUnloaded, arg0: import_ir_inst67, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst161:     {kind: ImplWitnessTable, arg0: inst_block47, arg1: impl5}
// CHECK:STDOUT:     ir0.inst162:     {kind: ImplWitness, arg0: ir0.inst161, arg1: specific7, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst163:     {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst164:     {kind: PointerType, arg0: ir0.inst163, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst165:     {kind: ImplWitness, arg0: ir0.inst161, arg1: specific8, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst166:     {kind: FunctionDecl, arg0: function3, arg1: inst_block_empty, type: type(symbolic_constant81)}
// CHECK:STDOUT:     ir0.inst167:     {kind: FunctionType, arg0: function3, arg1: specific7, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst168:     {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant81)}
// CHECK:STDOUT:     ir0.inst169:     {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant85)}
// CHECK:STDOUT:     ir0.inst170:     {kind: OutParamPattern, arg0: ir0.inst169, arg1: call_param1, type: type(symbolic_constant85)}
// CHECK:STDOUT:     ir0.inst171:     {kind: BindingPattern, arg0: entity_name24, type: type(symbolic_constant85)}
// CHECK:STDOUT:     ir0.inst172:     {kind: ValueParamPattern, arg0: ir0.inst171, arg1: call_param0, type: type(symbolic_constant85)}
// CHECK:STDOUT:     ir0.inst173:     {kind: ImportRefLoaded, arg0: import_ir_inst77, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst174:     {kind: FunctionType, arg0: function3, arg1: specific8, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst175:     {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant88)}
// CHECK:STDOUT:     ir0.inst176:     {kind: RequireCompleteType, arg0: ir0.inst164, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst177:     {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst178:     {kind: PointerType, arg0: ir0.inst177, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst179:     {kind: PatternType, arg0: ir0.inst178, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst180:     {kind: ImportRefUnloaded, arg0: import_ir_inst84, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst181:     {kind: ImplDecl, arg0: impl6, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst182:     {kind: ImportRefLoaded, arg0: import_ir_inst86, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst183:     {kind: ImportRefLoaded, arg0: import_ir_inst87, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst184:     {kind: ImportRefUnloaded, arg0: import_ir_inst88, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst185:     {kind: ImplDecl, arg0: impl7, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst186:     {kind: ImportRefLoaded, arg0: import_ir_inst90, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst187:     {kind: ImportRefLoaded, arg0: import_ir_inst91, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst188:     {kind: ImportRefUnloaded, arg0: import_ir_inst92, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst189:     {kind: ImplDecl, arg0: impl8, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst190:     {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst191:     {kind: SymbolicBindingType, arg0: entity_name26, arg1: ir0.inst190, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst192:     {kind: TupleType, arg0: inst_block57, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst193:     {kind: SymbolicBindingPattern, arg0: entity_name28, type: type(ir0.inst98)}
// CHECK:STDOUT:     ir0.inst194:     {kind: SymbolicBindingPattern, arg0: entity_name29, type: type(ir0.inst98)}
// CHECK:STDOUT:     ir0.inst195:     {kind: ImportRefLoaded, arg0: import_ir_inst96, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst196:     {kind: ImportRefLoaded, arg0: import_ir_inst97, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst197:     {kind: ImportRefLoaded, arg0: import_ir_inst98, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst198:     {kind: ImportRefLoaded, arg0: import_ir_inst99, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst199:     {kind: ImportRefUnloaded, arg0: import_ir_inst100, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst200:     {kind: ImplWitnessTable, arg0: inst_block60, arg1: impl8}
// CHECK:STDOUT:     ir0.inst201:     {kind: ImplWitness, arg0: ir0.inst200, arg1: specific10, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst202:     {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst203:     {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst204:     {kind: SymbolicBindingType, arg0: entity_name1, arg1: ir0.inst202, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst205:     {kind: SymbolicBindingType, arg0: entity_name26, arg1: ir0.inst203, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst206:     {kind: TupleType, arg0: inst_block62, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst207:     {kind: ImplWitness, arg0: ir0.inst200, arg1: specific11, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst208:     {kind: FunctionDecl, arg0: function4, arg1: inst_block_empty, type: type(symbolic_constant114)}
// CHECK:STDOUT:     ir0.inst209:     {kind: FunctionType, arg0: function4, arg1: specific10, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst210:     {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant114)}
// CHECK:STDOUT:     ir0.inst211:     {kind: PatternType, arg0: ir0.inst192, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst212:     {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant119)}
// CHECK:STDOUT:     ir0.inst213:     {kind: OutParamPattern, arg0: ir0.inst212, arg1: call_param1, type: type(symbolic_constant119)}
// CHECK:STDOUT:     ir0.inst214:     {kind: BindingPattern, arg0: entity_name36, type: type(symbolic_constant119)}
// CHECK:STDOUT:     ir0.inst215:     {kind: ValueParamPattern, arg0: ir0.inst214, arg1: call_param0, type: type(symbolic_constant119)}
// CHECK:STDOUT:     ir0.inst216:     {kind: ImportRefLoaded, arg0: import_ir_inst113, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst217:     {kind: ImportRefLoaded, arg0: import_ir_inst114, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst218:     {kind: FunctionType, arg0: function4, arg1: specific11, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst219:     {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant121)}
// CHECK:STDOUT:     ir0.inst220:     {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst221:     {kind: SymbolicBindingType, arg0: entity_name1, arg1: ir0.inst220, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst222:     {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst223:     {kind: SymbolicBindingType, arg0: entity_name26, arg1: ir0.inst222, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst224:     {kind: TupleType, arg0: inst_block70, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst225:     {kind: PatternType, arg0: ir0.inst224, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst226:     {kind: RequireCompleteType, arg0: ir0.inst192, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst227:     {kind: RequireCompleteType, arg0: ir0.inst191, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst228:     {kind: LookupImplWitness, arg0: ir0.inst190, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst229:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst190, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst230:     {kind: ImplWitnessAccess, arg0: ir0.inst228, arg1: element0, type: type(symbolic_constant145)}
// CHECK:STDOUT:     ir0.inst231:     {kind: SpecificImplFunction, arg0: ir0.inst230, arg1: specific13, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst232:     {kind: RequireCompleteType, arg0: ir0.inst224, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst233:     {kind: RequireCompleteType, arg0: ir0.inst221, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst234:     {kind: LookupImplWitness, arg0: ir0.inst220, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst235:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst220, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst236:     {kind: ImplWitnessAccess, arg0: ir0.inst234, arg1: element0, type: type(symbolic_constant154)}
// CHECK:STDOUT:     ir0.inst237:     {kind: SpecificImplFunction, arg0: ir0.inst236, arg1: specific14, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst238:     {kind: RequireCompleteType, arg0: ir0.inst223, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst239:     {kind: LookupImplWitness, arg0: ir0.inst222, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst240:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst222, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst241:     {kind: ImplWitnessAccess, arg0: ir0.inst239, arg1: element0, type: type(symbolic_constant159)}
// CHECK:STDOUT:     ir0.inst242:     {kind: SpecificImplFunction, arg0: ir0.inst241, arg1: specific15, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst243:     {kind: PatternType, arg0: ir0.inst191, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst244:     {kind: ImportRefUnloaded, arg0: import_ir_inst134, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst245:     {kind: ImplDecl, arg0: impl9, arg1: inst_block_empty}
// CHECK:STDOUT:     ir0.inst246:     {kind: BindSymbolicName, arg0: entity_name41, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst247:     {kind: SymbolicBindingType, arg0: entity_name41, arg1: ir0.inst246, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst248:     {kind: TupleType, arg0: inst_block79, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst249:     {kind: SymbolicBindingPattern, arg0: entity_name43, type: type(ir0.inst98)}
// CHECK:STDOUT:     ir0.inst250:     {kind: SymbolicBindingPattern, arg0: entity_name44, type: type(ir0.inst98)}
// CHECK:STDOUT:     ir0.inst251:     {kind: SymbolicBindingPattern, arg0: entity_name45, type: type(ir0.inst98)}
// CHECK:STDOUT:     ir0.inst252:     {kind: ImportRefLoaded, arg0: import_ir_inst139, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst253:     {kind: ImportRefLoaded, arg0: import_ir_inst140, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst254:     {kind: ImportRefLoaded, arg0: import_ir_inst141, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst255:     {kind: ImportRefLoaded, arg0: import_ir_inst142, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst256:     {kind: ImportRefLoaded, arg0: import_ir_inst143, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst257:     {kind: ImportRefUnloaded, arg0: import_ir_inst144, arg1: entity_name<none>}
// CHECK:STDOUT:     ir0.inst258:     {kind: ImplWitnessTable, arg0: inst_block82, arg1: impl9}
// CHECK:STDOUT:     ir0.inst259:     {kind: ImplWitness, arg0: ir0.inst258, arg1: specific16, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst260:     {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst261:     {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst262:     {kind: BindSymbolicName, arg0: entity_name41, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst263:     {kind: SymbolicBindingType, arg0: entity_name1, arg1: ir0.inst260, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst264:     {kind: SymbolicBindingType, arg0: entity_name26, arg1: ir0.inst261, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst265:     {kind: SymbolicBindingType, arg0: entity_name41, arg1: ir0.inst262, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst266:     {kind: TupleType, arg0: inst_block84, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst267:     {kind: ImplWitness, arg0: ir0.inst258, arg1: specific17, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst268:     {kind: FunctionDecl, arg0: function5, arg1: inst_block_empty, type: type(symbolic_constant186)}
// CHECK:STDOUT:     ir0.inst269:     {kind: FunctionType, arg0: function5, arg1: specific16, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst270:     {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant186)}
// CHECK:STDOUT:     ir0.inst271:     {kind: PatternType, arg0: ir0.inst248, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst272:     {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant191)}
// CHECK:STDOUT:     ir0.inst273:     {kind: OutParamPattern, arg0: ir0.inst272, arg1: call_param1, type: type(symbolic_constant191)}
// CHECK:STDOUT:     ir0.inst274:     {kind: BindingPattern, arg0: entity_name55, type: type(symbolic_constant191)}
// CHECK:STDOUT:     ir0.inst275:     {kind: ValueParamPattern, arg0: ir0.inst274, arg1: call_param0, type: type(symbolic_constant191)}
// CHECK:STDOUT:     ir0.inst276:     {kind: ImportRefLoaded, arg0: import_ir_inst159, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst277:     {kind: ImportRefLoaded, arg0: import_ir_inst160, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst278:     {kind: ImportRefLoaded, arg0: import_ir_inst161, arg1: entity_name<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst279:     {kind: FunctionType, arg0: function5, arg1: specific17, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst280:     {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant193)}
// CHECK:STDOUT:     ir0.inst281:     {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst282:     {kind: SymbolicBindingType, arg0: entity_name1, arg1: ir0.inst281, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst283:     {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst284:     {kind: SymbolicBindingType, arg0: entity_name26, arg1: ir0.inst283, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst285:     {kind: BindSymbolicName, arg0: entity_name41, arg1: inst<none>, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst286:     {kind: SymbolicBindingType, arg0: entity_name41, arg1: ir0.inst285, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst287:     {kind: TupleType, arg0: inst_block92, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst288:     {kind: PatternType, arg0: ir0.inst287, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst289:     {kind: RequireCompleteType, arg0: ir0.inst248, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst290:     {kind: RequireCompleteType, arg0: ir0.inst247, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst291:     {kind: LookupImplWitness, arg0: ir0.inst246, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst292:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst246, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst293:     {kind: ImplWitnessAccess, arg0: ir0.inst291, arg1: element0, type: type(symbolic_constant226)}
// CHECK:STDOUT:     ir0.inst294:     {kind: SpecificImplFunction, arg0: ir0.inst293, arg1: specific19, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst295:     {kind: RequireCompleteType, arg0: ir0.inst287, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst296:     {kind: RequireCompleteType, arg0: ir0.inst282, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst297:     {kind: LookupImplWitness, arg0: ir0.inst281, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst298:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst281, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst299:     {kind: ImplWitnessAccess, arg0: ir0.inst297, arg1: element0, type: type(symbolic_constant235)}
// CHECK:STDOUT:     ir0.inst300:     {kind: SpecificImplFunction, arg0: ir0.inst299, arg1: specific20, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst301:     {kind: RequireCompleteType, arg0: ir0.inst284, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst302:     {kind: LookupImplWitness, arg0: ir0.inst283, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst303:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst283, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst304:     {kind: ImplWitnessAccess, arg0: ir0.inst302, arg1: element0, type: type(symbolic_constant240)}
// CHECK:STDOUT:     ir0.inst305:     {kind: SpecificImplFunction, arg0: ir0.inst304, arg1: specific21, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst306:     {kind: RequireCompleteType, arg0: ir0.inst286, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst307:     {kind: LookupImplWitness, arg0: ir0.inst285, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst308:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst285, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst309:     {kind: ImplWitnessAccess, arg0: ir0.inst307, arg1: element0, type: type(symbolic_constant245)}
// CHECK:STDOUT:     ir0.inst310:     {kind: SpecificImplFunction, arg0: ir0.inst309, arg1: specific22, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst311:     {kind: PatternType, arg0: ir0.inst247, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst312:     {kind: LookupImplWitness, arg0: ir0.inst27, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst313:     {kind: LookupImplWitness, arg0: ir0.inst28, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst314:     {kind: FacetValue, arg0: ir0.inst27, arg1: inst_block103, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst315:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst314, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst316:     {kind: ImplWitnessAccess, arg0: ir0.inst312, arg1: element0, type: type(symbolic_constant256)}
// CHECK:STDOUT:     ir0.inst317:     {kind: ImplWitnessAccess, arg0: ir0.inst312, arg1: element0, type: type(symbolic_constant253)}
// CHECK:STDOUT:     ir0.inst318:     {kind: FacetValue, arg0: ir0.inst28, arg1: inst_block104, type: type(ir0.inst71)}
// CHECK:STDOUT:     ir0.inst319:     {kind: FunctionTypeWithSelfType, arg0: ir0.inst80, arg1: ir0.inst318, type: type(TypeType)}
// CHECK:STDOUT:     ir0.inst320:     {kind: ImplWitnessAccess, arg0: ir0.inst313, arg1: element0, type: type(symbolic_constant256)}
// CHECK:STDOUT:     ir0.inst321:     {kind: BoundMethod, arg0: ir0.inst63, arg1: ir0.inst316, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     ir0.inst322:     {kind: SpecificImplFunction, arg0: ir0.inst316, arg1: specific23, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst323:     {kind: SpecificImplFunction, arg0: ir0.inst317, arg1: specific23, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst324:     {kind: SpecificImplFunction, arg0: ir0.inst320, arg1: specific24, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     ir0.inst325:     {kind: BoundMethod, arg0: ir0.inst63, arg1: ir0.inst322, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     ir0.inst326:     {kind: RequireCompleteType, arg0: ir0.inst27, type: type(inst(WitnessType))}
// CHECK:STDOUT:     ir0.inst327:     {kind: Call, arg0: ir0.inst325, arg1: inst_block108, type: type(symbolic_constant4)}
// CHECK:STDOUT:     ir0.inst328:     {kind: InitializeFrom, arg0: ir0.inst327, arg1: ir0.inst67, type: type(symbolic_constant4)}
// CHECK:STDOUT:     ir0.inst329:     {kind: TupleAccess, arg0: ir0.inst52, arg1: element1, type: type(ir0.inst36)}
// CHECK:STDOUT:     ir0.inst330:     {kind: TupleInit, arg0: inst_block_empty, arg1: ir0.inst329, type: type(ir0.inst36)}
// CHECK:STDOUT:     ir0.inst331:     {kind: TupleValue, arg0: inst_block_empty, type: type(ir0.inst36)}
// CHECK:STDOUT:     ir0.inst332:     {kind: Converted, arg0: ir0.inst64, arg1: ir0.inst330, type: type(ir0.inst36)}
// CHECK:STDOUT:     ir0.inst333:     {kind: TupleInit, arg0: inst_block109, arg1: ir0.inst52, type: type(symbolic_constant8)}
// CHECK:STDOUT:     ir0.inst334:     {kind: Converted, arg0: ir0.inst65, arg1: ir0.inst333, type: type(symbolic_constant8)}
// CHECK:STDOUT:     ir0.inst335:     {kind: ReturnExpr, arg0: ir0.inst334, arg1: ir0.inst52}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst14:          concrete_constant(inst14)
// CHECK:STDOUT:       ir0.inst16:      concrete_constant(ir0.inst16)
// CHECK:STDOUT:       ir0.inst17:      concrete_constant(ir0.inst17)
// CHECK:STDOUT:       ir0.inst18:      symbolic_constant0
// CHECK:STDOUT:       ir0.inst19:      symbolic_constant0
// CHECK:STDOUT:       ir0.inst20:      symbolic_constant2
// CHECK:STDOUT:       ir0.inst21:      symbolic_constant1
// CHECK:STDOUT:       ir0.inst22:      symbolic_constant2
// CHECK:STDOUT:       ir0.inst23:      concrete_constant(ir0.inst23)
// CHECK:STDOUT:       ir0.inst24:      concrete_constant(ir0.inst24)
// CHECK:STDOUT:       ir0.inst25:      symbolic_constant2
// CHECK:STDOUT:       ir0.inst26:      symbolic_constant4
// CHECK:STDOUT:       ir0.inst27:      symbolic_constant3
// CHECK:STDOUT:       ir0.inst28:      symbolic_constant4
// CHECK:STDOUT:       ir0.inst30:      symbolic_constant5
// CHECK:STDOUT:       ir0.inst31:      concrete_constant(ir0.inst31)
// CHECK:STDOUT:       ir0.inst32:      symbolic_constant6
// CHECK:STDOUT:       ir0.inst33:      concrete_constant(ir0.inst33)
// CHECK:STDOUT:       ir0.inst34:      symbolic_constant2
// CHECK:STDOUT:       ir0.inst35:      symbolic_constant4
// CHECK:STDOUT:       ir0.inst36:      concrete_constant(ir0.inst36)
// CHECK:STDOUT:       ir0.inst38:      concrete_constant(ir0.inst38)
// CHECK:STDOUT:       ir0.inst40:      concrete_constant(ir0.inst40)
// CHECK:STDOUT:       ir0.inst41:      concrete_constant(ir0.inst36)
// CHECK:STDOUT:       ir0.inst42:      symbolic_constant7
// CHECK:STDOUT:       ir0.inst43:      symbolic_constant8
// CHECK:STDOUT:       ir0.inst44:      symbolic_constant8
// CHECK:STDOUT:       ir0.inst45:      symbolic_constant9
// CHECK:STDOUT:       ir0.inst46:      concrete_constant(ir0.inst46)
// CHECK:STDOUT:       ir0.inst47:      symbolic_constant10
// CHECK:STDOUT:       ir0.inst48:      concrete_constant(ir0.inst48)
// CHECK:STDOUT:       ir0.inst50:      symbolic_constant4
// CHECK:STDOUT:       ir0.inst53:      concrete_constant(ir0.inst55)
// CHECK:STDOUT:       ir0.inst54:      concrete_constant(ir0.inst54)
// CHECK:STDOUT:       ir0.inst55:      concrete_constant(ir0.inst55)
// CHECK:STDOUT:       ir0.inst56:      symbolic_constant11
// CHECK:STDOUT:       ir0.inst57:      symbolic_constant13
// CHECK:STDOUT:       ir0.inst58:      symbolic_constant12
// CHECK:STDOUT:       ir0.inst59:      symbolic_constant13
// CHECK:STDOUT:       ir0.inst60:      symbolic_constant15
// CHECK:STDOUT:       ir0.inst61:      symbolic_constant14
// CHECK:STDOUT:       ir0.inst62:      symbolic_constant15
// CHECK:STDOUT:       ir0.inst66:      symbolic_constant13
// CHECK:STDOUT:       ir0.inst68:      symbolic_constant15
// CHECK:STDOUT:       ir0.inst69:      concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst70:      concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst71:      concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst72:      symbolic_constant16
// CHECK:STDOUT:       ir0.inst73:      constant<none>
// CHECK:STDOUT:       ir0.inst74:      concrete_constant(ir0.inst78)
// CHECK:STDOUT:       ir0.inst75:      constant<none>
// CHECK:STDOUT:       ir0.inst76:      concrete_constant(ir0.inst76)
// CHECK:STDOUT:       ir0.inst77:      concrete_constant(ir0.inst81)
// CHECK:STDOUT:       ir0.inst78:      concrete_constant(ir0.inst78)
// CHECK:STDOUT:       ir0.inst79:      concrete_constant(ir0.inst81)
// CHECK:STDOUT:       ir0.inst80:      concrete_constant(ir0.inst80)
// CHECK:STDOUT:       ir0.inst81:      concrete_constant(ir0.inst81)
// CHECK:STDOUT:       ir0.inst82:      symbolic_constant17
// CHECK:STDOUT:       ir0.inst83:      symbolic_constant18
// CHECK:STDOUT:       ir0.inst84:      concrete_constant(ir0.inst84)
// CHECK:STDOUT:       ir0.inst85:      concrete_constant(ir0.inst85)
// CHECK:STDOUT:       ir0.inst86:      concrete_constant(ir0.inst86)
// CHECK:STDOUT:       ir0.inst87:      concrete_constant(ir0.inst87)
// CHECK:STDOUT:       ir0.inst88:      symbolic_constant16
// CHECK:STDOUT:       ir0.inst89:      symbolic_constant22
// CHECK:STDOUT:       ir0.inst90:      symbolic_constant23
// CHECK:STDOUT:       ir0.inst91:      symbolic_constant24
// CHECK:STDOUT:       ir0.inst92:      symbolic_constant251
// CHECK:STDOUT:       ir0.inst93:      constant<none>
// CHECK:STDOUT:       ir0.inst94:      concrete_constant(ir0.inst94)
// CHECK:STDOUT:       ir0.inst95:      symbolic_constant25
// CHECK:STDOUT:       ir0.inst96:      symbolic_constant26
// CHECK:STDOUT:       ir0.inst97:      symbolic_constant27
// CHECK:STDOUT:       ir0.inst98:      concrete_constant(ir0.inst98)
// CHECK:STDOUT:       ir0.inst99:      concrete_constant(ir0.inst99)
// CHECK:STDOUT:       ir0.inst100:     symbolic_constant29
// CHECK:STDOUT:       ir0.inst101:     symbolic_constant28
// CHECK:STDOUT:       ir0.inst102:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst103:     constant<none>
// CHECK:STDOUT:       ir0.inst104:     concrete_constant(ir0.inst104)
// CHECK:STDOUT:       ir0.inst105:     symbolic_constant32
// CHECK:STDOUT:       ir0.inst106:     symbolic_constant34
// CHECK:STDOUT:       ir0.inst107:     symbolic_constant35
// CHECK:STDOUT:       ir0.inst108:     symbolic_constant36
// CHECK:STDOUT:       ir0.inst109:     symbolic_constant37
// CHECK:STDOUT:       ir0.inst110:     symbolic_constant40
// CHECK:STDOUT:       ir0.inst111:     symbolic_constant38
// CHECK:STDOUT:       ir0.inst112:     symbolic_constant39
// CHECK:STDOUT:       ir0.inst113:     symbolic_constant42
// CHECK:STDOUT:       ir0.inst114:     concrete_constant(ir0.inst114)
// CHECK:STDOUT:       ir0.inst115:     concrete_constant(ir0.inst115)
// CHECK:STDOUT:       ir0.inst116:     concrete_constant(ir0.inst116)
// CHECK:STDOUT:       ir0.inst117:     concrete_constant(ir0.inst117)
// CHECK:STDOUT:       ir0.inst118:     symbolic_constant29
// CHECK:STDOUT:       ir0.inst119:     symbolic_constant45
// CHECK:STDOUT:       ir0.inst120:     symbolic_constant46
// CHECK:STDOUT:       ir0.inst121:     symbolic_constant50
// CHECK:STDOUT:       ir0.inst122:     symbolic_constant51
// CHECK:STDOUT:       ir0.inst123:     symbolic_constant52
// CHECK:STDOUT:       ir0.inst124:     symbolic_constant53
// CHECK:STDOUT:       ir0.inst125:     symbolic_constant54
// CHECK:STDOUT:       ir0.inst126:     symbolic_constant56
// CHECK:STDOUT:       ir0.inst127:     symbolic_constant58
// CHECK:STDOUT:       ir0.inst128:     symbolic_constant60
// CHECK:STDOUT:       ir0.inst129:     symbolic_constant62
// CHECK:STDOUT:       ir0.inst130:     symbolic_constant64
// CHECK:STDOUT:       ir0.inst131:     symbolic_constant66
// CHECK:STDOUT:       ir0.inst132:     symbolic_constant67
// CHECK:STDOUT:       ir0.inst133:     symbolic_constant68
// CHECK:STDOUT:       ir0.inst134:     symbolic_constant69
// CHECK:STDOUT:       ir0.inst135:     symbolic_constant70
// CHECK:STDOUT:       ir0.inst136:     symbolic_constant71
// CHECK:STDOUT:       ir0.inst137:     symbolic_constant72
// CHECK:STDOUT:       ir0.inst138:     constant<none>
// CHECK:STDOUT:       ir0.inst139:     concrete_constant(ir0.inst139)
// CHECK:STDOUT:       ir0.inst140:     concrete_constant(inst(BoolType))
// CHECK:STDOUT:       ir0.inst141:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst142:     constant<none>
// CHECK:STDOUT:       ir0.inst143:     concrete_constant(ir0.inst143)
// CHECK:STDOUT:       ir0.inst144:     concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       ir0.inst145:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst146:     constant<none>
// CHECK:STDOUT:       ir0.inst147:     concrete_constant(ir0.inst147)
// CHECK:STDOUT:       ir0.inst148:     concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       ir0.inst149:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst150:     constant<none>
// CHECK:STDOUT:       ir0.inst151:     concrete_constant(ir0.inst151)
// CHECK:STDOUT:       ir0.inst152:     concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       ir0.inst153:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst154:     symbolic_constant249
// CHECK:STDOUT:       ir0.inst155:     concrete_constant(ir0.inst155)
// CHECK:STDOUT:       ir0.inst156:     concrete_constant(ir0.inst156)
// CHECK:STDOUT:       ir0.inst157:     symbolic_constant74
// CHECK:STDOUT:       ir0.inst158:     symbolic_constant73
// CHECK:STDOUT:       ir0.inst159:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst160:     constant<none>
// CHECK:STDOUT:       ir0.inst161:     concrete_constant(ir0.inst161)
// CHECK:STDOUT:       ir0.inst162:     symbolic_constant76
// CHECK:STDOUT:       ir0.inst163:     symbolic_constant78
// CHECK:STDOUT:       ir0.inst164:     symbolic_constant79
// CHECK:STDOUT:       ir0.inst165:     symbolic_constant80
// CHECK:STDOUT:       ir0.inst166:     symbolic_constant83
// CHECK:STDOUT:       ir0.inst167:     symbolic_constant81
// CHECK:STDOUT:       ir0.inst168:     symbolic_constant82
// CHECK:STDOUT:       ir0.inst169:     concrete_constant(ir0.inst169)
// CHECK:STDOUT:       ir0.inst170:     concrete_constant(ir0.inst170)
// CHECK:STDOUT:       ir0.inst171:     concrete_constant(ir0.inst171)
// CHECK:STDOUT:       ir0.inst172:     concrete_constant(ir0.inst172)
// CHECK:STDOUT:       ir0.inst173:     symbolic_constant74
// CHECK:STDOUT:       ir0.inst174:     symbolic_constant88
// CHECK:STDOUT:       ir0.inst175:     symbolic_constant89
// CHECK:STDOUT:       ir0.inst176:     symbolic_constant90
// CHECK:STDOUT:       ir0.inst177:     symbolic_constant93
// CHECK:STDOUT:       ir0.inst178:     symbolic_constant94
// CHECK:STDOUT:       ir0.inst179:     symbolic_constant95
// CHECK:STDOUT:       ir0.inst180:     constant<none>
// CHECK:STDOUT:       ir0.inst181:     concrete_constant(ir0.inst181)
// CHECK:STDOUT:       ir0.inst182:     concrete_constant(inst(TypeType))
// CHECK:STDOUT:       ir0.inst183:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst184:     constant<none>
// CHECK:STDOUT:       ir0.inst185:     concrete_constant(ir0.inst185)
// CHECK:STDOUT:       ir0.inst186:     concrete_constant(ir0.inst36)
// CHECK:STDOUT:       ir0.inst187:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst188:     constant<none>
// CHECK:STDOUT:       ir0.inst189:     concrete_constant(ir0.inst189)
// CHECK:STDOUT:       ir0.inst190:     symbolic_constant96
// CHECK:STDOUT:       ir0.inst191:     symbolic_constant97
// CHECK:STDOUT:       ir0.inst192:     symbolic_constant98
// CHECK:STDOUT:       ir0.inst193:     concrete_constant(ir0.inst193)
// CHECK:STDOUT:       ir0.inst194:     concrete_constant(ir0.inst194)
// CHECK:STDOUT:       ir0.inst195:     symbolic_constant100
// CHECK:STDOUT:       ir0.inst196:     symbolic_constant101
// CHECK:STDOUT:       ir0.inst197:     symbolic_constant99
// CHECK:STDOUT:       ir0.inst198:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst199:     constant<none>
// CHECK:STDOUT:       ir0.inst200:     concrete_constant(ir0.inst200)
// CHECK:STDOUT:       ir0.inst201:     symbolic_constant106
// CHECK:STDOUT:       ir0.inst202:     symbolic_constant108
// CHECK:STDOUT:       ir0.inst203:     symbolic_constant109
// CHECK:STDOUT:       ir0.inst204:     symbolic_constant110
// CHECK:STDOUT:       ir0.inst205:     symbolic_constant111
// CHECK:STDOUT:       ir0.inst206:     symbolic_constant112
// CHECK:STDOUT:       ir0.inst207:     symbolic_constant113
// CHECK:STDOUT:       ir0.inst208:     symbolic_constant116
// CHECK:STDOUT:       ir0.inst209:     symbolic_constant114
// CHECK:STDOUT:       ir0.inst210:     symbolic_constant115
// CHECK:STDOUT:       ir0.inst211:     symbolic_constant118
// CHECK:STDOUT:       ir0.inst212:     concrete_constant(ir0.inst212)
// CHECK:STDOUT:       ir0.inst213:     concrete_constant(ir0.inst213)
// CHECK:STDOUT:       ir0.inst214:     concrete_constant(ir0.inst214)
// CHECK:STDOUT:       ir0.inst215:     concrete_constant(ir0.inst215)
// CHECK:STDOUT:       ir0.inst216:     symbolic_constant100
// CHECK:STDOUT:       ir0.inst217:     symbolic_constant101
// CHECK:STDOUT:       ir0.inst218:     symbolic_constant121
// CHECK:STDOUT:       ir0.inst219:     symbolic_constant122
// CHECK:STDOUT:       ir0.inst220:     symbolic_constant128
// CHECK:STDOUT:       ir0.inst221:     symbolic_constant129
// CHECK:STDOUT:       ir0.inst222:     symbolic_constant130
// CHECK:STDOUT:       ir0.inst223:     symbolic_constant131
// CHECK:STDOUT:       ir0.inst224:     symbolic_constant132
// CHECK:STDOUT:       ir0.inst225:     symbolic_constant133
// CHECK:STDOUT:       ir0.inst226:     symbolic_constant134
// CHECK:STDOUT:       ir0.inst227:     symbolic_constant141
// CHECK:STDOUT:       ir0.inst228:     symbolic_constant143
// CHECK:STDOUT:       ir0.inst229:     symbolic_constant145
// CHECK:STDOUT:       ir0.inst230:     symbolic_constant147
// CHECK:STDOUT:       ir0.inst231:     symbolic_constant149
// CHECK:STDOUT:       ir0.inst232:     symbolic_constant151
// CHECK:STDOUT:       ir0.inst233:     symbolic_constant152
// CHECK:STDOUT:       ir0.inst234:     symbolic_constant153
// CHECK:STDOUT:       ir0.inst235:     symbolic_constant154
// CHECK:STDOUT:       ir0.inst236:     symbolic_constant155
// CHECK:STDOUT:       ir0.inst237:     symbolic_constant156
// CHECK:STDOUT:       ir0.inst238:     symbolic_constant157
// CHECK:STDOUT:       ir0.inst239:     symbolic_constant158
// CHECK:STDOUT:       ir0.inst240:     symbolic_constant159
// CHECK:STDOUT:       ir0.inst241:     symbolic_constant160
// CHECK:STDOUT:       ir0.inst242:     symbolic_constant161
// CHECK:STDOUT:       ir0.inst243:     symbolic_constant162
// CHECK:STDOUT:       ir0.inst244:     constant<none>
// CHECK:STDOUT:       ir0.inst245:     concrete_constant(ir0.inst245)
// CHECK:STDOUT:       ir0.inst246:     symbolic_constant163
// CHECK:STDOUT:       ir0.inst247:     symbolic_constant164
// CHECK:STDOUT:       ir0.inst248:     symbolic_constant165
// CHECK:STDOUT:       ir0.inst249:     concrete_constant(ir0.inst249)
// CHECK:STDOUT:       ir0.inst250:     concrete_constant(ir0.inst250)
// CHECK:STDOUT:       ir0.inst251:     concrete_constant(ir0.inst251)
// CHECK:STDOUT:       ir0.inst252:     symbolic_constant167
// CHECK:STDOUT:       ir0.inst253:     symbolic_constant168
// CHECK:STDOUT:       ir0.inst254:     symbolic_constant169
// CHECK:STDOUT:       ir0.inst255:     symbolic_constant166
// CHECK:STDOUT:       ir0.inst256:     concrete_constant(ir0.inst71)
// CHECK:STDOUT:       ir0.inst257:     constant<none>
// CHECK:STDOUT:       ir0.inst258:     concrete_constant(ir0.inst258)
// CHECK:STDOUT:       ir0.inst259:     symbolic_constant176
// CHECK:STDOUT:       ir0.inst260:     symbolic_constant178
// CHECK:STDOUT:       ir0.inst261:     symbolic_constant179
// CHECK:STDOUT:       ir0.inst262:     symbolic_constant180
// CHECK:STDOUT:       ir0.inst263:     symbolic_constant181
// CHECK:STDOUT:       ir0.inst264:     symbolic_constant182
// CHECK:STDOUT:       ir0.inst265:     symbolic_constant183
// CHECK:STDOUT:       ir0.inst266:     symbolic_constant184
// CHECK:STDOUT:       ir0.inst267:     symbolic_constant185
// CHECK:STDOUT:       ir0.inst268:     symbolic_constant188
// CHECK:STDOUT:       ir0.inst269:     symbolic_constant186
// CHECK:STDOUT:       ir0.inst270:     symbolic_constant187
// CHECK:STDOUT:       ir0.inst271:     symbolic_constant190
// CHECK:STDOUT:       ir0.inst272:     concrete_constant(ir0.inst272)
// CHECK:STDOUT:       ir0.inst273:     concrete_constant(ir0.inst273)
// CHECK:STDOUT:       ir0.inst274:     concrete_constant(ir0.inst274)
// CHECK:STDOUT:       ir0.inst275:     concrete_constant(ir0.inst275)
// CHECK:STDOUT:       ir0.inst276:     symbolic_constant167
// CHECK:STDOUT:       ir0.inst277:     symbolic_constant168
// CHECK:STDOUT:       ir0.inst278:     symbolic_constant169
// CHECK:STDOUT:       ir0.inst279:     symbolic_constant193
// CHECK:STDOUT:       ir0.inst280:     symbolic_constant194
// CHECK:STDOUT:       ir0.inst281:     symbolic_constant202
// CHECK:STDOUT:       ir0.inst282:     symbolic_constant203
// CHECK:STDOUT:       ir0.inst283:     symbolic_constant204
// CHECK:STDOUT:       ir0.inst284:     symbolic_constant205
// CHECK:STDOUT:       ir0.inst285:     symbolic_constant206
// CHECK:STDOUT:       ir0.inst286:     symbolic_constant207
// CHECK:STDOUT:       ir0.inst287:     symbolic_constant208
// CHECK:STDOUT:       ir0.inst288:     symbolic_constant209
// CHECK:STDOUT:       ir0.inst289:     symbolic_constant210
// CHECK:STDOUT:       ir0.inst290:     symbolic_constant222
// CHECK:STDOUT:       ir0.inst291:     symbolic_constant224
// CHECK:STDOUT:       ir0.inst292:     symbolic_constant226
// CHECK:STDOUT:       ir0.inst293:     symbolic_constant228
// CHECK:STDOUT:       ir0.inst294:     symbolic_constant230
// CHECK:STDOUT:       ir0.inst295:     symbolic_constant232
// CHECK:STDOUT:       ir0.inst296:     symbolic_constant233
// CHECK:STDOUT:       ir0.inst297:     symbolic_constant234
// CHECK:STDOUT:       ir0.inst298:     symbolic_constant235
// CHECK:STDOUT:       ir0.inst299:     symbolic_constant236
// CHECK:STDOUT:       ir0.inst300:     symbolic_constant237
// CHECK:STDOUT:       ir0.inst301:     symbolic_constant238
// CHECK:STDOUT:       ir0.inst302:     symbolic_constant239
// CHECK:STDOUT:       ir0.inst303:     symbolic_constant240
// CHECK:STDOUT:       ir0.inst304:     symbolic_constant241
// CHECK:STDOUT:       ir0.inst305:     symbolic_constant242
// CHECK:STDOUT:       ir0.inst306:     symbolic_constant243
// CHECK:STDOUT:       ir0.inst307:     symbolic_constant244
// CHECK:STDOUT:       ir0.inst308:     symbolic_constant245
// CHECK:STDOUT:       ir0.inst309:     symbolic_constant246
// CHECK:STDOUT:       ir0.inst310:     symbolic_constant247
// CHECK:STDOUT:       ir0.inst311:     symbolic_constant248
// CHECK:STDOUT:       ir0.inst312:     symbolic_constant250
// CHECK:STDOUT:       ir0.inst313:     symbolic_constant251
// CHECK:STDOUT:       ir0.inst314:     symbolic_constant252
// CHECK:STDOUT:       ir0.inst315:     symbolic_constant253
// CHECK:STDOUT:       ir0.inst316:     symbolic_constant257
// CHECK:STDOUT:       ir0.inst317:     symbolic_constant254
// CHECK:STDOUT:       ir0.inst318:     symbolic_constant255
// CHECK:STDOUT:       ir0.inst319:     symbolic_constant256
// CHECK:STDOUT:       ir0.inst320:     symbolic_constant257
// CHECK:STDOUT:       ir0.inst322:     symbolic_constant259
// CHECK:STDOUT:       ir0.inst323:     symbolic_constant258
// CHECK:STDOUT:       ir0.inst324:     symbolic_constant259
// CHECK:STDOUT:       ir0.inst326:     symbolic_constant15
// CHECK:STDOUT:       ir0.inst330:     concrete_constant(ir0.inst331)
// CHECK:STDOUT:       ir0.inst331:     concrete_constant(ir0.inst331)
// CHECK:STDOUT:       ir0.inst332:     concrete_constant(ir0.inst331)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: ir0.inst19, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant1: {inst: ir0.inst21, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: ir0.inst21, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: ir0.inst27, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: ir0.inst27, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: ir0.inst30, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: ir0.inst30, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: ir0.inst42, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: ir0.inst42, generic: generic0, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: ir0.inst45, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant10: {inst: ir0.inst45, generic: generic0, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant11: {inst: ir0.inst56, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant12: {inst: ir0.inst58, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant13: {inst: ir0.inst58, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant14: {inst: ir0.inst61, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant15: {inst: ir0.inst61, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant16: {inst: ir0.inst72, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant17: {inst: ir0.inst82, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant18: {inst: ir0.inst83, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant19: {inst: ir0.inst83, generic: generic1, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant20: {inst: ir0.inst72, generic: generic1, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant21: {inst: ir0.inst82, generic: generic1, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant22: {inst: ir0.inst72, generic: generic1, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant23: {inst: ir0.inst82, generic: generic1, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant24: {inst: ir0.inst83, generic: generic1, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant25: {inst: ir0.inst95, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant26: {inst: ir0.inst96, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant27: {inst: ir0.inst97, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant28: {inst: ir0.inst97, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant29: {inst: ir0.inst95, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant30: {inst: ir0.inst95, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant31: {inst: ir0.inst96, generic: generic2, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant32: {inst: ir0.inst105, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant33: {inst: ir0.inst105, generic: generic2, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant34: {inst: ir0.inst95, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant35: {inst: ir0.inst96, generic: generic2, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant36: {inst: ir0.inst97, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant37: {inst: ir0.inst105, generic: generic2, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant38: {inst: ir0.inst111, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant39: {inst: ir0.inst112, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant40: {inst: ir0.inst112, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant41: {inst: ir0.inst111, generic: generic2, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant42: {inst: ir0.inst113, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant43: {inst: ir0.inst113, generic: generic3, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant44: {inst: ir0.inst112, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant45: {inst: ir0.inst111, generic: generic2, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant46: {inst: ir0.inst112, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant47: {inst: ir0.inst95, generic: generic3, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant48: {inst: ir0.inst96, generic: generic3, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant49: {inst: ir0.inst97, generic: generic3, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant50: {inst: ir0.inst95, generic: generic3, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant51: {inst: ir0.inst96, generic: generic3, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant52: {inst: ir0.inst97, generic: generic3, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant53: {inst: ir0.inst113, generic: generic3, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant54: {inst: ir0.inst125, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant55: {inst: ir0.inst125, generic: generic3, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant56: {inst: ir0.inst126, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant57: {inst: ir0.inst126, generic: generic3, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant58: {inst: ir0.inst127, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant59: {inst: ir0.inst127, generic: generic3, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60: {inst: ir0.inst128, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant61: {inst: ir0.inst128, generic: generic3, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant62: {inst: ir0.inst129, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant63: {inst: ir0.inst129, generic: generic3, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant64: {inst: ir0.inst130, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant65: {inst: ir0.inst130, generic: generic3, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant66: {inst: ir0.inst125, generic: generic3, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant67: {inst: ir0.inst126, generic: generic3, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant68: {inst: ir0.inst127, generic: generic3, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant69: {inst: ir0.inst128, generic: generic3, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant70: {inst: ir0.inst129, generic: generic3, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant71: {inst: ir0.inst130, generic: generic3, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant72: {inst: ir0.inst137, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant73: {inst: ir0.inst27, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant74: {inst: ir0.inst21, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant75: {inst: ir0.inst21, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant76: {inst: ir0.inst162, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant77: {inst: ir0.inst162, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant78: {inst: ir0.inst21, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant79: {inst: ir0.inst27, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant80: {inst: ir0.inst162, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant81: {inst: ir0.inst167, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant82: {inst: ir0.inst168, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant83: {inst: ir0.inst168, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant84: {inst: ir0.inst167, generic: generic4, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant85: {inst: ir0.inst30, generic: generic5, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant86: {inst: ir0.inst168, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant87: {inst: ir0.inst61, generic: generic4, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant88: {inst: ir0.inst167, generic: generic4, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant89: {inst: ir0.inst168, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant90: {inst: ir0.inst61, generic: generic4, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant91: {inst: ir0.inst21, generic: generic5, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant92: {inst: ir0.inst27, generic: generic5, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant93: {inst: ir0.inst21, generic: generic5, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant94: {inst: ir0.inst27, generic: generic5, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant95: {inst: ir0.inst30, generic: generic5, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant96: {inst: ir0.inst190, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant97: {inst: ir0.inst191, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant98: {inst: ir0.inst192, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant99: {inst: ir0.inst192, generic: generic6, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant100: {inst: ir0.inst95, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant101: {inst: ir0.inst190, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant102: {inst: ir0.inst95, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant103: {inst: ir0.inst190, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant104: {inst: ir0.inst96, generic: generic6, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant105: {inst: ir0.inst191, generic: generic6, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant106: {inst: ir0.inst201, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant107: {inst: ir0.inst201, generic: generic6, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant108: {inst: ir0.inst95, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant109: {inst: ir0.inst190, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant110: {inst: ir0.inst96, generic: generic6, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant111: {inst: ir0.inst191, generic: generic6, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant112: {inst: ir0.inst192, generic: generic6, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant113: {inst: ir0.inst201, generic: generic6, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant114: {inst: ir0.inst209, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant115: {inst: ir0.inst210, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant116: {inst: ir0.inst210, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant117: {inst: ir0.inst209, generic: generic6, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant118: {inst: ir0.inst211, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant119: {inst: ir0.inst211, generic: generic7, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant120: {inst: ir0.inst210, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant121: {inst: ir0.inst209, generic: generic6, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant122: {inst: ir0.inst210, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant123: {inst: ir0.inst95, generic: generic7, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant124: {inst: ir0.inst96, generic: generic7, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant125: {inst: ir0.inst190, generic: generic7, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant126: {inst: ir0.inst191, generic: generic7, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant127: {inst: ir0.inst192, generic: generic7, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant128: {inst: ir0.inst95, generic: generic7, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant129: {inst: ir0.inst96, generic: generic7, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant130: {inst: ir0.inst190, generic: generic7, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant131: {inst: ir0.inst191, generic: generic7, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant132: {inst: ir0.inst192, generic: generic7, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant133: {inst: ir0.inst211, generic: generic7, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant134: {inst: ir0.inst226, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant135: {inst: ir0.inst226, generic: generic7, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant136: {inst: ir0.inst126, generic: generic7, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant137: {inst: ir0.inst127, generic: generic7, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant138: {inst: ir0.inst128, generic: generic7, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant139: {inst: ir0.inst129, generic: generic7, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant140: {inst: ir0.inst130, generic: generic7, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant141: {inst: ir0.inst227, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant142: {inst: ir0.inst227, generic: generic7, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant143: {inst: ir0.inst228, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant144: {inst: ir0.inst228, generic: generic7, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant145: {inst: ir0.inst229, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant146: {inst: ir0.inst229, generic: generic7, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant147: {inst: ir0.inst230, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant148: {inst: ir0.inst230, generic: generic7, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant149: {inst: ir0.inst231, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant150: {inst: ir0.inst231, generic: generic7, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant151: {inst: ir0.inst226, generic: generic7, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant152: {inst: ir0.inst126, generic: generic7, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant153: {inst: ir0.inst127, generic: generic7, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant154: {inst: ir0.inst128, generic: generic7, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant155: {inst: ir0.inst129, generic: generic7, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant156: {inst: ir0.inst130, generic: generic7, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant157: {inst: ir0.inst227, generic: generic7, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant158: {inst: ir0.inst228, generic: generic7, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant159: {inst: ir0.inst229, generic: generic7, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant160: {inst: ir0.inst230, generic: generic7, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant161: {inst: ir0.inst231, generic: generic7, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant162: {inst: ir0.inst243, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant163: {inst: ir0.inst246, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant164: {inst: ir0.inst247, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant165: {inst: ir0.inst248, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant166: {inst: ir0.inst248, generic: generic8, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant167: {inst: ir0.inst95, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant168: {inst: ir0.inst190, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant169: {inst: ir0.inst246, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant170: {inst: ir0.inst95, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant171: {inst: ir0.inst190, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant172: {inst: ir0.inst246, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant173: {inst: ir0.inst96, generic: generic8, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant174: {inst: ir0.inst191, generic: generic8, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant175: {inst: ir0.inst247, generic: generic8, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant176: {inst: ir0.inst259, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant177: {inst: ir0.inst259, generic: generic8, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant178: {inst: ir0.inst95, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant179: {inst: ir0.inst190, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant180: {inst: ir0.inst246, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant181: {inst: ir0.inst96, generic: generic8, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant182: {inst: ir0.inst191, generic: generic8, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant183: {inst: ir0.inst247, generic: generic8, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant184: {inst: ir0.inst248, generic: generic8, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant185: {inst: ir0.inst259, generic: generic8, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant186: {inst: ir0.inst269, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant187: {inst: ir0.inst270, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant188: {inst: ir0.inst270, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant189: {inst: ir0.inst269, generic: generic8, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant190: {inst: ir0.inst271, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant191: {inst: ir0.inst271, generic: generic9, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant192: {inst: ir0.inst270, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant193: {inst: ir0.inst269, generic: generic8, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant194: {inst: ir0.inst270, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant195: {inst: ir0.inst95, generic: generic9, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant196: {inst: ir0.inst96, generic: generic9, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant197: {inst: ir0.inst190, generic: generic9, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant198: {inst: ir0.inst191, generic: generic9, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant199: {inst: ir0.inst246, generic: generic9, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant200: {inst: ir0.inst247, generic: generic9, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant201: {inst: ir0.inst248, generic: generic9, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant202: {inst: ir0.inst95, generic: generic9, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant203: {inst: ir0.inst96, generic: generic9, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant204: {inst: ir0.inst190, generic: generic9, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant205: {inst: ir0.inst191, generic: generic9, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant206: {inst: ir0.inst246, generic: generic9, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant207: {inst: ir0.inst247, generic: generic9, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant208: {inst: ir0.inst248, generic: generic9, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant209: {inst: ir0.inst271, generic: generic9, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant210: {inst: ir0.inst289, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant211: {inst: ir0.inst289, generic: generic9, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant212: {inst: ir0.inst126, generic: generic9, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant213: {inst: ir0.inst127, generic: generic9, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant214: {inst: ir0.inst128, generic: generic9, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant215: {inst: ir0.inst129, generic: generic9, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant216: {inst: ir0.inst130, generic: generic9, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant217: {inst: ir0.inst227, generic: generic9, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant218: {inst: ir0.inst228, generic: generic9, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant219: {inst: ir0.inst229, generic: generic9, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant220: {inst: ir0.inst230, generic: generic9, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant221: {inst: ir0.inst231, generic: generic9, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant222: {inst: ir0.inst290, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant223: {inst: ir0.inst290, generic: generic9, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant224: {inst: ir0.inst291, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant225: {inst: ir0.inst291, generic: generic9, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant226: {inst: ir0.inst292, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant227: {inst: ir0.inst292, generic: generic9, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant228: {inst: ir0.inst293, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant229: {inst: ir0.inst293, generic: generic9, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant230: {inst: ir0.inst294, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant231: {inst: ir0.inst294, generic: generic9, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant232: {inst: ir0.inst289, generic: generic9, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant233: {inst: ir0.inst126, generic: generic9, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant234: {inst: ir0.inst127, generic: generic9, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant235: {inst: ir0.inst128, generic: generic9, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant236: {inst: ir0.inst129, generic: generic9, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant237: {inst: ir0.inst130, generic: generic9, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant238: {inst: ir0.inst227, generic: generic9, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant239: {inst: ir0.inst228, generic: generic9, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant240: {inst: ir0.inst229, generic: generic9, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant241: {inst: ir0.inst230, generic: generic9, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant242: {inst: ir0.inst231, generic: generic9, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant243: {inst: ir0.inst290, generic: generic9, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant244: {inst: ir0.inst291, generic: generic9, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant245: {inst: ir0.inst292, generic: generic9, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant246: {inst: ir0.inst293, generic: generic9, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant247: {inst: ir0.inst294, generic: generic9, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant248: {inst: ir0.inst311, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant249: {inst: ir0.inst162, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant250: {inst: ir0.inst312, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant251: {inst: ir0.inst312, generic: generic0, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant252: {inst: ir0.inst314, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant253: {inst: ir0.inst315, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant254: {inst: ir0.inst317, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant255: {inst: ir0.inst314, generic: generic0, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant256: {inst: ir0.inst315, generic: generic0, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant257: {inst: ir0.inst317, generic: generic0, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant258: {inst: ir0.inst323, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant259: {inst: ir0.inst323, generic: generic0, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               ir0.inst53
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               ir0.inst16
// CHECK:STDOUT:       1:               ir0.inst69
// CHECK:STDOUT:       2:               ir0.inst70
// CHECK:STDOUT:       3:               ir0.inst73
// CHECK:STDOUT:       4:               ir0.inst74
// CHECK:STDOUT:       5:               ir0.inst75
// CHECK:STDOUT:       6:               ir0.inst77
// CHECK:STDOUT:       7:               ir0.inst79
// CHECK:STDOUT:       8:               ir0.inst84
// CHECK:STDOUT:       9:               ir0.inst85
// CHECK:STDOUT:       10:              ir0.inst86
// CHECK:STDOUT:       11:              ir0.inst87
// CHECK:STDOUT:       12:              ir0.inst88
// CHECK:STDOUT:       13:              ir0.inst93
// CHECK:STDOUT:       14:              ir0.inst94
// CHECK:STDOUT:       15:              ir0.inst99
// CHECK:STDOUT:       16:              ir0.inst100
// CHECK:STDOUT:       17:              ir0.inst101
// CHECK:STDOUT:       18:              ir0.inst102
// CHECK:STDOUT:       19:              ir0.inst103
// CHECK:STDOUT:       20:              ir0.inst104
// CHECK:STDOUT:       21:              ir0.inst110
// CHECK:STDOUT:       22:              ir0.inst114
// CHECK:STDOUT:       23:              ir0.inst115
// CHECK:STDOUT:       24:              ir0.inst116
// CHECK:STDOUT:       25:              ir0.inst117
// CHECK:STDOUT:       26:              ir0.inst118
// CHECK:STDOUT:       27:              ir0.inst138
// CHECK:STDOUT:       28:              ir0.inst139
// CHECK:STDOUT:       29:              ir0.inst140
// CHECK:STDOUT:       30:              ir0.inst141
// CHECK:STDOUT:       31:              ir0.inst142
// CHECK:STDOUT:       32:              ir0.inst143
// CHECK:STDOUT:       33:              ir0.inst144
// CHECK:STDOUT:       34:              ir0.inst145
// CHECK:STDOUT:       35:              ir0.inst146
// CHECK:STDOUT:       36:              ir0.inst147
// CHECK:STDOUT:       37:              ir0.inst148
// CHECK:STDOUT:       38:              ir0.inst149
// CHECK:STDOUT:       39:              ir0.inst150
// CHECK:STDOUT:       40:              ir0.inst151
// CHECK:STDOUT:       41:              ir0.inst152
// CHECK:STDOUT:       42:              ir0.inst153
// CHECK:STDOUT:       43:              ir0.inst154
// CHECK:STDOUT:       44:              ir0.inst155
// CHECK:STDOUT:       45:              ir0.inst156
// CHECK:STDOUT:       46:              ir0.inst157
// CHECK:STDOUT:       47:              ir0.inst158
// CHECK:STDOUT:       48:              ir0.inst159
// CHECK:STDOUT:       49:              ir0.inst160
// CHECK:STDOUT:       50:              ir0.inst161
// CHECK:STDOUT:       51:              ir0.inst166
// CHECK:STDOUT:       52:              ir0.inst169
// CHECK:STDOUT:       53:              ir0.inst170
// CHECK:STDOUT:       54:              ir0.inst171
// CHECK:STDOUT:       55:              ir0.inst172
// CHECK:STDOUT:       56:              ir0.inst173
// CHECK:STDOUT:       57:              ir0.inst180
// CHECK:STDOUT:       58:              ir0.inst181
// CHECK:STDOUT:       59:              ir0.inst182
// CHECK:STDOUT:       60:              ir0.inst183
// CHECK:STDOUT:       61:              ir0.inst184
// CHECK:STDOUT:       62:              ir0.inst185
// CHECK:STDOUT:       63:              ir0.inst186
// CHECK:STDOUT:       64:              ir0.inst187
// CHECK:STDOUT:       65:              ir0.inst188
// CHECK:STDOUT:       66:              ir0.inst189
// CHECK:STDOUT:       67:              ir0.inst193
// CHECK:STDOUT:       68:              ir0.inst194
// CHECK:STDOUT:       69:              ir0.inst195
// CHECK:STDOUT:       70:              ir0.inst196
// CHECK:STDOUT:       71:              ir0.inst197
// CHECK:STDOUT:       72:              ir0.inst198
// CHECK:STDOUT:       73:              ir0.inst199
// CHECK:STDOUT:       74:              ir0.inst200
// CHECK:STDOUT:       75:              ir0.inst208
// CHECK:STDOUT:       76:              ir0.inst212
// CHECK:STDOUT:       77:              ir0.inst213
// CHECK:STDOUT:       78:              ir0.inst214
// CHECK:STDOUT:       79:              ir0.inst215
// CHECK:STDOUT:       80:              ir0.inst216
// CHECK:STDOUT:       81:              ir0.inst217
// CHECK:STDOUT:       82:              ir0.inst244
// CHECK:STDOUT:       83:              ir0.inst245
// CHECK:STDOUT:       84:              ir0.inst249
// CHECK:STDOUT:       85:              ir0.inst250
// CHECK:STDOUT:       86:              ir0.inst251
// CHECK:STDOUT:       87:              ir0.inst252
// CHECK:STDOUT:       88:              ir0.inst253
// CHECK:STDOUT:       89:              ir0.inst254
// CHECK:STDOUT:       90:              ir0.inst255
// CHECK:STDOUT:       91:              ir0.inst256
// CHECK:STDOUT:       92:              ir0.inst257
// CHECK:STDOUT:       93:              ir0.inst258
// CHECK:STDOUT:       94:              ir0.inst268
// CHECK:STDOUT:       95:              ir0.inst272
// CHECK:STDOUT:       96:              ir0.inst273
// CHECK:STDOUT:       97:              ir0.inst274
// CHECK:STDOUT:       98:              ir0.inst275
// CHECK:STDOUT:       99:              ir0.inst276
// CHECK:STDOUT:       100:             ir0.inst277
// CHECK:STDOUT:       101:             ir0.inst278
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               ir0.inst18
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               ir0.inst24
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               ir0.inst25
// CHECK:STDOUT:       1:               ir0.inst26
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               ir0.inst33
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               ir0.inst35
// CHECK:STDOUT:       1:               ir0.inst37
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               ir0.inst36
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               ir0.inst35
// CHECK:STDOUT:       1:               ir0.inst41
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               ir0.inst27
// CHECK:STDOUT:       1:               ir0.inst36
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               ir0.inst28
// CHECK:STDOUT:       1:               ir0.inst36
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               ir0.inst49
// CHECK:STDOUT:       1:               ir0.inst51
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               ir0.inst24
// CHECK:STDOUT:       1:               ir0.inst31
// CHECK:STDOUT:       2:               ir0.inst33
// CHECK:STDOUT:       3:               ir0.inst46
// CHECK:STDOUT:       4:               ir0.inst48
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               ir0.inst34
// CHECK:STDOUT:       1:               ir0.inst35
// CHECK:STDOUT:       2:               ir0.inst37
// CHECK:STDOUT:       3:               ir0.inst39
// CHECK:STDOUT:       4:               ir0.inst41
// CHECK:STDOUT:       5:               ir0.inst43
// CHECK:STDOUT:       6:               ir0.inst18
// CHECK:STDOUT:       7:               ir0.inst20
// CHECK:STDOUT:       8:               ir0.inst49
// CHECK:STDOUT:       9:               ir0.inst50
// CHECK:STDOUT:       10:              ir0.inst29
// CHECK:STDOUT:       11:              ir0.inst51
// CHECK:STDOUT:       12:              ir0.inst52
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               ir0.inst20
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               ir0.inst21
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               ir0.inst22
// CHECK:STDOUT:       1:               ir0.inst28
// CHECK:STDOUT:       2:               ir0.inst32
// CHECK:STDOUT:       3:               ir0.inst44
// CHECK:STDOUT:       4:               ir0.inst47
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               ir0.inst21
// CHECK:STDOUT:       1:               ir0.inst27
// CHECK:STDOUT:       2:               ir0.inst30
// CHECK:STDOUT:       3:               ir0.inst42
// CHECK:STDOUT:       4:               ir0.inst45
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               ir0.inst63
// CHECK:STDOUT:       1:               ir0.inst64
// CHECK:STDOUT:       2:               ir0.inst65
// CHECK:STDOUT:       3:               ir0.inst316
// CHECK:STDOUT:       4:               ir0.inst321
// CHECK:STDOUT:       5:               ir0.inst322
// CHECK:STDOUT:       6:               ir0.inst325
// CHECK:STDOUT:       7:               ir0.inst327
// CHECK:STDOUT:       8:               ir0.inst67
// CHECK:STDOUT:       9:               ir0.inst328
// CHECK:STDOUT:       10:              ir0.inst329
// CHECK:STDOUT:       11:              ir0.inst330
// CHECK:STDOUT:       12:              ir0.inst332
// CHECK:STDOUT:       13:              ir0.inst333
// CHECK:STDOUT:       14:              ir0.inst334
// CHECK:STDOUT:       15:              ir0.inst335
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               ir0.inst63
// CHECK:STDOUT:       1:               ir0.inst64
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               ir0.inst75
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               ir0.inst87
// CHECK:STDOUT:     inst_block24:
// CHECK:STDOUT:       0:               ir0.inst88
// CHECK:STDOUT:     inst_block25:
// CHECK:STDOUT:       0:               ir0.inst89
// CHECK:STDOUT:       1:               ir0.inst90
// CHECK:STDOUT:       2:               ir0.inst91
// CHECK:STDOUT:     inst_block26:
// CHECK:STDOUT:       0:               ir0.inst72
// CHECK:STDOUT:     inst_block27:
// CHECK:STDOUT:       0:               ir0.inst72
// CHECK:STDOUT:       1:               ir0.inst82
// CHECK:STDOUT:       2:               ir0.inst83
// CHECK:STDOUT:     inst_block28:
// CHECK:STDOUT:       0:               ir0.inst99
// CHECK:STDOUT:     inst_block29:
// CHECK:STDOUT:       0:               ir0.inst100
// CHECK:STDOUT:     inst_block30:
// CHECK:STDOUT:       0:               ir0.inst103
// CHECK:STDOUT:     inst_block31:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:     inst_block32:
// CHECK:STDOUT:       0:               ir0.inst106
// CHECK:STDOUT:     inst_block33:
// CHECK:STDOUT:       0:               ir0.inst106
// CHECK:STDOUT:       1:               ir0.inst107
// CHECK:STDOUT:       2:               ir0.inst108
// CHECK:STDOUT:       3:               ir0.inst109
// CHECK:STDOUT:     inst_block34:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst96
// CHECK:STDOUT:       2:               ir0.inst97
// CHECK:STDOUT:       3:               ir0.inst105
// CHECK:STDOUT:     inst_block35:
// CHECK:STDOUT:       0:               ir0.inst117
// CHECK:STDOUT:     inst_block36:
// CHECK:STDOUT:       0:               ir0.inst118
// CHECK:STDOUT:     inst_block37:
// CHECK:STDOUT:       0:               ir0.inst106
// CHECK:STDOUT:     inst_block38:
// CHECK:STDOUT:       0:               ir0.inst119
// CHECK:STDOUT:       1:               ir0.inst120
// CHECK:STDOUT:     inst_block39:
// CHECK:STDOUT:       0:               ir0.inst121
// CHECK:STDOUT:       1:               ir0.inst122
// CHECK:STDOUT:       2:               ir0.inst123
// CHECK:STDOUT:       3:               ir0.inst124
// CHECK:STDOUT:     inst_block40:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst96
// CHECK:STDOUT:       2:               ir0.inst97
// CHECK:STDOUT:       3:               ir0.inst113
// CHECK:STDOUT:     inst_block41:
// CHECK:STDOUT:       0:               ir0.inst121
// CHECK:STDOUT:     inst_block42:
// CHECK:STDOUT:       0:               ir0.inst131
// CHECK:STDOUT:       1:               ir0.inst132
// CHECK:STDOUT:       2:               ir0.inst133
// CHECK:STDOUT:       3:               ir0.inst134
// CHECK:STDOUT:       4:               ir0.inst135
// CHECK:STDOUT:       5:               ir0.inst136
// CHECK:STDOUT:     inst_block43:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst96
// CHECK:STDOUT:       2:               ir0.inst137
// CHECK:STDOUT:     inst_block44:
// CHECK:STDOUT:       0:               ir0.inst111
// CHECK:STDOUT:       1:               ir0.inst112
// CHECK:STDOUT:     inst_block45:
// CHECK:STDOUT:       0:               ir0.inst156
// CHECK:STDOUT:     inst_block46:
// CHECK:STDOUT:       0:               ir0.inst157
// CHECK:STDOUT:     inst_block47:
// CHECK:STDOUT:       0:               ir0.inst160
// CHECK:STDOUT:     inst_block48:
// CHECK:STDOUT:       0:               ir0.inst163
// CHECK:STDOUT:     inst_block49:
// CHECK:STDOUT:       0:               ir0.inst163
// CHECK:STDOUT:       1:               ir0.inst164
// CHECK:STDOUT:       2:               ir0.inst165
// CHECK:STDOUT:     inst_block50:
// CHECK:STDOUT:       0:               ir0.inst21
// CHECK:STDOUT:       1:               ir0.inst27
// CHECK:STDOUT:       2:               ir0.inst162
// CHECK:STDOUT:     inst_block51:
// CHECK:STDOUT:       0:               ir0.inst172
// CHECK:STDOUT:     inst_block52:
// CHECK:STDOUT:       0:               ir0.inst173
// CHECK:STDOUT:     inst_block53:
// CHECK:STDOUT:       0:               ir0.inst163
// CHECK:STDOUT:     inst_block54:
// CHECK:STDOUT:       0:               ir0.inst174
// CHECK:STDOUT:       1:               ir0.inst175
// CHECK:STDOUT:       2:               ir0.inst176
// CHECK:STDOUT:     inst_block55:
// CHECK:STDOUT:       0:               ir0.inst177
// CHECK:STDOUT:       1:               ir0.inst178
// CHECK:STDOUT:       2:               ir0.inst179
// CHECK:STDOUT:     inst_block56:
// CHECK:STDOUT:       0:               ir0.inst21
// CHECK:STDOUT:       1:               ir0.inst27
// CHECK:STDOUT:       2:               ir0.inst30
// CHECK:STDOUT:     inst_block57:
// CHECK:STDOUT:       0:               ir0.inst96
// CHECK:STDOUT:       1:               ir0.inst191
// CHECK:STDOUT:     inst_block58:
// CHECK:STDOUT:       0:               ir0.inst194
// CHECK:STDOUT:       1:               ir0.inst193
// CHECK:STDOUT:     inst_block59:
// CHECK:STDOUT:       0:               ir0.inst195
// CHECK:STDOUT:       1:               ir0.inst196
// CHECK:STDOUT:     inst_block60:
// CHECK:STDOUT:       0:               ir0.inst199
// CHECK:STDOUT:     inst_block61:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst190
// CHECK:STDOUT:     inst_block62:
// CHECK:STDOUT:       0:               ir0.inst204
// CHECK:STDOUT:       1:               ir0.inst205
// CHECK:STDOUT:     inst_block63:
// CHECK:STDOUT:       0:               ir0.inst202
// CHECK:STDOUT:       1:               ir0.inst203
// CHECK:STDOUT:     inst_block64:
// CHECK:STDOUT:       0:               ir0.inst202
// CHECK:STDOUT:       1:               ir0.inst203
// CHECK:STDOUT:       2:               ir0.inst204
// CHECK:STDOUT:       3:               ir0.inst205
// CHECK:STDOUT:       4:               ir0.inst206
// CHECK:STDOUT:       5:               ir0.inst207
// CHECK:STDOUT:     inst_block65:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst190
// CHECK:STDOUT:       2:               ir0.inst96
// CHECK:STDOUT:       3:               ir0.inst191
// CHECK:STDOUT:       4:               ir0.inst192
// CHECK:STDOUT:       5:               ir0.inst201
// CHECK:STDOUT:     inst_block66:
// CHECK:STDOUT:       0:               ir0.inst215
// CHECK:STDOUT:     inst_block67:
// CHECK:STDOUT:       0:               ir0.inst216
// CHECK:STDOUT:       1:               ir0.inst217
// CHECK:STDOUT:     inst_block68:
// CHECK:STDOUT:       0:               ir0.inst202
// CHECK:STDOUT:       1:               ir0.inst203
// CHECK:STDOUT:     inst_block69:
// CHECK:STDOUT:       0:               ir0.inst218
// CHECK:STDOUT:       1:               ir0.inst219
// CHECK:STDOUT:     inst_block70:
// CHECK:STDOUT:       0:               ir0.inst221
// CHECK:STDOUT:       1:               ir0.inst223
// CHECK:STDOUT:     inst_block71:
// CHECK:STDOUT:       0:               ir0.inst220
// CHECK:STDOUT:       1:               ir0.inst221
// CHECK:STDOUT:       2:               ir0.inst222
// CHECK:STDOUT:       3:               ir0.inst223
// CHECK:STDOUT:       4:               ir0.inst224
// CHECK:STDOUT:       5:               ir0.inst225
// CHECK:STDOUT:     inst_block72:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst96
// CHECK:STDOUT:       2:               ir0.inst190
// CHECK:STDOUT:       3:               ir0.inst191
// CHECK:STDOUT:       4:               ir0.inst192
// CHECK:STDOUT:       5:               ir0.inst211
// CHECK:STDOUT:     inst_block73:
// CHECK:STDOUT:       0:               ir0.inst190
// CHECK:STDOUT:     inst_block74:
// CHECK:STDOUT:       0:               ir0.inst220
// CHECK:STDOUT:     inst_block75:
// CHECK:STDOUT:       0:               ir0.inst222
// CHECK:STDOUT:     inst_block76:
// CHECK:STDOUT:       0:               ir0.inst232
// CHECK:STDOUT:       1:               ir0.inst233
// CHECK:STDOUT:       2:               ir0.inst234
// CHECK:STDOUT:       3:               ir0.inst235
// CHECK:STDOUT:       4:               ir0.inst236
// CHECK:STDOUT:       5:               ir0.inst237
// CHECK:STDOUT:       6:               ir0.inst238
// CHECK:STDOUT:       7:               ir0.inst239
// CHECK:STDOUT:       8:               ir0.inst240
// CHECK:STDOUT:       9:               ir0.inst241
// CHECK:STDOUT:       10:              ir0.inst242
// CHECK:STDOUT:     inst_block77:
// CHECK:STDOUT:       0:               ir0.inst190
// CHECK:STDOUT:       1:               ir0.inst191
// CHECK:STDOUT:       2:               ir0.inst243
// CHECK:STDOUT:     inst_block78:
// CHECK:STDOUT:       0:               ir0.inst209
// CHECK:STDOUT:       1:               ir0.inst210
// CHECK:STDOUT:     inst_block79:
// CHECK:STDOUT:       0:               ir0.inst96
// CHECK:STDOUT:       1:               ir0.inst191
// CHECK:STDOUT:       2:               ir0.inst247
// CHECK:STDOUT:     inst_block80:
// CHECK:STDOUT:       0:               ir0.inst251
// CHECK:STDOUT:       1:               ir0.inst250
// CHECK:STDOUT:       2:               ir0.inst249
// CHECK:STDOUT:     inst_block81:
// CHECK:STDOUT:       0:               ir0.inst252
// CHECK:STDOUT:       1:               ir0.inst253
// CHECK:STDOUT:       2:               ir0.inst254
// CHECK:STDOUT:     inst_block82:
// CHECK:STDOUT:       0:               ir0.inst257
// CHECK:STDOUT:     inst_block83:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst190
// CHECK:STDOUT:       2:               ir0.inst246
// CHECK:STDOUT:     inst_block84:
// CHECK:STDOUT:       0:               ir0.inst263
// CHECK:STDOUT:       1:               ir0.inst264
// CHECK:STDOUT:       2:               ir0.inst265
// CHECK:STDOUT:     inst_block85:
// CHECK:STDOUT:       0:               ir0.inst260
// CHECK:STDOUT:       1:               ir0.inst261
// CHECK:STDOUT:       2:               ir0.inst262
// CHECK:STDOUT:     inst_block86:
// CHECK:STDOUT:       0:               ir0.inst260
// CHECK:STDOUT:       1:               ir0.inst261
// CHECK:STDOUT:       2:               ir0.inst262
// CHECK:STDOUT:       3:               ir0.inst263
// CHECK:STDOUT:       4:               ir0.inst264
// CHECK:STDOUT:       5:               ir0.inst265
// CHECK:STDOUT:       6:               ir0.inst266
// CHECK:STDOUT:       7:               ir0.inst267
// CHECK:STDOUT:     inst_block87:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst190
// CHECK:STDOUT:       2:               ir0.inst246
// CHECK:STDOUT:       3:               ir0.inst96
// CHECK:STDOUT:       4:               ir0.inst191
// CHECK:STDOUT:       5:               ir0.inst247
// CHECK:STDOUT:       6:               ir0.inst248
// CHECK:STDOUT:       7:               ir0.inst259
// CHECK:STDOUT:     inst_block88:
// CHECK:STDOUT:       0:               ir0.inst275
// CHECK:STDOUT:     inst_block89:
// CHECK:STDOUT:       0:               ir0.inst276
// CHECK:STDOUT:       1:               ir0.inst277
// CHECK:STDOUT:       2:               ir0.inst278
// CHECK:STDOUT:     inst_block90:
// CHECK:STDOUT:       0:               ir0.inst260
// CHECK:STDOUT:       1:               ir0.inst261
// CHECK:STDOUT:       2:               ir0.inst262
// CHECK:STDOUT:     inst_block91:
// CHECK:STDOUT:       0:               ir0.inst279
// CHECK:STDOUT:       1:               ir0.inst280
// CHECK:STDOUT:     inst_block92:
// CHECK:STDOUT:       0:               ir0.inst282
// CHECK:STDOUT:       1:               ir0.inst284
// CHECK:STDOUT:       2:               ir0.inst286
// CHECK:STDOUT:     inst_block93:
// CHECK:STDOUT:       0:               ir0.inst281
// CHECK:STDOUT:       1:               ir0.inst282
// CHECK:STDOUT:       2:               ir0.inst283
// CHECK:STDOUT:       3:               ir0.inst284
// CHECK:STDOUT:       4:               ir0.inst285
// CHECK:STDOUT:       5:               ir0.inst286
// CHECK:STDOUT:       6:               ir0.inst287
// CHECK:STDOUT:       7:               ir0.inst288
// CHECK:STDOUT:     inst_block94:
// CHECK:STDOUT:       0:               ir0.inst95
// CHECK:STDOUT:       1:               ir0.inst96
// CHECK:STDOUT:       2:               ir0.inst190
// CHECK:STDOUT:       3:               ir0.inst191
// CHECK:STDOUT:       4:               ir0.inst246
// CHECK:STDOUT:       5:               ir0.inst247
// CHECK:STDOUT:       6:               ir0.inst248
// CHECK:STDOUT:       7:               ir0.inst271
// CHECK:STDOUT:     inst_block95:
// CHECK:STDOUT:       0:               ir0.inst246
// CHECK:STDOUT:     inst_block96:
// CHECK:STDOUT:       0:               ir0.inst281
// CHECK:STDOUT:     inst_block97:
// CHECK:STDOUT:       0:               ir0.inst283
// CHECK:STDOUT:     inst_block98:
// CHECK:STDOUT:       0:               ir0.inst285
// CHECK:STDOUT:     inst_block99:
// CHECK:STDOUT:       0:               ir0.inst295
// CHECK:STDOUT:       1:               ir0.inst296
// CHECK:STDOUT:       2:               ir0.inst297
// CHECK:STDOUT:       3:               ir0.inst298
// CHECK:STDOUT:       4:               ir0.inst299
// CHECK:STDOUT:       5:               ir0.inst300
// CHECK:STDOUT:       6:               ir0.inst301
// CHECK:STDOUT:       7:               ir0.inst302
// CHECK:STDOUT:       8:               ir0.inst303
// CHECK:STDOUT:       9:               ir0.inst304
// CHECK:STDOUT:       10:              ir0.inst305
// CHECK:STDOUT:       11:              ir0.inst306
// CHECK:STDOUT:       12:              ir0.inst307
// CHECK:STDOUT:       13:              ir0.inst308
// CHECK:STDOUT:       14:              ir0.inst309
// CHECK:STDOUT:       15:              ir0.inst310
// CHECK:STDOUT:     inst_block100:
// CHECK:STDOUT:       0:               ir0.inst246
// CHECK:STDOUT:       1:               ir0.inst247
// CHECK:STDOUT:       2:               ir0.inst311
// CHECK:STDOUT:     inst_block101:
// CHECK:STDOUT:       0:               ir0.inst269
// CHECK:STDOUT:       1:               ir0.inst270
// CHECK:STDOUT:     inst_block102:
// CHECK:STDOUT:       0:               ir0.inst167
// CHECK:STDOUT:       1:               ir0.inst168
// CHECK:STDOUT:       2:               ir0.inst61
// CHECK:STDOUT:     inst_block103:
// CHECK:STDOUT:       0:               ir0.inst312
// CHECK:STDOUT:     inst_block104:
// CHECK:STDOUT:       0:               ir0.inst313
// CHECK:STDOUT:     inst_block105:
// CHECK:STDOUT:       0:               ir0.inst314
// CHECK:STDOUT:     inst_block106:
// CHECK:STDOUT:       0:               ir0.inst314
// CHECK:STDOUT:       1:               ir0.inst27
// CHECK:STDOUT:       2:               ir0.inst30
// CHECK:STDOUT:     inst_block107:
// CHECK:STDOUT:       0:               ir0.inst318
// CHECK:STDOUT:     inst_block108:
// CHECK:STDOUT:       0:               ir0.inst63
// CHECK:STDOUT:     inst_block109:
// CHECK:STDOUT:       0:               ir0.inst328
// CHECK:STDOUT:       1:               ir0.inst332
// CHECK:STDOUT:     inst_block110:
// CHECK:STDOUT:       0:               ir0.inst59
// CHECK:STDOUT:       1:               ir0.inst62
// CHECK:STDOUT:       2:               ir0.inst313
// CHECK:STDOUT:       3:               ir0.inst318
// CHECK:STDOUT:       4:               ir0.inst319
// CHECK:STDOUT:       5:               ir0.inst320
// CHECK:STDOUT:       6:               ir0.inst324
// CHECK:STDOUT:     inst_block111:
// CHECK:STDOUT:       0:               inst14
// CHECK:STDOUT:       1:               ir0.inst15
// CHECK:STDOUT:       2:               ir0.inst53
// CHECK:STDOUT: ...
