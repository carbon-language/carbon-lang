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
// CHECK:STDOUT:     import_ir2:      {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir3:      {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir4:      {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir5:      {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir4, inst_id: inst4800000F}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir4, inst_id: inst4800000F}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir4, inst_id: inst48000011}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir4, inst_id: inst4800002D}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir4, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: import_ir4, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: import_ir4, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: import_ir4, inst_id: inst48000022}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: import_ir4, inst_id: inst48000023}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: import_ir4, inst_id: inst4800001C}
// CHECK:STDOUT:     import_ir_inst10: {ir_id: import_ir4, inst_id: inst4800001E}
// CHECK:STDOUT:     import_ir_inst11: {ir_id: import_ir4, inst_id: inst48000011}
// CHECK:STDOUT:     import_ir_inst12: {ir_id: import_ir4, inst_id: inst48000015}
// CHECK:STDOUT:     import_ir_inst13: {ir_id: import_ir4, inst_id: inst48000018}
// CHECK:STDOUT:     import_ir_inst14: {ir_id: import_ir4, inst_id: inst4800001D}
// CHECK:STDOUT:     import_ir_inst15: {ir_id: import_ir4, inst_id: inst48000061}
// CHECK:STDOUT:     import_ir_inst16: {ir_id: import_ir4, inst_id: inst4800005F}
// CHECK:STDOUT:     import_ir_inst17: {ir_id: import_ir4, inst_id: inst48000054}
// CHECK:STDOUT:     import_ir_inst18: {ir_id: import_ir4, inst_id: inst48000050}
// CHECK:STDOUT:     import_ir_inst19: {ir_id: import_ir4, inst_id: inst4800005A}
// CHECK:STDOUT:     import_ir_inst20: {ir_id: import_ir4, inst_id: inst4800005D}
// CHECK:STDOUT:     import_ir_inst21: {ir_id: import_ir4, inst_id: inst48000075}
// CHECK:STDOUT:     import_ir_inst22: {ir_id: import_ir4, inst_id: inst48000060}
// CHECK:STDOUT:     import_ir_inst23: {ir_id: import_ir4, inst_id: inst48000052}
// CHECK:STDOUT:     import_ir_inst24: {ir_id: import_ir4, inst_id: inst48000058}
// CHECK:STDOUT:     import_ir_inst25: {ir_id: import_ir4, inst_id: inst4800005C}
// CHECK:STDOUT:     import_ir_inst26: {ir_id: import_ir4, inst_id: inst48000063}
// CHECK:STDOUT:     import_ir_inst27: {ir_id: import_ir4, inst_id: inst48000075}
// CHECK:STDOUT:     import_ir_inst28: {ir_id: import_ir4, inst_id: inst48000070}
// CHECK:STDOUT:     import_ir_inst29: {ir_id: import_ir4, inst_id: inst48000071}
// CHECK:STDOUT:     import_ir_inst30: {ir_id: import_ir4, inst_id: inst4800006C}
// CHECK:STDOUT:     import_ir_inst31: {ir_id: import_ir4, inst_id: inst4800006E}
// CHECK:STDOUT:     import_ir_inst32: {ir_id: import_ir4, inst_id: inst48000050}
// CHECK:STDOUT:     import_ir_inst33: {ir_id: import_ir4, inst_id: inst48000078}
// CHECK:STDOUT:     import_ir_inst34: {ir_id: import_ir4, inst_id: inst48000079}
// CHECK:STDOUT:     import_ir_inst35: {ir_id: import_ir4, inst_id: inst48000067}
// CHECK:STDOUT:     import_ir_inst36: {ir_id: import_ir4, inst_id: inst48000068}
// CHECK:STDOUT:     import_ir_inst37: {ir_id: import_ir4, inst_id: inst48000069}
// CHECK:STDOUT:     import_ir_inst38: {ir_id: import_ir4, inst_id: inst4800006D}
// CHECK:STDOUT:     import_ir_inst39: {ir_id: import_ir4, inst_id: inst4800007D}
// CHECK:STDOUT:     import_ir_inst40: {ir_id: import_ir4, inst_id: inst48000085}
// CHECK:STDOUT:     import_ir_inst41: {ir_id: import_ir4, inst_id: inst4800008C}
// CHECK:STDOUT:     import_ir_inst42: {ir_id: import_ir4, inst_id: inst48000090}
// CHECK:STDOUT:     import_ir_inst43: {ir_id: import_ir4, inst_id: inst48000091}
// CHECK:STDOUT:     import_ir_inst44: {ir_id: import_ir4, inst_id: inst48000096}
// CHECK:STDOUT:     import_ir_inst45: {ir_id: import_ir4, inst_id: inst480000AB}
// CHECK:STDOUT:     import_ir_inst46: {ir_id: import_ir4, inst_id: inst480000A9}
// CHECK:STDOUT:     import_ir_inst47: {ir_id: import_ir4, inst_id: inst480000A7}
// CHECK:STDOUT:     import_ir_inst48: {ir_id: import_ir4, inst_id: inst480000A8}
// CHECK:STDOUT:     import_ir_inst49: {ir_id: import_ir4, inst_id: inst480000C3}
// CHECK:STDOUT:     import_ir_inst50: {ir_id: import_ir4, inst_id: inst480000C1}
// CHECK:STDOUT:     import_ir_inst51: {ir_id: import_ir4, inst_id: inst480000BF}
// CHECK:STDOUT:     import_ir_inst52: {ir_id: import_ir4, inst_id: inst480000C0}
// CHECK:STDOUT:     import_ir_inst53: {ir_id: import_ir4, inst_id: inst480000DB}
// CHECK:STDOUT:     import_ir_inst54: {ir_id: import_ir4, inst_id: inst480000D9}
// CHECK:STDOUT:     import_ir_inst55: {ir_id: import_ir4, inst_id: inst480000D7}
// CHECK:STDOUT:     import_ir_inst56: {ir_id: import_ir4, inst_id: inst480000D8}
// CHECK:STDOUT:     import_ir_inst57: {ir_id: import_ir4, inst_id: inst480000F3}
// CHECK:STDOUT:     import_ir_inst58: {ir_id: import_ir4, inst_id: inst480000F1}
// CHECK:STDOUT:     import_ir_inst59: {ir_id: import_ir4, inst_id: inst480000EF}
// CHECK:STDOUT:     import_ir_inst60: {ir_id: import_ir4, inst_id: inst480000F0}
// CHECK:STDOUT:     import_ir_inst61: {ir_id: import_ir4, inst_id: inst48000110}
// CHECK:STDOUT:     import_ir_inst62: {ir_id: import_ir4, inst_id: inst4800010E}
// CHECK:STDOUT:     import_ir_inst63: {ir_id: import_ir4, inst_id: inst48000108}
// CHECK:STDOUT:     import_ir_inst64: {ir_id: import_ir4, inst_id: inst48000105}
// CHECK:STDOUT:     import_ir_inst65: {ir_id: import_ir4, inst_id: inst4800010A}
// CHECK:STDOUT:     import_ir_inst66: {ir_id: import_ir4, inst_id: inst4800010D}
// CHECK:STDOUT:     import_ir_inst67: {ir_id: import_ir4, inst_id: inst48000121}
// CHECK:STDOUT:     import_ir_inst68: {ir_id: import_ir4, inst_id: inst4800010F}
// CHECK:STDOUT:     import_ir_inst69: {ir_id: import_ir4, inst_id: inst48000107}
// CHECK:STDOUT:     import_ir_inst70: {ir_id: import_ir4, inst_id: inst4800010C}
// CHECK:STDOUT:     import_ir_inst71: {ir_id: import_ir4, inst_id: inst48000112}
// CHECK:STDOUT:     import_ir_inst72: {ir_id: import_ir4, inst_id: inst48000121}
// CHECK:STDOUT:     import_ir_inst73: {ir_id: import_ir4, inst_id: inst4800011C}
// CHECK:STDOUT:     import_ir_inst74: {ir_id: import_ir4, inst_id: inst4800011D}
// CHECK:STDOUT:     import_ir_inst75: {ir_id: import_ir4, inst_id: inst48000118}
// CHECK:STDOUT:     import_ir_inst76: {ir_id: import_ir4, inst_id: inst4800011A}
// CHECK:STDOUT:     import_ir_inst77: {ir_id: import_ir4, inst_id: inst48000105}
// CHECK:STDOUT:     import_ir_inst78: {ir_id: import_ir4, inst_id: inst48000124}
// CHECK:STDOUT:     import_ir_inst79: {ir_id: import_ir4, inst_id: inst48000125}
// CHECK:STDOUT:     import_ir_inst80: {ir_id: import_ir4, inst_id: inst48000128}
// CHECK:STDOUT:     import_ir_inst81: {ir_id: import_ir4, inst_id: inst48000114}
// CHECK:STDOUT:     import_ir_inst82: {ir_id: import_ir4, inst_id: inst48000115}
// CHECK:STDOUT:     import_ir_inst83: {ir_id: import_ir4, inst_id: inst48000119}
// CHECK:STDOUT:     import_ir_inst84: {ir_id: import_ir4, inst_id: inst4800012E}
// CHECK:STDOUT:     import_ir_inst85: {ir_id: import_ir4, inst_id: inst4800012C}
// CHECK:STDOUT:     import_ir_inst86: {ir_id: import_ir4, inst_id: inst(TypeType)}
// CHECK:STDOUT:     import_ir_inst87: {ir_id: import_ir4, inst_id: inst4800012B}
// CHECK:STDOUT:     import_ir_inst88: {ir_id: import_ir4, inst_id: inst48000143}
// CHECK:STDOUT:     import_ir_inst89: {ir_id: import_ir4, inst_id: inst48000141}
// CHECK:STDOUT:     import_ir_inst90: {ir_id: import_ir4, inst_id: inst4800013F}
// CHECK:STDOUT:     import_ir_inst91: {ir_id: import_ir4, inst_id: inst48000140}
// CHECK:STDOUT:     import_ir_inst92: {ir_id: import_ir4, inst_id: inst48000173}
// CHECK:STDOUT:     import_ir_inst93: {ir_id: import_ir4, inst_id: inst48000171}
// CHECK:STDOUT:     import_ir_inst94: {ir_id: import_ir4, inst_id: inst4800015E}
// CHECK:STDOUT:     import_ir_inst95: {ir_id: import_ir4, inst_id: inst48000158}
// CHECK:STDOUT:     import_ir_inst96: {ir_id: import_ir4, inst_id: inst48000156}
// CHECK:STDOUT:     import_ir_inst97: {ir_id: import_ir4, inst_id: inst4800015B}
// CHECK:STDOUT:     import_ir_inst98: {ir_id: import_ir4, inst_id: inst4800016C}
// CHECK:STDOUT:     import_ir_inst99: {ir_id: import_ir4, inst_id: inst4800016E}
// CHECK:STDOUT:     import_ir_inst100: {ir_id: import_ir4, inst_id: inst4800018B}
// CHECK:STDOUT:     import_ir_inst101: {ir_id: import_ir4, inst_id: inst48000172}
// CHECK:STDOUT:     import_ir_inst102: {ir_id: import_ir4, inst_id: inst48000157}
// CHECK:STDOUT:     import_ir_inst103: {ir_id: import_ir4, inst_id: inst4800015D}
// CHECK:STDOUT:     import_ir_inst104: {ir_id: import_ir4, inst_id: inst48000165}
// CHECK:STDOUT:     import_ir_inst105: {ir_id: import_ir4, inst_id: inst48000169}
// CHECK:STDOUT:     import_ir_inst106: {ir_id: import_ir4, inst_id: inst4800016D}
// CHECK:STDOUT:     import_ir_inst107: {ir_id: import_ir4, inst_id: inst48000175}
// CHECK:STDOUT:     import_ir_inst108: {ir_id: import_ir4, inst_id: inst4800018B}
// CHECK:STDOUT:     import_ir_inst109: {ir_id: import_ir4, inst_id: inst48000186}
// CHECK:STDOUT:     import_ir_inst110: {ir_id: import_ir4, inst_id: inst48000187}
// CHECK:STDOUT:     import_ir_inst111: {ir_id: import_ir4, inst_id: inst48000182}
// CHECK:STDOUT:     import_ir_inst112: {ir_id: import_ir4, inst_id: inst48000184}
// CHECK:STDOUT:     import_ir_inst113: {ir_id: import_ir4, inst_id: inst48000156}
// CHECK:STDOUT:     import_ir_inst114: {ir_id: import_ir4, inst_id: inst4800015B}
// CHECK:STDOUT:     import_ir_inst115: {ir_id: import_ir4, inst_id: inst4800018E}
// CHECK:STDOUT:     import_ir_inst116: {ir_id: import_ir4, inst_id: inst4800018F}
// CHECK:STDOUT:     import_ir_inst117: {ir_id: import_ir4, inst_id: inst4800017B}
// CHECK:STDOUT:     import_ir_inst118: {ir_id: import_ir4, inst_id: inst4800017C}
// CHECK:STDOUT:     import_ir_inst119: {ir_id: import_ir4, inst_id: inst4800017D}
// CHECK:STDOUT:     import_ir_inst120: {ir_id: import_ir4, inst_id: inst4800017E}
// CHECK:STDOUT:     import_ir_inst121: {ir_id: import_ir4, inst_id: inst4800017F}
// CHECK:STDOUT:     import_ir_inst122: {ir_id: import_ir4, inst_id: inst48000183}
// CHECK:STDOUT:     import_ir_inst123: {ir_id: import_ir4, inst_id: inst48000194}
// CHECK:STDOUT:     import_ir_inst124: {ir_id: import_ir4, inst_id: inst4800019C}
// CHECK:STDOUT:     import_ir_inst125: {ir_id: import_ir4, inst_id: inst480001A0}
// CHECK:STDOUT:     import_ir_inst126: {ir_id: import_ir4, inst_id: inst480001A2}
// CHECK:STDOUT:     import_ir_inst127: {ir_id: import_ir4, inst_id: inst480001A3}
// CHECK:STDOUT:     import_ir_inst128: {ir_id: import_ir4, inst_id: inst480001A6}
// CHECK:STDOUT:     import_ir_inst129: {ir_id: import_ir4, inst_id: inst480001B2}
// CHECK:STDOUT:     import_ir_inst130: {ir_id: import_ir4, inst_id: inst480001B7}
// CHECK:STDOUT:     import_ir_inst131: {ir_id: import_ir4, inst_id: inst480001BB}
// CHECK:STDOUT:     import_ir_inst132: {ir_id: import_ir4, inst_id: inst480001BC}
// CHECK:STDOUT:     import_ir_inst133: {ir_id: import_ir4, inst_id: inst480001C1}
// CHECK:STDOUT:     import_ir_inst134: {ir_id: import_ir4, inst_id: inst480001FA}
// CHECK:STDOUT:     import_ir_inst135: {ir_id: import_ir4, inst_id: inst480001F8}
// CHECK:STDOUT:     import_ir_inst136: {ir_id: import_ir4, inst_id: inst480001E0}
// CHECK:STDOUT:     import_ir_inst137: {ir_id: import_ir4, inst_id: inst480001DA}
// CHECK:STDOUT:     import_ir_inst138: {ir_id: import_ir4, inst_id: inst480001D5}
// CHECK:STDOUT:     import_ir_inst139: {ir_id: import_ir4, inst_id: inst480001D3}
// CHECK:STDOUT:     import_ir_inst140: {ir_id: import_ir4, inst_id: inst480001D8}
// CHECK:STDOUT:     import_ir_inst141: {ir_id: import_ir4, inst_id: inst480001DD}
// CHECK:STDOUT:     import_ir_inst142: {ir_id: import_ir4, inst_id: inst480001F2}
// CHECK:STDOUT:     import_ir_inst143: {ir_id: import_ir4, inst_id: inst480001F4}
// CHECK:STDOUT:     import_ir_inst144: {ir_id: import_ir4, inst_id: inst48000216}
// CHECK:STDOUT:     import_ir_inst145: {ir_id: import_ir4, inst_id: inst480001F9}
// CHECK:STDOUT:     import_ir_inst146: {ir_id: import_ir4, inst_id: inst480001D4}
// CHECK:STDOUT:     import_ir_inst147: {ir_id: import_ir4, inst_id: inst480001D9}
// CHECK:STDOUT:     import_ir_inst148: {ir_id: import_ir4, inst_id: inst480001DF}
// CHECK:STDOUT:     import_ir_inst149: {ir_id: import_ir4, inst_id: inst480001E8}
// CHECK:STDOUT:     import_ir_inst150: {ir_id: import_ir4, inst_id: inst480001EB}
// CHECK:STDOUT:     import_ir_inst151: {ir_id: import_ir4, inst_id: inst480001EF}
// CHECK:STDOUT:     import_ir_inst152: {ir_id: import_ir4, inst_id: inst480001F3}
// CHECK:STDOUT:     import_ir_inst153: {ir_id: import_ir4, inst_id: inst480001FC}
// CHECK:STDOUT:     import_ir_inst154: {ir_id: import_ir4, inst_id: inst48000216}
// CHECK:STDOUT:     import_ir_inst155: {ir_id: import_ir4, inst_id: inst48000211}
// CHECK:STDOUT:     import_ir_inst156: {ir_id: import_ir4, inst_id: inst48000212}
// CHECK:STDOUT:     import_ir_inst157: {ir_id: import_ir4, inst_id: inst4800020D}
// CHECK:STDOUT:     import_ir_inst158: {ir_id: import_ir4, inst_id: inst4800020F}
// CHECK:STDOUT:     import_ir_inst159: {ir_id: import_ir4, inst_id: inst480001D3}
// CHECK:STDOUT:     import_ir_inst160: {ir_id: import_ir4, inst_id: inst480001D8}
// CHECK:STDOUT:     import_ir_inst161: {ir_id: import_ir4, inst_id: inst480001DD}
// CHECK:STDOUT:     import_ir_inst162: {ir_id: import_ir4, inst_id: inst48000219}
// CHECK:STDOUT:     import_ir_inst163: {ir_id: import_ir4, inst_id: inst4800021A}
// CHECK:STDOUT:     import_ir_inst164: {ir_id: import_ir4, inst_id: inst48000204}
// CHECK:STDOUT:     import_ir_inst165: {ir_id: import_ir4, inst_id: inst48000205}
// CHECK:STDOUT:     import_ir_inst166: {ir_id: import_ir4, inst_id: inst48000206}
// CHECK:STDOUT:     import_ir_inst167: {ir_id: import_ir4, inst_id: inst48000207}
// CHECK:STDOUT:     import_ir_inst168: {ir_id: import_ir4, inst_id: inst48000208}
// CHECK:STDOUT:     import_ir_inst169: {ir_id: import_ir4, inst_id: inst48000209}
// CHECK:STDOUT:     import_ir_inst170: {ir_id: import_ir4, inst_id: inst4800020A}
// CHECK:STDOUT:     import_ir_inst171: {ir_id: import_ir4, inst_id: inst4800020E}
// CHECK:STDOUT:     import_ir_inst172: {ir_id: import_ir4, inst_id: inst4800021F}
// CHECK:STDOUT:     import_ir_inst173: {ir_id: import_ir4, inst_id: inst48000226}
// CHECK:STDOUT:     import_ir_inst174: {ir_id: import_ir4, inst_id: inst4800022A}
// CHECK:STDOUT:     import_ir_inst175: {ir_id: import_ir4, inst_id: inst4800022C}
// CHECK:STDOUT:     import_ir_inst176: {ir_id: import_ir4, inst_id: inst4800022D}
// CHECK:STDOUT:     import_ir_inst177: {ir_id: import_ir4, inst_id: inst48000230}
// CHECK:STDOUT:     import_ir_inst178: {ir_id: import_ir4, inst_id: inst4800023A}
// CHECK:STDOUT:     import_ir_inst179: {ir_id: import_ir4, inst_id: inst4800023E}
// CHECK:STDOUT:     import_ir_inst180: {ir_id: import_ir4, inst_id: inst48000240}
// CHECK:STDOUT:     import_ir_inst181: {ir_id: import_ir4, inst_id: inst48000241}
// CHECK:STDOUT:     import_ir_inst182: {ir_id: import_ir4, inst_id: inst48000244}
// CHECK:STDOUT:     import_ir_inst183: {ir_id: import_ir4, inst_id: inst48000250}
// CHECK:STDOUT:     import_ir_inst184: {ir_id: import_ir4, inst_id: inst48000255}
// CHECK:STDOUT:     import_ir_inst185: {ir_id: import_ir4, inst_id: inst48000259}
// CHECK:STDOUT:     import_ir_inst186: {ir_id: import_ir4, inst_id: inst4800025A}
// CHECK:STDOUT:     import_ir_inst187: {ir_id: import_ir4, inst_id: inst4800025F}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst0000000E, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst60000010, name0: inst60000035}}
// CHECK:STDOUT:     name_scope1:     {inst: inst60000010, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: inst60000045}}
// CHECK:STDOUT:     name_scope2:     {inst: inst60000046, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {name(SelfType): inst60000049, name4: inst6000004A}}
// CHECK:STDOUT:     name_scope3:     {inst: inst6000005E, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope4:     {inst: inst6000008B, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope5:     {inst: inst6000008F, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6:     {inst: inst60000093, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7:     {inst: inst60000097, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope8:     {inst: inst6000009B, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope9:     {inst: inst600000B5, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope10:    {inst: inst600000B9, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope11:    {inst: inst600000BD, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope12:    {inst: inst600000F5, parent_scope: name_scope1, has_error: false, extended_scopes: [], names: {}}
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
// CHECK:STDOUT:     function60000000: {name: name0, parent_scope: name_scope0, call_params_id: inst_block13, return_slot_pattern: inst60000030, body: [inst_block20]}
// CHECK:STDOUT:     function60000001: {name: name4, parent_scope: name_scope2, return_slot_pattern: inst60000055}
// CHECK:STDOUT:     function60000002: {name: name4, parent_scope: name_scope3, return_slot_pattern: inst60000073}
// CHECK:STDOUT:     function60000003: {name: name4, parent_scope: name_scope8, return_slot_pattern: inst600000AA}
// CHECK:STDOUT:     function60000004: {name: name4, parent_scope: name_scope11, return_slot_pattern: inst600000D5}
// CHECK:STDOUT:     function60000005: {name: name4, parent_scope: name_scope12, return_slot_pattern: inst60000111}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic0:        {decl: inst60000035, bindings: inst_block16}
// CHECK:STDOUT:     generic1:        {decl: inst6000004F, bindings: inst_block24}
// CHECK:STDOUT:     generic2:        {decl: inst6000005E, bindings: inst_block29}
// CHECK:STDOUT:     generic3:        {decl: inst6000006E, bindings: inst_block36}
// CHECK:STDOUT:     generic4:        {decl: inst6000009B, bindings: inst_block46}
// CHECK:STDOUT:     generic5:        {decl: inst600000A6, bindings: inst_block52}
// CHECK:STDOUT:     generic6:        {decl: inst600000BD, bindings: inst_block59}
// CHECK:STDOUT:     generic7:        {decl: inst600000D0, bindings: inst_block67}
// CHECK:STDOUT:     generic8:        {decl: inst600000F5, bindings: inst_block81}
// CHECK:STDOUT:     generic9:        {decl: inst6000010C, bindings: inst_block89}
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
// CHECK:STDOUT:     'type(inst60000024)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(inst60000026)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst60000028)}
// CHECK:STDOUT:     'type(inst60000028)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst60000028)}
// CHECK:STDOUT:     'type(inst60000036)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
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
// CHECK:STDOUT:     'type(inst60000047)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst60000047)}
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     'type(symbolic_constant253)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(symbolic_constant256)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst0000000E:    {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst6000000F:    {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst60000010:    {kind: Namespace, arg0: name_scope1, arg1: inst6000000F, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000011:    {kind: FacetType, arg0: facet_type0, type: type(TypeType)}
// CHECK:STDOUT:     inst60000012:    {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000013:    {kind: BindSymbolicName, arg0: entity_name0, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000014:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000015:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000016:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000017:    {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst60000018:    {kind: SymbolicBindingPattern, arg0: entity_name1, type: type(inst60000017)}
// CHECK:STDOUT:     inst60000019:    {kind: NameRef, arg0: name1, arg1: inst60000014, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001A:    {kind: PointerType, arg0: inst60000019, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001B:    {kind: PointerType, arg0: inst60000015, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001C:    {kind: PointerType, arg0: inst60000016, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001D:    {kind: BindName, arg0: entity_name2, arg1: inst60000031, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst6000001E:    {kind: PatternType, arg0: inst6000001B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001F:    {kind: BindingPattern, arg0: entity_name2, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst60000020:    {kind: PatternType, arg0: inst6000001C, type: type(TypeType)}
// CHECK:STDOUT:     inst60000021:    {kind: ValueParamPattern, arg0: inst6000001F, arg1: call_param0, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst60000022:    {kind: NameRef, arg0: name1, arg1: inst60000014, type: type(TypeType)}
// CHECK:STDOUT:     inst60000023:    {kind: PointerType, arg0: inst60000022, type: type(TypeType)}
// CHECK:STDOUT:     inst60000024:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000025:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000026:    {kind: TupleType, arg0: inst_block9, type: type(TypeType)}
// CHECK:STDOUT:     inst60000027:    {kind: TupleLiteral, arg0: inst_block8, type: type(inst60000026)}
// CHECK:STDOUT:     inst60000028:    {kind: PointerType, arg0: inst60000026, type: type(TypeType)}
// CHECK:STDOUT:     inst60000029:    {kind: Converted, arg0: inst60000025, arg1: inst60000024, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002A:    {kind: TupleType, arg0: inst_block11, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002B:    {kind: Converted, arg0: inst60000027, arg1: inst6000002A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002C:    {kind: TupleType, arg0: inst_block12, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002D:    {kind: PatternType, arg0: inst6000002A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002E:    {kind: ReturnSlotPattern, arg0: inst6000002B, type: type(symbolic_constant10)}
// CHECK:STDOUT:     inst6000002F:    {kind: PatternType, arg0: inst6000002C, type: type(TypeType)}
// CHECK:STDOUT:     inst60000030:    {kind: OutParamPattern, arg0: inst6000002E, arg1: call_param1, type: type(symbolic_constant10)}
// CHECK:STDOUT:     inst60000031:    {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000032:    {kind: SpliceBlock, arg0: inst_block6, arg1: inst6000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000033:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst60000034:    {kind: ReturnSlot, arg0: inst6000002A, arg1: inst60000033, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst60000035:    {kind: FunctionDecl, arg0: function60000000, arg1: inst_block15, type: type(inst60000036)}
// CHECK:STDOUT:     inst60000036:    {kind: FunctionType, arg0: function60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000037:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000036)}
// CHECK:STDOUT:     inst60000038:    {kind: PointerType, arg0: inst6000002A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000039:    {kind: RequireCompleteType, arg0: inst6000002A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003A:    {kind: RequireCompleteType, arg0: inst6000002A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003B:    {kind: RequireCompleteType, arg0: inst6000002C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003C:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003D:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003E:    {kind: RequireCompleteType, arg0: inst6000001C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003F:    {kind: NameRef, arg0: name2, arg1: inst6000001D, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000040:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000041:    {kind: TupleLiteral, arg0: inst_block21, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst60000042:    {kind: RequireCompleteType, arg0: inst6000002A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000043:    {kind: TupleAccess, arg0: inst60000034, arg1: element0, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000044:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000045:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name3, type: type(TypeType)}
// CHECK:STDOUT:     inst60000046:    {kind: InterfaceDecl, arg0: interface0, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000047:    {kind: FacetType, arg0: facet_type1, type: type(TypeType)}
// CHECK:STDOUT:     inst60000048:    {kind: BindSymbolicName, arg0: entity_name4, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000049:    {kind: ImportRefUnloaded, arg0: import_ir_inst2, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000004A:    {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst6000004C)}
// CHECK:STDOUT:     inst6000004B:    {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name5}
// CHECK:STDOUT:     inst6000004C:    {kind: AssociatedEntityType, arg0: interface0, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000004D:    {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(inst60000050)}
// CHECK:STDOUT:     inst6000004E:    {kind: AssociatedEntity, arg0: element0, arg1: inst6000004D, type: type(inst6000004C)}
// CHECK:STDOUT:     inst6000004F:    {kind: FunctionDecl, arg0: function60000001, arg1: inst_block_empty, type: type(inst60000050)}
// CHECK:STDOUT:     inst60000050:    {kind: FunctionType, arg0: function60000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000051:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000050)}
// CHECK:STDOUT:     inst60000052:    {kind: SymbolicBindingType, arg0: entity_name4, arg1: inst60000048, type: type(TypeType)}
// CHECK:STDOUT:     inst60000053:    {kind: PatternType, arg0: inst60000052, type: type(TypeType)}
// CHECK:STDOUT:     inst60000054:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst60000055:    {kind: OutParamPattern, arg0: inst60000054, arg1: call_param1, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst60000056:    {kind: BindingPattern, arg0: entity_name8, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst60000057:    {kind: ValueParamPattern, arg0: inst60000056, arg1: call_param0, type: type(symbolic_constant19)}
// CHECK:STDOUT:     inst60000058:    {kind: ImportRefLoaded, arg0: import_ir_inst11, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000059:    {kind: BindSymbolicName, arg0: entity_name4, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000005A:    {kind: SymbolicBindingType, arg0: entity_name4, arg1: inst60000059, type: type(TypeType)}
// CHECK:STDOUT:     inst6000005B:    {kind: PatternType, arg0: inst6000005A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000005C:    {kind: LookupImplWitness, arg0: inst6000001B, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000005D:    {kind: ImportRefUnloaded, arg0: import_ir_inst15, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000005E:    {kind: ImplDecl, arg0: impl0, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000005F:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000060:    {kind: SymbolicBindingType, arg0: entity_name1, arg1: inst6000005F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000061:    {kind: ConstType, arg0: inst60000060, type: type(TypeType)}
// CHECK:STDOUT:     inst60000062:    {kind: PatternType, arg0: inst60000047, type: type(TypeType)}
// CHECK:STDOUT:     inst60000063:    {kind: SymbolicBindingPattern, arg0: entity_name13, type: type(inst60000062)}
// CHECK:STDOUT:     inst60000064:    {kind: ImportRefLoaded, arg0: import_ir_inst18, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000065:    {kind: ImportRefLoaded, arg0: import_ir_inst19, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000066:    {kind: ImportRefLoaded, arg0: import_ir_inst20, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000067:    {kind: ImportRefUnloaded, arg0: import_ir_inst21, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000068:    {kind: ImplWitnessTable, arg0: inst_block30, arg1: impl0}
// CHECK:STDOUT:     inst60000069:    {kind: ImplWitness, arg0: inst60000068, arg1: specific2, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000006A:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000006B:    {kind: SymbolicBindingType, arg0: entity_name1, arg1: inst6000006A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000006C:    {kind: ConstType, arg0: inst6000006B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000006D:    {kind: ImplWitness, arg0: inst60000068, arg1: specific3, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000006E:    {kind: FunctionDecl, arg0: function60000002, arg1: inst_block_empty, type: type(symbolic_constant38)}
// CHECK:STDOUT:     inst6000006F:    {kind: FunctionType, arg0: function60000002, arg1: specific2, type: type(TypeType)}
// CHECK:STDOUT:     inst60000070:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant38)}
// CHECK:STDOUT:     inst60000071:    {kind: PatternType, arg0: inst60000061, type: type(TypeType)}
// CHECK:STDOUT:     inst60000072:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant43)}
// CHECK:STDOUT:     inst60000073:    {kind: OutParamPattern, arg0: inst60000072, arg1: call_param1, type: type(symbolic_constant43)}
// CHECK:STDOUT:     inst60000074:    {kind: BindingPattern, arg0: entity_name17, type: type(symbolic_constant43)}
// CHECK:STDOUT:     inst60000075:    {kind: ValueParamPattern, arg0: inst60000074, arg1: call_param0, type: type(symbolic_constant43)}
// CHECK:STDOUT:     inst60000076:    {kind: ImportRefLoaded, arg0: import_ir_inst32, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000077:    {kind: FunctionType, arg0: function60000002, arg1: specific3, type: type(TypeType)}
// CHECK:STDOUT:     inst60000078:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant45)}
// CHECK:STDOUT:     inst60000079:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000007A:    {kind: SymbolicBindingType, arg0: entity_name1, arg1: inst60000079, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007B:    {kind: ConstType, arg0: inst6000007A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007C:    {kind: PatternType, arg0: inst6000007B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007D:    {kind: RequireCompleteType, arg0: inst60000061, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000007E:    {kind: RequireCompleteType, arg0: inst60000060, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000007F:    {kind: LookupImplWitness, arg0: inst6000005F, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000080:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000005F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000081:    {kind: ImplWitnessAccess, arg0: inst6000007F, arg1: element0, type: type(symbolic_constant60)}
// CHECK:STDOUT:     inst60000082:    {kind: SpecificImplFunction, arg0: inst60000081, arg1: specific5, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000083:    {kind: RequireCompleteType, arg0: inst6000007B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000084:    {kind: RequireCompleteType, arg0: inst6000007A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000085:    {kind: LookupImplWitness, arg0: inst60000079, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000086:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst60000079, type: type(TypeType)}
// CHECK:STDOUT:     inst60000087:    {kind: ImplWitnessAccess, arg0: inst60000085, arg1: element0, type: type(symbolic_constant69)}
// CHECK:STDOUT:     inst60000088:    {kind: SpecificImplFunction, arg0: inst60000087, arg1: specific6, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000089:    {kind: PatternType, arg0: inst60000060, type: type(TypeType)}
// CHECK:STDOUT:     inst6000008A:    {kind: ImportRefUnloaded, arg0: import_ir_inst45, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000008B:    {kind: ImplDecl, arg0: impl1, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000008C:    {kind: ImportRefLoaded, arg0: import_ir_inst47, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000008D:    {kind: ImportRefLoaded, arg0: import_ir_inst48, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000008E:    {kind: ImportRefUnloaded, arg0: import_ir_inst49, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000008F:    {kind: ImplDecl, arg0: impl2, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000090:    {kind: ImportRefLoaded, arg0: import_ir_inst51, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000091:    {kind: ImportRefLoaded, arg0: import_ir_inst52, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000092:    {kind: ImportRefUnloaded, arg0: import_ir_inst53, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000093:    {kind: ImplDecl, arg0: impl3, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000094:    {kind: ImportRefLoaded, arg0: import_ir_inst55, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000095:    {kind: ImportRefLoaded, arg0: import_ir_inst56, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000096:    {kind: ImportRefUnloaded, arg0: import_ir_inst57, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000097:    {kind: ImplDecl, arg0: impl4, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000098:    {kind: ImportRefLoaded, arg0: import_ir_inst59, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000099:    {kind: ImportRefLoaded, arg0: import_ir_inst60, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000009A:    {kind: ImportRefLoaded, arg0: import_ir_inst61, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000009B:    {kind: ImplDecl, arg0: impl5, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000009C:    {kind: SymbolicBindingPattern, arg0: entity_name21, type: type(inst60000017)}
// CHECK:STDOUT:     inst6000009D:    {kind: ImportRefLoaded, arg0: import_ir_inst64, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000009E:    {kind: ImportRefLoaded, arg0: import_ir_inst65, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000009F:    {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A0:    {kind: ImportRefUnloaded, arg0: import_ir_inst67, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000A1:    {kind: ImplWitnessTable, arg0: inst_block47, arg1: impl5}
// CHECK:STDOUT:     inst600000A2:    {kind: ImplWitness, arg0: inst600000A1, arg1: specific7, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000A3:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A4:    {kind: PointerType, arg0: inst600000A3, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A5:    {kind: ImplWitness, arg0: inst600000A1, arg1: specific8, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000A6:    {kind: FunctionDecl, arg0: function60000003, arg1: inst_block_empty, type: type(symbolic_constant81)}
// CHECK:STDOUT:     inst600000A7:    {kind: FunctionType, arg0: function60000003, arg1: specific7, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A8:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant81)}
// CHECK:STDOUT:     inst600000A9:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant85)}
// CHECK:STDOUT:     inst600000AA:    {kind: OutParamPattern, arg0: inst600000A9, arg1: call_param1, type: type(symbolic_constant85)}
// CHECK:STDOUT:     inst600000AB:    {kind: BindingPattern, arg0: entity_name24, type: type(symbolic_constant85)}
// CHECK:STDOUT:     inst600000AC:    {kind: ValueParamPattern, arg0: inst600000AB, arg1: call_param0, type: type(symbolic_constant85)}
// CHECK:STDOUT:     inst600000AD:    {kind: ImportRefLoaded, arg0: import_ir_inst77, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000AE:    {kind: FunctionType, arg0: function60000003, arg1: specific8, type: type(TypeType)}
// CHECK:STDOUT:     inst600000AF:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant88)}
// CHECK:STDOUT:     inst600000B0:    {kind: RequireCompleteType, arg0: inst600000A4, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000B1:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B2:    {kind: PointerType, arg0: inst600000B1, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B3:    {kind: PatternType, arg0: inst600000B2, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B4:    {kind: ImportRefUnloaded, arg0: import_ir_inst84, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000B5:    {kind: ImplDecl, arg0: impl6, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000B6:    {kind: ImportRefLoaded, arg0: import_ir_inst86, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B7:    {kind: ImportRefLoaded, arg0: import_ir_inst87, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B8:    {kind: ImportRefUnloaded, arg0: import_ir_inst88, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000B9:    {kind: ImplDecl, arg0: impl7, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000BA:    {kind: ImportRefLoaded, arg0: import_ir_inst90, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000BB:    {kind: ImportRefLoaded, arg0: import_ir_inst91, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000BC:    {kind: ImportRefUnloaded, arg0: import_ir_inst92, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000BD:    {kind: ImplDecl, arg0: impl8, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000BE:    {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000BF:    {kind: SymbolicBindingType, arg0: entity_name26, arg1: inst600000BE, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C0:    {kind: TupleType, arg0: inst_block57, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C1:    {kind: SymbolicBindingPattern, arg0: entity_name28, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000C2:    {kind: SymbolicBindingPattern, arg0: entity_name29, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000C3:    {kind: ImportRefLoaded, arg0: import_ir_inst96, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000C4:    {kind: ImportRefLoaded, arg0: import_ir_inst97, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000C5:    {kind: ImportRefLoaded, arg0: import_ir_inst98, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C6:    {kind: ImportRefLoaded, arg0: import_ir_inst99, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C7:    {kind: ImportRefUnloaded, arg0: import_ir_inst100, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000C8:    {kind: ImplWitnessTable, arg0: inst_block60, arg1: impl8}
// CHECK:STDOUT:     inst600000C9:    {kind: ImplWitness, arg0: inst600000C8, arg1: specific10, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000CA:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000CB:    {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000CC:    {kind: SymbolicBindingType, arg0: entity_name1, arg1: inst600000CA, type: type(TypeType)}
// CHECK:STDOUT:     inst600000CD:    {kind: SymbolicBindingType, arg0: entity_name26, arg1: inst600000CB, type: type(TypeType)}
// CHECK:STDOUT:     inst600000CE:    {kind: TupleType, arg0: inst_block62, type: type(TypeType)}
// CHECK:STDOUT:     inst600000CF:    {kind: ImplWitness, arg0: inst600000C8, arg1: specific11, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000D0:    {kind: FunctionDecl, arg0: function60000004, arg1: inst_block_empty, type: type(symbolic_constant114)}
// CHECK:STDOUT:     inst600000D1:    {kind: FunctionType, arg0: function60000004, arg1: specific10, type: type(TypeType)}
// CHECK:STDOUT:     inst600000D2:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant114)}
// CHECK:STDOUT:     inst600000D3:    {kind: PatternType, arg0: inst600000C0, type: type(TypeType)}
// CHECK:STDOUT:     inst600000D4:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant119)}
// CHECK:STDOUT:     inst600000D5:    {kind: OutParamPattern, arg0: inst600000D4, arg1: call_param1, type: type(symbolic_constant119)}
// CHECK:STDOUT:     inst600000D6:    {kind: BindingPattern, arg0: entity_name36, type: type(symbolic_constant119)}
// CHECK:STDOUT:     inst600000D7:    {kind: ValueParamPattern, arg0: inst600000D6, arg1: call_param0, type: type(symbolic_constant119)}
// CHECK:STDOUT:     inst600000D8:    {kind: ImportRefLoaded, arg0: import_ir_inst113, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000D9:    {kind: ImportRefLoaded, arg0: import_ir_inst114, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000DA:    {kind: FunctionType, arg0: function60000004, arg1: specific11, type: type(TypeType)}
// CHECK:STDOUT:     inst600000DB:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant121)}
// CHECK:STDOUT:     inst600000DC:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000DD:    {kind: SymbolicBindingType, arg0: entity_name1, arg1: inst600000DC, type: type(TypeType)}
// CHECK:STDOUT:     inst600000DE:    {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000DF:    {kind: SymbolicBindingType, arg0: entity_name26, arg1: inst600000DE, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E0:    {kind: TupleType, arg0: inst_block70, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E1:    {kind: PatternType, arg0: inst600000E0, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E2:    {kind: RequireCompleteType, arg0: inst600000C0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E3:    {kind: RequireCompleteType, arg0: inst600000BF, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E4:    {kind: LookupImplWitness, arg0: inst600000BE, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E5:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst600000BE, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E6:    {kind: ImplWitnessAccess, arg0: inst600000E4, arg1: element0, type: type(symbolic_constant145)}
// CHECK:STDOUT:     inst600000E7:    {kind: SpecificImplFunction, arg0: inst600000E6, arg1: specific13, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000E8:    {kind: RequireCompleteType, arg0: inst600000E0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E9:    {kind: RequireCompleteType, arg0: inst600000DD, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EA:    {kind: LookupImplWitness, arg0: inst600000DC, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EB:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst600000DC, type: type(TypeType)}
// CHECK:STDOUT:     inst600000EC:    {kind: ImplWitnessAccess, arg0: inst600000EA, arg1: element0, type: type(symbolic_constant154)}
// CHECK:STDOUT:     inst600000ED:    {kind: SpecificImplFunction, arg0: inst600000EC, arg1: specific14, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000EE:    {kind: RequireCompleteType, arg0: inst600000DF, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EF:    {kind: LookupImplWitness, arg0: inst600000DE, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000F0:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst600000DE, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F1:    {kind: ImplWitnessAccess, arg0: inst600000EF, arg1: element0, type: type(symbolic_constant159)}
// CHECK:STDOUT:     inst600000F2:    {kind: SpecificImplFunction, arg0: inst600000F1, arg1: specific15, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000F3:    {kind: PatternType, arg0: inst600000BF, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F4:    {kind: ImportRefUnloaded, arg0: import_ir_inst134, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000F5:    {kind: ImplDecl, arg0: impl9, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000F6:    {kind: BindSymbolicName, arg0: entity_name41, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000F7:    {kind: SymbolicBindingType, arg0: entity_name41, arg1: inst600000F6, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F8:    {kind: TupleType, arg0: inst_block79, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F9:    {kind: SymbolicBindingPattern, arg0: entity_name43, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000FA:    {kind: SymbolicBindingPattern, arg0: entity_name44, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000FB:    {kind: SymbolicBindingPattern, arg0: entity_name45, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000FC:    {kind: ImportRefLoaded, arg0: import_ir_inst139, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000FD:    {kind: ImportRefLoaded, arg0: import_ir_inst140, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000FE:    {kind: ImportRefLoaded, arg0: import_ir_inst141, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000FF:    {kind: ImportRefLoaded, arg0: import_ir_inst142, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000100:    {kind: ImportRefLoaded, arg0: import_ir_inst143, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000101:    {kind: ImportRefUnloaded, arg0: import_ir_inst144, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000102:    {kind: ImplWitnessTable, arg0: inst_block82, arg1: impl9}
// CHECK:STDOUT:     inst60000103:    {kind: ImplWitness, arg0: inst60000102, arg1: specific16, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000104:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000105:    {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000106:    {kind: BindSymbolicName, arg0: entity_name41, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000107:    {kind: SymbolicBindingType, arg0: entity_name1, arg1: inst60000104, type: type(TypeType)}
// CHECK:STDOUT:     inst60000108:    {kind: SymbolicBindingType, arg0: entity_name26, arg1: inst60000105, type: type(TypeType)}
// CHECK:STDOUT:     inst60000109:    {kind: SymbolicBindingType, arg0: entity_name41, arg1: inst60000106, type: type(TypeType)}
// CHECK:STDOUT:     inst6000010A:    {kind: TupleType, arg0: inst_block84, type: type(TypeType)}
// CHECK:STDOUT:     inst6000010B:    {kind: ImplWitness, arg0: inst60000102, arg1: specific17, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000010C:    {kind: FunctionDecl, arg0: function60000005, arg1: inst_block_empty, type: type(symbolic_constant186)}
// CHECK:STDOUT:     inst6000010D:    {kind: FunctionType, arg0: function60000005, arg1: specific16, type: type(TypeType)}
// CHECK:STDOUT:     inst6000010E:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant186)}
// CHECK:STDOUT:     inst6000010F:    {kind: PatternType, arg0: inst600000F8, type: type(TypeType)}
// CHECK:STDOUT:     inst60000110:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant191)}
// CHECK:STDOUT:     inst60000111:    {kind: OutParamPattern, arg0: inst60000110, arg1: call_param1, type: type(symbolic_constant191)}
// CHECK:STDOUT:     inst60000112:    {kind: BindingPattern, arg0: entity_name55, type: type(symbolic_constant191)}
// CHECK:STDOUT:     inst60000113:    {kind: ValueParamPattern, arg0: inst60000112, arg1: call_param0, type: type(symbolic_constant191)}
// CHECK:STDOUT:     inst60000114:    {kind: ImportRefLoaded, arg0: import_ir_inst159, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000115:    {kind: ImportRefLoaded, arg0: import_ir_inst160, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000116:    {kind: ImportRefLoaded, arg0: import_ir_inst161, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000117:    {kind: FunctionType, arg0: function60000005, arg1: specific17, type: type(TypeType)}
// CHECK:STDOUT:     inst60000118:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant193)}
// CHECK:STDOUT:     inst60000119:    {kind: BindSymbolicName, arg0: entity_name1, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000011A:    {kind: SymbolicBindingType, arg0: entity_name1, arg1: inst60000119, type: type(TypeType)}
// CHECK:STDOUT:     inst6000011B:    {kind: BindSymbolicName, arg0: entity_name26, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000011C:    {kind: SymbolicBindingType, arg0: entity_name26, arg1: inst6000011B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000011D:    {kind: BindSymbolicName, arg0: entity_name41, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000011E:    {kind: SymbolicBindingType, arg0: entity_name41, arg1: inst6000011D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000011F:    {kind: TupleType, arg0: inst_block92, type: type(TypeType)}
// CHECK:STDOUT:     inst60000120:    {kind: PatternType, arg0: inst6000011F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000121:    {kind: RequireCompleteType, arg0: inst600000F8, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000122:    {kind: RequireCompleteType, arg0: inst600000F7, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000123:    {kind: LookupImplWitness, arg0: inst600000F6, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000124:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst600000F6, type: type(TypeType)}
// CHECK:STDOUT:     inst60000125:    {kind: ImplWitnessAccess, arg0: inst60000123, arg1: element0, type: type(symbolic_constant226)}
// CHECK:STDOUT:     inst60000126:    {kind: SpecificImplFunction, arg0: inst60000125, arg1: specific19, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000127:    {kind: RequireCompleteType, arg0: inst6000011F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000128:    {kind: RequireCompleteType, arg0: inst6000011A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000129:    {kind: LookupImplWitness, arg0: inst60000119, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012A:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst60000119, type: type(TypeType)}
// CHECK:STDOUT:     inst6000012B:    {kind: ImplWitnessAccess, arg0: inst60000129, arg1: element0, type: type(symbolic_constant235)}
// CHECK:STDOUT:     inst6000012C:    {kind: SpecificImplFunction, arg0: inst6000012B, arg1: specific20, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst6000012D:    {kind: RequireCompleteType, arg0: inst6000011C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012E:    {kind: LookupImplWitness, arg0: inst6000011B, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012F:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000011B, type: type(TypeType)}
// CHECK:STDOUT:     inst60000130:    {kind: ImplWitnessAccess, arg0: inst6000012E, arg1: element0, type: type(symbolic_constant240)}
// CHECK:STDOUT:     inst60000131:    {kind: SpecificImplFunction, arg0: inst60000130, arg1: specific21, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000132:    {kind: RequireCompleteType, arg0: inst6000011E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000133:    {kind: LookupImplWitness, arg0: inst6000011D, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000134:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000011D, type: type(TypeType)}
// CHECK:STDOUT:     inst60000135:    {kind: ImplWitnessAccess, arg0: inst60000133, arg1: element0, type: type(symbolic_constant245)}
// CHECK:STDOUT:     inst60000136:    {kind: SpecificImplFunction, arg0: inst60000135, arg1: specific22, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000137:    {kind: PatternType, arg0: inst600000F7, type: type(TypeType)}
// CHECK:STDOUT:     inst60000138:    {kind: LookupImplWitness, arg0: inst6000001B, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000139:    {kind: LookupImplWitness, arg0: inst6000001C, arg1: specific_interface0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000013A:    {kind: FacetValue, arg0: inst6000001B, arg1: inst_block103, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000013B:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000013A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000013C:    {kind: ImplWitnessAccess, arg0: inst60000138, arg1: element0, type: type(symbolic_constant256)}
// CHECK:STDOUT:     inst6000013D:    {kind: ImplWitnessAccess, arg0: inst60000138, arg1: element0, type: type(symbolic_constant253)}
// CHECK:STDOUT:     inst6000013E:    {kind: FacetValue, arg0: inst6000001C, arg1: inst_block104, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000013F:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000013E, type: type(TypeType)}
// CHECK:STDOUT:     inst60000140:    {kind: ImplWitnessAccess, arg0: inst60000139, arg1: element0, type: type(symbolic_constant256)}
// CHECK:STDOUT:     inst60000141:    {kind: BoundMethod, arg0: inst6000003F, arg1: inst6000013C, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst60000142:    {kind: SpecificImplFunction, arg0: inst6000013C, arg1: specific23, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000143:    {kind: SpecificImplFunction, arg0: inst6000013D, arg1: specific23, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000144:    {kind: SpecificImplFunction, arg0: inst60000140, arg1: specific24, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000145:    {kind: BoundMethod, arg0: inst6000003F, arg1: inst60000142, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst60000146:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000147:    {kind: Call, arg0: inst60000145, arg1: inst_block108, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000148:    {kind: InitializeFrom, arg0: inst60000147, arg1: inst60000043, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000149:    {kind: TupleAccess, arg0: inst60000034, arg1: element1, type: type(inst60000024)}
// CHECK:STDOUT:     inst6000014A:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst60000149, type: type(inst60000024)}
// CHECK:STDOUT:     inst6000014B:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst6000014C:    {kind: Converted, arg0: inst60000040, arg1: inst6000014A, type: type(inst60000024)}
// CHECK:STDOUT:     inst6000014D:    {kind: TupleInit, arg0: inst_block109, arg1: inst60000034, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst6000014E:    {kind: Converted, arg0: inst60000041, arg1: inst6000014D, type: type(symbolic_constant8)}
// CHECK:STDOUT:     inst6000014F:    {kind: ReturnExpr, arg0: inst6000014E, arg1: inst60000034}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst0000000E:    concrete_constant(inst0000000E)
// CHECK:STDOUT:       inst60000010:    concrete_constant(inst60000010)
// CHECK:STDOUT:       inst60000011:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000012:    symbolic_constant0
// CHECK:STDOUT:       inst60000013:    symbolic_constant0
// CHECK:STDOUT:       inst60000014:    symbolic_constant2
// CHECK:STDOUT:       inst60000015:    symbolic_constant1
// CHECK:STDOUT:       inst60000016:    symbolic_constant2
// CHECK:STDOUT:       inst60000017:    concrete_constant(inst60000017)
// CHECK:STDOUT:       inst60000018:    concrete_constant(inst60000018)
// CHECK:STDOUT:       inst60000019:    symbolic_constant2
// CHECK:STDOUT:       inst6000001A:    symbolic_constant4
// CHECK:STDOUT:       inst6000001B:    symbolic_constant3
// CHECK:STDOUT:       inst6000001C:    symbolic_constant4
// CHECK:STDOUT:       inst6000001E:    symbolic_constant5
// CHECK:STDOUT:       inst6000001F:    concrete_constant(inst6000001F)
// CHECK:STDOUT:       inst60000020:    symbolic_constant6
// CHECK:STDOUT:       inst60000021:    concrete_constant(inst60000021)
// CHECK:STDOUT:       inst60000022:    symbolic_constant2
// CHECK:STDOUT:       inst60000023:    symbolic_constant4
// CHECK:STDOUT:       inst60000024:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst60000026:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000028:    concrete_constant(inst60000028)
// CHECK:STDOUT:       inst60000029:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst6000002A:    symbolic_constant7
// CHECK:STDOUT:       inst6000002B:    symbolic_constant8
// CHECK:STDOUT:       inst6000002C:    symbolic_constant8
// CHECK:STDOUT:       inst6000002D:    symbolic_constant9
// CHECK:STDOUT:       inst6000002E:    concrete_constant(inst6000002E)
// CHECK:STDOUT:       inst6000002F:    symbolic_constant10
// CHECK:STDOUT:       inst60000030:    concrete_constant(inst60000030)
// CHECK:STDOUT:       inst60000032:    symbolic_constant4
// CHECK:STDOUT:       inst60000035:    concrete_constant(inst60000037)
// CHECK:STDOUT:       inst60000036:    concrete_constant(inst60000036)
// CHECK:STDOUT:       inst60000037:    concrete_constant(inst60000037)
// CHECK:STDOUT:       inst60000038:    symbolic_constant11
// CHECK:STDOUT:       inst60000039:    symbolic_constant13
// CHECK:STDOUT:       inst6000003A:    symbolic_constant12
// CHECK:STDOUT:       inst6000003B:    symbolic_constant13
// CHECK:STDOUT:       inst6000003C:    symbolic_constant15
// CHECK:STDOUT:       inst6000003D:    symbolic_constant14
// CHECK:STDOUT:       inst6000003E:    symbolic_constant15
// CHECK:STDOUT:       inst60000042:    symbolic_constant13
// CHECK:STDOUT:       inst60000044:    symbolic_constant15
// CHECK:STDOUT:       inst60000045:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000046:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000047:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000048:    symbolic_constant16
// CHECK:STDOUT:       inst60000049:    constant<none>
// CHECK:STDOUT:       inst6000004A:    concrete_constant(inst6000004E)
// CHECK:STDOUT:       inst6000004B:    constant<none>
// CHECK:STDOUT:       inst6000004C:    concrete_constant(inst6000004C)
// CHECK:STDOUT:       inst6000004D:    concrete_constant(inst60000051)
// CHECK:STDOUT:       inst6000004E:    concrete_constant(inst6000004E)
// CHECK:STDOUT:       inst6000004F:    concrete_constant(inst60000051)
// CHECK:STDOUT:       inst60000050:    concrete_constant(inst60000050)
// CHECK:STDOUT:       inst60000051:    concrete_constant(inst60000051)
// CHECK:STDOUT:       inst60000052:    symbolic_constant17
// CHECK:STDOUT:       inst60000053:    symbolic_constant18
// CHECK:STDOUT:       inst60000054:    concrete_constant(inst60000054)
// CHECK:STDOUT:       inst60000055:    concrete_constant(inst60000055)
// CHECK:STDOUT:       inst60000056:    concrete_constant(inst60000056)
// CHECK:STDOUT:       inst60000057:    concrete_constant(inst60000057)
// CHECK:STDOUT:       inst60000058:    symbolic_constant16
// CHECK:STDOUT:       inst60000059:    symbolic_constant22
// CHECK:STDOUT:       inst6000005A:    symbolic_constant23
// CHECK:STDOUT:       inst6000005B:    symbolic_constant24
// CHECK:STDOUT:       inst6000005C:    symbolic_constant251
// CHECK:STDOUT:       inst6000005D:    constant<none>
// CHECK:STDOUT:       inst6000005E:    concrete_constant(inst6000005E)
// CHECK:STDOUT:       inst6000005F:    symbolic_constant25
// CHECK:STDOUT:       inst60000060:    symbolic_constant26
// CHECK:STDOUT:       inst60000061:    symbolic_constant27
// CHECK:STDOUT:       inst60000062:    concrete_constant(inst60000062)
// CHECK:STDOUT:       inst60000063:    concrete_constant(inst60000063)
// CHECK:STDOUT:       inst60000064:    symbolic_constant29
// CHECK:STDOUT:       inst60000065:    symbolic_constant28
// CHECK:STDOUT:       inst60000066:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000067:    constant<none>
// CHECK:STDOUT:       inst60000068:    concrete_constant(inst60000068)
// CHECK:STDOUT:       inst60000069:    symbolic_constant32
// CHECK:STDOUT:       inst6000006A:    symbolic_constant34
// CHECK:STDOUT:       inst6000006B:    symbolic_constant35
// CHECK:STDOUT:       inst6000006C:    symbolic_constant36
// CHECK:STDOUT:       inst6000006D:    symbolic_constant37
// CHECK:STDOUT:       inst6000006E:    symbolic_constant40
// CHECK:STDOUT:       inst6000006F:    symbolic_constant38
// CHECK:STDOUT:       inst60000070:    symbolic_constant39
// CHECK:STDOUT:       inst60000071:    symbolic_constant42
// CHECK:STDOUT:       inst60000072:    concrete_constant(inst60000072)
// CHECK:STDOUT:       inst60000073:    concrete_constant(inst60000073)
// CHECK:STDOUT:       inst60000074:    concrete_constant(inst60000074)
// CHECK:STDOUT:       inst60000075:    concrete_constant(inst60000075)
// CHECK:STDOUT:       inst60000076:    symbolic_constant29
// CHECK:STDOUT:       inst60000077:    symbolic_constant45
// CHECK:STDOUT:       inst60000078:    symbolic_constant46
// CHECK:STDOUT:       inst60000079:    symbolic_constant50
// CHECK:STDOUT:       inst6000007A:    symbolic_constant51
// CHECK:STDOUT:       inst6000007B:    symbolic_constant52
// CHECK:STDOUT:       inst6000007C:    symbolic_constant53
// CHECK:STDOUT:       inst6000007D:    symbolic_constant54
// CHECK:STDOUT:       inst6000007E:    symbolic_constant56
// CHECK:STDOUT:       inst6000007F:    symbolic_constant58
// CHECK:STDOUT:       inst60000080:    symbolic_constant60
// CHECK:STDOUT:       inst60000081:    symbolic_constant62
// CHECK:STDOUT:       inst60000082:    symbolic_constant64
// CHECK:STDOUT:       inst60000083:    symbolic_constant66
// CHECK:STDOUT:       inst60000084:    symbolic_constant67
// CHECK:STDOUT:       inst60000085:    symbolic_constant68
// CHECK:STDOUT:       inst60000086:    symbolic_constant69
// CHECK:STDOUT:       inst60000087:    symbolic_constant70
// CHECK:STDOUT:       inst60000088:    symbolic_constant71
// CHECK:STDOUT:       inst60000089:    symbolic_constant72
// CHECK:STDOUT:       inst6000008A:    constant<none>
// CHECK:STDOUT:       inst6000008B:    concrete_constant(inst6000008B)
// CHECK:STDOUT:       inst6000008C:    concrete_constant(inst(BoolType))
// CHECK:STDOUT:       inst6000008D:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst6000008E:    constant<none>
// CHECK:STDOUT:       inst6000008F:    concrete_constant(inst6000008F)
// CHECK:STDOUT:       inst60000090:    concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       inst60000091:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000092:    constant<none>
// CHECK:STDOUT:       inst60000093:    concrete_constant(inst60000093)
// CHECK:STDOUT:       inst60000094:    concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       inst60000095:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000096:    constant<none>
// CHECK:STDOUT:       inst60000097:    concrete_constant(inst60000097)
// CHECK:STDOUT:       inst60000098:    concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       inst60000099:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst6000009A:    symbolic_constant249
// CHECK:STDOUT:       inst6000009B:    concrete_constant(inst6000009B)
// CHECK:STDOUT:       inst6000009C:    concrete_constant(inst6000009C)
// CHECK:STDOUT:       inst6000009D:    symbolic_constant74
// CHECK:STDOUT:       inst6000009E:    symbolic_constant73
// CHECK:STDOUT:       inst6000009F:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst600000A0:    constant<none>
// CHECK:STDOUT:       inst600000A1:    concrete_constant(inst600000A1)
// CHECK:STDOUT:       inst600000A2:    symbolic_constant76
// CHECK:STDOUT:       inst600000A3:    symbolic_constant78
// CHECK:STDOUT:       inst600000A4:    symbolic_constant79
// CHECK:STDOUT:       inst600000A5:    symbolic_constant80
// CHECK:STDOUT:       inst600000A6:    symbolic_constant83
// CHECK:STDOUT:       inst600000A7:    symbolic_constant81
// CHECK:STDOUT:       inst600000A8:    symbolic_constant82
// CHECK:STDOUT:       inst600000A9:    concrete_constant(inst600000A9)
// CHECK:STDOUT:       inst600000AA:    concrete_constant(inst600000AA)
// CHECK:STDOUT:       inst600000AB:    concrete_constant(inst600000AB)
// CHECK:STDOUT:       inst600000AC:    concrete_constant(inst600000AC)
// CHECK:STDOUT:       inst600000AD:    symbolic_constant74
// CHECK:STDOUT:       inst600000AE:    symbolic_constant88
// CHECK:STDOUT:       inst600000AF:    symbolic_constant89
// CHECK:STDOUT:       inst600000B0:    symbolic_constant90
// CHECK:STDOUT:       inst600000B1:    symbolic_constant93
// CHECK:STDOUT:       inst600000B2:    symbolic_constant94
// CHECK:STDOUT:       inst600000B3:    symbolic_constant95
// CHECK:STDOUT:       inst600000B4:    constant<none>
// CHECK:STDOUT:       inst600000B5:    concrete_constant(inst600000B5)
// CHECK:STDOUT:       inst600000B6:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst600000B7:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst600000B8:    constant<none>
// CHECK:STDOUT:       inst600000B9:    concrete_constant(inst600000B9)
// CHECK:STDOUT:       inst600000BA:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst600000BB:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst600000BC:    constant<none>
// CHECK:STDOUT:       inst600000BD:    concrete_constant(inst600000BD)
// CHECK:STDOUT:       inst600000BE:    symbolic_constant96
// CHECK:STDOUT:       inst600000BF:    symbolic_constant97
// CHECK:STDOUT:       inst600000C0:    symbolic_constant98
// CHECK:STDOUT:       inst600000C1:    concrete_constant(inst600000C1)
// CHECK:STDOUT:       inst600000C2:    concrete_constant(inst600000C2)
// CHECK:STDOUT:       inst600000C3:    symbolic_constant100
// CHECK:STDOUT:       inst600000C4:    symbolic_constant101
// CHECK:STDOUT:       inst600000C5:    symbolic_constant99
// CHECK:STDOUT:       inst600000C6:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst600000C7:    constant<none>
// CHECK:STDOUT:       inst600000C8:    concrete_constant(inst600000C8)
// CHECK:STDOUT:       inst600000C9:    symbolic_constant106
// CHECK:STDOUT:       inst600000CA:    symbolic_constant108
// CHECK:STDOUT:       inst600000CB:    symbolic_constant109
// CHECK:STDOUT:       inst600000CC:    symbolic_constant110
// CHECK:STDOUT:       inst600000CD:    symbolic_constant111
// CHECK:STDOUT:       inst600000CE:    symbolic_constant112
// CHECK:STDOUT:       inst600000CF:    symbolic_constant113
// CHECK:STDOUT:       inst600000D0:    symbolic_constant116
// CHECK:STDOUT:       inst600000D1:    symbolic_constant114
// CHECK:STDOUT:       inst600000D2:    symbolic_constant115
// CHECK:STDOUT:       inst600000D3:    symbolic_constant118
// CHECK:STDOUT:       inst600000D4:    concrete_constant(inst600000D4)
// CHECK:STDOUT:       inst600000D5:    concrete_constant(inst600000D5)
// CHECK:STDOUT:       inst600000D6:    concrete_constant(inst600000D6)
// CHECK:STDOUT:       inst600000D7:    concrete_constant(inst600000D7)
// CHECK:STDOUT:       inst600000D8:    symbolic_constant100
// CHECK:STDOUT:       inst600000D9:    symbolic_constant101
// CHECK:STDOUT:       inst600000DA:    symbolic_constant121
// CHECK:STDOUT:       inst600000DB:    symbolic_constant122
// CHECK:STDOUT:       inst600000DC:    symbolic_constant128
// CHECK:STDOUT:       inst600000DD:    symbolic_constant129
// CHECK:STDOUT:       inst600000DE:    symbolic_constant130
// CHECK:STDOUT:       inst600000DF:    symbolic_constant131
// CHECK:STDOUT:       inst600000E0:    symbolic_constant132
// CHECK:STDOUT:       inst600000E1:    symbolic_constant133
// CHECK:STDOUT:       inst600000E2:    symbolic_constant134
// CHECK:STDOUT:       inst600000E3:    symbolic_constant141
// CHECK:STDOUT:       inst600000E4:    symbolic_constant143
// CHECK:STDOUT:       inst600000E5:    symbolic_constant145
// CHECK:STDOUT:       inst600000E6:    symbolic_constant147
// CHECK:STDOUT:       inst600000E7:    symbolic_constant149
// CHECK:STDOUT:       inst600000E8:    symbolic_constant151
// CHECK:STDOUT:       inst600000E9:    symbolic_constant152
// CHECK:STDOUT:       inst600000EA:    symbolic_constant153
// CHECK:STDOUT:       inst600000EB:    symbolic_constant154
// CHECK:STDOUT:       inst600000EC:    symbolic_constant155
// CHECK:STDOUT:       inst600000ED:    symbolic_constant156
// CHECK:STDOUT:       inst600000EE:    symbolic_constant157
// CHECK:STDOUT:       inst600000EF:    symbolic_constant158
// CHECK:STDOUT:       inst600000F0:    symbolic_constant159
// CHECK:STDOUT:       inst600000F1:    symbolic_constant160
// CHECK:STDOUT:       inst600000F2:    symbolic_constant161
// CHECK:STDOUT:       inst600000F3:    symbolic_constant162
// CHECK:STDOUT:       inst600000F4:    constant<none>
// CHECK:STDOUT:       inst600000F5:    concrete_constant(inst600000F5)
// CHECK:STDOUT:       inst600000F6:    symbolic_constant163
// CHECK:STDOUT:       inst600000F7:    symbolic_constant164
// CHECK:STDOUT:       inst600000F8:    symbolic_constant165
// CHECK:STDOUT:       inst600000F9:    concrete_constant(inst600000F9)
// CHECK:STDOUT:       inst600000FA:    concrete_constant(inst600000FA)
// CHECK:STDOUT:       inst600000FB:    concrete_constant(inst600000FB)
// CHECK:STDOUT:       inst600000FC:    symbolic_constant167
// CHECK:STDOUT:       inst600000FD:    symbolic_constant168
// CHECK:STDOUT:       inst600000FE:    symbolic_constant169
// CHECK:STDOUT:       inst600000FF:    symbolic_constant166
// CHECK:STDOUT:       inst60000100:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000101:    constant<none>
// CHECK:STDOUT:       inst60000102:    concrete_constant(inst60000102)
// CHECK:STDOUT:       inst60000103:    symbolic_constant176
// CHECK:STDOUT:       inst60000104:    symbolic_constant178
// CHECK:STDOUT:       inst60000105:    symbolic_constant179
// CHECK:STDOUT:       inst60000106:    symbolic_constant180
// CHECK:STDOUT:       inst60000107:    symbolic_constant181
// CHECK:STDOUT:       inst60000108:    symbolic_constant182
// CHECK:STDOUT:       inst60000109:    symbolic_constant183
// CHECK:STDOUT:       inst6000010A:    symbolic_constant184
// CHECK:STDOUT:       inst6000010B:    symbolic_constant185
// CHECK:STDOUT:       inst6000010C:    symbolic_constant188
// CHECK:STDOUT:       inst6000010D:    symbolic_constant186
// CHECK:STDOUT:       inst6000010E:    symbolic_constant187
// CHECK:STDOUT:       inst6000010F:    symbolic_constant190
// CHECK:STDOUT:       inst60000110:    concrete_constant(inst60000110)
// CHECK:STDOUT:       inst60000111:    concrete_constant(inst60000111)
// CHECK:STDOUT:       inst60000112:    concrete_constant(inst60000112)
// CHECK:STDOUT:       inst60000113:    concrete_constant(inst60000113)
// CHECK:STDOUT:       inst60000114:    symbolic_constant167
// CHECK:STDOUT:       inst60000115:    symbolic_constant168
// CHECK:STDOUT:       inst60000116:    symbolic_constant169
// CHECK:STDOUT:       inst60000117:    symbolic_constant193
// CHECK:STDOUT:       inst60000118:    symbolic_constant194
// CHECK:STDOUT:       inst60000119:    symbolic_constant202
// CHECK:STDOUT:       inst6000011A:    symbolic_constant203
// CHECK:STDOUT:       inst6000011B:    symbolic_constant204
// CHECK:STDOUT:       inst6000011C:    symbolic_constant205
// CHECK:STDOUT:       inst6000011D:    symbolic_constant206
// CHECK:STDOUT:       inst6000011E:    symbolic_constant207
// CHECK:STDOUT:       inst6000011F:    symbolic_constant208
// CHECK:STDOUT:       inst60000120:    symbolic_constant209
// CHECK:STDOUT:       inst60000121:    symbolic_constant210
// CHECK:STDOUT:       inst60000122:    symbolic_constant222
// CHECK:STDOUT:       inst60000123:    symbolic_constant224
// CHECK:STDOUT:       inst60000124:    symbolic_constant226
// CHECK:STDOUT:       inst60000125:    symbolic_constant228
// CHECK:STDOUT:       inst60000126:    symbolic_constant230
// CHECK:STDOUT:       inst60000127:    symbolic_constant232
// CHECK:STDOUT:       inst60000128:    symbolic_constant233
// CHECK:STDOUT:       inst60000129:    symbolic_constant234
// CHECK:STDOUT:       inst6000012A:    symbolic_constant235
// CHECK:STDOUT:       inst6000012B:    symbolic_constant236
// CHECK:STDOUT:       inst6000012C:    symbolic_constant237
// CHECK:STDOUT:       inst6000012D:    symbolic_constant238
// CHECK:STDOUT:       inst6000012E:    symbolic_constant239
// CHECK:STDOUT:       inst6000012F:    symbolic_constant240
// CHECK:STDOUT:       inst60000130:    symbolic_constant241
// CHECK:STDOUT:       inst60000131:    symbolic_constant242
// CHECK:STDOUT:       inst60000132:    symbolic_constant243
// CHECK:STDOUT:       inst60000133:    symbolic_constant244
// CHECK:STDOUT:       inst60000134:    symbolic_constant245
// CHECK:STDOUT:       inst60000135:    symbolic_constant246
// CHECK:STDOUT:       inst60000136:    symbolic_constant247
// CHECK:STDOUT:       inst60000137:    symbolic_constant248
// CHECK:STDOUT:       inst60000138:    symbolic_constant250
// CHECK:STDOUT:       inst60000139:    symbolic_constant251
// CHECK:STDOUT:       inst6000013A:    symbolic_constant252
// CHECK:STDOUT:       inst6000013B:    symbolic_constant253
// CHECK:STDOUT:       inst6000013C:    symbolic_constant257
// CHECK:STDOUT:       inst6000013D:    symbolic_constant254
// CHECK:STDOUT:       inst6000013E:    symbolic_constant255
// CHECK:STDOUT:       inst6000013F:    symbolic_constant256
// CHECK:STDOUT:       inst60000140:    symbolic_constant257
// CHECK:STDOUT:       inst60000142:    symbolic_constant259
// CHECK:STDOUT:       inst60000143:    symbolic_constant258
// CHECK:STDOUT:       inst60000144:    symbolic_constant259
// CHECK:STDOUT:       inst60000146:    symbolic_constant15
// CHECK:STDOUT:       inst6000014A:    concrete_constant(inst6000014B)
// CHECK:STDOUT:       inst6000014B:    concrete_constant(inst6000014B)
// CHECK:STDOUT:       inst6000014C:    concrete_constant(inst6000014B)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: inst60000013, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant1: {inst: inst60000015, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: inst60000015, generic: generic0, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: inst6000001B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: inst6000001B, generic: generic0, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: inst6000001E, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: inst6000001E, generic: generic0, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: inst6000002A, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: inst6000002A, generic: generic0, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: inst6000002D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant10: {inst: inst6000002D, generic: generic0, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant11: {inst: inst60000038, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant12: {inst: inst6000003A, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant13: {inst: inst6000003A, generic: generic0, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant14: {inst: inst6000003D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant15: {inst: inst6000003D, generic: generic0, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant16: {inst: inst60000048, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant17: {inst: inst60000052, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant18: {inst: inst60000053, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant19: {inst: inst60000053, generic: generic1, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant20: {inst: inst60000048, generic: generic1, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant21: {inst: inst60000052, generic: generic1, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant22: {inst: inst60000048, generic: generic1, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant23: {inst: inst60000052, generic: generic1, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant24: {inst: inst60000053, generic: generic1, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant25: {inst: inst6000005F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant26: {inst: inst60000060, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant27: {inst: inst60000061, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant28: {inst: inst60000061, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant29: {inst: inst6000005F, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant30: {inst: inst6000005F, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant31: {inst: inst60000060, generic: generic2, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant32: {inst: inst60000069, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant33: {inst: inst60000069, generic: generic2, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant34: {inst: inst6000005F, generic: generic2, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant35: {inst: inst60000060, generic: generic2, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant36: {inst: inst60000061, generic: generic2, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant37: {inst: inst60000069, generic: generic2, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant38: {inst: inst6000006F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant39: {inst: inst60000070, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant40: {inst: inst60000070, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant41: {inst: inst6000006F, generic: generic2, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant42: {inst: inst60000071, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant43: {inst: inst60000071, generic: generic3, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant44: {inst: inst60000070, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant45: {inst: inst6000006F, generic: generic2, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant46: {inst: inst60000070, generic: generic2, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant47: {inst: inst6000005F, generic: generic3, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant48: {inst: inst60000060, generic: generic3, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant49: {inst: inst60000061, generic: generic3, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant50: {inst: inst6000005F, generic: generic3, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant51: {inst: inst60000060, generic: generic3, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant52: {inst: inst60000061, generic: generic3, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant53: {inst: inst60000071, generic: generic3, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant54: {inst: inst6000007D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant55: {inst: inst6000007D, generic: generic3, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant56: {inst: inst6000007E, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant57: {inst: inst6000007E, generic: generic3, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant58: {inst: inst6000007F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant59: {inst: inst6000007F, generic: generic3, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60: {inst: inst60000080, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant61: {inst: inst60000080, generic: generic3, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant62: {inst: inst60000081, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant63: {inst: inst60000081, generic: generic3, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant64: {inst: inst60000082, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant65: {inst: inst60000082, generic: generic3, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant66: {inst: inst6000007D, generic: generic3, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant67: {inst: inst6000007E, generic: generic3, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant68: {inst: inst6000007F, generic: generic3, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant69: {inst: inst60000080, generic: generic3, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant70: {inst: inst60000081, generic: generic3, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant71: {inst: inst60000082, generic: generic3, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant72: {inst: inst60000089, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant73: {inst: inst6000001B, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant74: {inst: inst60000015, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant75: {inst: inst60000015, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant76: {inst: inst600000A2, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant77: {inst: inst600000A2, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant78: {inst: inst60000015, generic: generic4, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant79: {inst: inst6000001B, generic: generic4, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant80: {inst: inst600000A2, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant81: {inst: inst600000A7, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant82: {inst: inst600000A8, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant83: {inst: inst600000A8, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant84: {inst: inst600000A7, generic: generic4, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant85: {inst: inst6000001E, generic: generic5, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant86: {inst: inst600000A8, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant87: {inst: inst6000003D, generic: generic4, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant88: {inst: inst600000A7, generic: generic4, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant89: {inst: inst600000A8, generic: generic4, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant90: {inst: inst6000003D, generic: generic4, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant91: {inst: inst60000015, generic: generic5, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant92: {inst: inst6000001B, generic: generic5, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant93: {inst: inst60000015, generic: generic5, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant94: {inst: inst6000001B, generic: generic5, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant95: {inst: inst6000001E, generic: generic5, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant96: {inst: inst600000BE, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant97: {inst: inst600000BF, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant98: {inst: inst600000C0, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant99: {inst: inst600000C0, generic: generic6, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant100: {inst: inst6000005F, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant101: {inst: inst600000BE, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant102: {inst: inst6000005F, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant103: {inst: inst600000BE, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant104: {inst: inst60000060, generic: generic6, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant105: {inst: inst600000BF, generic: generic6, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant106: {inst: inst600000C9, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant107: {inst: inst600000C9, generic: generic6, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant108: {inst: inst6000005F, generic: generic6, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant109: {inst: inst600000BE, generic: generic6, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant110: {inst: inst60000060, generic: generic6, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant111: {inst: inst600000BF, generic: generic6, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant112: {inst: inst600000C0, generic: generic6, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant113: {inst: inst600000C9, generic: generic6, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant114: {inst: inst600000D1, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant115: {inst: inst600000D2, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant116: {inst: inst600000D2, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant117: {inst: inst600000D1, generic: generic6, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant118: {inst: inst600000D3, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant119: {inst: inst600000D3, generic: generic7, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant120: {inst: inst600000D2, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant121: {inst: inst600000D1, generic: generic6, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant122: {inst: inst600000D2, generic: generic6, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant123: {inst: inst6000005F, generic: generic7, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant124: {inst: inst60000060, generic: generic7, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant125: {inst: inst600000BE, generic: generic7, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant126: {inst: inst600000BF, generic: generic7, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant127: {inst: inst600000C0, generic: generic7, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant128: {inst: inst6000005F, generic: generic7, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant129: {inst: inst60000060, generic: generic7, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant130: {inst: inst600000BE, generic: generic7, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant131: {inst: inst600000BF, generic: generic7, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant132: {inst: inst600000C0, generic: generic7, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant133: {inst: inst600000D3, generic: generic7, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant134: {inst: inst600000E2, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant135: {inst: inst600000E2, generic: generic7, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant136: {inst: inst6000007E, generic: generic7, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant137: {inst: inst6000007F, generic: generic7, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant138: {inst: inst60000080, generic: generic7, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant139: {inst: inst60000081, generic: generic7, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant140: {inst: inst60000082, generic: generic7, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant141: {inst: inst600000E3, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant142: {inst: inst600000E3, generic: generic7, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant143: {inst: inst600000E4, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant144: {inst: inst600000E4, generic: generic7, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant145: {inst: inst600000E5, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant146: {inst: inst600000E5, generic: generic7, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant147: {inst: inst600000E6, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant148: {inst: inst600000E6, generic: generic7, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant149: {inst: inst600000E7, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant150: {inst: inst600000E7, generic: generic7, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant151: {inst: inst600000E2, generic: generic7, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant152: {inst: inst6000007E, generic: generic7, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant153: {inst: inst6000007F, generic: generic7, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant154: {inst: inst60000080, generic: generic7, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant155: {inst: inst60000081, generic: generic7, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant156: {inst: inst60000082, generic: generic7, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant157: {inst: inst600000E3, generic: generic7, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant158: {inst: inst600000E4, generic: generic7, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant159: {inst: inst600000E5, generic: generic7, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant160: {inst: inst600000E6, generic: generic7, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant161: {inst: inst600000E7, generic: generic7, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant162: {inst: inst600000F3, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant163: {inst: inst600000F6, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant164: {inst: inst600000F7, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant165: {inst: inst600000F8, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant166: {inst: inst600000F8, generic: generic8, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant167: {inst: inst6000005F, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant168: {inst: inst600000BE, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant169: {inst: inst600000F6, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant170: {inst: inst6000005F, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant171: {inst: inst600000BE, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant172: {inst: inst600000F6, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant173: {inst: inst60000060, generic: generic8, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant174: {inst: inst600000BF, generic: generic8, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant175: {inst: inst600000F7, generic: generic8, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant176: {inst: inst60000103, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant177: {inst: inst60000103, generic: generic8, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant178: {inst: inst6000005F, generic: generic8, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant179: {inst: inst600000BE, generic: generic8, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant180: {inst: inst600000F6, generic: generic8, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant181: {inst: inst60000060, generic: generic8, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant182: {inst: inst600000BF, generic: generic8, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant183: {inst: inst600000F7, generic: generic8, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant184: {inst: inst600000F8, generic: generic8, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant185: {inst: inst60000103, generic: generic8, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant186: {inst: inst6000010D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant187: {inst: inst6000010E, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant188: {inst: inst6000010E, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant189: {inst: inst6000010D, generic: generic8, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant190: {inst: inst6000010F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant191: {inst: inst6000010F, generic: generic9, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant192: {inst: inst6000010E, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant193: {inst: inst6000010D, generic: generic8, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant194: {inst: inst6000010E, generic: generic8, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant195: {inst: inst6000005F, generic: generic9, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant196: {inst: inst60000060, generic: generic9, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant197: {inst: inst600000BE, generic: generic9, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant198: {inst: inst600000BF, generic: generic9, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant199: {inst: inst600000F6, generic: generic9, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant200: {inst: inst600000F7, generic: generic9, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant201: {inst: inst600000F8, generic: generic9, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant202: {inst: inst6000005F, generic: generic9, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant203: {inst: inst60000060, generic: generic9, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant204: {inst: inst600000BE, generic: generic9, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant205: {inst: inst600000BF, generic: generic9, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant206: {inst: inst600000F6, generic: generic9, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant207: {inst: inst600000F7, generic: generic9, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant208: {inst: inst600000F8, generic: generic9, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant209: {inst: inst6000010F, generic: generic9, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant210: {inst: inst60000121, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant211: {inst: inst60000121, generic: generic9, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant212: {inst: inst6000007E, generic: generic9, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant213: {inst: inst6000007F, generic: generic9, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant214: {inst: inst60000080, generic: generic9, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant215: {inst: inst60000081, generic: generic9, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant216: {inst: inst60000082, generic: generic9, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant217: {inst: inst600000E3, generic: generic9, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant218: {inst: inst600000E4, generic: generic9, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant219: {inst: inst600000E5, generic: generic9, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant220: {inst: inst600000E6, generic: generic9, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant221: {inst: inst600000E7, generic: generic9, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant222: {inst: inst60000122, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant223: {inst: inst60000122, generic: generic9, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant224: {inst: inst60000123, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant225: {inst: inst60000123, generic: generic9, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant226: {inst: inst60000124, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant227: {inst: inst60000124, generic: generic9, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant228: {inst: inst60000125, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant229: {inst: inst60000125, generic: generic9, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant230: {inst: inst60000126, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant231: {inst: inst60000126, generic: generic9, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant232: {inst: inst60000121, generic: generic9, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant233: {inst: inst6000007E, generic: generic9, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant234: {inst: inst6000007F, generic: generic9, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant235: {inst: inst60000080, generic: generic9, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant236: {inst: inst60000081, generic: generic9, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant237: {inst: inst60000082, generic: generic9, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant238: {inst: inst600000E3, generic: generic9, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant239: {inst: inst600000E4, generic: generic9, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant240: {inst: inst600000E5, generic: generic9, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant241: {inst: inst600000E6, generic: generic9, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant242: {inst: inst600000E7, generic: generic9, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant243: {inst: inst60000122, generic: generic9, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant244: {inst: inst60000123, generic: generic9, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant245: {inst: inst60000124, generic: generic9, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant246: {inst: inst60000125, generic: generic9, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant247: {inst: inst60000126, generic: generic9, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant248: {inst: inst60000137, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant249: {inst: inst600000A2, generic: generic4, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant250: {inst: inst60000138, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant251: {inst: inst60000138, generic: generic0, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant252: {inst: inst6000013A, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant253: {inst: inst6000013B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant254: {inst: inst6000013D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant255: {inst: inst6000013A, generic: generic0, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant256: {inst: inst6000013B, generic: generic0, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant257: {inst: inst6000013D, generic: generic0, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant258: {inst: inst60000143, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant259: {inst: inst60000143, generic: generic0, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst60000035
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst60000010
// CHECK:STDOUT:       1:               inst60000045
// CHECK:STDOUT:       2:               inst60000046
// CHECK:STDOUT:       3:               inst60000049
// CHECK:STDOUT:       4:               inst6000004A
// CHECK:STDOUT:       5:               inst6000004B
// CHECK:STDOUT:       6:               inst6000004D
// CHECK:STDOUT:       7:               inst6000004F
// CHECK:STDOUT:       8:               inst60000054
// CHECK:STDOUT:       9:               inst60000055
// CHECK:STDOUT:       10:              inst60000056
// CHECK:STDOUT:       11:              inst60000057
// CHECK:STDOUT:       12:              inst60000058
// CHECK:STDOUT:       13:              inst6000005D
// CHECK:STDOUT:       14:              inst6000005E
// CHECK:STDOUT:       15:              inst60000063
// CHECK:STDOUT:       16:              inst60000064
// CHECK:STDOUT:       17:              inst60000065
// CHECK:STDOUT:       18:              inst60000066
// CHECK:STDOUT:       19:              inst60000067
// CHECK:STDOUT:       20:              inst60000068
// CHECK:STDOUT:       21:              inst6000006E
// CHECK:STDOUT:       22:              inst60000072
// CHECK:STDOUT:       23:              inst60000073
// CHECK:STDOUT:       24:              inst60000074
// CHECK:STDOUT:       25:              inst60000075
// CHECK:STDOUT:       26:              inst60000076
// CHECK:STDOUT:       27:              inst6000008A
// CHECK:STDOUT:       28:              inst6000008B
// CHECK:STDOUT:       29:              inst6000008C
// CHECK:STDOUT:       30:              inst6000008D
// CHECK:STDOUT:       31:              inst6000008E
// CHECK:STDOUT:       32:              inst6000008F
// CHECK:STDOUT:       33:              inst60000090
// CHECK:STDOUT:       34:              inst60000091
// CHECK:STDOUT:       35:              inst60000092
// CHECK:STDOUT:       36:              inst60000093
// CHECK:STDOUT:       37:              inst60000094
// CHECK:STDOUT:       38:              inst60000095
// CHECK:STDOUT:       39:              inst60000096
// CHECK:STDOUT:       40:              inst60000097
// CHECK:STDOUT:       41:              inst60000098
// CHECK:STDOUT:       42:              inst60000099
// CHECK:STDOUT:       43:              inst6000009A
// CHECK:STDOUT:       44:              inst6000009B
// CHECK:STDOUT:       45:              inst6000009C
// CHECK:STDOUT:       46:              inst6000009D
// CHECK:STDOUT:       47:              inst6000009E
// CHECK:STDOUT:       48:              inst6000009F
// CHECK:STDOUT:       49:              inst600000A0
// CHECK:STDOUT:       50:              inst600000A1
// CHECK:STDOUT:       51:              inst600000A6
// CHECK:STDOUT:       52:              inst600000A9
// CHECK:STDOUT:       53:              inst600000AA
// CHECK:STDOUT:       54:              inst600000AB
// CHECK:STDOUT:       55:              inst600000AC
// CHECK:STDOUT:       56:              inst600000AD
// CHECK:STDOUT:       57:              inst600000B4
// CHECK:STDOUT:       58:              inst600000B5
// CHECK:STDOUT:       59:              inst600000B6
// CHECK:STDOUT:       60:              inst600000B7
// CHECK:STDOUT:       61:              inst600000B8
// CHECK:STDOUT:       62:              inst600000B9
// CHECK:STDOUT:       63:              inst600000BA
// CHECK:STDOUT:       64:              inst600000BB
// CHECK:STDOUT:       65:              inst600000BC
// CHECK:STDOUT:       66:              inst600000BD
// CHECK:STDOUT:       67:              inst600000C1
// CHECK:STDOUT:       68:              inst600000C2
// CHECK:STDOUT:       69:              inst600000C3
// CHECK:STDOUT:       70:              inst600000C4
// CHECK:STDOUT:       71:              inst600000C5
// CHECK:STDOUT:       72:              inst600000C6
// CHECK:STDOUT:       73:              inst600000C7
// CHECK:STDOUT:       74:              inst600000C8
// CHECK:STDOUT:       75:              inst600000D0
// CHECK:STDOUT:       76:              inst600000D4
// CHECK:STDOUT:       77:              inst600000D5
// CHECK:STDOUT:       78:              inst600000D6
// CHECK:STDOUT:       79:              inst600000D7
// CHECK:STDOUT:       80:              inst600000D8
// CHECK:STDOUT:       81:              inst600000D9
// CHECK:STDOUT:       82:              inst600000F4
// CHECK:STDOUT:       83:              inst600000F5
// CHECK:STDOUT:       84:              inst600000F9
// CHECK:STDOUT:       85:              inst600000FA
// CHECK:STDOUT:       86:              inst600000FB
// CHECK:STDOUT:       87:              inst600000FC
// CHECK:STDOUT:       88:              inst600000FD
// CHECK:STDOUT:       89:              inst600000FE
// CHECK:STDOUT:       90:              inst600000FF
// CHECK:STDOUT:       91:              inst60000100
// CHECK:STDOUT:       92:              inst60000101
// CHECK:STDOUT:       93:              inst60000102
// CHECK:STDOUT:       94:              inst6000010C
// CHECK:STDOUT:       95:              inst60000110
// CHECK:STDOUT:       96:              inst60000111
// CHECK:STDOUT:       97:              inst60000112
// CHECK:STDOUT:       98:              inst60000113
// CHECK:STDOUT:       99:              inst60000114
// CHECK:STDOUT:       100:             inst60000115
// CHECK:STDOUT:       101:             inst60000116
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block4:
// CHECK:STDOUT:       0:               inst60000012
// CHECK:STDOUT:     inst_block5:
// CHECK:STDOUT:       0:               inst60000018
// CHECK:STDOUT:     inst_block6:
// CHECK:STDOUT:       0:               inst60000019
// CHECK:STDOUT:       1:               inst6000001A
// CHECK:STDOUT:     inst_block7:
// CHECK:STDOUT:       0:               inst60000021
// CHECK:STDOUT:     inst_block8:
// CHECK:STDOUT:       0:               inst60000023
// CHECK:STDOUT:       1:               inst60000025
// CHECK:STDOUT:     inst_block9:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst60000024
// CHECK:STDOUT:     inst_block10:
// CHECK:STDOUT:       0:               inst60000023
// CHECK:STDOUT:       1:               inst60000029
// CHECK:STDOUT:     inst_block11:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:       1:               inst60000024
// CHECK:STDOUT:     inst_block12:
// CHECK:STDOUT:       0:               inst6000001C
// CHECK:STDOUT:       1:               inst60000024
// CHECK:STDOUT:     inst_block13:
// CHECK:STDOUT:       0:               inst60000031
// CHECK:STDOUT:       1:               inst60000033
// CHECK:STDOUT:     inst_block14:
// CHECK:STDOUT:       0:               inst60000018
// CHECK:STDOUT:       1:               inst6000001F
// CHECK:STDOUT:       2:               inst60000021
// CHECK:STDOUT:       3:               inst6000002E
// CHECK:STDOUT:       4:               inst60000030
// CHECK:STDOUT:     inst_block15:
// CHECK:STDOUT:       0:               inst60000022
// CHECK:STDOUT:       1:               inst60000023
// CHECK:STDOUT:       2:               inst60000025
// CHECK:STDOUT:       3:               inst60000027
// CHECK:STDOUT:       4:               inst60000029
// CHECK:STDOUT:       5:               inst6000002B
// CHECK:STDOUT:       6:               inst60000012
// CHECK:STDOUT:       7:               inst60000014
// CHECK:STDOUT:       8:               inst60000031
// CHECK:STDOUT:       9:               inst60000032
// CHECK:STDOUT:       10:              inst6000001D
// CHECK:STDOUT:       11:              inst60000033
// CHECK:STDOUT:       12:              inst60000034
// CHECK:STDOUT:     inst_block16:
// CHECK:STDOUT:       0:               inst60000014
// CHECK:STDOUT:     inst_block17:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:     inst_block18:
// CHECK:STDOUT:       0:               inst60000016
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:       2:               inst60000020
// CHECK:STDOUT:       3:               inst6000002C
// CHECK:STDOUT:       4:               inst6000002F
// CHECK:STDOUT:     inst_block19:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:       3:               inst6000002A
// CHECK:STDOUT:       4:               inst6000002D
// CHECK:STDOUT:     inst_block20:
// CHECK:STDOUT:       0:               inst6000003F
// CHECK:STDOUT:       1:               inst60000040
// CHECK:STDOUT:       2:               inst60000041
// CHECK:STDOUT:       3:               inst6000013C
// CHECK:STDOUT:       4:               inst60000141
// CHECK:STDOUT:       5:               inst60000142
// CHECK:STDOUT:       6:               inst60000145
// CHECK:STDOUT:       7:               inst60000147
// CHECK:STDOUT:       8:               inst60000043
// CHECK:STDOUT:       9:               inst60000148
// CHECK:STDOUT:       10:              inst60000149
// CHECK:STDOUT:       11:              inst6000014A
// CHECK:STDOUT:       12:              inst6000014C
// CHECK:STDOUT:       13:              inst6000014D
// CHECK:STDOUT:       14:              inst6000014E
// CHECK:STDOUT:       15:              inst6000014F
// CHECK:STDOUT:     inst_block21:
// CHECK:STDOUT:       0:               inst6000003F
// CHECK:STDOUT:       1:               inst60000040
// CHECK:STDOUT:     inst_block22:
// CHECK:STDOUT:       0:               inst6000004B
// CHECK:STDOUT:     inst_block23:
// CHECK:STDOUT:       0:               inst60000057
// CHECK:STDOUT:     inst_block24:
// CHECK:STDOUT:       0:               inst60000058
// CHECK:STDOUT:     inst_block25:
// CHECK:STDOUT:       0:               inst60000059
// CHECK:STDOUT:       1:               inst6000005A
// CHECK:STDOUT:       2:               inst6000005B
// CHECK:STDOUT:     inst_block26:
// CHECK:STDOUT:       0:               inst60000048
// CHECK:STDOUT:     inst_block27:
// CHECK:STDOUT:       0:               inst60000048
// CHECK:STDOUT:       1:               inst60000052
// CHECK:STDOUT:       2:               inst60000053
// CHECK:STDOUT:     inst_block28:
// CHECK:STDOUT:       0:               inst60000063
// CHECK:STDOUT:     inst_block29:
// CHECK:STDOUT:       0:               inst60000064
// CHECK:STDOUT:     inst_block30:
// CHECK:STDOUT:       0:               inst60000067
// CHECK:STDOUT:     inst_block31:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:     inst_block32:
// CHECK:STDOUT:       0:               inst6000006A
// CHECK:STDOUT:     inst_block33:
// CHECK:STDOUT:       0:               inst6000006A
// CHECK:STDOUT:       1:               inst6000006B
// CHECK:STDOUT:       2:               inst6000006C
// CHECK:STDOUT:       3:               inst6000006D
// CHECK:STDOUT:     inst_block34:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst60000061
// CHECK:STDOUT:       3:               inst60000069
// CHECK:STDOUT:     inst_block35:
// CHECK:STDOUT:       0:               inst60000075
// CHECK:STDOUT:     inst_block36:
// CHECK:STDOUT:       0:               inst60000076
// CHECK:STDOUT:     inst_block37:
// CHECK:STDOUT:       0:               inst6000006A
// CHECK:STDOUT:     inst_block38:
// CHECK:STDOUT:       0:               inst60000077
// CHECK:STDOUT:       1:               inst60000078
// CHECK:STDOUT:     inst_block39:
// CHECK:STDOUT:       0:               inst60000079
// CHECK:STDOUT:       1:               inst6000007A
// CHECK:STDOUT:       2:               inst6000007B
// CHECK:STDOUT:       3:               inst6000007C
// CHECK:STDOUT:     inst_block40:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst60000061
// CHECK:STDOUT:       3:               inst60000071
// CHECK:STDOUT:     inst_block41:
// CHECK:STDOUT:       0:               inst60000079
// CHECK:STDOUT:     inst_block42:
// CHECK:STDOUT:       0:               inst60000083
// CHECK:STDOUT:       1:               inst60000084
// CHECK:STDOUT:       2:               inst60000085
// CHECK:STDOUT:       3:               inst60000086
// CHECK:STDOUT:       4:               inst60000087
// CHECK:STDOUT:       5:               inst60000088
// CHECK:STDOUT:     inst_block43:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst60000089
// CHECK:STDOUT:     inst_block44:
// CHECK:STDOUT:       0:               inst6000006F
// CHECK:STDOUT:       1:               inst60000070
// CHECK:STDOUT:     inst_block45:
// CHECK:STDOUT:       0:               inst6000009C
// CHECK:STDOUT:     inst_block46:
// CHECK:STDOUT:       0:               inst6000009D
// CHECK:STDOUT:     inst_block47:
// CHECK:STDOUT:       0:               inst600000A0
// CHECK:STDOUT:     inst_block48:
// CHECK:STDOUT:       0:               inst600000A3
// CHECK:STDOUT:     inst_block49:
// CHECK:STDOUT:       0:               inst600000A3
// CHECK:STDOUT:       1:               inst600000A4
// CHECK:STDOUT:       2:               inst600000A5
// CHECK:STDOUT:     inst_block50:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst600000A2
// CHECK:STDOUT:     inst_block51:
// CHECK:STDOUT:       0:               inst600000AC
// CHECK:STDOUT:     inst_block52:
// CHECK:STDOUT:       0:               inst600000AD
// CHECK:STDOUT:     inst_block53:
// CHECK:STDOUT:       0:               inst600000A3
// CHECK:STDOUT:     inst_block54:
// CHECK:STDOUT:       0:               inst600000AE
// CHECK:STDOUT:       1:               inst600000AF
// CHECK:STDOUT:       2:               inst600000B0
// CHECK:STDOUT:     inst_block55:
// CHECK:STDOUT:       0:               inst600000B1
// CHECK:STDOUT:       1:               inst600000B2
// CHECK:STDOUT:       2:               inst600000B3
// CHECK:STDOUT:     inst_block56:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:     inst_block57:
// CHECK:STDOUT:       0:               inst60000060
// CHECK:STDOUT:       1:               inst600000BF
// CHECK:STDOUT:     inst_block58:
// CHECK:STDOUT:       0:               inst600000C2
// CHECK:STDOUT:       1:               inst600000C1
// CHECK:STDOUT:     inst_block59:
// CHECK:STDOUT:       0:               inst600000C3
// CHECK:STDOUT:       1:               inst600000C4
// CHECK:STDOUT:     inst_block60:
// CHECK:STDOUT:       0:               inst600000C7
// CHECK:STDOUT:     inst_block61:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst600000BE
// CHECK:STDOUT:     inst_block62:
// CHECK:STDOUT:       0:               inst600000CC
// CHECK:STDOUT:       1:               inst600000CD
// CHECK:STDOUT:     inst_block63:
// CHECK:STDOUT:       0:               inst600000CA
// CHECK:STDOUT:       1:               inst600000CB
// CHECK:STDOUT:     inst_block64:
// CHECK:STDOUT:       0:               inst600000CA
// CHECK:STDOUT:       1:               inst600000CB
// CHECK:STDOUT:       2:               inst600000CC
// CHECK:STDOUT:       3:               inst600000CD
// CHECK:STDOUT:       4:               inst600000CE
// CHECK:STDOUT:       5:               inst600000CF
// CHECK:STDOUT:     inst_block65:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst600000BE
// CHECK:STDOUT:       2:               inst60000060
// CHECK:STDOUT:       3:               inst600000BF
// CHECK:STDOUT:       4:               inst600000C0
// CHECK:STDOUT:       5:               inst600000C9
// CHECK:STDOUT:     inst_block66:
// CHECK:STDOUT:       0:               inst600000D7
// CHECK:STDOUT:     inst_block67:
// CHECK:STDOUT:       0:               inst600000D8
// CHECK:STDOUT:       1:               inst600000D9
// CHECK:STDOUT:     inst_block68:
// CHECK:STDOUT:       0:               inst600000CA
// CHECK:STDOUT:       1:               inst600000CB
// CHECK:STDOUT:     inst_block69:
// CHECK:STDOUT:       0:               inst600000DA
// CHECK:STDOUT:       1:               inst600000DB
// CHECK:STDOUT:     inst_block70:
// CHECK:STDOUT:       0:               inst600000DD
// CHECK:STDOUT:       1:               inst600000DF
// CHECK:STDOUT:     inst_block71:
// CHECK:STDOUT:       0:               inst600000DC
// CHECK:STDOUT:       1:               inst600000DD
// CHECK:STDOUT:       2:               inst600000DE
// CHECK:STDOUT:       3:               inst600000DF
// CHECK:STDOUT:       4:               inst600000E0
// CHECK:STDOUT:       5:               inst600000E1
// CHECK:STDOUT:     inst_block72:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst600000BE
// CHECK:STDOUT:       3:               inst600000BF
// CHECK:STDOUT:       4:               inst600000C0
// CHECK:STDOUT:       5:               inst600000D3
// CHECK:STDOUT:     inst_block73:
// CHECK:STDOUT:       0:               inst600000BE
// CHECK:STDOUT:     inst_block74:
// CHECK:STDOUT:       0:               inst600000DC
// CHECK:STDOUT:     inst_block75:
// CHECK:STDOUT:       0:               inst600000DE
// CHECK:STDOUT:     inst_block76:
// CHECK:STDOUT:       0:               inst600000E8
// CHECK:STDOUT:       1:               inst600000E9
// CHECK:STDOUT:       2:               inst600000EA
// CHECK:STDOUT:       3:               inst600000EB
// CHECK:STDOUT:       4:               inst600000EC
// CHECK:STDOUT:       5:               inst600000ED
// CHECK:STDOUT:       6:               inst600000EE
// CHECK:STDOUT:       7:               inst600000EF
// CHECK:STDOUT:       8:               inst600000F0
// CHECK:STDOUT:       9:               inst600000F1
// CHECK:STDOUT:       10:              inst600000F2
// CHECK:STDOUT:     inst_block77:
// CHECK:STDOUT:       0:               inst600000BE
// CHECK:STDOUT:       1:               inst600000BF
// CHECK:STDOUT:       2:               inst600000F3
// CHECK:STDOUT:     inst_block78:
// CHECK:STDOUT:       0:               inst600000D1
// CHECK:STDOUT:       1:               inst600000D2
// CHECK:STDOUT:     inst_block79:
// CHECK:STDOUT:       0:               inst60000060
// CHECK:STDOUT:       1:               inst600000BF
// CHECK:STDOUT:       2:               inst600000F7
// CHECK:STDOUT:     inst_block80:
// CHECK:STDOUT:       0:               inst600000FB
// CHECK:STDOUT:       1:               inst600000FA
// CHECK:STDOUT:       2:               inst600000F9
// CHECK:STDOUT:     inst_block81:
// CHECK:STDOUT:       0:               inst600000FC
// CHECK:STDOUT:       1:               inst600000FD
// CHECK:STDOUT:       2:               inst600000FE
// CHECK:STDOUT:     inst_block82:
// CHECK:STDOUT:       0:               inst60000101
// CHECK:STDOUT:     inst_block83:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst600000BE
// CHECK:STDOUT:       2:               inst600000F6
// CHECK:STDOUT:     inst_block84:
// CHECK:STDOUT:       0:               inst60000107
// CHECK:STDOUT:       1:               inst60000108
// CHECK:STDOUT:       2:               inst60000109
// CHECK:STDOUT:     inst_block85:
// CHECK:STDOUT:       0:               inst60000104
// CHECK:STDOUT:       1:               inst60000105
// CHECK:STDOUT:       2:               inst60000106
// CHECK:STDOUT:     inst_block86:
// CHECK:STDOUT:       0:               inst60000104
// CHECK:STDOUT:       1:               inst60000105
// CHECK:STDOUT:       2:               inst60000106
// CHECK:STDOUT:       3:               inst60000107
// CHECK:STDOUT:       4:               inst60000108
// CHECK:STDOUT:       5:               inst60000109
// CHECK:STDOUT:       6:               inst6000010A
// CHECK:STDOUT:       7:               inst6000010B
// CHECK:STDOUT:     inst_block87:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst600000BE
// CHECK:STDOUT:       2:               inst600000F6
// CHECK:STDOUT:       3:               inst60000060
// CHECK:STDOUT:       4:               inst600000BF
// CHECK:STDOUT:       5:               inst600000F7
// CHECK:STDOUT:       6:               inst600000F8
// CHECK:STDOUT:       7:               inst60000103
// CHECK:STDOUT:     inst_block88:
// CHECK:STDOUT:       0:               inst60000113
// CHECK:STDOUT:     inst_block89:
// CHECK:STDOUT:       0:               inst60000114
// CHECK:STDOUT:       1:               inst60000115
// CHECK:STDOUT:       2:               inst60000116
// CHECK:STDOUT:     inst_block90:
// CHECK:STDOUT:       0:               inst60000104
// CHECK:STDOUT:       1:               inst60000105
// CHECK:STDOUT:       2:               inst60000106
// CHECK:STDOUT:     inst_block91:
// CHECK:STDOUT:       0:               inst60000117
// CHECK:STDOUT:       1:               inst60000118
// CHECK:STDOUT:     inst_block92:
// CHECK:STDOUT:       0:               inst6000011A
// CHECK:STDOUT:       1:               inst6000011C
// CHECK:STDOUT:       2:               inst6000011E
// CHECK:STDOUT:     inst_block93:
// CHECK:STDOUT:       0:               inst60000119
// CHECK:STDOUT:       1:               inst6000011A
// CHECK:STDOUT:       2:               inst6000011B
// CHECK:STDOUT:       3:               inst6000011C
// CHECK:STDOUT:       4:               inst6000011D
// CHECK:STDOUT:       5:               inst6000011E
// CHECK:STDOUT:       6:               inst6000011F
// CHECK:STDOUT:       7:               inst60000120
// CHECK:STDOUT:     inst_block94:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst600000BE
// CHECK:STDOUT:       3:               inst600000BF
// CHECK:STDOUT:       4:               inst600000F6
// CHECK:STDOUT:       5:               inst600000F7
// CHECK:STDOUT:       6:               inst600000F8
// CHECK:STDOUT:       7:               inst6000010F
// CHECK:STDOUT:     inst_block95:
// CHECK:STDOUT:       0:               inst600000F6
// CHECK:STDOUT:     inst_block96:
// CHECK:STDOUT:       0:               inst60000119
// CHECK:STDOUT:     inst_block97:
// CHECK:STDOUT:       0:               inst6000011B
// CHECK:STDOUT:     inst_block98:
// CHECK:STDOUT:       0:               inst6000011D
// CHECK:STDOUT:     inst_block99:
// CHECK:STDOUT:       0:               inst60000127
// CHECK:STDOUT:       1:               inst60000128
// CHECK:STDOUT:       2:               inst60000129
// CHECK:STDOUT:       3:               inst6000012A
// CHECK:STDOUT:       4:               inst6000012B
// CHECK:STDOUT:       5:               inst6000012C
// CHECK:STDOUT:       6:               inst6000012D
// CHECK:STDOUT:       7:               inst6000012E
// CHECK:STDOUT:       8:               inst6000012F
// CHECK:STDOUT:       9:               inst60000130
// CHECK:STDOUT:       10:              inst60000131
// CHECK:STDOUT:       11:              inst60000132
// CHECK:STDOUT:       12:              inst60000133
// CHECK:STDOUT:       13:              inst60000134
// CHECK:STDOUT:       14:              inst60000135
// CHECK:STDOUT:       15:              inst60000136
// CHECK:STDOUT:     inst_block100:
// CHECK:STDOUT:       0:               inst600000F6
// CHECK:STDOUT:       1:               inst600000F7
// CHECK:STDOUT:       2:               inst60000137
// CHECK:STDOUT:     inst_block101:
// CHECK:STDOUT:       0:               inst6000010D
// CHECK:STDOUT:       1:               inst6000010E
// CHECK:STDOUT:     inst_block102:
// CHECK:STDOUT:       0:               inst600000A7
// CHECK:STDOUT:       1:               inst600000A8
// CHECK:STDOUT:       2:               inst6000003D
// CHECK:STDOUT:     inst_block103:
// CHECK:STDOUT:       0:               inst60000138
// CHECK:STDOUT:     inst_block104:
// CHECK:STDOUT:       0:               inst60000139
// CHECK:STDOUT:     inst_block105:
// CHECK:STDOUT:       0:               inst6000013A
// CHECK:STDOUT:     inst_block106:
// CHECK:STDOUT:       0:               inst6000013A
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:     inst_block107:
// CHECK:STDOUT:       0:               inst6000013E
// CHECK:STDOUT:     inst_block108:
// CHECK:STDOUT:       0:               inst6000003F
// CHECK:STDOUT:     inst_block109:
// CHECK:STDOUT:       0:               inst60000148
// CHECK:STDOUT:       1:               inst6000014C
// CHECK:STDOUT:     inst_block110:
// CHECK:STDOUT:       0:               inst6000003B
// CHECK:STDOUT:       1:               inst6000003E
// CHECK:STDOUT:       2:               inst60000139
// CHECK:STDOUT:       3:               inst6000013E
// CHECK:STDOUT:       4:               inst6000013F
// CHECK:STDOUT:       5:               inst60000140
// CHECK:STDOUT:       6:               inst60000144
// CHECK:STDOUT:     inst_block111:
// CHECK:STDOUT:       0:               inst0000000E
// CHECK:STDOUT:       1:               inst6000000F
// CHECK:STDOUT:       2:               inst60000035
// CHECK:STDOUT: ...
