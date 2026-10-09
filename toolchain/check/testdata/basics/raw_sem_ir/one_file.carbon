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

fn Foo[T: type](p: T*) -> (T*, ()) {
  return (p, ());
}


// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        one_file.carbon
// CHECK:STDOUT: sem_ir:
// CHECK:STDOUT:   names:
// CHECK:STDOUT:     name0:           Foo
// CHECK:STDOUT:     name1:           T
// CHECK:STDOUT:     name2:           p
// CHECK:STDOUT:     name3:           Copy
// CHECK:STDOUT:     name4:           Op
// CHECK:STDOUT:     name5:           U
// CHECK:STDOUT:     name6:           V
// CHECK:STDOUT:   import_irs:
// CHECK:STDOUT:     'import_ir(Local)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     import_ir78000003: {decl_id: inst7800000F, is_export: false}
// CHECK:STDOUT:     import_ir78000004: {decl_id: inst7800000F, is_export: false}
// CHECK:STDOUT:     import_ir78000005: {decl_id: inst7800000F, is_export: false}
// CHECK:STDOUT:     import_ir78000006: {decl_id: inst7800000F, is_export: false}
// CHECK:STDOUT:     import_ir78000007: {decl_id: inst7800000F, is_export: false}
// CHECK:STDOUT:     import_ir78000008: {decl_id: inst7800000F, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir78000006, inst_id: inst4800000F}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir78000006, inst_id: inst48000014}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir78000006, inst_id: inst4800003C}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir78000006, inst_id: inst48000034}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir78000006, inst_id: inst48000011}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: import_ir78000006, inst_id: inst4800002B}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: import_ir78000006, inst_id: inst4800001D}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: import_ir78000006, inst_id: inst4800001C}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: import_ir78000006, inst_id: inst4800002A}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: import_ir78000006, inst_id: inst48000020}
// CHECK:STDOUT:     import_ir_instA: {ir_id: import_ir78000006, inst_id: inst48000026}
// CHECK:STDOUT:     import_ir_instB: {ir_id: import_ir78000006, inst_id: inst48000027}
// CHECK:STDOUT:     import_ir_instC: {ir_id: import_ir78000006, inst_id: inst4800002D}
// CHECK:STDOUT:     import_ir_instD: {ir_id: import_ir78000006, inst_id: inst48000016}
// CHECK:STDOUT:     import_ir_instE: {ir_id: import_ir78000006, inst_id: inst48000019}
// CHECK:STDOUT:     import_ir_instF: {ir_id: import_ir78000006, inst_id: inst4800001E}
// CHECK:STDOUT:     import_ir_inst10: {ir_id: import_ir78000006, inst_id: inst4800001F}
// CHECK:STDOUT:     import_ir_inst11: {ir_id: import_ir78000006, inst_id: inst48000022}
// CHECK:STDOUT:     import_ir_inst12: {ir_id: import_ir78000006, inst_id: inst48000029}
// CHECK:STDOUT:     import_ir_inst13: {ir_id: import_ir78000006, inst_id: inst4800002C}
// CHECK:STDOUT:     import_ir_inst14: {ir_id: import_ir78000006, inst_id: inst4800002F}
// CHECK:STDOUT:     import_ir_inst15: {ir_id: import_ir78000006, inst_id: inst48000093}
// CHECK:STDOUT:     import_ir_inst16: {ir_id: import_ir78000006, inst_id: inst4800008E}
// CHECK:STDOUT:     import_ir_inst17: {ir_id: import_ir78000006, inst_id: inst480000B5}
// CHECK:STDOUT:     import_ir_inst18: {ir_id: import_ir78000006, inst_id: inst48000092}
// CHECK:STDOUT:     import_ir_inst19: {ir_id: import_ir78000006, inst_id: inst480000AC}
// CHECK:STDOUT:     import_ir_inst1A: {ir_id: import_ir78000006, inst_id: inst4800009F}
// CHECK:STDOUT:     import_ir_inst1B: {ir_id: import_ir78000006, inst_id: inst4800009E}
// CHECK:STDOUT:     import_ir_inst1C: {ir_id: import_ir78000006, inst_id: inst480000AB}
// CHECK:STDOUT:     import_ir_inst1D: {ir_id: import_ir78000006, inst_id: inst480000A2}
// CHECK:STDOUT:     import_ir_inst1E: {ir_id: import_ir78000006, inst_id: inst480000A7}
// CHECK:STDOUT:     import_ir_inst1F: {ir_id: import_ir78000006, inst_id: inst480000A8}
// CHECK:STDOUT:     import_ir_inst20: {ir_id: import_ir78000006, inst_id: inst480000AE}
// CHECK:STDOUT:     import_ir_inst21: {ir_id: import_ir78000006, inst_id: inst48000081}
// CHECK:STDOUT:     import_ir_inst22: {ir_id: import_ir78000006, inst_id: inst48000099}
// CHECK:STDOUT:     import_ir_inst23: {ir_id: import_ir78000006, inst_id: inst4800009A}
// CHECK:STDOUT:     import_ir_inst24: {ir_id: import_ir78000006, inst_id: inst4800009B}
// CHECK:STDOUT:     import_ir_inst25: {ir_id: import_ir78000006, inst_id: inst480000A0}
// CHECK:STDOUT:     import_ir_inst26: {ir_id: import_ir78000006, inst_id: inst480000A1}
// CHECK:STDOUT:     import_ir_inst27: {ir_id: import_ir78000006, inst_id: inst480000A4}
// CHECK:STDOUT:     import_ir_inst28: {ir_id: import_ir78000006, inst_id: inst480000AA}
// CHECK:STDOUT:     import_ir_inst29: {ir_id: import_ir78000006, inst_id: inst480000AD}
// CHECK:STDOUT:     import_ir_inst2A: {ir_id: import_ir78000006, inst_id: inst480000B0}
// CHECK:STDOUT:     import_ir_inst2B: {ir_id: import_ir78000006, inst_id: inst4800007E}
// CHECK:STDOUT:     import_ir_inst2C: {ir_id: import_ir78000006, inst_id: inst48000089}
// CHECK:STDOUT:     import_ir_inst2D: {ir_id: import_ir78000006, inst_id: inst4800008C}
// CHECK:STDOUT:     import_ir_inst2E: {ir_id: import_ir78000006, inst_id: inst4800008F}
// CHECK:STDOUT:     import_ir_inst2F: {ir_id: import_ir78000006, inst_id: inst48000080}
// CHECK:STDOUT:     import_ir_inst30: {ir_id: import_ir78000006, inst_id: inst48000083}
// CHECK:STDOUT:     import_ir_inst31: {ir_id: import_ir78000006, inst_id: inst48000087}
// CHECK:STDOUT:     import_ir_inst32: {ir_id: import_ir78000006, inst_id: inst4800008B}
// CHECK:STDOUT:     import_ir_inst33: {ir_id: import_ir78000006, inst_id: inst48000091}
// CHECK:STDOUT:     import_ir_inst34: {ir_id: import_ir78000006, inst_id: inst48000095}
// CHECK:STDOUT:     import_ir_inst35: {ir_id: import_ir78000006, inst_id: inst480000F5}
// CHECK:STDOUT:     import_ir_inst36: {ir_id: import_ir78000006, inst_id: inst480000F1}
// CHECK:STDOUT:     import_ir_inst37: {ir_id: import_ir78000006, inst_id: inst480000EF}
// CHECK:STDOUT:     import_ir_inst38: {ir_id: import_ir78000006, inst_id: inst480000F0}
// CHECK:STDOUT:     import_ir_inst39: {ir_id: import_ir78000006, inst_id: inst480000F2}
// CHECK:STDOUT:     import_ir_inst3A: {ir_id: import_ir78000006, inst_id: inst48000114}
// CHECK:STDOUT:     import_ir_inst3B: {ir_id: import_ir78000006, inst_id: inst48000110}
// CHECK:STDOUT:     import_ir_inst3C: {ir_id: import_ir78000006, inst_id: inst4800010E}
// CHECK:STDOUT:     import_ir_inst3D: {ir_id: import_ir78000006, inst_id: inst4800010F}
// CHECK:STDOUT:     import_ir_inst3E: {ir_id: import_ir78000006, inst_id: inst48000111}
// CHECK:STDOUT:     import_ir_inst3F: {ir_id: import_ir78000006, inst_id: inst48000133}
// CHECK:STDOUT:     import_ir_inst40: {ir_id: import_ir78000006, inst_id: inst4800012F}
// CHECK:STDOUT:     import_ir_inst41: {ir_id: import_ir78000006, inst_id: inst4800012D}
// CHECK:STDOUT:     import_ir_inst42: {ir_id: import_ir78000006, inst_id: inst4800012E}
// CHECK:STDOUT:     import_ir_inst43: {ir_id: import_ir78000006, inst_id: inst48000130}
// CHECK:STDOUT:     import_ir_inst44: {ir_id: import_ir78000006, inst_id: inst48000152}
// CHECK:STDOUT:     import_ir_inst45: {ir_id: import_ir78000006, inst_id: inst4800014E}
// CHECK:STDOUT:     import_ir_inst46: {ir_id: import_ir78000006, inst_id: inst4800014C}
// CHECK:STDOUT:     import_ir_inst47: {ir_id: import_ir78000006, inst_id: inst4800014D}
// CHECK:STDOUT:     import_ir_inst48: {ir_id: import_ir78000006, inst_id: inst4800014F}
// CHECK:STDOUT:     import_ir_inst49: {ir_id: import_ir78000006, inst_id: inst4800017E}
// CHECK:STDOUT:     import_ir_inst4A: {ir_id: import_ir78000006, inst_id: inst48000179}
// CHECK:STDOUT:     import_ir_inst4B: {ir_id: import_ir78000006, inst_id: inst4800019D}
// CHECK:STDOUT:     import_ir_inst4C: {ir_id: import_ir78000006, inst_id: inst4800017D}
// CHECK:STDOUT:     import_ir_inst4D: {ir_id: import_ir78000006, inst_id: inst4800016D}
// CHECK:STDOUT:     import_ir_inst4E: {ir_id: import_ir78000006, inst_id: inst48000174}
// CHECK:STDOUT:     import_ir_inst4F: {ir_id: import_ir78000006, inst_id: inst48000177}
// CHECK:STDOUT:     import_ir_inst50: {ir_id: import_ir78000006, inst_id: inst4800017A}
// CHECK:STDOUT:     import_ir_inst51: {ir_id: import_ir78000006, inst_id: inst48000170}
// CHECK:STDOUT:     import_ir_inst52: {ir_id: import_ir78000006, inst_id: inst4800016F}
// CHECK:STDOUT:     import_ir_inst53: {ir_id: import_ir78000006, inst_id: inst48000172}
// CHECK:STDOUT:     import_ir_inst54: {ir_id: import_ir78000006, inst_id: inst48000176}
// CHECK:STDOUT:     import_ir_inst55: {ir_id: import_ir78000006, inst_id: inst4800017C}
// CHECK:STDOUT:     import_ir_inst56: {ir_id: import_ir78000006, inst_id: inst48000180}
// CHECK:STDOUT:     import_ir_inst57: {ir_id: import_ir78000006, inst_id: inst480001AE}
// CHECK:STDOUT:     import_ir_inst58: {ir_id: import_ir78000006, inst_id: inst480001AA}
// CHECK:STDOUT:     import_ir_inst59: {ir_id: import_ir78000006, inst_id: inst480001A8}
// CHECK:STDOUT:     import_ir_inst5A: {ir_id: import_ir78000006, inst_id: inst480001A9}
// CHECK:STDOUT:     import_ir_inst5B: {ir_id: import_ir78000006, inst_id: inst480001AB}
// CHECK:STDOUT:     import_ir_inst5C: {ir_id: import_ir78000006, inst_id: inst480001CB}
// CHECK:STDOUT:     import_ir_inst5D: {ir_id: import_ir78000006, inst_id: inst480001C7}
// CHECK:STDOUT:     import_ir_inst5E: {ir_id: import_ir78000006, inst_id: inst480001C5}
// CHECK:STDOUT:     import_ir_inst5F: {ir_id: import_ir78000006, inst_id: inst480001C6}
// CHECK:STDOUT:     import_ir_inst60: {ir_id: import_ir78000006, inst_id: inst480001C8}
// CHECK:STDOUT:     import_ir_inst61: {ir_id: import_ir78000006, inst_id: inst4800020B}
// CHECK:STDOUT:     import_ir_inst62: {ir_id: import_ir78000006, inst_id: inst48000206}
// CHECK:STDOUT:     import_ir_inst63: {ir_id: import_ir78000006, inst_id: inst48000231}
// CHECK:STDOUT:     import_ir_inst64: {ir_id: import_ir78000006, inst_id: inst4800020A}
// CHECK:STDOUT:     import_ir_inst65: {ir_id: import_ir78000006, inst_id: inst48000228}
// CHECK:STDOUT:     import_ir_inst66: {ir_id: import_ir78000006, inst_id: inst4800021B}
// CHECK:STDOUT:     import_ir_inst67: {ir_id: import_ir78000006, inst_id: inst4800021A}
// CHECK:STDOUT:     import_ir_inst68: {ir_id: import_ir78000006, inst_id: inst48000227}
// CHECK:STDOUT:     import_ir_inst69: {ir_id: import_ir78000006, inst_id: inst4800021E}
// CHECK:STDOUT:     import_ir_inst6A: {ir_id: import_ir78000006, inst_id: inst48000223}
// CHECK:STDOUT:     import_ir_inst6B: {ir_id: import_ir78000006, inst_id: inst48000224}
// CHECK:STDOUT:     import_ir_inst6C: {ir_id: import_ir78000006, inst_id: inst4800022A}
// CHECK:STDOUT:     import_ir_inst6D: {ir_id: import_ir78000006, inst_id: inst480001E8}
// CHECK:STDOUT:     import_ir_inst6E: {ir_id: import_ir78000006, inst_id: inst480001EF}
// CHECK:STDOUT:     import_ir_inst6F: {ir_id: import_ir78000006, inst_id: inst48000213}
// CHECK:STDOUT:     import_ir_inst70: {ir_id: import_ir78000006, inst_id: inst48000214}
// CHECK:STDOUT:     import_ir_inst71: {ir_id: import_ir78000006, inst_id: inst48000215}
// CHECK:STDOUT:     import_ir_inst72: {ir_id: import_ir78000006, inst_id: inst48000216}
// CHECK:STDOUT:     import_ir_inst73: {ir_id: import_ir78000006, inst_id: inst48000217}
// CHECK:STDOUT:     import_ir_inst74: {ir_id: import_ir78000006, inst_id: inst4800021C}
// CHECK:STDOUT:     import_ir_inst75: {ir_id: import_ir78000006, inst_id: inst4800021D}
// CHECK:STDOUT:     import_ir_inst76: {ir_id: import_ir78000006, inst_id: inst48000220}
// CHECK:STDOUT:     import_ir_inst77: {ir_id: import_ir78000006, inst_id: inst48000226}
// CHECK:STDOUT:     import_ir_inst78: {ir_id: import_ir78000006, inst_id: inst48000229}
// CHECK:STDOUT:     import_ir_inst79: {ir_id: import_ir78000006, inst_id: inst4800022C}
// CHECK:STDOUT:     import_ir_inst7A: {ir_id: import_ir78000006, inst_id: inst480001E6}
// CHECK:STDOUT:     import_ir_inst7B: {ir_id: import_ir78000006, inst_id: inst480001EC}
// CHECK:STDOUT:     import_ir_inst7C: {ir_id: import_ir78000006, inst_id: inst48000201}
// CHECK:STDOUT:     import_ir_inst7D: {ir_id: import_ir78000006, inst_id: inst48000203}
// CHECK:STDOUT:     import_ir_inst7E: {ir_id: import_ir78000006, inst_id: inst48000207}
// CHECK:STDOUT:     import_ir_inst7F: {ir_id: import_ir78000006, inst_id: inst480001E7}
// CHECK:STDOUT:     import_ir_inst80: {ir_id: import_ir78000006, inst_id: inst480001E9}
// CHECK:STDOUT:     import_ir_inst81: {ir_id: import_ir78000006, inst_id: inst480001EE}
// CHECK:STDOUT:     import_ir_inst82: {ir_id: import_ir78000006, inst_id: inst480001F1}
// CHECK:STDOUT:     import_ir_inst83: {ir_id: import_ir78000006, inst_id: inst480001F7}
// CHECK:STDOUT:     import_ir_inst84: {ir_id: import_ir78000006, inst_id: inst480001FA}
// CHECK:STDOUT:     import_ir_inst85: {ir_id: import_ir78000006, inst_id: inst480001FE}
// CHECK:STDOUT:     import_ir_inst86: {ir_id: import_ir78000006, inst_id: inst48000202}
// CHECK:STDOUT:     import_ir_inst87: {ir_id: import_ir78000006, inst_id: inst48000209}
// CHECK:STDOUT:     import_ir_inst88: {ir_id: import_ir78000006, inst_id: inst4800020D}
// CHECK:STDOUT:     import_ir_inst89: {ir_id: import_ir78000006, inst_id: inst480002B1}
// CHECK:STDOUT:     import_ir_inst8A: {ir_id: import_ir78000006, inst_id: inst480002AC}
// CHECK:STDOUT:     import_ir_inst8B: {ir_id: import_ir78000006, inst_id: inst480002DB}
// CHECK:STDOUT:     import_ir_inst8C: {ir_id: import_ir78000006, inst_id: inst480002B0}
// CHECK:STDOUT:     import_ir_inst8D: {ir_id: import_ir78000006, inst_id: inst480002D2}
// CHECK:STDOUT:     import_ir_inst8E: {ir_id: import_ir78000006, inst_id: inst480002C5}
// CHECK:STDOUT:     import_ir_inst8F: {ir_id: import_ir78000006, inst_id: inst480002C4}
// CHECK:STDOUT:     import_ir_inst90: {ir_id: import_ir78000006, inst_id: inst480002D1}
// CHECK:STDOUT:     import_ir_inst91: {ir_id: import_ir78000006, inst_id: inst480002C8}
// CHECK:STDOUT:     import_ir_inst92: {ir_id: import_ir78000006, inst_id: inst480002CD}
// CHECK:STDOUT:     import_ir_inst93: {ir_id: import_ir78000006, inst_id: inst480002CE}
// CHECK:STDOUT:     import_ir_inst94: {ir_id: import_ir78000006, inst_id: inst480002D4}
// CHECK:STDOUT:     import_ir_inst95: {ir_id: import_ir78000006, inst_id: inst48000283}
// CHECK:STDOUT:     import_ir_inst96: {ir_id: import_ir78000006, inst_id: inst48000289}
// CHECK:STDOUT:     import_ir_inst97: {ir_id: import_ir78000006, inst_id: inst48000290}
// CHECK:STDOUT:     import_ir_inst98: {ir_id: import_ir78000006, inst_id: inst480002BB}
// CHECK:STDOUT:     import_ir_inst99: {ir_id: import_ir78000006, inst_id: inst480002BC}
// CHECK:STDOUT:     import_ir_inst9A: {ir_id: import_ir78000006, inst_id: inst480002BD}
// CHECK:STDOUT:     import_ir_inst9B: {ir_id: import_ir78000006, inst_id: inst480002BE}
// CHECK:STDOUT:     import_ir_inst9C: {ir_id: import_ir78000006, inst_id: inst480002BF}
// CHECK:STDOUT:     import_ir_inst9D: {ir_id: import_ir78000006, inst_id: inst480002C0}
// CHECK:STDOUT:     import_ir_inst9E: {ir_id: import_ir78000006, inst_id: inst480002C1}
// CHECK:STDOUT:     import_ir_inst9F: {ir_id: import_ir78000006, inst_id: inst480002C6}
// CHECK:STDOUT:     import_ir_instA0: {ir_id: import_ir78000006, inst_id: inst480002C7}
// CHECK:STDOUT:     import_ir_instA1: {ir_id: import_ir78000006, inst_id: inst480002CA}
// CHECK:STDOUT:     import_ir_instA2: {ir_id: import_ir78000006, inst_id: inst480002D0}
// CHECK:STDOUT:     import_ir_instA3: {ir_id: import_ir78000006, inst_id: inst480002D3}
// CHECK:STDOUT:     import_ir_instA4: {ir_id: import_ir78000006, inst_id: inst480002D6}
// CHECK:STDOUT:     import_ir_instA5: {ir_id: import_ir78000006, inst_id: inst48000281}
// CHECK:STDOUT:     import_ir_instA6: {ir_id: import_ir78000006, inst_id: inst48000287}
// CHECK:STDOUT:     import_ir_instA7: {ir_id: import_ir78000006, inst_id: inst4800028D}
// CHECK:STDOUT:     import_ir_instA8: {ir_id: import_ir78000006, inst_id: inst480002A6}
// CHECK:STDOUT:     import_ir_instA9: {ir_id: import_ir78000006, inst_id: inst480002A8}
// CHECK:STDOUT:     import_ir_instAA: {ir_id: import_ir78000006, inst_id: inst480002AD}
// CHECK:STDOUT:     import_ir_instAB: {ir_id: import_ir78000006, inst_id: inst48000282}
// CHECK:STDOUT:     import_ir_instAC: {ir_id: import_ir78000006, inst_id: inst48000284}
// CHECK:STDOUT:     import_ir_instAD: {ir_id: import_ir78000006, inst_id: inst48000288}
// CHECK:STDOUT:     import_ir_instAE: {ir_id: import_ir78000006, inst_id: inst4800028A}
// CHECK:STDOUT:     import_ir_instAF: {ir_id: import_ir78000006, inst_id: inst4800028F}
// CHECK:STDOUT:     import_ir_instB0: {ir_id: import_ir78000006, inst_id: inst48000292}
// CHECK:STDOUT:     import_ir_instB1: {ir_id: import_ir78000006, inst_id: inst48000299}
// CHECK:STDOUT:     import_ir_instB2: {ir_id: import_ir78000006, inst_id: inst4800029C}
// CHECK:STDOUT:     import_ir_instB3: {ir_id: import_ir78000006, inst_id: inst4800029F}
// CHECK:STDOUT:     import_ir_instB4: {ir_id: import_ir78000006, inst_id: inst480002A3}
// CHECK:STDOUT:     import_ir_instB5: {ir_id: import_ir78000006, inst_id: inst480002A7}
// CHECK:STDOUT:     import_ir_instB6: {ir_id: import_ir78000006, inst_id: inst480002AF}
// CHECK:STDOUT:     import_ir_instB7: {ir_id: import_ir78000006, inst_id: inst480002B3}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   clang_decl_signatures: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst(Package), parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst78000010, name0: inst78000046}}
// CHECK:STDOUT:     name_scope78000001: {inst: inst78000010, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: inst78000056}}
// CHECK:STDOUT:     name_scope78000002: {inst: inst78000057, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {name(SelfType): inst7800007E}}
// CHECK:STDOUT:     name_scope78000003: {inst: inst78000059, parent_scope: name_scope78000002, has_error: false, extended_scopes: [], names: {name4: inst7800005B}}
// CHECK:STDOUT:     name_scope78000004: {inst: inst78000084, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000005: {inst: inst780000BA, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000006: {inst: inst780000C0, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000007: {inst: inst780000C6, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000008: {inst: inst780000CC, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000009: {inst: inst780000D2, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000A: {inst: inst780000E2, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000B: {inst: inst780000E8, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000C: {inst: inst780000EE, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000D: {inst: inst7800012E, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name78000000: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, is_frozen_period_self: 1, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000001: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst78000013}
// CHECK:STDOUT:     entity_name78000002: {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst7800001C}
// CHECK:STDOUT:     entity_name78000003: {name: name3, parent_scope: name_scope78000001, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000004: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000005: {name: name4, parent_scope: name_scope78000003, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000006: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000007: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000008: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000009: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000A: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000B: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000C: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000D: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000E: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000F: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000010: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000011: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000012: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000013: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000014: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000015: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000016: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000017: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000018: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000019: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001A: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001B: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001C: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001D: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001E: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001F: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000020: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000021: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000022: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000023: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000024: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000025: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000026: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000027: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000028: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000029: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002A: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002B: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002C: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002D: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002E: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002F: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000030: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000031: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000032: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000033: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000034: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000035: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000036: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000037: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000038: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000039: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003A: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003B: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003C: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003D: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function78000000: {name: name0, parent_scope: name_scope0, call_param_patterns_id: inst_block78000011, call_params_id: inst_block78000012, return_type_inst_id: inst78000034, return_form_inst_id: inst78000036, return_pattern_id: inst7800003E, body: [inst_block78000019]}
// CHECK:STDOUT:     function78000001: {name: name4, parent_scope: name_scope78000003, call_param_patterns_id: inst_block7800001F, return_type_inst_id: inst78000070, return_form_inst_id: inst78000071, return_pattern_id: inst78000072}
// CHECK:STDOUT:     function78000002: {name: name4, parent_scope: name_scope78000004, call_param_patterns_id: inst_block78000028, return_type_inst_id: inst7800009F, return_form_inst_id: inst780000A0, return_pattern_id: inst780000A1}
// CHECK:STDOUT:     function78000003: {name: name4, parent_scope: name_scope7800000C, call_param_patterns_id: inst_block7800003D, return_type_inst_id: inst7800010A, return_form_inst_id: inst7800010B, return_pattern_id: inst7800010C}
// CHECK:STDOUT:     function78000004: {name: name4, parent_scope: name_scope7800000D, call_param_patterns_id: inst_block7800004F, return_type_inst_id: inst7800014A, return_form_inst_id: inst7800014B, return_pattern_id: inst7800014C}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:
// CHECK:STDOUT:     interface78000000: {name: name3, parent_scope: name_scope78000001, require_impls_block_id: require_block_empty, core_interface: Copy}
// CHECK:STDOUT:   associated_constants: {}
// CHECK:STDOUT:   impls:
// CHECK:STDOUT:     impl78000000:    {self: inst780000AF, constraint: inst780000B0, witness: inst78000083}
// CHECK:STDOUT:     impl78000001:    {self: inst780000BC, constraint: inst780000BD, witness: inst780000B9}
// CHECK:STDOUT:     impl78000002:    {self: inst780000C2, constraint: inst780000C3, witness: inst780000BF}
// CHECK:STDOUT:     impl78000003:    {self: inst780000C8, constraint: inst780000C9, witness: inst780000C5}
// CHECK:STDOUT:     impl78000004:    {self: inst780000CE, constraint: inst780000CF, witness: inst780000CB}
// CHECK:STDOUT:     impl78000005:    {self: inst780000D8, constraint: inst780000D9, witness: inst780000D1}
// CHECK:STDOUT:     impl78000006:    {self: inst780000E4, constraint: inst780000E5, witness: inst780000E1}
// CHECK:STDOUT:     impl78000007:    {self: inst780000EA, constraint: inst780000EB, witness: inst780000E7}
// CHECK:STDOUT:     impl78000008:    {self: inst7800011E, constraint: inst7800011F, witness: inst780000ED}
// CHECK:STDOUT:     impl78000009:    {self: inst78000162, constraint: inst78000163, witness: inst7800012D}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic78000000: {decl: inst78000046, bindings: inst_block78000015, self_specific_id: specific78000000, decl_block_id: inst_block78000017, definition_ir_id: import_ir(Local), definition_block_id: inst_block78000064}
// CHECK:STDOUT:     generic78000001: {decl: inst78000059, bindings: inst_block7800001C, self_specific_id: specific78000001, decl_block_id: inst_block_empty, definition_ir_id: import_ir78000006, definition_block_id: inst_block48000015}
// CHECK:STDOUT:     generic78000002: {decl: inst7800005E, bindings: inst_block78000021, self_specific_id: specific78000002, decl_block_id: inst_block78000022, definition_ir_id: import_ir<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     generic78000003: {decl: inst78000084, bindings: inst_block7800002E, self_specific_id: specific78000003, decl_block_id: inst_block78000030, definition_ir_id: import_ir78000006, definition_block_id: inst_block48000046}
// CHECK:STDOUT:     generic78000004: {decl: inst7800008C, bindings: inst_block7800002A, self_specific_id: specific78000004, decl_block_id: inst_block7800002B, definition_ir_id: import_ir78000006, definition_block_id: inst_block4800004F}
// CHECK:STDOUT:     generic78000005: {decl: inst780000D2, bindings: inst_block78000034, self_specific_id: specific78000006, decl_block_id: inst_block78000036, definition_ir_id: import_ir78000006, definition_block_id: inst_block480000A7}
// CHECK:STDOUT:     generic78000006: {decl: inst780000EE, bindings: inst_block78000044, self_specific_id: specific78000008, decl_block_id: inst_block78000048, definition_ir_id: import_ir78000006, definition_block_id: inst_block480000E9}
// CHECK:STDOUT:     generic78000007: {decl: inst780000F6, bindings: inst_block7800003F, self_specific_id: specific78000009, decl_block_id: inst_block78000041, definition_ir_id: import_ir78000006, definition_block_id: inst_block480000F8}
// CHECK:STDOUT:     generic78000008: {decl: inst7800012E, bindings: inst_block78000056, self_specific_id: specific7800000B, decl_block_id: inst_block7800005A, definition_ir_id: import_ir78000006, definition_block_id: inst_block4800011D}
// CHECK:STDOUT:     generic78000009: {decl: inst78000136, bindings: inst_block78000051, self_specific_id: specific7800000C, decl_block_id: inst_block78000053, definition_ir_id: import_ir78000006, definition_block_id: inst_block4800012F}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific78000000: {generic: generic78000000, args: inst_block78000016, identified: 0, decl_block_id: inst_block78000018, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000001: {generic: generic78000001, args: inst_block7800001D, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block7800001E, definition_has_error: 0}
// CHECK:STDOUT:     specific78000002: {generic: generic78000002, args: inst_block7800001D, identified: 0, decl_block_id: inst_block78000023, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000003: {generic: generic78000003, args: inst_block78000025, identified: 0, decl_block_id: inst_block78000026, decl_has_error: 0, definition_block_id: inst_block78000027, definition_has_error: 0}
// CHECK:STDOUT:     specific78000004: {generic: generic78000004, args: inst_block78000025, identified: 0, decl_block_id: inst_block7800002C, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000005: {generic: generic78000003, args: inst_block7800002F, identified: 0, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000006: {generic: generic78000005, args: inst_block78000016, identified: 0, decl_block_id: inst_block78000032, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000007: {generic: generic78000005, args: inst_block78000035, identified: 0, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000008: {generic: generic78000006, args: inst_block78000039, identified: 0, decl_block_id: inst_block7800003B, decl_has_error: 0, definition_block_id: inst_block7800003C, definition_has_error: 0}
// CHECK:STDOUT:     specific78000009: {generic: generic78000007, args: inst_block78000039, identified: 0, decl_block_id: inst_block78000042, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000A: {generic: generic78000006, args: inst_block78000045, identified: 0, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000B: {generic: generic78000008, args: inst_block7800004B, identified: 0, decl_block_id: inst_block7800004D, decl_has_error: 0, definition_block_id: inst_block7800004E, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000C: {generic: generic78000009, args: inst_block7800004B, identified: 0, decl_block_id: inst_block78000054, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000D: {generic: generic78000008, args: inst_block78000057, identified: 0, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000E: {generic: generic78000001, args: inst_block7800005C, identified: 0, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block7800005D, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000F: {generic: generic78000001, args: inst_block7800005F, identified: 0, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000010: {generic: generic78000002, args: inst_block7800005C, identified: 0, decl_block_id: inst_block78000060, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000011: {generic: generic78000002, args: inst_block7800005F, identified: 0, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:   specific_interfaces:
// CHECK:STDOUT:     specific_interface78000000: {interface_id: interface78000000, specific_id: specific<none>}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields_empty: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(FormType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(FormType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(InstType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(InstType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst7800002A)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst7800002A)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst78000031)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst78000031)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst7800002D)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst78000031)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst78000047)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst7800002A)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant78000005)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant78000005)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant7800001B)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant7800001B)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(symbolic_constant7800000F)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant7800001B)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(symbolic_constant78000006)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(symbolic_constant78000010)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant7800001B)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst78000058)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst78000058)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant78000127)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst7800002A)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant7800012C)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst7800002A)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:   declared_facet_types:
// CHECK:STDOUT:     'declared_facet_type(Empty)': {}
// CHECK:STDOUT:     declared_facet_type78000001: {extends interface: [interface78000000]}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst(TypeType)':  {kind: FacetType, arg0: declared_facet_type(Empty), type: type(TypeType)}
// CHECK:STDOUT:     'inst(Package)':   {kind: Namespace, arg0: name_scope0, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst7800000F:    {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst78000010:    {kind: Namespace, arg0: name_scope78000001, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst78000011:    {kind: SymbolicBinding, arg0: entity_name78000000, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000012:    {kind: SymbolicBinding, arg0: entity_name78000000, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000013:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst78000014:    {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst78000015:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000014)}
// CHECK:STDOUT:     inst78000016:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000014)}
// CHECK:STDOUT:     inst78000017:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000014)}
// CHECK:STDOUT:     inst78000018:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000019:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001A:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001B:    {kind: NameRef, arg0: name1, arg1: inst78000018, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001C:    {kind: PointerType, arg0: inst7800001B, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001D:    {kind: PointerType, arg0: inst78000019, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001E:    {kind: PointerType, arg0: inst7800001A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001F:    {kind: PatternType, arg0: inst7800001D, type: type(TypeType)}
// CHECK:STDOUT:     inst78000020:    {kind: ValueParamPattern, arg0: name2, type: type(symbolic_constant78000009)}
// CHECK:STDOUT:     inst78000021:    {kind: ValueParamPattern, arg0: name2, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000022:    {kind: PatternType, arg0: inst7800001E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000023:    {kind: ValueParamPattern, arg0: name2, type: type(symbolic_constant78000009)}
// CHECK:STDOUT:     inst78000024:    {kind: WrapperBindingPattern, arg0: entity_name78000002, arg1: inst78000020, type: type(symbolic_constant78000009)}
// CHECK:STDOUT:     inst78000025:    {kind: WrapperBindingPattern, arg0: entity_name78000002, arg1: inst78000021, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000026:    {kind: WrapperBindingPattern, arg0: entity_name78000002, arg1: inst78000023, type: type(symbolic_constant78000009)}
// CHECK:STDOUT:     inst78000027:    {kind: WrapperBinding, arg0: entity_name78000002, arg1: inst78000042, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000028:    {kind: NameRef, arg0: name1, arg1: inst78000018, type: type(TypeType)}
// CHECK:STDOUT:     inst78000029:    {kind: PointerType, arg0: inst78000028, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002A:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002B:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst7800002A)}
// CHECK:STDOUT:     inst7800002C:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst7800002A)}
// CHECK:STDOUT:     inst7800002D:    {kind: TupleType, arg0: inst_block7800000B, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002E:    {kind: TupleLiteral, arg0: inst_block7800000A, type: type(inst7800002D)}
// CHECK:STDOUT:     inst7800002F:    {kind: TupleValue, arg0: inst_block7800000C, type: type(inst7800002D)}
// CHECK:STDOUT:     inst78000030:    {kind: TupleValue, arg0: inst_block7800000D, type: type(inst7800002D)}
// CHECK:STDOUT:     inst78000031:    {kind: PointerType, arg0: inst7800002D, type: type(TypeType)}
// CHECK:STDOUT:     inst78000032:    {kind: Converted, arg0: inst7800002C, arg1: inst7800002A, type: type(TypeType)}
// CHECK:STDOUT:     inst78000033:    {kind: TupleType, arg0: inst_block7800000F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000034:    {kind: Converted, arg0: inst7800002E, arg1: inst78000033, type: type(TypeType)}
// CHECK:STDOUT:     inst78000035:    {kind: TupleType, arg0: inst_block78000010, type: type(TypeType)}
// CHECK:STDOUT:     inst78000036:    {kind: InitForm, arg0: inst78000034, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000037:    {kind: InitForm, arg0: inst78000033, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000038:    {kind: InitForm, arg0: inst78000035, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000039:    {kind: PatternType, arg0: inst78000033, type: type(TypeType)}
// CHECK:STDOUT:     inst7800003A:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst7800003B:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000013)}
// CHECK:STDOUT:     inst7800003C:    {kind: PatternType, arg0: inst78000035, type: type(TypeType)}
// CHECK:STDOUT:     inst7800003D:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst7800003E:    {kind: ReturnSlotPattern, arg0: inst7800003A, arg1: inst78000034, type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst7800003F:    {kind: ReturnSlotPattern, arg0: inst7800003B, arg1: inst78000033, type: type(symbolic_constant78000013)}
// CHECK:STDOUT:     inst78000040:    {kind: ReturnSlotPattern, arg0: inst7800003D, arg1: inst78000035, type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst78000041:    {kind: SpliceBlock, arg0: inst_block78000006, arg1: inst78000013, type: type(TypeType)}
// CHECK:STDOUT:     inst78000042:    {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000043:    {kind: SpliceBlock, arg0: inst_block78000008, arg1: inst7800001C, type: type(TypeType)}
// CHECK:STDOUT:     inst78000044:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst78000045:    {kind: ReturnSlot, arg0: inst78000033, arg1: inst78000044, type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst78000046:    {kind: FunctionDecl, arg0: function78000000, arg1: inst_block78000014, type: type(inst78000047)}
// CHECK:STDOUT:     inst78000047:    {kind: FunctionType, arg0: function78000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000048:    {kind: StructValue, arg0: inst_block_empty, type: type(inst78000047)}
// CHECK:STDOUT:     inst78000049:    {kind: RequireCompleteType, arg0: inst7800001D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004A:    {kind: RequireCompleteType, arg0: inst7800001D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004B:    {kind: RequireCompleteType, arg0: inst7800001E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004C:    {kind: PointerType, arg0: inst78000033, type: type(TypeType)}
// CHECK:STDOUT:     inst7800004D:    {kind: RequireCompleteType, arg0: inst78000033, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004E:    {kind: RequireCompleteType, arg0: inst78000033, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004F:    {kind: RequireCompleteType, arg0: inst78000035, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000050:    {kind: NameRef, arg0: name2, arg1: inst78000027, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000051:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst7800002A)}
// CHECK:STDOUT:     inst78000052:    {kind: TupleLiteral, arg0: inst_block7800001A, type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst78000053:    {kind: RequireCompleteType, arg0: inst78000033, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000054:    {kind: TupleAccess, arg0: inst78000044, arg1: element0, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000055:    {kind: RequireCompleteType, arg0: inst7800001D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000056:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name78000003, type: type(TypeType)}
// CHECK:STDOUT:     inst78000057:    {kind: InterfaceDecl, arg0: interface78000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst78000058:    {kind: FacetType, arg0: declared_facet_type78000001, type: type(TypeType)}
// CHECK:STDOUT:     inst78000059:    {kind: InterfaceWithSelfDecl, arg0: interface78000000}
// CHECK:STDOUT:     inst7800005A:    {kind: SymbolicBinding, arg0: entity_name78000004, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800005B:    {kind: ImportRefLoaded, arg0: import_ir_inst2, arg1: entity_name<none>, type: type(inst7800007F)}
// CHECK:STDOUT:     inst7800005C:    {kind: ImportRefUnloaded, arg0: import_ir_inst3, arg1: entity_name78000005}
// CHECK:STDOUT:     inst7800005D:    {kind: ImportRefLoaded, arg0: import_ir_inst4, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800005E:    {kind: FunctionDecl, arg0: function78000001, arg1: inst_block_empty, type: type(symbolic_constant7800001F)}
// CHECK:STDOUT:     inst7800005F:    {kind: FunctionType, arg0: function78000001, arg1: specific78000001, type: type(TypeType)}
// CHECK:STDOUT:     inst78000060:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant7800001F)}
// CHECK:STDOUT:     inst78000061:    {kind: FacetAccessType, arg0: inst7800005A, type: type(TypeType)}
// CHECK:STDOUT:     inst78000062:    {kind: PatternType, arg0: inst78000061, type: type(TypeType)}
// CHECK:STDOUT:     inst78000063:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000064:    {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000065:    {kind: ReturnSlotPattern, arg0: inst78000063, arg1: inst<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000066:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000067:    {kind: ImportRefLoaded, arg0: import_ir_inst6, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000068:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000066, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000069:    {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800006A:    {kind: InitForm, arg0: inst78000061, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800006B:    {kind: ImportRefLoaded, arg0: import_ir_inst6, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800006C:    {kind: ImportRefLoaded, arg0: import_ir_inst7, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst7800006D:    {kind: ImportRefLoaded, arg0: import_ir_inst8, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst7800006E:    {kind: ImportRefLoaded, arg0: import_ir_inst9, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst7800006F:    {kind: ImportRefLoaded, arg0: import_ir_inst9, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst78000070:    {kind: ImportRefLoaded, arg0: import_ir_instA, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000071:    {kind: ImportRefLoaded, arg0: import_ir_instB, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000072:    {kind: ImportRefLoaded, arg0: import_ir_instC, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst78000073:    {kind: ImportRefLoaded, arg0: import_ir_inst4, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000074:    {kind: SymbolicBinding, arg0: entity_name78000004, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000075:    {kind: FacetAccessType, arg0: inst78000074, type: type(TypeType)}
// CHECK:STDOUT:     inst78000076:    {kind: PatternType, arg0: inst78000075, type: type(TypeType)}
// CHECK:STDOUT:     inst78000077:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant7800003A)}
// CHECK:STDOUT:     inst78000078:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000077, type: type(symbolic_constant7800003A)}
// CHECK:STDOUT:     inst78000079:    {kind: InitForm, arg0: inst78000075, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800007A:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant7800003A)}
// CHECK:STDOUT:     inst7800007B:    {kind: ReturnSlotPattern, arg0: inst7800007A, arg1: inst<none>, type: type(symbolic_constant7800003A)}
// CHECK:STDOUT:     inst7800007C:    {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800007D:    {kind: ImportRefLoaded, arg0: import_ir_inst6, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800007E:    {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name<none>}
// CHECK:STDOUT:     inst7800007F:    {kind: AssociatedEntityType, arg0: interface78000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000080:    {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(symbolic_constant78000025)}
// CHECK:STDOUT:     inst78000081:    {kind: AssociatedEntity, arg0: element0, arg1: inst78000080, type: type(inst7800007F)}
// CHECK:STDOUT:     inst78000082:    {kind: LookupImplWitness, arg0: inst7800001D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000083:    {kind: ImportRefUnloaded, arg0: import_ir_inst15, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000084:    {kind: ImplDecl, arg0: impl78000000, arg1: inst_block_empty}
// CHECK:STDOUT:     inst78000085:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000086:    {kind: FacetAccessType, arg0: inst78000085, type: type(TypeType)}
// CHECK:STDOUT:     inst78000087:    {kind: ConstType, arg0: inst78000086, type: type(TypeType)}
// CHECK:STDOUT:     inst78000088:    {kind: ImplSelfWitness, arg0: inst78000087, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000089:    {kind: ImportRefUnloaded, arg0: import_ir_inst17, arg1: entity_name<none>}
// CHECK:STDOUT:     inst7800008A:    {kind: ImplWitnessTable, arg0: inst_block78000024, arg1: impl78000000}
// CHECK:STDOUT:     inst7800008B:    {kind: ImplWitness, arg0: inst7800008A, arg1: specific78000003, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800008C:    {kind: FunctionDecl, arg0: function78000002, arg1: inst_block_empty, type: type(symbolic_constant78000048)}
// CHECK:STDOUT:     inst7800008D:    {kind: FunctionType, arg0: function78000002, arg1: specific78000003, type: type(TypeType)}
// CHECK:STDOUT:     inst7800008E:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000048)}
// CHECK:STDOUT:     inst7800008F:    {kind: PatternType, arg0: inst78000087, type: type(TypeType)}
// CHECK:STDOUT:     inst78000090:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst78000091:    {kind: PatternType, arg0: inst78000058, type: type(TypeType)}
// CHECK:STDOUT:     inst78000092:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000091)}
// CHECK:STDOUT:     inst78000093:    {kind: ImportRefLoaded, arg0: import_ir_inst19, arg1: entity_name<none>, type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst78000094:    {kind: ReturnSlotPattern, arg0: inst78000090, arg1: inst<none>, type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst78000095:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst78000096:    {kind: ImportRefLoaded, arg0: import_ir_inst1A, arg1: entity_name<none>, type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst78000097:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000095, type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst78000098:    {kind: ImportRefLoaded, arg0: import_ir_inst19, arg1: entity_name<none>, type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst78000099:    {kind: InitForm, arg0: inst78000087, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800009A:    {kind: ImportRefLoaded, arg0: import_ir_inst1A, arg1: entity_name<none>, type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst7800009B:    {kind: ImportRefLoaded, arg0: import_ir_inst1B, arg1: entity_name<none>, type: type(symbolic_constant78000051)}
// CHECK:STDOUT:     inst7800009C:    {kind: ImportRefLoaded, arg0: import_ir_inst1C, arg1: entity_name<none>, type: type(symbolic_constant78000051)}
// CHECK:STDOUT:     inst7800009D:    {kind: ImportRefLoaded, arg0: import_ir_inst1D, arg1: entity_name<none>, type: type(symbolic_constant78000051)}
// CHECK:STDOUT:     inst7800009E:    {kind: ImportRefLoaded, arg0: import_ir_inst1D, arg1: entity_name<none>, type: type(symbolic_constant78000051)}
// CHECK:STDOUT:     inst7800009F:    {kind: ImportRefLoaded, arg0: import_ir_inst1E, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A0:    {kind: ImportRefLoaded, arg0: import_ir_inst1F, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000A1:    {kind: ImportRefLoaded, arg0: import_ir_inst20, arg1: entity_name<none>, type: type(symbolic_constant78000051)}
// CHECK:STDOUT:     inst780000A2:    {kind: ImportRefLoaded, arg0: import_ir_inst21, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst780000A3:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst780000A4:    {kind: FacetAccessType, arg0: inst780000A3, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A5:    {kind: ConstType, arg0: inst780000A4, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A6:    {kind: PatternType, arg0: inst780000A5, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A7:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000066)}
// CHECK:STDOUT:     inst780000A8:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst780000A7, type: type(symbolic_constant78000066)}
// CHECK:STDOUT:     inst780000A9:    {kind: InitForm, arg0: inst780000A5, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000AA:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000066)}
// CHECK:STDOUT:     inst780000AB:    {kind: ReturnSlotPattern, arg0: inst780000AA, arg1: inst<none>, type: type(symbolic_constant78000066)}
// CHECK:STDOUT:     inst780000AC:    {kind: ImportRefLoaded, arg0: import_ir_inst19, arg1: entity_name<none>, type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst780000AD:    {kind: ImportRefLoaded, arg0: import_ir_inst1A, arg1: entity_name<none>, type: type(symbolic_constant7800004B)}
// CHECK:STDOUT:     inst780000AE:    {kind: ImportRefLoaded, arg0: import_ir_inst2B, arg1: entity_name<none>, type: type(inst78000091)}
// CHECK:STDOUT:     inst780000AF:    {kind: ImportRefLoaded, arg0: import_ir_inst2C, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B0:    {kind: ImportRefLoaded, arg0: import_ir_inst2D, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B1:    {kind: ImportRefLoaded, arg0: import_ir_inst2E, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000B2:    {kind: ImportRefLoaded, arg0: import_ir_inst21, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst780000B3:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000091)}
// CHECK:STDOUT:     inst780000B4:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst780000B5:    {kind: FacetAccessType, arg0: inst780000B4, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B6:    {kind: ConstType, arg0: inst780000B5, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B7:    {kind: ImplSelfWitness, arg0: inst780000B6, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000B8:    {kind: ImplWitness, arg0: inst7800008A, arg1: specific78000005, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000B9:    {kind: ImportRefUnloaded, arg0: import_ir_inst35, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000BA:    {kind: ImplDecl, arg0: impl78000001, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000BB:    {kind: ImplSelfWitness, arg0: inst(BoolType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000BC:    {kind: ImportRefLoaded, arg0: import_ir_inst37, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000BD:    {kind: ImportRefLoaded, arg0: import_ir_inst38, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000BE:    {kind: ImportRefLoaded, arg0: import_ir_inst39, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000BF:    {kind: ImportRefUnloaded, arg0: import_ir_inst3A, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000C0:    {kind: ImplDecl, arg0: impl78000002, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000C1:    {kind: ImplSelfWitness, arg0: inst(CharLiteralType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000C2:    {kind: ImportRefLoaded, arg0: import_ir_inst3C, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000C3:    {kind: ImportRefLoaded, arg0: import_ir_inst3D, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000C4:    {kind: ImportRefLoaded, arg0: import_ir_inst3E, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000C5:    {kind: ImportRefUnloaded, arg0: import_ir_inst3F, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000C6:    {kind: ImplDecl, arg0: impl78000003, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000C7:    {kind: ImplSelfWitness, arg0: inst(FloatLiteralType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000C8:    {kind: ImportRefLoaded, arg0: import_ir_inst41, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000C9:    {kind: ImportRefLoaded, arg0: import_ir_inst42, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000CA:    {kind: ImportRefLoaded, arg0: import_ir_inst43, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000CB:    {kind: ImportRefUnloaded, arg0: import_ir_inst44, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000CC:    {kind: ImplDecl, arg0: impl78000004, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000CD:    {kind: ImplSelfWitness, arg0: inst(IntLiteralType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000CE:    {kind: ImportRefLoaded, arg0: import_ir_inst46, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000CF:    {kind: ImportRefLoaded, arg0: import_ir_inst47, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D0:    {kind: ImportRefLoaded, arg0: import_ir_inst48, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000D1:    {kind: ImportRefLoaded, arg0: import_ir_inst49, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000D2:    {kind: ImplDecl, arg0: impl78000005, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000D3:    {kind: ImplSelfWitness, arg0: inst7800001D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000D4:    {kind: ImportRefUnloaded, arg0: import_ir_inst4B, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000D5:    {kind: ImplWitnessTable, arg0: inst_block78000031, arg1: impl78000005}
// CHECK:STDOUT:     inst780000D6:    {kind: ImplWitness, arg0: inst780000D5, arg1: specific78000006, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000D7:    {kind: ImportRefLoaded, arg0: import_ir_inst4D, arg1: entity_name<none>, type: type(inst78000014)}
// CHECK:STDOUT:     inst780000D8:    {kind: ImportRefLoaded, arg0: import_ir_inst4E, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D9:    {kind: ImportRefLoaded, arg0: import_ir_inst4F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DA:    {kind: ImportRefLoaded, arg0: import_ir_inst50, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000DB:    {kind: ImportRefLoaded, arg0: import_ir_inst51, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DC:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000014)}
// CHECK:STDOUT:     inst780000DD:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DE:    {kind: PointerType, arg0: inst780000DD, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DF:    {kind: ImplSelfWitness, arg0: inst780000DE, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000E0:    {kind: ImplWitness, arg0: inst780000D5, arg1: specific78000007, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000E1:    {kind: ImportRefUnloaded, arg0: import_ir_inst57, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000E2:    {kind: ImplDecl, arg0: impl78000006, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000E3:    {kind: ImplSelfWitness, arg0: inst(TypeType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000E4:    {kind: ImportRefLoaded, arg0: import_ir_inst59, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E5:    {kind: ImportRefLoaded, arg0: import_ir_inst5A, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E6:    {kind: ImportRefLoaded, arg0: import_ir_inst5B, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000E7:    {kind: ImportRefUnloaded, arg0: import_ir_inst5C, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000E8:    {kind: ImplDecl, arg0: impl78000007, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000E9:    {kind: ImplSelfWitness, arg0: inst7800002A, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000EA:    {kind: ImportRefLoaded, arg0: import_ir_inst5E, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000EB:    {kind: ImportRefLoaded, arg0: import_ir_inst5F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000EC:    {kind: ImportRefLoaded, arg0: import_ir_inst60, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000ED:    {kind: ImportRefUnloaded, arg0: import_ir_inst61, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000EE:    {kind: ImplDecl, arg0: impl78000008, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000EF:    {kind: SymbolicBinding, arg0: entity_name7800001B, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst780000F0:    {kind: FacetAccessType, arg0: inst780000EF, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F1:    {kind: TupleType, arg0: inst_block78000037, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F2:    {kind: ImplSelfWitness, arg0: inst780000F1, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000F3:    {kind: ImportRefUnloaded, arg0: import_ir_inst63, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000F4:    {kind: ImplWitnessTable, arg0: inst_block78000038, arg1: impl78000008}
// CHECK:STDOUT:     inst780000F5:    {kind: ImplWitness, arg0: inst780000F4, arg1: specific78000008, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000F6:    {kind: FunctionDecl, arg0: function78000003, arg1: inst_block_empty, type: type(symbolic_constant78000090)}
// CHECK:STDOUT:     inst780000F7:    {kind: FunctionType, arg0: function78000003, arg1: specific78000008, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F8:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000090)}
// CHECK:STDOUT:     inst780000F9:    {kind: PatternType, arg0: inst780000F1, type: type(TypeType)}
// CHECK:STDOUT:     inst780000FA:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst780000FB:    {kind: TupleType, arg0: inst_block7800003A, type: type(TypeType)}
// CHECK:STDOUT:     inst780000FC:    {kind: TupleValue, arg0: inst_block78000039, type: type(inst780000FB)}
// CHECK:STDOUT:     inst780000FD:    {kind: SymbolicBindingPattern, arg0: entity_name7800001B, type: type(inst78000091)}
// CHECK:STDOUT:     inst780000FE:    {kind: ImportRefLoaded, arg0: import_ir_inst65, arg1: entity_name<none>, type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst780000FF:    {kind: ReturnSlotPattern, arg0: inst780000FA, arg1: inst<none>, type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst78000100:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst78000101:    {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst78000102:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000100, type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst78000103:    {kind: ImportRefLoaded, arg0: import_ir_inst65, arg1: entity_name<none>, type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst78000104:    {kind: InitForm, arg0: inst780000F1, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000105:    {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst78000106:    {kind: ImportRefLoaded, arg0: import_ir_inst67, arg1: entity_name<none>, type: type(symbolic_constant7800009A)}
// CHECK:STDOUT:     inst78000107:    {kind: ImportRefLoaded, arg0: import_ir_inst68, arg1: entity_name<none>, type: type(symbolic_constant7800009A)}
// CHECK:STDOUT:     inst78000108:    {kind: ImportRefLoaded, arg0: import_ir_inst69, arg1: entity_name<none>, type: type(symbolic_constant7800009A)}
// CHECK:STDOUT:     inst78000109:    {kind: ImportRefLoaded, arg0: import_ir_inst69, arg1: entity_name<none>, type: type(symbolic_constant7800009A)}
// CHECK:STDOUT:     inst7800010A:    {kind: ImportRefLoaded, arg0: import_ir_inst6A, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800010B:    {kind: ImportRefLoaded, arg0: import_ir_inst6B, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800010C:    {kind: ImportRefLoaded, arg0: import_ir_inst6C, arg1: entity_name<none>, type: type(symbolic_constant7800009A)}
// CHECK:STDOUT:     inst7800010D:    {kind: ImportRefLoaded, arg0: import_ir_inst6D, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800010E:    {kind: ImportRefLoaded, arg0: import_ir_inst6E, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800010F:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000110:    {kind: FacetAccessType, arg0: inst7800010F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000111:    {kind: SymbolicBinding, arg0: entity_name7800001B, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000112:    {kind: FacetAccessType, arg0: inst78000111, type: type(TypeType)}
// CHECK:STDOUT:     inst78000113:    {kind: TupleType, arg0: inst_block78000040, type: type(TypeType)}
// CHECK:STDOUT:     inst78000114:    {kind: PatternType, arg0: inst78000113, type: type(TypeType)}
// CHECK:STDOUT:     inst78000115:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst78000116:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000115, type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst78000117:    {kind: InitForm, arg0: inst78000113, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000118:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst78000119:    {kind: ReturnSlotPattern, arg0: inst78000118, arg1: inst<none>, type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst7800011A:    {kind: ImportRefLoaded, arg0: import_ir_inst65, arg1: entity_name<none>, type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst7800011B:    {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(symbolic_constant78000093)}
// CHECK:STDOUT:     inst7800011C:    {kind: ImportRefLoaded, arg0: import_ir_inst7A, arg1: entity_name<none>, type: type(inst78000091)}
// CHECK:STDOUT:     inst7800011D:    {kind: ImportRefLoaded, arg0: import_ir_inst7B, arg1: entity_name<none>, type: type(inst78000091)}
// CHECK:STDOUT:     inst7800011E:    {kind: ImportRefLoaded, arg0: import_ir_inst7C, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800011F:    {kind: ImportRefLoaded, arg0: import_ir_inst7D, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000120:    {kind: ImportRefLoaded, arg0: import_ir_inst7E, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000121:    {kind: ImportRefLoaded, arg0: import_ir_inst6D, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000122:    {kind: ImportRefLoaded, arg0: import_ir_inst6E, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000123:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000091)}
// CHECK:STDOUT:     inst78000124:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000125:    {kind: SymbolicBindingPattern, arg0: entity_name7800001B, type: type(inst78000091)}
// CHECK:STDOUT:     inst78000126:    {kind: SymbolicBinding, arg0: entity_name7800001B, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000127:    {kind: TupleValue, arg0: inst_block78000045, type: type(inst780000FB)}
// CHECK:STDOUT:     inst78000128:    {kind: FacetAccessType, arg0: inst78000124, type: type(TypeType)}
// CHECK:STDOUT:     inst78000129:    {kind: FacetAccessType, arg0: inst78000126, type: type(TypeType)}
// CHECK:STDOUT:     inst7800012A:    {kind: TupleType, arg0: inst_block78000046, type: type(TypeType)}
// CHECK:STDOUT:     inst7800012B:    {kind: ImplSelfWitness, arg0: inst7800012A, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800012C:    {kind: ImplWitness, arg0: inst780000F4, arg1: specific7800000A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800012D:    {kind: ImportRefUnloaded, arg0: import_ir_inst89, arg1: entity_name<none>}
// CHECK:STDOUT:     inst7800012E:    {kind: ImplDecl, arg0: impl78000009, arg1: inst_block_empty}
// CHECK:STDOUT:     inst7800012F:    {kind: SymbolicBinding, arg0: entity_name7800002A, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000130:    {kind: FacetAccessType, arg0: inst7800012F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000131:    {kind: TupleType, arg0: inst_block78000049, type: type(TypeType)}
// CHECK:STDOUT:     inst78000132:    {kind: ImplSelfWitness, arg0: inst78000131, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000133:    {kind: ImportRefUnloaded, arg0: import_ir_inst8B, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000134:    {kind: ImplWitnessTable, arg0: inst_block7800004A, arg1: impl78000009}
// CHECK:STDOUT:     inst78000135:    {kind: ImplWitness, arg0: inst78000134, arg1: specific7800000B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000136:    {kind: FunctionDecl, arg0: function78000004, arg1: inst_block_empty, type: type(symbolic_constant780000D7)}
// CHECK:STDOUT:     inst78000137:    {kind: FunctionType, arg0: function78000004, arg1: specific7800000B, type: type(TypeType)}
// CHECK:STDOUT:     inst78000138:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780000D7)}
// CHECK:STDOUT:     inst78000139:    {kind: PatternType, arg0: inst78000131, type: type(TypeType)}
// CHECK:STDOUT:     inst7800013A:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst7800013B:    {kind: TupleType, arg0: inst_block7800004C, type: type(TypeType)}
// CHECK:STDOUT:     inst7800013C:    {kind: TupleValue, arg0: inst_block7800004B, type: type(inst7800013B)}
// CHECK:STDOUT:     inst7800013D:    {kind: SymbolicBindingPattern, arg0: entity_name7800002A, type: type(inst78000091)}
// CHECK:STDOUT:     inst7800013E:    {kind: ImportRefLoaded, arg0: import_ir_inst8D, arg1: entity_name<none>, type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst7800013F:    {kind: ReturnSlotPattern, arg0: inst7800013A, arg1: inst<none>, type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst78000140:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst78000141:    {kind: ImportRefLoaded, arg0: import_ir_inst8E, arg1: entity_name<none>, type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst78000142:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000140, type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst78000143:    {kind: ImportRefLoaded, arg0: import_ir_inst8D, arg1: entity_name<none>, type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst78000144:    {kind: InitForm, arg0: inst78000131, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000145:    {kind: ImportRefLoaded, arg0: import_ir_inst8E, arg1: entity_name<none>, type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst78000146:    {kind: ImportRefLoaded, arg0: import_ir_inst8F, arg1: entity_name<none>, type: type(symbolic_constant780000E1)}
// CHECK:STDOUT:     inst78000147:    {kind: ImportRefLoaded, arg0: import_ir_inst90, arg1: entity_name<none>, type: type(symbolic_constant780000E1)}
// CHECK:STDOUT:     inst78000148:    {kind: ImportRefLoaded, arg0: import_ir_inst91, arg1: entity_name<none>, type: type(symbolic_constant780000E1)}
// CHECK:STDOUT:     inst78000149:    {kind: ImportRefLoaded, arg0: import_ir_inst91, arg1: entity_name<none>, type: type(symbolic_constant780000E1)}
// CHECK:STDOUT:     inst7800014A:    {kind: ImportRefLoaded, arg0: import_ir_inst92, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800014B:    {kind: ImportRefLoaded, arg0: import_ir_inst93, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800014C:    {kind: ImportRefLoaded, arg0: import_ir_inst94, arg1: entity_name<none>, type: type(symbolic_constant780000E1)}
// CHECK:STDOUT:     inst7800014D:    {kind: ImportRefLoaded, arg0: import_ir_inst95, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800014E:    {kind: ImportRefLoaded, arg0: import_ir_inst96, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800014F:    {kind: ImportRefLoaded, arg0: import_ir_inst97, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000150:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000151:    {kind: FacetAccessType, arg0: inst78000150, type: type(TypeType)}
// CHECK:STDOUT:     inst78000152:    {kind: SymbolicBinding, arg0: entity_name7800001B, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000153:    {kind: FacetAccessType, arg0: inst78000152, type: type(TypeType)}
// CHECK:STDOUT:     inst78000154:    {kind: SymbolicBinding, arg0: entity_name7800002A, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000155:    {kind: FacetAccessType, arg0: inst78000154, type: type(TypeType)}
// CHECK:STDOUT:     inst78000156:    {kind: TupleType, arg0: inst_block78000052, type: type(TypeType)}
// CHECK:STDOUT:     inst78000157:    {kind: PatternType, arg0: inst78000156, type: type(TypeType)}
// CHECK:STDOUT:     inst78000158:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000100)}
// CHECK:STDOUT:     inst78000159:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000158, type: type(symbolic_constant78000100)}
// CHECK:STDOUT:     inst7800015A:    {kind: InitForm, arg0: inst78000156, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800015B:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000100)}
// CHECK:STDOUT:     inst7800015C:    {kind: ReturnSlotPattern, arg0: inst7800015B, arg1: inst<none>, type: type(symbolic_constant78000100)}
// CHECK:STDOUT:     inst7800015D:    {kind: ImportRefLoaded, arg0: import_ir_inst8D, arg1: entity_name<none>, type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst7800015E:    {kind: ImportRefLoaded, arg0: import_ir_inst8E, arg1: entity_name<none>, type: type(symbolic_constant780000DA)}
// CHECK:STDOUT:     inst7800015F:    {kind: ImportRefLoaded, arg0: import_ir_instA5, arg1: entity_name<none>, type: type(inst78000091)}
// CHECK:STDOUT:     inst78000160:    {kind: ImportRefLoaded, arg0: import_ir_instA6, arg1: entity_name<none>, type: type(inst78000091)}
// CHECK:STDOUT:     inst78000161:    {kind: ImportRefLoaded, arg0: import_ir_instA7, arg1: entity_name<none>, type: type(inst78000091)}
// CHECK:STDOUT:     inst78000162:    {kind: ImportRefLoaded, arg0: import_ir_instA8, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000163:    {kind: ImportRefLoaded, arg0: import_ir_instA9, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000164:    {kind: ImportRefLoaded, arg0: import_ir_instAA, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000165:    {kind: ImportRefLoaded, arg0: import_ir_inst95, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000166:    {kind: ImportRefLoaded, arg0: import_ir_inst96, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000167:    {kind: ImportRefLoaded, arg0: import_ir_inst97, arg1: entity_name<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000168:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000091)}
// CHECK:STDOUT:     inst78000169:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800016A:    {kind: SymbolicBindingPattern, arg0: entity_name7800001B, type: type(inst78000091)}
// CHECK:STDOUT:     inst7800016B:    {kind: SymbolicBinding, arg0: entity_name7800001B, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800016C:    {kind: SymbolicBindingPattern, arg0: entity_name7800002A, type: type(inst78000091)}
// CHECK:STDOUT:     inst7800016D:    {kind: SymbolicBinding, arg0: entity_name7800002A, arg1: inst<none>, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800016E:    {kind: TupleValue, arg0: inst_block78000057, type: type(inst7800013B)}
// CHECK:STDOUT:     inst7800016F:    {kind: FacetAccessType, arg0: inst78000169, type: type(TypeType)}
// CHECK:STDOUT:     inst78000170:    {kind: FacetAccessType, arg0: inst7800016B, type: type(TypeType)}
// CHECK:STDOUT:     inst78000171:    {kind: FacetAccessType, arg0: inst7800016D, type: type(TypeType)}
// CHECK:STDOUT:     inst78000172:    {kind: TupleType, arg0: inst_block78000058, type: type(TypeType)}
// CHECK:STDOUT:     inst78000173:    {kind: ImplSelfWitness, arg0: inst78000172, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000174:    {kind: ImplWitness, arg0: inst78000134, arg1: specific7800000D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000175:    {kind: LookupImplWitness, arg0: inst7800001D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000176:    {kind: FacetValue, arg0: inst7800001D, arg1: inst_block7800005B, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000177:    {kind: FunctionType, arg0: function78000001, arg1: specific7800000E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000178:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000125)}
// CHECK:STDOUT:     inst78000179:    {kind: FunctionTypeWithSelfType, arg0: inst78000177, arg1: inst78000176, type: type(TypeType)}
// CHECK:STDOUT:     inst7800017A:    {kind: ImplWitnessAccess, arg0: inst78000175, arg1: element0, type: type(symbolic_constant7800012C)}
// CHECK:STDOUT:     inst7800017B:    {kind: ImplWitnessAccess, arg0: inst78000175, arg1: element0, type: type(symbolic_constant78000127)}
// CHECK:STDOUT:     inst7800017C:    {kind: LookupImplWitness, arg0: inst7800001E, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800017D:    {kind: FacetValue, arg0: inst7800001E, arg1: inst_block7800005E, type: type(inst78000058)}
// CHECK:STDOUT:     inst7800017E:    {kind: FunctionType, arg0: function78000001, arg1: specific7800000F, type: type(TypeType)}
// CHECK:STDOUT:     inst7800017F:    {kind: FunctionTypeWithSelfType, arg0: inst7800017E, arg1: inst7800017D, type: type(TypeType)}
// CHECK:STDOUT:     inst78000180:    {kind: ImplWitnessAccess, arg0: inst7800017C, arg1: element0, type: type(symbolic_constant7800012C)}
// CHECK:STDOUT:     inst78000181:    {kind: BoundMethod, arg0: inst78000050, arg1: inst7800017A, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst78000182:    {kind: LookupImplWitness, arg0: inst7800001D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000183:    {kind: FacetValue, arg0: inst7800001D, arg1: inst_block7800005B, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000184:    {kind: Converted, arg0: inst7800001D, arg1: inst78000183, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000185:    {kind: LookupImplWitness, arg0: inst7800001D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000186:    {kind: FacetValue, arg0: inst7800001D, arg1: inst_block7800005B, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000187:    {kind: Converted, arg0: inst7800001D, arg1: inst78000186, type: type(inst78000058)}
// CHECK:STDOUT:     inst78000188:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000189:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000188, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst7800018A:    {kind: InitForm, arg0: inst7800001D, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800018B:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst7800018C:    {kind: ReturnSlotPattern, arg0: inst7800018B, arg1: inst<none>, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst7800018D:    {kind: SpecificImplFunction, arg0: inst7800017A, arg1: specific78000010, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst7800018E:    {kind: SpecificImplFunction, arg0: inst7800017B, arg1: specific78000010, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst7800018F:    {kind: SpecificImplFunction, arg0: inst78000180, arg1: specific78000011, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000190:    {kind: BoundMethod, arg0: inst78000050, arg1: inst7800018D, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst78000191:    {kind: Call, arg0: inst78000190, arg1: inst_block78000062, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000192:    {kind: InPlaceInit, arg0: inst78000191, arg1: inst78000054, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000193:    {kind: TupleAccess, arg0: inst78000044, arg1: element1, type: type(inst7800002A)}
// CHECK:STDOUT:     inst78000194:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst78000193, type: type(inst7800002A)}
// CHECK:STDOUT:     inst78000195:    {kind: Converted, arg0: inst78000051, arg1: inst78000194, type: type(inst7800002A)}
// CHECK:STDOUT:     inst78000196:    {kind: InPlaceInit, arg0: inst78000195, arg1: inst78000193, type: type(inst7800002A)}
// CHECK:STDOUT:     inst78000197:    {kind: TupleInit, arg0: inst_block78000063, arg1: inst78000044, type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst78000198:    {kind: Converted, arg0: inst78000052, arg1: inst78000197, type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst78000199:    {kind: ReturnExpr, arg0: inst78000198, arg1: inst78000044}
// CHECK:STDOUT:   bundles:         {}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       'inst(TypeType)':  concrete_constant(inst(TypeType))
// CHECK:STDOUT:       'inst(Package)':   concrete_constant(inst(Package))
// CHECK:STDOUT:       inst78000010:    concrete_constant(inst78000010)
// CHECK:STDOUT:       inst78000011:    symbolic_constant78000000
// CHECK:STDOUT:       inst78000012:    symbolic_constant78000000
// CHECK:STDOUT:       inst78000013:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst78000014:    concrete_constant(inst78000014)
// CHECK:STDOUT:       inst78000015:    symbolic_constant78000002
// CHECK:STDOUT:       inst78000016:    symbolic_constant78000001
// CHECK:STDOUT:       inst78000017:    symbolic_constant78000002
// CHECK:STDOUT:       inst78000018:    symbolic_constant78000004
// CHECK:STDOUT:       inst78000019:    symbolic_constant78000003
// CHECK:STDOUT:       inst7800001A:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800001B:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800001C:    symbolic_constant78000006
// CHECK:STDOUT:       inst7800001D:    symbolic_constant78000005
// CHECK:STDOUT:       inst7800001E:    symbolic_constant78000006
// CHECK:STDOUT:       inst7800001F:    symbolic_constant78000007
// CHECK:STDOUT:       inst78000020:    symbolic_constant7800000A
// CHECK:STDOUT:       inst78000021:    symbolic_constant78000008
// CHECK:STDOUT:       inst78000022:    symbolic_constant78000009
// CHECK:STDOUT:       inst78000023:    symbolic_constant7800000A
// CHECK:STDOUT:       inst78000024:    symbolic_constant7800000C
// CHECK:STDOUT:       inst78000025:    symbolic_constant7800000B
// CHECK:STDOUT:       inst78000026:    symbolic_constant7800000C
// CHECK:STDOUT:       inst78000028:    symbolic_constant78000004
// CHECK:STDOUT:       inst78000029:    symbolic_constant78000006
// CHECK:STDOUT:       inst7800002A:    concrete_constant(inst7800002A)
// CHECK:STDOUT:       inst7800002B:    concrete_constant(inst7800002C)
// CHECK:STDOUT:       inst7800002C:    concrete_constant(inst7800002C)
// CHECK:STDOUT:       inst7800002D:    concrete_constant(inst7800002D)
// CHECK:STDOUT:       inst7800002E:    symbolic_constant7800000E
// CHECK:STDOUT:       inst7800002F:    symbolic_constant7800000D
// CHECK:STDOUT:       inst78000030:    symbolic_constant7800000E
// CHECK:STDOUT:       inst78000031:    concrete_constant(inst78000031)
// CHECK:STDOUT:       inst78000032:    concrete_constant(inst7800002A)
// CHECK:STDOUT:       inst78000033:    symbolic_constant7800000F
// CHECK:STDOUT:       inst78000034:    symbolic_constant78000010
// CHECK:STDOUT:       inst78000035:    symbolic_constant78000010
// CHECK:STDOUT:       inst78000036:    symbolic_constant78000012
// CHECK:STDOUT:       inst78000037:    symbolic_constant78000011
// CHECK:STDOUT:       inst78000038:    symbolic_constant78000012
// CHECK:STDOUT:       inst78000039:    symbolic_constant78000013
// CHECK:STDOUT:       inst7800003A:    symbolic_constant78000016
// CHECK:STDOUT:       inst7800003B:    symbolic_constant78000014
// CHECK:STDOUT:       inst7800003C:    symbolic_constant78000015
// CHECK:STDOUT:       inst7800003D:    symbolic_constant78000016
// CHECK:STDOUT:       inst7800003E:    symbolic_constant78000018
// CHECK:STDOUT:       inst7800003F:    symbolic_constant78000017
// CHECK:STDOUT:       inst78000040:    symbolic_constant78000018
// CHECK:STDOUT:       inst78000041:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst78000043:    symbolic_constant78000006
// CHECK:STDOUT:       inst78000046:    concrete_constant(inst78000048)
// CHECK:STDOUT:       inst78000047:    concrete_constant(inst78000047)
// CHECK:STDOUT:       inst78000048:    concrete_constant(inst78000048)
// CHECK:STDOUT:       inst78000049:    symbolic_constant7800001A
// CHECK:STDOUT:       inst7800004A:    symbolic_constant78000019
// CHECK:STDOUT:       inst7800004B:    symbolic_constant7800001A
// CHECK:STDOUT:       inst7800004C:    symbolic_constant7800001B
// CHECK:STDOUT:       inst7800004D:    symbolic_constant7800001D
// CHECK:STDOUT:       inst7800004E:    symbolic_constant7800001C
// CHECK:STDOUT:       inst7800004F:    symbolic_constant7800001D
// CHECK:STDOUT:       inst78000051:    concrete_constant(inst7800002C)
// CHECK:STDOUT:       inst78000053:    symbolic_constant7800001D
// CHECK:STDOUT:       inst78000055:    symbolic_constant7800001A
// CHECK:STDOUT:       inst78000056:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst78000057:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst78000058:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst78000059:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst7800005A:    symbolic_constant7800001E
// CHECK:STDOUT:       inst7800005B:    concrete_constant(inst78000081)
// CHECK:STDOUT:       inst7800005C:    constant<none>
// CHECK:STDOUT:       inst7800005D:    symbolic_constant7800001E
// CHECK:STDOUT:       inst7800005E:    symbolic_constant78000021
// CHECK:STDOUT:       inst7800005F:    symbolic_constant7800001F
// CHECK:STDOUT:       inst78000060:    symbolic_constant78000020
// CHECK:STDOUT:       inst78000061:    symbolic_constant78000022
// CHECK:STDOUT:       inst78000062:    symbolic_constant78000023
// CHECK:STDOUT:       inst78000063:    symbolic_constant78000024
// CHECK:STDOUT:       inst78000064:    symbolic_constant78000024
// CHECK:STDOUT:       inst78000065:    symbolic_constant78000026
// CHECK:STDOUT:       inst78000066:    symbolic_constant78000029
// CHECK:STDOUT:       inst78000067:    symbolic_constant78000029
// CHECK:STDOUT:       inst78000068:    symbolic_constant7800002A
// CHECK:STDOUT:       inst78000069:    symbolic_constant78000024
// CHECK:STDOUT:       inst7800006A:    symbolic_constant7800002E
// CHECK:STDOUT:       inst7800006B:    symbolic_constant78000029
// CHECK:STDOUT:       inst7800006C:    symbolic_constant78000037
// CHECK:STDOUT:       inst7800006D:    symbolic_constant78000036
// CHECK:STDOUT:       inst7800006E:    symbolic_constant7800002B
// CHECK:STDOUT:       inst7800006F:    symbolic_constant7800002B
// CHECK:STDOUT:       inst78000070:    symbolic_constant78000035
// CHECK:STDOUT:       inst78000071:    symbolic_constant78000034
// CHECK:STDOUT:       inst78000072:    symbolic_constant78000027
// CHECK:STDOUT:       inst78000073:    symbolic_constant7800001E
// CHECK:STDOUT:       inst78000074:    symbolic_constant78000038
// CHECK:STDOUT:       inst78000075:    symbolic_constant78000039
// CHECK:STDOUT:       inst78000076:    symbolic_constant7800003A
// CHECK:STDOUT:       inst78000077:    symbolic_constant7800003B
// CHECK:STDOUT:       inst78000078:    symbolic_constant7800003C
// CHECK:STDOUT:       inst78000079:    symbolic_constant7800003D
// CHECK:STDOUT:       inst7800007A:    symbolic_constant7800003E
// CHECK:STDOUT:       inst7800007B:    symbolic_constant7800003F
// CHECK:STDOUT:       inst7800007C:    symbolic_constant78000024
// CHECK:STDOUT:       inst7800007D:    symbolic_constant78000029
// CHECK:STDOUT:       inst7800007E:    constant<none>
// CHECK:STDOUT:       inst7800007F:    concrete_constant(inst7800007F)
// CHECK:STDOUT:       inst78000080:    symbolic_constant78000021
// CHECK:STDOUT:       inst78000081:    concrete_constant(inst78000081)
// CHECK:STDOUT:       inst78000083:    constant<none>
// CHECK:STDOUT:       inst78000084:    concrete_constant(inst78000084)
// CHECK:STDOUT:       inst78000085:    symbolic_constant78000040
// CHECK:STDOUT:       inst78000086:    symbolic_constant78000041
// CHECK:STDOUT:       inst78000087:    symbolic_constant78000042
// CHECK:STDOUT:       inst78000088:    symbolic_constant78000043
// CHECK:STDOUT:       inst78000089:    constant<none>
// CHECK:STDOUT:       inst7800008A:    concrete_constant(inst7800008A)
// CHECK:STDOUT:       inst7800008B:    symbolic_constant78000046
// CHECK:STDOUT:       inst7800008C:    symbolic_constant7800004A
// CHECK:STDOUT:       inst7800008D:    symbolic_constant78000048
// CHECK:STDOUT:       inst7800008E:    symbolic_constant78000049
// CHECK:STDOUT:       inst7800008F:    symbolic_constant7800004B
// CHECK:STDOUT:       inst78000090:    symbolic_constant7800004C
// CHECK:STDOUT:       inst78000091:    concrete_constant(inst78000091)
// CHECK:STDOUT:       inst78000092:    symbolic_constant7800004E
// CHECK:STDOUT:       inst78000093:    symbolic_constant7800004C
// CHECK:STDOUT:       inst78000094:    symbolic_constant7800004F
// CHECK:STDOUT:       inst78000095:    symbolic_constant78000052
// CHECK:STDOUT:       inst78000096:    symbolic_constant78000052
// CHECK:STDOUT:       inst78000097:    symbolic_constant78000053
// CHECK:STDOUT:       inst78000098:    symbolic_constant7800004C
// CHECK:STDOUT:       inst78000099:    symbolic_constant78000057
// CHECK:STDOUT:       inst7800009A:    symbolic_constant78000052
// CHECK:STDOUT:       inst7800009B:    symbolic_constant78000062
// CHECK:STDOUT:       inst7800009C:    symbolic_constant78000061
// CHECK:STDOUT:       inst7800009D:    symbolic_constant78000054
// CHECK:STDOUT:       inst7800009E:    symbolic_constant78000054
// CHECK:STDOUT:       inst7800009F:    symbolic_constant78000060
// CHECK:STDOUT:       inst780000A0:    symbolic_constant7800005F
// CHECK:STDOUT:       inst780000A1:    symbolic_constant78000050
// CHECK:STDOUT:       inst780000A2:    symbolic_constant7800005E
// CHECK:STDOUT:       inst780000A3:    symbolic_constant78000063
// CHECK:STDOUT:       inst780000A4:    symbolic_constant78000064
// CHECK:STDOUT:       inst780000A5:    symbolic_constant78000065
// CHECK:STDOUT:       inst780000A6:    symbolic_constant78000066
// CHECK:STDOUT:       inst780000A7:    symbolic_constant78000067
// CHECK:STDOUT:       inst780000A8:    symbolic_constant78000068
// CHECK:STDOUT:       inst780000A9:    symbolic_constant78000069
// CHECK:STDOUT:       inst780000AA:    symbolic_constant7800006A
// CHECK:STDOUT:       inst780000AB:    symbolic_constant7800006B
// CHECK:STDOUT:       inst780000AC:    symbolic_constant7800004C
// CHECK:STDOUT:       inst780000AD:    symbolic_constant78000052
// CHECK:STDOUT:       inst780000AE:    symbolic_constant78000071
// CHECK:STDOUT:       inst780000AF:    symbolic_constant78000045
// CHECK:STDOUT:       inst780000B0:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst780000B1:    symbolic_constant78000044
// CHECK:STDOUT:       inst780000B2:    symbolic_constant7800005E
// CHECK:STDOUT:       inst780000B3:    symbolic_constant78000072
// CHECK:STDOUT:       inst780000B4:    symbolic_constant78000073
// CHECK:STDOUT:       inst780000B5:    symbolic_constant78000074
// CHECK:STDOUT:       inst780000B6:    symbolic_constant78000075
// CHECK:STDOUT:       inst780000B7:    symbolic_constant78000076
// CHECK:STDOUT:       inst780000B8:    symbolic_constant78000077
// CHECK:STDOUT:       inst780000B9:    constant<none>
// CHECK:STDOUT:       inst780000BA:    concrete_constant(inst780000BA)
// CHECK:STDOUT:       inst780000BB:    concrete_constant(inst780000BB)
// CHECK:STDOUT:       inst780000BC:    concrete_constant(inst(BoolType))
// CHECK:STDOUT:       inst780000BD:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst780000BE:    concrete_constant(inst780000BB)
// CHECK:STDOUT:       inst780000BF:    constant<none>
// CHECK:STDOUT:       inst780000C0:    concrete_constant(inst780000C0)
// CHECK:STDOUT:       inst780000C1:    concrete_constant(inst780000C1)
// CHECK:STDOUT:       inst780000C2:    concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       inst780000C3:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst780000C4:    concrete_constant(inst780000C1)
// CHECK:STDOUT:       inst780000C5:    constant<none>
// CHECK:STDOUT:       inst780000C6:    concrete_constant(inst780000C6)
// CHECK:STDOUT:       inst780000C7:    concrete_constant(inst780000C7)
// CHECK:STDOUT:       inst780000C8:    concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       inst780000C9:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst780000CA:    concrete_constant(inst780000C7)
// CHECK:STDOUT:       inst780000CB:    constant<none>
// CHECK:STDOUT:       inst780000CC:    concrete_constant(inst780000CC)
// CHECK:STDOUT:       inst780000CD:    concrete_constant(inst780000CD)
// CHECK:STDOUT:       inst780000CE:    concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       inst780000CF:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst780000D0:    concrete_constant(inst780000CD)
// CHECK:STDOUT:       inst780000D1:    symbolic_constant78000123
// CHECK:STDOUT:       inst780000D2:    concrete_constant(inst780000D2)
// CHECK:STDOUT:       inst780000D3:    symbolic_constant78000078
// CHECK:STDOUT:       inst780000D4:    constant<none>
// CHECK:STDOUT:       inst780000D5:    concrete_constant(inst780000D5)
// CHECK:STDOUT:       inst780000D6:    symbolic_constant7800007B
// CHECK:STDOUT:       inst780000D7:    symbolic_constant78000082
// CHECK:STDOUT:       inst780000D8:    symbolic_constant7800007A
// CHECK:STDOUT:       inst780000D9:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst780000DA:    symbolic_constant78000079
// CHECK:STDOUT:       inst780000DB:    symbolic_constant78000081
// CHECK:STDOUT:       inst780000DC:    symbolic_constant78000083
// CHECK:STDOUT:       inst780000DD:    symbolic_constant78000084
// CHECK:STDOUT:       inst780000DE:    symbolic_constant78000085
// CHECK:STDOUT:       inst780000DF:    symbolic_constant78000086
// CHECK:STDOUT:       inst780000E0:    symbolic_constant78000087
// CHECK:STDOUT:       inst780000E1:    constant<none>
// CHECK:STDOUT:       inst780000E2:    concrete_constant(inst780000E2)
// CHECK:STDOUT:       inst780000E3:    concrete_constant(inst780000E3)
// CHECK:STDOUT:       inst780000E4:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst780000E5:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst780000E6:    concrete_constant(inst780000E3)
// CHECK:STDOUT:       inst780000E7:    constant<none>
// CHECK:STDOUT:       inst780000E8:    concrete_constant(inst780000E8)
// CHECK:STDOUT:       inst780000E9:    concrete_constant(inst780000E9)
// CHECK:STDOUT:       inst780000EA:    concrete_constant(inst7800002A)
// CHECK:STDOUT:       inst780000EB:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst780000EC:    concrete_constant(inst780000E9)
// CHECK:STDOUT:       inst780000ED:    constant<none>
// CHECK:STDOUT:       inst780000EE:    concrete_constant(inst780000EE)
// CHECK:STDOUT:       inst780000EF:    symbolic_constant78000088
// CHECK:STDOUT:       inst780000F0:    symbolic_constant78000089
// CHECK:STDOUT:       inst780000F1:    symbolic_constant7800008A
// CHECK:STDOUT:       inst780000F2:    symbolic_constant7800008B
// CHECK:STDOUT:       inst780000F3:    constant<none>
// CHECK:STDOUT:       inst780000F4:    concrete_constant(inst780000F4)
// CHECK:STDOUT:       inst780000F5:    symbolic_constant7800008E
// CHECK:STDOUT:       inst780000F6:    symbolic_constant78000092
// CHECK:STDOUT:       inst780000F7:    symbolic_constant78000090
// CHECK:STDOUT:       inst780000F8:    symbolic_constant78000091
// CHECK:STDOUT:       inst780000F9:    symbolic_constant78000093
// CHECK:STDOUT:       inst780000FA:    symbolic_constant78000094
// CHECK:STDOUT:       inst780000FB:    concrete_constant(inst780000FB)
// CHECK:STDOUT:       inst780000FC:    symbolic_constant78000096
// CHECK:STDOUT:       inst780000FD:    symbolic_constant78000097
// CHECK:STDOUT:       inst780000FE:    symbolic_constant78000094
// CHECK:STDOUT:       inst780000FF:    symbolic_constant78000098
// CHECK:STDOUT:       inst78000100:    symbolic_constant7800009B
// CHECK:STDOUT:       inst78000101:    symbolic_constant7800009B
// CHECK:STDOUT:       inst78000102:    symbolic_constant7800009C
// CHECK:STDOUT:       inst78000103:    symbolic_constant78000094
// CHECK:STDOUT:       inst78000104:    symbolic_constant780000A0
// CHECK:STDOUT:       inst78000105:    symbolic_constant7800009B
// CHECK:STDOUT:       inst78000106:    symbolic_constant780000AE
// CHECK:STDOUT:       inst78000107:    symbolic_constant780000AD
// CHECK:STDOUT:       inst78000108:    symbolic_constant7800009D
// CHECK:STDOUT:       inst78000109:    symbolic_constant7800009D
// CHECK:STDOUT:       inst7800010A:    symbolic_constant780000AC
// CHECK:STDOUT:       inst7800010B:    symbolic_constant780000AB
// CHECK:STDOUT:       inst7800010C:    symbolic_constant78000099
// CHECK:STDOUT:       inst7800010D:    symbolic_constant780000AA
// CHECK:STDOUT:       inst7800010E:    symbolic_constant780000A9
// CHECK:STDOUT:       inst7800010F:    symbolic_constant780000AF
// CHECK:STDOUT:       inst78000110:    symbolic_constant780000B0
// CHECK:STDOUT:       inst78000111:    symbolic_constant780000B1
// CHECK:STDOUT:       inst78000112:    symbolic_constant780000B2
// CHECK:STDOUT:       inst78000113:    symbolic_constant780000B3
// CHECK:STDOUT:       inst78000114:    symbolic_constant780000B4
// CHECK:STDOUT:       inst78000115:    symbolic_constant780000B5
// CHECK:STDOUT:       inst78000116:    symbolic_constant780000B6
// CHECK:STDOUT:       inst78000117:    symbolic_constant780000B7
// CHECK:STDOUT:       inst78000118:    symbolic_constant780000B8
// CHECK:STDOUT:       inst78000119:    symbolic_constant780000B9
// CHECK:STDOUT:       inst7800011A:    symbolic_constant78000094
// CHECK:STDOUT:       inst7800011B:    symbolic_constant7800009B
// CHECK:STDOUT:       inst7800011C:    symbolic_constant780000C4
// CHECK:STDOUT:       inst7800011D:    symbolic_constant780000C3
// CHECK:STDOUT:       inst7800011E:    symbolic_constant7800008D
// CHECK:STDOUT:       inst7800011F:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst78000120:    symbolic_constant7800008C
// CHECK:STDOUT:       inst78000121:    symbolic_constant780000AA
// CHECK:STDOUT:       inst78000122:    symbolic_constant780000A9
// CHECK:STDOUT:       inst78000123:    symbolic_constant780000C5
// CHECK:STDOUT:       inst78000124:    symbolic_constant780000C6
// CHECK:STDOUT:       inst78000125:    symbolic_constant780000C7
// CHECK:STDOUT:       inst78000126:    symbolic_constant780000C8
// CHECK:STDOUT:       inst78000127:    symbolic_constant780000C9
// CHECK:STDOUT:       inst78000128:    symbolic_constant780000CA
// CHECK:STDOUT:       inst78000129:    symbolic_constant780000CB
// CHECK:STDOUT:       inst7800012A:    symbolic_constant780000CC
// CHECK:STDOUT:       inst7800012B:    symbolic_constant780000CD
// CHECK:STDOUT:       inst7800012C:    symbolic_constant780000CE
// CHECK:STDOUT:       inst7800012D:    constant<none>
// CHECK:STDOUT:       inst7800012E:    concrete_constant(inst7800012E)
// CHECK:STDOUT:       inst7800012F:    symbolic_constant780000CF
// CHECK:STDOUT:       inst78000130:    symbolic_constant780000D0
// CHECK:STDOUT:       inst78000131:    symbolic_constant780000D1
// CHECK:STDOUT:       inst78000132:    symbolic_constant780000D2
// CHECK:STDOUT:       inst78000133:    constant<none>
// CHECK:STDOUT:       inst78000134:    concrete_constant(inst78000134)
// CHECK:STDOUT:       inst78000135:    symbolic_constant780000D5
// CHECK:STDOUT:       inst78000136:    symbolic_constant780000D9
// CHECK:STDOUT:       inst78000137:    symbolic_constant780000D7
// CHECK:STDOUT:       inst78000138:    symbolic_constant780000D8
// CHECK:STDOUT:       inst78000139:    symbolic_constant780000DA
// CHECK:STDOUT:       inst7800013A:    symbolic_constant780000DB
// CHECK:STDOUT:       inst7800013B:    concrete_constant(inst7800013B)
// CHECK:STDOUT:       inst7800013C:    symbolic_constant780000DD
// CHECK:STDOUT:       inst7800013D:    symbolic_constant780000DE
// CHECK:STDOUT:       inst7800013E:    symbolic_constant780000DB
// CHECK:STDOUT:       inst7800013F:    symbolic_constant780000DF
// CHECK:STDOUT:       inst78000140:    symbolic_constant780000E2
// CHECK:STDOUT:       inst78000141:    symbolic_constant780000E2
// CHECK:STDOUT:       inst78000142:    symbolic_constant780000E3
// CHECK:STDOUT:       inst78000143:    symbolic_constant780000DB
// CHECK:STDOUT:       inst78000144:    symbolic_constant780000E7
// CHECK:STDOUT:       inst78000145:    symbolic_constant780000E2
// CHECK:STDOUT:       inst78000146:    symbolic_constant780000F8
// CHECK:STDOUT:       inst78000147:    symbolic_constant780000F7
// CHECK:STDOUT:       inst78000148:    symbolic_constant780000E4
// CHECK:STDOUT:       inst78000149:    symbolic_constant780000E4
// CHECK:STDOUT:       inst7800014A:    symbolic_constant780000F6
// CHECK:STDOUT:       inst7800014B:    symbolic_constant780000F5
// CHECK:STDOUT:       inst7800014C:    symbolic_constant780000E0
// CHECK:STDOUT:       inst7800014D:    symbolic_constant780000F4
// CHECK:STDOUT:       inst7800014E:    symbolic_constant780000F3
// CHECK:STDOUT:       inst7800014F:    symbolic_constant780000F2
// CHECK:STDOUT:       inst78000150:    symbolic_constant780000F9
// CHECK:STDOUT:       inst78000151:    symbolic_constant780000FA
// CHECK:STDOUT:       inst78000152:    symbolic_constant780000FB
// CHECK:STDOUT:       inst78000153:    symbolic_constant780000FC
// CHECK:STDOUT:       inst78000154:    symbolic_constant780000FD
// CHECK:STDOUT:       inst78000155:    symbolic_constant780000FE
// CHECK:STDOUT:       inst78000156:    symbolic_constant780000FF
// CHECK:STDOUT:       inst78000157:    symbolic_constant78000100
// CHECK:STDOUT:       inst78000158:    symbolic_constant78000101
// CHECK:STDOUT:       inst78000159:    symbolic_constant78000102
// CHECK:STDOUT:       inst7800015A:    symbolic_constant78000103
// CHECK:STDOUT:       inst7800015B:    symbolic_constant78000104
// CHECK:STDOUT:       inst7800015C:    symbolic_constant78000105
// CHECK:STDOUT:       inst7800015D:    symbolic_constant780000DB
// CHECK:STDOUT:       inst7800015E:    symbolic_constant780000E2
// CHECK:STDOUT:       inst7800015F:    symbolic_constant78000114
// CHECK:STDOUT:       inst78000160:    symbolic_constant78000113
// CHECK:STDOUT:       inst78000161:    symbolic_constant78000112
// CHECK:STDOUT:       inst78000162:    symbolic_constant780000D4
// CHECK:STDOUT:       inst78000163:    concrete_constant(inst78000058)
// CHECK:STDOUT:       inst78000164:    symbolic_constant780000D3
// CHECK:STDOUT:       inst78000165:    symbolic_constant780000F4
// CHECK:STDOUT:       inst78000166:    symbolic_constant780000F3
// CHECK:STDOUT:       inst78000167:    symbolic_constant780000F2
// CHECK:STDOUT:       inst78000168:    symbolic_constant78000115
// CHECK:STDOUT:       inst78000169:    symbolic_constant78000116
// CHECK:STDOUT:       inst7800016A:    symbolic_constant78000117
// CHECK:STDOUT:       inst7800016B:    symbolic_constant78000118
// CHECK:STDOUT:       inst7800016C:    symbolic_constant78000119
// CHECK:STDOUT:       inst7800016D:    symbolic_constant7800011A
// CHECK:STDOUT:       inst7800016E:    symbolic_constant7800011B
// CHECK:STDOUT:       inst7800016F:    symbolic_constant7800011C
// CHECK:STDOUT:       inst78000170:    symbolic_constant7800011D
// CHECK:STDOUT:       inst78000171:    symbolic_constant7800011E
// CHECK:STDOUT:       inst78000172:    symbolic_constant7800011F
// CHECK:STDOUT:       inst78000173:    symbolic_constant78000120
// CHECK:STDOUT:       inst78000174:    symbolic_constant78000121
// CHECK:STDOUT:       inst78000175:    symbolic_constant78000122
// CHECK:STDOUT:       inst78000176:    symbolic_constant78000124
// CHECK:STDOUT:       inst78000177:    symbolic_constant78000125
// CHECK:STDOUT:       inst78000178:    symbolic_constant78000126
// CHECK:STDOUT:       inst78000179:    symbolic_constant78000127
// CHECK:STDOUT:       inst7800017A:    symbolic_constant7800012D
// CHECK:STDOUT:       inst7800017B:    symbolic_constant78000128
// CHECK:STDOUT:       inst7800017C:    symbolic_constant78000129
// CHECK:STDOUT:       inst7800017D:    symbolic_constant7800012A
// CHECK:STDOUT:       inst7800017E:    symbolic_constant7800012B
// CHECK:STDOUT:       inst7800017F:    symbolic_constant7800012C
// CHECK:STDOUT:       inst78000180:    symbolic_constant7800012D
// CHECK:STDOUT:       inst78000183:    symbolic_constant7800012A
// CHECK:STDOUT:       inst78000184:    symbolic_constant7800012A
// CHECK:STDOUT:       inst78000186:    symbolic_constant7800012A
// CHECK:STDOUT:       inst78000187:    symbolic_constant7800012A
// CHECK:STDOUT:       inst78000188:    symbolic_constant7800012E
// CHECK:STDOUT:       inst78000189:    symbolic_constant7800012F
// CHECK:STDOUT:       inst7800018A:    symbolic_constant78000130
// CHECK:STDOUT:       inst7800018B:    symbolic_constant78000131
// CHECK:STDOUT:       inst7800018C:    symbolic_constant78000132
// CHECK:STDOUT:       inst7800018D:    symbolic_constant78000134
// CHECK:STDOUT:       inst7800018E:    symbolic_constant78000133
// CHECK:STDOUT:       inst7800018F:    symbolic_constant78000134
// CHECK:STDOUT:       inst78000194:    concrete_constant(inst7800002C)
// CHECK:STDOUT:       inst78000195:    concrete_constant(inst7800002C)
// CHECK:STDOUT:       inst78000196:    concrete_constant(inst7800002C)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant78000000: {inst: inst78000012, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant78000001: {inst: inst78000016, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000002: {inst: inst78000016, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000003: {inst: inst78000019, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000004: {inst: inst78000019, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000005: {inst: inst7800001D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000006: {inst: inst7800001D, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000007: {inst: inst7800001F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000008: {inst: inst78000021, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000009: {inst: inst7800001F, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800000A: {inst: inst78000021, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800000B: {inst: inst78000025, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800000C: {inst: inst78000025, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800000D: {inst: inst7800002F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800000E: {inst: inst7800002F, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800000F: {inst: inst78000033, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000010: {inst: inst78000033, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000011: {inst: inst78000037, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000012: {inst: inst78000037, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000013: {inst: inst78000039, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000014: {inst: inst7800003B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000015: {inst: inst78000039, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant78000016: {inst: inst7800003B, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000017: {inst: inst7800003F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000018: {inst: inst7800003F, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant78000019: {inst: inst7800004A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001A: {inst: inst7800004A, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800001B: {inst: inst7800004C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001C: {inst: inst7800004E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001D: {inst: inst7800004E, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800001E: {inst: inst7800005A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001F: {inst: inst7800005F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000020: {inst: inst78000060, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000021: {inst: inst78000060, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000022: {inst: inst78000061, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000023: {inst: inst78000062, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000024: {inst: inst78000063, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000025: {inst: inst7800005F, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000026: {inst: inst78000065, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000027: {inst: inst78000065, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000028: {inst: inst78000062, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000029: {inst: inst78000066, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002A: {inst: inst78000068, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002B: {inst: inst78000068, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800002C: {inst: inst78000065, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800002D: {inst: inst78000063, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800002E: {inst: inst7800006A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002F: {inst: inst7800006A, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000030: {inst: inst78000068, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000031: {inst: inst78000066, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000032: {inst: inst78000061, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000033: {inst: inst7800005A, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000034: {inst: inst7800006A, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000035: {inst: inst78000061, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000036: {inst: inst78000063, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000037: {inst: inst78000066, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000038: {inst: inst7800005A, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000039: {inst: inst78000061, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800003A: {inst: inst78000062, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800003B: {inst: inst78000066, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800003C: {inst: inst78000068, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800003D: {inst: inst7800006A, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800003E: {inst: inst78000063, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800003F: {inst: inst78000065, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000040: {inst: inst78000085, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000041: {inst: inst78000086, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000042: {inst: inst78000087, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000043: {inst: inst78000088, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000044: {inst: inst78000088, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000045: {inst: inst78000087, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000046: {inst: inst7800008B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000047: {inst: inst7800008B, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000048: {inst: inst7800008D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000049: {inst: inst7800008E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004A: {inst: inst7800008E, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800004B: {inst: inst7800008F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004C: {inst: inst78000090, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004D: {inst: inst7800008D, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800004E: {inst: inst78000092, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004F: {inst: inst78000094, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000050: {inst: inst78000094, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000051: {inst: inst7800008F, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000052: {inst: inst78000095, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000053: {inst: inst78000097, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000054: {inst: inst78000097, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000055: {inst: inst78000094, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000056: {inst: inst78000090, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000057: {inst: inst78000099, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000058: {inst: inst78000099, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000059: {inst: inst78000097, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800005A: {inst: inst78000095, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800005B: {inst: inst78000087, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800005C: {inst: inst78000086, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800005D: {inst: inst78000085, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800005E: {inst: inst78000085, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800005F: {inst: inst78000099, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000060: {inst: inst78000087, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000061: {inst: inst78000090, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000062: {inst: inst78000095, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000063: {inst: inst78000085, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000064: {inst: inst78000086, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000065: {inst: inst78000087, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000066: {inst: inst7800008F, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000067: {inst: inst78000095, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000068: {inst: inst78000097, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000069: {inst: inst78000099, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800006A: {inst: inst78000090, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800006B: {inst: inst78000094, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800006C: {inst: inst78000088, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800006D: {inst: inst78000087, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800006E: {inst: inst78000086, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800006F: {inst: inst78000085, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000070: {inst: inst78000092, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000071: {inst: inst78000092, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000072: {inst: inst78000092, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000073: {inst: inst78000085, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000074: {inst: inst78000086, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000075: {inst: inst78000087, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000076: {inst: inst78000088, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000077: {inst: inst7800008B, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000078: {inst: inst780000D3, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000079: {inst: inst780000D3, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800007A: {inst: inst7800001D, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800007B: {inst: inst780000D6, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800007C: {inst: inst780000D6, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800007D: {inst: inst780000D3, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800007E: {inst: inst7800001D, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800007F: {inst: inst78000019, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000080: {inst: inst78000016, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000081: {inst: inst78000019, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000082: {inst: inst78000016, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000083: {inst: inst78000016, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000084: {inst: inst78000019, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000085: {inst: inst7800001D, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000086: {inst: inst780000D3, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000087: {inst: inst780000D6, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000088: {inst: inst780000EF, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000089: {inst: inst780000F0, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800008A: {inst: inst780000F1, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800008B: {inst: inst780000F2, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800008C: {inst: inst780000F2, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800008D: {inst: inst780000F1, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800008E: {inst: inst780000F5, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800008F: {inst: inst780000F5, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant78000090: {inst: inst780000F7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000091: {inst: inst780000F8, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000092: {inst: inst780000F8, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000093: {inst: inst780000F9, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000094: {inst: inst780000FA, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000095: {inst: inst780000F7, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000096: {inst: inst780000FC, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000097: {inst: inst780000FD, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000098: {inst: inst780000FF, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000099: {inst: inst780000FF, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant7800009A: {inst: inst780000F9, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800009B: {inst: inst78000100, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800009C: {inst: inst78000102, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800009D: {inst: inst78000102, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800009E: {inst: inst780000FF, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant7800009F: {inst: inst780000FA, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000A0: {inst: inst78000104, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000A1: {inst: inst78000104, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000A2: {inst: inst78000102, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000A3: {inst: inst78000100, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000A4: {inst: inst780000F1, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000A5: {inst: inst780000F0, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000A6: {inst: inst780000EF, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000A7: {inst: inst78000086, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000A8: {inst: inst78000085, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000A9: {inst: inst780000EF, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000AA: {inst: inst78000085, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000AB: {inst: inst78000104, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000AC: {inst: inst780000F1, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000AD: {inst: inst780000FA, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000AE: {inst: inst78000100, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000AF: {inst: inst78000085, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000B0: {inst: inst78000086, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000B1: {inst: inst780000EF, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000B2: {inst: inst780000F0, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000B3: {inst: inst780000F1, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000B4: {inst: inst780000F9, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000B5: {inst: inst78000100, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000B6: {inst: inst78000102, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000B7: {inst: inst78000104, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000B8: {inst: inst780000FA, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000B9: {inst: inst780000FF, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant780000BA: {inst: inst780000F2, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000BB: {inst: inst780000F1, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000BC: {inst: inst780000F0, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000BD: {inst: inst78000086, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000BE: {inst: inst780000FC, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000BF: {inst: inst780000EF, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000C0: {inst: inst780000FD, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000C1: {inst: inst78000085, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000C2: {inst: inst78000092, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000C3: {inst: inst780000FD, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000C4: {inst: inst78000092, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000C5: {inst: inst78000092, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000C6: {inst: inst78000085, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000C7: {inst: inst780000FD, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000C8: {inst: inst780000EF, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000C9: {inst: inst780000FC, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000CA: {inst: inst78000086, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000CB: {inst: inst780000F0, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000CC: {inst: inst780000F1, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000CD: {inst: inst780000F2, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000CE: {inst: inst780000F5, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000CF: {inst: inst7800012F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D0: {inst: inst78000130, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D1: {inst: inst78000131, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D2: {inst: inst78000132, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D3: {inst: inst78000132, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant780000D4: {inst: inst78000131, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant780000D5: {inst: inst78000135, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D6: {inst: inst78000135, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant780000D7: {inst: inst78000137, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D8: {inst: inst78000138, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D9: {inst: inst78000138, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000DA: {inst: inst78000139, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DB: {inst: inst7800013A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DC: {inst: inst78000137, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780000DD: {inst: inst7800013C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DE: {inst: inst7800013D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DF: {inst: inst7800013F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E0: {inst: inst7800013F, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant780000E1: {inst: inst78000139, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000E2: {inst: inst78000140, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E3: {inst: inst78000142, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E4: {inst: inst78000142, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000E5: {inst: inst7800013F, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant780000E6: {inst: inst7800013A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant780000E7: {inst: inst78000144, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E8: {inst: inst78000144, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant780000E9: {inst: inst78000142, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000EA: {inst: inst78000140, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000EB: {inst: inst78000131, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000EC: {inst: inst78000130, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000ED: {inst: inst7800012F, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000EE: {inst: inst780000F0, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000EF: {inst: inst780000EF, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000F0: {inst: inst78000086, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000F1: {inst: inst78000085, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000F2: {inst: inst7800012F, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000F3: {inst: inst780000EF, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000F4: {inst: inst78000085, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000F5: {inst: inst78000144, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant780000F6: {inst: inst78000131, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000F7: {inst: inst7800013A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant780000F8: {inst: inst78000140, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000F9: {inst: inst78000085, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000FA: {inst: inst78000086, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000FB: {inst: inst780000EF, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000FC: {inst: inst780000F0, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000FD: {inst: inst7800012F, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000FE: {inst: inst78000130, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000FF: {inst: inst78000131, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000100: {inst: inst78000139, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000101: {inst: inst78000140, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000102: {inst: inst78000142, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant78000103: {inst: inst78000144, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000104: {inst: inst7800013A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant78000105: {inst: inst7800013F, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant78000106: {inst: inst78000132, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant78000107: {inst: inst78000131, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000108: {inst: inst78000130, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant78000109: {inst: inst780000F0, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800010A: {inst: inst78000086, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800010B: {inst: inst7800013C, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800010C: {inst: inst7800012F, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800010D: {inst: inst7800013D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800010E: {inst: inst780000EF, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800010F: {inst: inst780000FD, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000110: {inst: inst78000085, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000111: {inst: inst78000092, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000112: {inst: inst7800013D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000113: {inst: inst780000FD, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000114: {inst: inst78000092, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000115: {inst: inst78000092, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000116: {inst: inst78000085, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000117: {inst: inst780000FD, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000118: {inst: inst780000EF, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000119: {inst: inst7800013D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800011A: {inst: inst7800012F, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800011B: {inst: inst7800013C, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800011C: {inst: inst78000086, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800011D: {inst: inst780000F0, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800011E: {inst: inst78000130, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant7800011F: {inst: inst78000131, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000120: {inst: inst78000132, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant78000121: {inst: inst78000135, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant78000122: {inst: inst78000175, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000123: {inst: inst780000D6, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000124: {inst: inst78000176, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000125: {inst: inst78000177, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000126: {inst: inst78000178, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000127: {inst: inst78000179, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000128: {inst: inst7800017B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000129: {inst: inst78000175, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant7800012A: {inst: inst78000176, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant7800012B: {inst: inst78000177, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant7800012C: {inst: inst78000179, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant7800012D: {inst: inst7800017B, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant7800012E: {inst: inst78000188, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800012F: {inst: inst78000189, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000130: {inst: inst7800018A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000131: {inst: inst7800018B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000132: {inst: inst7800018C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000133: {inst: inst7800018E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000134: {inst: inst7800018E, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def7}}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst78000046
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst78000010
// CHECK:STDOUT:       1:               inst78000056
// CHECK:STDOUT:       2:               inst78000057
// CHECK:STDOUT:       3:               inst78000059
// CHECK:STDOUT:       4:               inst7800005B
// CHECK:STDOUT:       5:               inst7800005C
// CHECK:STDOUT:       6:               inst7800005D
// CHECK:STDOUT:       7:               inst7800005E
// CHECK:STDOUT:       8:               inst78000064
// CHECK:STDOUT:       9:               inst78000067
// CHECK:STDOUT:       10:              inst78000069
// CHECK:STDOUT:       11:              inst7800006B
// CHECK:STDOUT:       12:              inst7800006C
// CHECK:STDOUT:       13:              inst7800006D
// CHECK:STDOUT:       14:              inst7800006E
// CHECK:STDOUT:       15:              inst7800006F
// CHECK:STDOUT:       16:              inst78000070
// CHECK:STDOUT:       17:              inst78000071
// CHECK:STDOUT:       18:              inst78000072
// CHECK:STDOUT:       19:              inst78000073
// CHECK:STDOUT:       20:              inst7800007C
// CHECK:STDOUT:       21:              inst7800007D
// CHECK:STDOUT:       22:              inst7800007E
// CHECK:STDOUT:       23:              inst78000080
// CHECK:STDOUT:       24:              inst78000083
// CHECK:STDOUT:       25:              inst78000084
// CHECK:STDOUT:       26:              inst78000089
// CHECK:STDOUT:       27:              inst7800008A
// CHECK:STDOUT:       28:              inst7800008C
// CHECK:STDOUT:       29:              inst78000093
// CHECK:STDOUT:       30:              inst78000096
// CHECK:STDOUT:       31:              inst78000098
// CHECK:STDOUT:       32:              inst7800009A
// CHECK:STDOUT:       33:              inst7800009B
// CHECK:STDOUT:       34:              inst7800009C
// CHECK:STDOUT:       35:              inst7800009D
// CHECK:STDOUT:       36:              inst7800009E
// CHECK:STDOUT:       37:              inst7800009F
// CHECK:STDOUT:       38:              inst780000A0
// CHECK:STDOUT:       39:              inst780000A1
// CHECK:STDOUT:       40:              inst780000A2
// CHECK:STDOUT:       41:              inst780000AC
// CHECK:STDOUT:       42:              inst780000AD
// CHECK:STDOUT:       43:              inst780000AE
// CHECK:STDOUT:       44:              inst780000AF
// CHECK:STDOUT:       45:              inst780000B0
// CHECK:STDOUT:       46:              inst780000B1
// CHECK:STDOUT:       47:              inst780000B2
// CHECK:STDOUT:       48:              inst780000B9
// CHECK:STDOUT:       49:              inst780000BA
// CHECK:STDOUT:       50:              inst780000BC
// CHECK:STDOUT:       51:              inst780000BD
// CHECK:STDOUT:       52:              inst780000BE
// CHECK:STDOUT:       53:              inst780000BF
// CHECK:STDOUT:       54:              inst780000C0
// CHECK:STDOUT:       55:              inst780000C2
// CHECK:STDOUT:       56:              inst780000C3
// CHECK:STDOUT:       57:              inst780000C4
// CHECK:STDOUT:       58:              inst780000C5
// CHECK:STDOUT:       59:              inst780000C6
// CHECK:STDOUT:       60:              inst780000C8
// CHECK:STDOUT:       61:              inst780000C9
// CHECK:STDOUT:       62:              inst780000CA
// CHECK:STDOUT:       63:              inst780000CB
// CHECK:STDOUT:       64:              inst780000CC
// CHECK:STDOUT:       65:              inst780000CE
// CHECK:STDOUT:       66:              inst780000CF
// CHECK:STDOUT:       67:              inst780000D0
// CHECK:STDOUT:       68:              inst780000D1
// CHECK:STDOUT:       69:              inst780000D2
// CHECK:STDOUT:       70:              inst780000D4
// CHECK:STDOUT:       71:              inst780000D5
// CHECK:STDOUT:       72:              inst780000D7
// CHECK:STDOUT:       73:              inst780000D8
// CHECK:STDOUT:       74:              inst780000D9
// CHECK:STDOUT:       75:              inst780000DA
// CHECK:STDOUT:       76:              inst780000DB
// CHECK:STDOUT:       77:              inst780000E1
// CHECK:STDOUT:       78:              inst780000E2
// CHECK:STDOUT:       79:              inst780000E4
// CHECK:STDOUT:       80:              inst780000E5
// CHECK:STDOUT:       81:              inst780000E6
// CHECK:STDOUT:       82:              inst780000E7
// CHECK:STDOUT:       83:              inst780000E8
// CHECK:STDOUT:       84:              inst780000EA
// CHECK:STDOUT:       85:              inst780000EB
// CHECK:STDOUT:       86:              inst780000EC
// CHECK:STDOUT:       87:              inst780000ED
// CHECK:STDOUT:       88:              inst780000EE
// CHECK:STDOUT:       89:              inst780000F3
// CHECK:STDOUT:       90:              inst780000F4
// CHECK:STDOUT:       91:              inst780000F6
// CHECK:STDOUT:       92:              inst780000FE
// CHECK:STDOUT:       93:              inst78000101
// CHECK:STDOUT:       94:              inst78000103
// CHECK:STDOUT:       95:              inst78000105
// CHECK:STDOUT:       96:              inst78000106
// CHECK:STDOUT:       97:              inst78000107
// CHECK:STDOUT:       98:              inst78000108
// CHECK:STDOUT:       99:              inst78000109
// CHECK:STDOUT:       100:             inst7800010A
// CHECK:STDOUT:       101:             inst7800010B
// CHECK:STDOUT:       102:             inst7800010C
// CHECK:STDOUT:       103:             inst7800010D
// CHECK:STDOUT:       104:             inst7800010E
// CHECK:STDOUT:       105:             inst7800011A
// CHECK:STDOUT:       106:             inst7800011B
// CHECK:STDOUT:       107:             inst7800011C
// CHECK:STDOUT:       108:             inst7800011D
// CHECK:STDOUT:       109:             inst7800011E
// CHECK:STDOUT:       110:             inst7800011F
// CHECK:STDOUT:       111:             inst78000120
// CHECK:STDOUT:       112:             inst78000121
// CHECK:STDOUT:       113:             inst78000122
// CHECK:STDOUT:       114:             inst7800012D
// CHECK:STDOUT:       115:             inst7800012E
// CHECK:STDOUT:       116:             inst78000133
// CHECK:STDOUT:       117:             inst78000134
// CHECK:STDOUT:       118:             inst78000136
// CHECK:STDOUT:       119:             inst7800013E
// CHECK:STDOUT:       120:             inst78000141
// CHECK:STDOUT:       121:             inst78000143
// CHECK:STDOUT:       122:             inst78000145
// CHECK:STDOUT:       123:             inst78000146
// CHECK:STDOUT:       124:             inst78000147
// CHECK:STDOUT:       125:             inst78000148
// CHECK:STDOUT:       126:             inst78000149
// CHECK:STDOUT:       127:             inst7800014A
// CHECK:STDOUT:       128:             inst7800014B
// CHECK:STDOUT:       129:             inst7800014C
// CHECK:STDOUT:       130:             inst7800014D
// CHECK:STDOUT:       131:             inst7800014E
// CHECK:STDOUT:       132:             inst7800014F
// CHECK:STDOUT:       133:             inst7800015D
// CHECK:STDOUT:       134:             inst7800015E
// CHECK:STDOUT:       135:             inst7800015F
// CHECK:STDOUT:       136:             inst78000160
// CHECK:STDOUT:       137:             inst78000161
// CHECK:STDOUT:       138:             inst78000162
// CHECK:STDOUT:       139:             inst78000163
// CHECK:STDOUT:       140:             inst78000164
// CHECK:STDOUT:       141:             inst78000165
// CHECK:STDOUT:       142:             inst78000166
// CHECK:STDOUT:       143:             inst78000167
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     extra_global_variables: {}
// CHECK:STDOUT:     inst_block78000006:
// CHECK:STDOUT:       0:               inst78000011
// CHECK:STDOUT:       1:               inst78000013
// CHECK:STDOUT:     inst_block78000007:
// CHECK:STDOUT:       0:               inst78000015
// CHECK:STDOUT:     inst_block78000008:
// CHECK:STDOUT:       0:               inst7800001B
// CHECK:STDOUT:       1:               inst7800001C
// CHECK:STDOUT:     inst_block78000009:
// CHECK:STDOUT:       0:               inst78000024
// CHECK:STDOUT:     inst_block7800000A:
// CHECK:STDOUT:       0:               inst78000029
// CHECK:STDOUT:       1:               inst7800002B
// CHECK:STDOUT:     inst_block7800000B:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst7800002A
// CHECK:STDOUT:     inst_block7800000C:
// CHECK:STDOUT:       0:               inst7800001D
// CHECK:STDOUT:       1:               inst7800002C
// CHECK:STDOUT:     inst_block7800000D:
// CHECK:STDOUT:       0:               inst7800001E
// CHECK:STDOUT:       1:               inst7800002C
// CHECK:STDOUT:     inst_block7800000E:
// CHECK:STDOUT:       0:               inst7800001D
// CHECK:STDOUT:       1:               inst78000032
// CHECK:STDOUT:     inst_block7800000F:
// CHECK:STDOUT:       0:               inst7800001D
// CHECK:STDOUT:       1:               inst7800002A
// CHECK:STDOUT:     inst_block78000010:
// CHECK:STDOUT:       0:               inst7800001E
// CHECK:STDOUT:       1:               inst7800002A
// CHECK:STDOUT:     inst_block78000011:
// CHECK:STDOUT:       0:               inst78000020
// CHECK:STDOUT:       1:               inst7800003A
// CHECK:STDOUT:     inst_block78000012:
// CHECK:STDOUT:       0:               inst78000042
// CHECK:STDOUT:       1:               inst78000044
// CHECK:STDOUT:     inst_block78000013:
// CHECK:STDOUT:       0:               inst78000015
// CHECK:STDOUT:       1:               inst78000020
// CHECK:STDOUT:       2:               inst78000024
// CHECK:STDOUT:       3:               inst7800003A
// CHECK:STDOUT:       4:               inst7800003E
// CHECK:STDOUT:     inst_block78000014:
// CHECK:STDOUT:       0:               inst78000028
// CHECK:STDOUT:       1:               inst78000029
// CHECK:STDOUT:       2:               inst7800002B
// CHECK:STDOUT:       3:               inst7800002E
// CHECK:STDOUT:       4:               inst78000032
// CHECK:STDOUT:       5:               inst78000034
// CHECK:STDOUT:       6:               inst78000036
// CHECK:STDOUT:       7:               inst78000041
// CHECK:STDOUT:       8:               inst78000018
// CHECK:STDOUT:       9:               inst78000042
// CHECK:STDOUT:       10:              inst78000043
// CHECK:STDOUT:       11:              inst78000027
// CHECK:STDOUT:       12:              inst78000044
// CHECK:STDOUT:       13:              inst78000045
// CHECK:STDOUT:     inst_block78000015:
// CHECK:STDOUT:       0:               inst78000018
// CHECK:STDOUT:     inst_block78000016:
// CHECK:STDOUT:       0:               inst78000019
// CHECK:STDOUT:     inst_block78000017:
// CHECK:STDOUT:       0:               inst78000017
// CHECK:STDOUT:       1:               inst7800001A
// CHECK:STDOUT:       2:               inst7800001E
// CHECK:STDOUT:       3:               inst78000022
// CHECK:STDOUT:       4:               inst78000023
// CHECK:STDOUT:       5:               inst78000026
// CHECK:STDOUT:       6:               inst78000030
// CHECK:STDOUT:       7:               inst78000035
// CHECK:STDOUT:       8:               inst78000038
// CHECK:STDOUT:       9:               inst7800003C
// CHECK:STDOUT:       10:              inst7800003D
// CHECK:STDOUT:       11:              inst78000040
// CHECK:STDOUT:     inst_block78000018:
// CHECK:STDOUT:       0:               inst78000016
// CHECK:STDOUT:       1:               inst78000019
// CHECK:STDOUT:       2:               inst7800001D
// CHECK:STDOUT:       3:               inst7800001F
// CHECK:STDOUT:       4:               inst78000021
// CHECK:STDOUT:       5:               inst78000025
// CHECK:STDOUT:       6:               inst7800002F
// CHECK:STDOUT:       7:               inst78000033
// CHECK:STDOUT:       8:               inst78000037
// CHECK:STDOUT:       9:               inst78000039
// CHECK:STDOUT:       10:              inst7800003B
// CHECK:STDOUT:       11:              inst7800003F
// CHECK:STDOUT:     inst_block78000019:
// CHECK:STDOUT:       0:               inst78000050
// CHECK:STDOUT:       1:               inst78000051
// CHECK:STDOUT:       2:               inst78000052
// CHECK:STDOUT:       3:               inst7800017A
// CHECK:STDOUT:       4:               inst78000181
// CHECK:STDOUT:       5:               inst78000183
// CHECK:STDOUT:       6:               inst78000184
// CHECK:STDOUT:       7:               inst78000186
// CHECK:STDOUT:       8:               inst78000187
// CHECK:STDOUT:       9:               inst7800018D
// CHECK:STDOUT:       10:              inst78000190
// CHECK:STDOUT:       11:              inst78000191
// CHECK:STDOUT:       12:              inst78000054
// CHECK:STDOUT:       13:              inst78000192
// CHECK:STDOUT:       14:              inst78000193
// CHECK:STDOUT:       15:              inst78000194
// CHECK:STDOUT:       16:              inst78000195
// CHECK:STDOUT:       17:              inst78000196
// CHECK:STDOUT:       18:              inst78000197
// CHECK:STDOUT:       19:              inst78000198
// CHECK:STDOUT:       20:              inst78000199
// CHECK:STDOUT:     inst_block7800001A:
// CHECK:STDOUT:       0:               inst78000050
// CHECK:STDOUT:       1:               inst78000051
// CHECK:STDOUT:     inst_block7800001B:
// CHECK:STDOUT:       0:               inst7800005C
// CHECK:STDOUT:     inst_block7800001C:
// CHECK:STDOUT:       0:               inst7800005D
// CHECK:STDOUT:     inst_block7800001D:
// CHECK:STDOUT:       0:               inst7800005A
// CHECK:STDOUT:     inst_block7800001E:
// CHECK:STDOUT:       0:               inst7800005A
// CHECK:STDOUT:       1:               inst7800005F
// CHECK:STDOUT:       2:               inst78000060
// CHECK:STDOUT:     inst_block7800001F:
// CHECK:STDOUT:       0:               inst7800006C
// CHECK:STDOUT:       1:               inst7800006D
// CHECK:STDOUT:     inst_block78000020:
// CHECK:STDOUT:       0:               inst7800006F
// CHECK:STDOUT:     inst_block78000021:
// CHECK:STDOUT:       0:               inst78000073
// CHECK:STDOUT:     inst_block78000022:
// CHECK:STDOUT:       0:               inst78000074
// CHECK:STDOUT:       1:               inst78000075
// CHECK:STDOUT:       2:               inst78000076
// CHECK:STDOUT:       3:               inst78000077
// CHECK:STDOUT:       4:               inst78000078
// CHECK:STDOUT:       5:               inst78000079
// CHECK:STDOUT:       6:               inst7800007A
// CHECK:STDOUT:       7:               inst7800007B
// CHECK:STDOUT:     inst_block78000023:
// CHECK:STDOUT:       0:               inst7800005A
// CHECK:STDOUT:       1:               inst78000061
// CHECK:STDOUT:       2:               inst78000062
// CHECK:STDOUT:       3:               inst78000066
// CHECK:STDOUT:       4:               inst78000068
// CHECK:STDOUT:       5:               inst7800006A
// CHECK:STDOUT:       6:               inst78000063
// CHECK:STDOUT:       7:               inst78000065
// CHECK:STDOUT:     inst_block78000024:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:     inst_block78000025:
// CHECK:STDOUT:       0:               inst78000085
// CHECK:STDOUT:     inst_block78000026:
// CHECK:STDOUT:       0:               inst78000092
// CHECK:STDOUT:       1:               inst78000085
// CHECK:STDOUT:       2:               inst78000086
// CHECK:STDOUT:       3:               inst78000087
// CHECK:STDOUT:       4:               inst78000088
// CHECK:STDOUT:       5:               inst7800008B
// CHECK:STDOUT:     inst_block78000027:
// CHECK:STDOUT:       0:               inst7800008D
// CHECK:STDOUT:       1:               inst7800008E
// CHECK:STDOUT:     inst_block78000028:
// CHECK:STDOUT:       0:               inst7800009B
// CHECK:STDOUT:       1:               inst7800009C
// CHECK:STDOUT:     inst_block78000029:
// CHECK:STDOUT:       0:               inst7800009E
// CHECK:STDOUT:     inst_block7800002A:
// CHECK:STDOUT:       0:               inst780000A2
// CHECK:STDOUT:     inst_block7800002B:
// CHECK:STDOUT:       0:               inst780000A3
// CHECK:STDOUT:       1:               inst780000A4
// CHECK:STDOUT:       2:               inst780000A5
// CHECK:STDOUT:       3:               inst780000A6
// CHECK:STDOUT:       4:               inst780000A7
// CHECK:STDOUT:       5:               inst780000A8
// CHECK:STDOUT:       6:               inst780000A9
// CHECK:STDOUT:       7:               inst780000AA
// CHECK:STDOUT:       8:               inst780000AB
// CHECK:STDOUT:     inst_block7800002C:
// CHECK:STDOUT:       0:               inst78000085
// CHECK:STDOUT:       1:               inst78000086
// CHECK:STDOUT:       2:               inst78000087
// CHECK:STDOUT:       3:               inst7800008F
// CHECK:STDOUT:       4:               inst78000095
// CHECK:STDOUT:       5:               inst78000097
// CHECK:STDOUT:       6:               inst78000099
// CHECK:STDOUT:       7:               inst78000090
// CHECK:STDOUT:       8:               inst78000094
// CHECK:STDOUT:     inst_block7800002D:
// CHECK:STDOUT:       0:               inst780000AE
// CHECK:STDOUT:     inst_block7800002E:
// CHECK:STDOUT:       0:               inst780000B2
// CHECK:STDOUT:     inst_block7800002F:
// CHECK:STDOUT:       0:               inst780000B4
// CHECK:STDOUT:     inst_block78000030:
// CHECK:STDOUT:       0:               inst780000B3
// CHECK:STDOUT:       1:               inst780000B4
// CHECK:STDOUT:       2:               inst780000B5
// CHECK:STDOUT:       3:               inst780000B6
// CHECK:STDOUT:       4:               inst780000B7
// CHECK:STDOUT:       5:               inst780000B8
// CHECK:STDOUT:     inst_block78000031:
// CHECK:STDOUT:       0:               inst780000D4
// CHECK:STDOUT:     inst_block78000032:
// CHECK:STDOUT:       0:               inst78000016
// CHECK:STDOUT:       1:               inst78000019
// CHECK:STDOUT:       2:               inst7800001D
// CHECK:STDOUT:       3:               inst780000D3
// CHECK:STDOUT:       4:               inst780000D6
// CHECK:STDOUT:     inst_block78000033:
// CHECK:STDOUT:       0:               inst780000D7
// CHECK:STDOUT:     inst_block78000034:
// CHECK:STDOUT:       0:               inst780000DB
// CHECK:STDOUT:     inst_block78000035:
// CHECK:STDOUT:       0:               inst780000DD
// CHECK:STDOUT:     inst_block78000036:
// CHECK:STDOUT:       0:               inst780000DC
// CHECK:STDOUT:       1:               inst780000DD
// CHECK:STDOUT:       2:               inst780000DE
// CHECK:STDOUT:       3:               inst780000DF
// CHECK:STDOUT:       4:               inst780000E0
// CHECK:STDOUT:     inst_block78000037:
// CHECK:STDOUT:       0:               inst78000086
// CHECK:STDOUT:       1:               inst780000F0
// CHECK:STDOUT:     inst_block78000038:
// CHECK:STDOUT:       0:               inst780000F3
// CHECK:STDOUT:     inst_block78000039:
// CHECK:STDOUT:       0:               inst78000085
// CHECK:STDOUT:       1:               inst780000EF
// CHECK:STDOUT:     inst_block7800003A:
// CHECK:STDOUT:       0:               inst78000058
// CHECK:STDOUT:       1:               inst78000058
// CHECK:STDOUT:     inst_block7800003B:
// CHECK:STDOUT:       0:               inst78000092
// CHECK:STDOUT:       1:               inst78000085
// CHECK:STDOUT:       2:               inst780000FD
// CHECK:STDOUT:       3:               inst780000EF
// CHECK:STDOUT:       4:               inst780000FC
// CHECK:STDOUT:       5:               inst78000086
// CHECK:STDOUT:       6:               inst780000F0
// CHECK:STDOUT:       7:               inst780000F1
// CHECK:STDOUT:       8:               inst780000F2
// CHECK:STDOUT:       9:               inst780000F5
// CHECK:STDOUT:     inst_block7800003C:
// CHECK:STDOUT:       0:               inst780000F7
// CHECK:STDOUT:       1:               inst780000F8
// CHECK:STDOUT:     inst_block7800003D:
// CHECK:STDOUT:       0:               inst78000106
// CHECK:STDOUT:       1:               inst78000107
// CHECK:STDOUT:     inst_block7800003E:
// CHECK:STDOUT:       0:               inst78000109
// CHECK:STDOUT:     inst_block7800003F:
// CHECK:STDOUT:       0:               inst7800010D
// CHECK:STDOUT:       1:               inst7800010E
// CHECK:STDOUT:     inst_block78000040:
// CHECK:STDOUT:       0:               inst78000110
// CHECK:STDOUT:       1:               inst78000112
// CHECK:STDOUT:     inst_block78000041:
// CHECK:STDOUT:       0:               inst7800010F
// CHECK:STDOUT:       1:               inst78000110
// CHECK:STDOUT:       2:               inst78000111
// CHECK:STDOUT:       3:               inst78000112
// CHECK:STDOUT:       4:               inst78000113
// CHECK:STDOUT:       5:               inst78000114
// CHECK:STDOUT:       6:               inst78000115
// CHECK:STDOUT:       7:               inst78000116
// CHECK:STDOUT:       8:               inst78000117
// CHECK:STDOUT:       9:               inst78000118
// CHECK:STDOUT:       10:              inst78000119
// CHECK:STDOUT:     inst_block78000042:
// CHECK:STDOUT:       0:               inst78000085
// CHECK:STDOUT:       1:               inst78000086
// CHECK:STDOUT:       2:               inst780000EF
// CHECK:STDOUT:       3:               inst780000F0
// CHECK:STDOUT:       4:               inst780000F1
// CHECK:STDOUT:       5:               inst780000F9
// CHECK:STDOUT:       6:               inst78000100
// CHECK:STDOUT:       7:               inst78000102
// CHECK:STDOUT:       8:               inst78000104
// CHECK:STDOUT:       9:               inst780000FA
// CHECK:STDOUT:       10:              inst780000FF
// CHECK:STDOUT:     inst_block78000043:
// CHECK:STDOUT:       0:               inst7800011C
// CHECK:STDOUT:       1:               inst7800011D
// CHECK:STDOUT:     inst_block78000044:
// CHECK:STDOUT:       0:               inst78000121
// CHECK:STDOUT:       1:               inst78000122
// CHECK:STDOUT:     inst_block78000045:
// CHECK:STDOUT:       0:               inst78000124
// CHECK:STDOUT:       1:               inst78000126
// CHECK:STDOUT:     inst_block78000046:
// CHECK:STDOUT:       0:               inst78000128
// CHECK:STDOUT:       1:               inst78000129
// CHECK:STDOUT:     inst_block78000047:
// CHECK:STDOUT:       0:               inst78000124
// CHECK:STDOUT:       1:               inst78000126
// CHECK:STDOUT:     inst_block78000048:
// CHECK:STDOUT:       0:               inst78000123
// CHECK:STDOUT:       1:               inst78000124
// CHECK:STDOUT:       2:               inst78000125
// CHECK:STDOUT:       3:               inst78000126
// CHECK:STDOUT:       4:               inst78000127
// CHECK:STDOUT:       5:               inst78000128
// CHECK:STDOUT:       6:               inst78000129
// CHECK:STDOUT:       7:               inst7800012A
// CHECK:STDOUT:       8:               inst7800012B
// CHECK:STDOUT:       9:               inst7800012C
// CHECK:STDOUT:     inst_block78000049:
// CHECK:STDOUT:       0:               inst78000086
// CHECK:STDOUT:       1:               inst780000F0
// CHECK:STDOUT:       2:               inst78000130
// CHECK:STDOUT:     inst_block7800004A:
// CHECK:STDOUT:       0:               inst78000133
// CHECK:STDOUT:     inst_block7800004B:
// CHECK:STDOUT:       0:               inst78000085
// CHECK:STDOUT:       1:               inst780000EF
// CHECK:STDOUT:       2:               inst7800012F
// CHECK:STDOUT:     inst_block7800004C:
// CHECK:STDOUT:       0:               inst78000058
// CHECK:STDOUT:       1:               inst78000058
// CHECK:STDOUT:       2:               inst78000058
// CHECK:STDOUT:     inst_block7800004D:
// CHECK:STDOUT:       0:               inst78000092
// CHECK:STDOUT:       1:               inst78000085
// CHECK:STDOUT:       2:               inst780000FD
// CHECK:STDOUT:       3:               inst780000EF
// CHECK:STDOUT:       4:               inst7800013D
// CHECK:STDOUT:       5:               inst7800012F
// CHECK:STDOUT:       6:               inst7800013C
// CHECK:STDOUT:       7:               inst78000086
// CHECK:STDOUT:       8:               inst780000F0
// CHECK:STDOUT:       9:               inst78000130
// CHECK:STDOUT:       10:              inst78000131
// CHECK:STDOUT:       11:              inst78000132
// CHECK:STDOUT:       12:              inst78000135
// CHECK:STDOUT:     inst_block7800004E:
// CHECK:STDOUT:       0:               inst78000137
// CHECK:STDOUT:       1:               inst78000138
// CHECK:STDOUT:     inst_block7800004F:
// CHECK:STDOUT:       0:               inst78000146
// CHECK:STDOUT:       1:               inst78000147
// CHECK:STDOUT:     inst_block78000050:
// CHECK:STDOUT:       0:               inst78000149
// CHECK:STDOUT:     inst_block78000051:
// CHECK:STDOUT:       0:               inst7800014D
// CHECK:STDOUT:       1:               inst7800014E
// CHECK:STDOUT:       2:               inst7800014F
// CHECK:STDOUT:     inst_block78000052:
// CHECK:STDOUT:       0:               inst78000151
// CHECK:STDOUT:       1:               inst78000153
// CHECK:STDOUT:       2:               inst78000155
// CHECK:STDOUT:     inst_block78000053:
// CHECK:STDOUT:       0:               inst78000150
// CHECK:STDOUT:       1:               inst78000151
// CHECK:STDOUT:       2:               inst78000152
// CHECK:STDOUT:       3:               inst78000153
// CHECK:STDOUT:       4:               inst78000154
// CHECK:STDOUT:       5:               inst78000155
// CHECK:STDOUT:       6:               inst78000156
// CHECK:STDOUT:       7:               inst78000157
// CHECK:STDOUT:       8:               inst78000158
// CHECK:STDOUT:       9:               inst78000159
// CHECK:STDOUT:       10:              inst7800015A
// CHECK:STDOUT:       11:              inst7800015B
// CHECK:STDOUT:       12:              inst7800015C
// CHECK:STDOUT:     inst_block78000054:
// CHECK:STDOUT:       0:               inst78000085
// CHECK:STDOUT:       1:               inst78000086
// CHECK:STDOUT:       2:               inst780000EF
// CHECK:STDOUT:       3:               inst780000F0
// CHECK:STDOUT:       4:               inst7800012F
// CHECK:STDOUT:       5:               inst78000130
// CHECK:STDOUT:       6:               inst78000131
// CHECK:STDOUT:       7:               inst78000139
// CHECK:STDOUT:       8:               inst78000140
// CHECK:STDOUT:       9:               inst78000142
// CHECK:STDOUT:       10:              inst78000144
// CHECK:STDOUT:       11:              inst7800013A
// CHECK:STDOUT:       12:              inst7800013F
// CHECK:STDOUT:     inst_block78000055:
// CHECK:STDOUT:       0:               inst7800015F
// CHECK:STDOUT:       1:               inst78000160
// CHECK:STDOUT:       2:               inst78000161
// CHECK:STDOUT:     inst_block78000056:
// CHECK:STDOUT:       0:               inst78000165
// CHECK:STDOUT:       1:               inst78000166
// CHECK:STDOUT:       2:               inst78000167
// CHECK:STDOUT:     inst_block78000057:
// CHECK:STDOUT:       0:               inst78000169
// CHECK:STDOUT:       1:               inst7800016B
// CHECK:STDOUT:       2:               inst7800016D
// CHECK:STDOUT:     inst_block78000058:
// CHECK:STDOUT:       0:               inst7800016F
// CHECK:STDOUT:       1:               inst78000170
// CHECK:STDOUT:       2:               inst78000171
// CHECK:STDOUT:     inst_block78000059:
// CHECK:STDOUT:       0:               inst78000169
// CHECK:STDOUT:       1:               inst7800016B
// CHECK:STDOUT:       2:               inst7800016D
// CHECK:STDOUT:     inst_block7800005A:
// CHECK:STDOUT:       0:               inst78000168
// CHECK:STDOUT:       1:               inst78000169
// CHECK:STDOUT:       2:               inst7800016A
// CHECK:STDOUT:       3:               inst7800016B
// CHECK:STDOUT:       4:               inst7800016C
// CHECK:STDOUT:       5:               inst7800016D
// CHECK:STDOUT:       6:               inst7800016E
// CHECK:STDOUT:       7:               inst7800016F
// CHECK:STDOUT:       8:               inst78000170
// CHECK:STDOUT:       9:               inst78000171
// CHECK:STDOUT:       10:              inst78000172
// CHECK:STDOUT:       11:              inst78000173
// CHECK:STDOUT:       12:              inst78000174
// CHECK:STDOUT:     inst_block7800005B:
// CHECK:STDOUT:       0:               inst78000175
// CHECK:STDOUT:     inst_block7800005C:
// CHECK:STDOUT:       0:               inst78000176
// CHECK:STDOUT:     inst_block7800005D:
// CHECK:STDOUT:       0:               inst78000176
// CHECK:STDOUT:       1:               inst78000177
// CHECK:STDOUT:       2:               inst78000178
// CHECK:STDOUT:     inst_block7800005E:
// CHECK:STDOUT:       0:               inst7800017C
// CHECK:STDOUT:     inst_block7800005F:
// CHECK:STDOUT:       0:               inst7800017D
// CHECK:STDOUT:     inst_block78000060:
// CHECK:STDOUT:       0:               inst78000176
// CHECK:STDOUT:       1:               inst7800001D
// CHECK:STDOUT:       2:               inst7800001F
// CHECK:STDOUT:       3:               inst78000188
// CHECK:STDOUT:       4:               inst78000189
// CHECK:STDOUT:       5:               inst7800018A
// CHECK:STDOUT:       6:               inst7800018B
// CHECK:STDOUT:       7:               inst7800018C
// CHECK:STDOUT:     inst_block78000061:
// CHECK:STDOUT:       0:               inst7800017D
// CHECK:STDOUT:     inst_block78000062:
// CHECK:STDOUT:       0:               inst78000050
// CHECK:STDOUT:     inst_block78000063:
// CHECK:STDOUT:       0:               inst78000192
// CHECK:STDOUT:       1:               inst78000196
// CHECK:STDOUT:     inst_block78000064:
// CHECK:STDOUT:       0:               inst7800004B
// CHECK:STDOUT:       1:               inst7800004F
// CHECK:STDOUT:       2:               inst7800017C
// CHECK:STDOUT:       3:               inst7800017D
// CHECK:STDOUT:       4:               inst7800017E
// CHECK:STDOUT:       5:               inst7800017F
// CHECK:STDOUT:       6:               inst78000180
// CHECK:STDOUT:       7:               inst7800018F
// CHECK:STDOUT:     inst_block78000065:
// CHECK:STDOUT:       0:               inst(Package)
// CHECK:STDOUT:       1:               inst7800000F
// CHECK:STDOUT:       2:               inst78000046
// CHECK:STDOUT:   value_stores:
// CHECK:STDOUT:     shared_values:
// CHECK:STDOUT:       ints:            {}
// CHECK:STDOUT:       reals:           {}
// CHECK:STDOUT:       floats:          {}
// CHECK:STDOUT:       identifiers:
// CHECK:STDOUT:         identifier0:     Foo
// CHECK:STDOUT:         identifier1:     T
// CHECK:STDOUT:         identifier2:     p
// CHECK:STDOUT:         identifier3:     Copy
// CHECK:STDOUT:         identifier4:     Op
// CHECK:STDOUT:         identifier5:     U
// CHECK:STDOUT:         identifier6:     V
// CHECK:STDOUT:       strings:
// CHECK:STDOUT:         string0:         prelude
// CHECK:STDOUT: ...
