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
// CHECK:STDOUT:     import_ir60000002: {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir60000003: {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir60000004: {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir60000005: {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir60000004, inst_id: inst4800000F}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir60000004, inst_id: inst4800000F}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir60000004, inst_id: inst48000011}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir60000004, inst_id: inst4800002D}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir60000004, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: import_ir60000004, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: import_ir60000004, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: import_ir60000004, inst_id: inst48000022}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: import_ir60000004, inst_id: inst48000023}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: import_ir60000004, inst_id: inst4800001C}
// CHECK:STDOUT:     import_ir_instA: {ir_id: import_ir60000004, inst_id: inst4800001E}
// CHECK:STDOUT:     import_ir_instB: {ir_id: import_ir60000004, inst_id: inst48000011}
// CHECK:STDOUT:     import_ir_instC: {ir_id: import_ir60000004, inst_id: inst48000015}
// CHECK:STDOUT:     import_ir_instD: {ir_id: import_ir60000004, inst_id: inst48000018}
// CHECK:STDOUT:     import_ir_instE: {ir_id: import_ir60000004, inst_id: inst4800001D}
// CHECK:STDOUT:     import_ir_instF: {ir_id: import_ir60000004, inst_id: inst48000061}
// CHECK:STDOUT:     import_ir_inst10: {ir_id: import_ir60000004, inst_id: inst4800005F}
// CHECK:STDOUT:     import_ir_inst11: {ir_id: import_ir60000004, inst_id: inst48000054}
// CHECK:STDOUT:     import_ir_inst12: {ir_id: import_ir60000004, inst_id: inst48000050}
// CHECK:STDOUT:     import_ir_inst13: {ir_id: import_ir60000004, inst_id: inst4800005A}
// CHECK:STDOUT:     import_ir_inst14: {ir_id: import_ir60000004, inst_id: inst4800005D}
// CHECK:STDOUT:     import_ir_inst15: {ir_id: import_ir60000004, inst_id: inst48000075}
// CHECK:STDOUT:     import_ir_inst16: {ir_id: import_ir60000004, inst_id: inst48000060}
// CHECK:STDOUT:     import_ir_inst17: {ir_id: import_ir60000004, inst_id: inst48000052}
// CHECK:STDOUT:     import_ir_inst18: {ir_id: import_ir60000004, inst_id: inst48000058}
// CHECK:STDOUT:     import_ir_inst19: {ir_id: import_ir60000004, inst_id: inst4800005C}
// CHECK:STDOUT:     import_ir_inst1A: {ir_id: import_ir60000004, inst_id: inst48000063}
// CHECK:STDOUT:     import_ir_inst1B: {ir_id: import_ir60000004, inst_id: inst48000075}
// CHECK:STDOUT:     import_ir_inst1C: {ir_id: import_ir60000004, inst_id: inst48000070}
// CHECK:STDOUT:     import_ir_inst1D: {ir_id: import_ir60000004, inst_id: inst48000071}
// CHECK:STDOUT:     import_ir_inst1E: {ir_id: import_ir60000004, inst_id: inst4800006C}
// CHECK:STDOUT:     import_ir_inst1F: {ir_id: import_ir60000004, inst_id: inst4800006E}
// CHECK:STDOUT:     import_ir_inst20: {ir_id: import_ir60000004, inst_id: inst48000050}
// CHECK:STDOUT:     import_ir_inst21: {ir_id: import_ir60000004, inst_id: inst48000078}
// CHECK:STDOUT:     import_ir_inst22: {ir_id: import_ir60000004, inst_id: inst48000079}
// CHECK:STDOUT:     import_ir_inst23: {ir_id: import_ir60000004, inst_id: inst48000067}
// CHECK:STDOUT:     import_ir_inst24: {ir_id: import_ir60000004, inst_id: inst48000068}
// CHECK:STDOUT:     import_ir_inst25: {ir_id: import_ir60000004, inst_id: inst48000069}
// CHECK:STDOUT:     import_ir_inst26: {ir_id: import_ir60000004, inst_id: inst4800006D}
// CHECK:STDOUT:     import_ir_inst27: {ir_id: import_ir60000004, inst_id: inst4800007D}
// CHECK:STDOUT:     import_ir_inst28: {ir_id: import_ir60000004, inst_id: inst48000085}
// CHECK:STDOUT:     import_ir_inst29: {ir_id: import_ir60000004, inst_id: inst4800008C}
// CHECK:STDOUT:     import_ir_inst2A: {ir_id: import_ir60000004, inst_id: inst48000090}
// CHECK:STDOUT:     import_ir_inst2B: {ir_id: import_ir60000004, inst_id: inst48000091}
// CHECK:STDOUT:     import_ir_inst2C: {ir_id: import_ir60000004, inst_id: inst48000096}
// CHECK:STDOUT:     import_ir_inst2D: {ir_id: import_ir60000004, inst_id: inst480000AB}
// CHECK:STDOUT:     import_ir_inst2E: {ir_id: import_ir60000004, inst_id: inst480000A9}
// CHECK:STDOUT:     import_ir_inst2F: {ir_id: import_ir60000004, inst_id: inst480000A7}
// CHECK:STDOUT:     import_ir_inst30: {ir_id: import_ir60000004, inst_id: inst480000A8}
// CHECK:STDOUT:     import_ir_inst31: {ir_id: import_ir60000004, inst_id: inst480000C3}
// CHECK:STDOUT:     import_ir_inst32: {ir_id: import_ir60000004, inst_id: inst480000C1}
// CHECK:STDOUT:     import_ir_inst33: {ir_id: import_ir60000004, inst_id: inst480000BF}
// CHECK:STDOUT:     import_ir_inst34: {ir_id: import_ir60000004, inst_id: inst480000C0}
// CHECK:STDOUT:     import_ir_inst35: {ir_id: import_ir60000004, inst_id: inst480000DB}
// CHECK:STDOUT:     import_ir_inst36: {ir_id: import_ir60000004, inst_id: inst480000D9}
// CHECK:STDOUT:     import_ir_inst37: {ir_id: import_ir60000004, inst_id: inst480000D7}
// CHECK:STDOUT:     import_ir_inst38: {ir_id: import_ir60000004, inst_id: inst480000D8}
// CHECK:STDOUT:     import_ir_inst39: {ir_id: import_ir60000004, inst_id: inst480000F3}
// CHECK:STDOUT:     import_ir_inst3A: {ir_id: import_ir60000004, inst_id: inst480000F1}
// CHECK:STDOUT:     import_ir_inst3B: {ir_id: import_ir60000004, inst_id: inst480000EF}
// CHECK:STDOUT:     import_ir_inst3C: {ir_id: import_ir60000004, inst_id: inst480000F0}
// CHECK:STDOUT:     import_ir_inst3D: {ir_id: import_ir60000004, inst_id: inst48000110}
// CHECK:STDOUT:     import_ir_inst3E: {ir_id: import_ir60000004, inst_id: inst4800010E}
// CHECK:STDOUT:     import_ir_inst3F: {ir_id: import_ir60000004, inst_id: inst48000108}
// CHECK:STDOUT:     import_ir_inst40: {ir_id: import_ir60000004, inst_id: inst48000105}
// CHECK:STDOUT:     import_ir_inst41: {ir_id: import_ir60000004, inst_id: inst4800010A}
// CHECK:STDOUT:     import_ir_inst42: {ir_id: import_ir60000004, inst_id: inst4800010D}
// CHECK:STDOUT:     import_ir_inst43: {ir_id: import_ir60000004, inst_id: inst48000121}
// CHECK:STDOUT:     import_ir_inst44: {ir_id: import_ir60000004, inst_id: inst4800010F}
// CHECK:STDOUT:     import_ir_inst45: {ir_id: import_ir60000004, inst_id: inst48000107}
// CHECK:STDOUT:     import_ir_inst46: {ir_id: import_ir60000004, inst_id: inst4800010C}
// CHECK:STDOUT:     import_ir_inst47: {ir_id: import_ir60000004, inst_id: inst48000112}
// CHECK:STDOUT:     import_ir_inst48: {ir_id: import_ir60000004, inst_id: inst48000121}
// CHECK:STDOUT:     import_ir_inst49: {ir_id: import_ir60000004, inst_id: inst4800011C}
// CHECK:STDOUT:     import_ir_inst4A: {ir_id: import_ir60000004, inst_id: inst4800011D}
// CHECK:STDOUT:     import_ir_inst4B: {ir_id: import_ir60000004, inst_id: inst48000118}
// CHECK:STDOUT:     import_ir_inst4C: {ir_id: import_ir60000004, inst_id: inst4800011A}
// CHECK:STDOUT:     import_ir_inst4D: {ir_id: import_ir60000004, inst_id: inst48000105}
// CHECK:STDOUT:     import_ir_inst4E: {ir_id: import_ir60000004, inst_id: inst48000124}
// CHECK:STDOUT:     import_ir_inst4F: {ir_id: import_ir60000004, inst_id: inst48000125}
// CHECK:STDOUT:     import_ir_inst50: {ir_id: import_ir60000004, inst_id: inst48000128}
// CHECK:STDOUT:     import_ir_inst51: {ir_id: import_ir60000004, inst_id: inst48000114}
// CHECK:STDOUT:     import_ir_inst52: {ir_id: import_ir60000004, inst_id: inst48000115}
// CHECK:STDOUT:     import_ir_inst53: {ir_id: import_ir60000004, inst_id: inst48000119}
// CHECK:STDOUT:     import_ir_inst54: {ir_id: import_ir60000004, inst_id: inst4800012E}
// CHECK:STDOUT:     import_ir_inst55: {ir_id: import_ir60000004, inst_id: inst4800012C}
// CHECK:STDOUT:     import_ir_inst56: {ir_id: import_ir60000004, inst_id: inst(TypeType)}
// CHECK:STDOUT:     import_ir_inst57: {ir_id: import_ir60000004, inst_id: inst4800012B}
// CHECK:STDOUT:     import_ir_inst58: {ir_id: import_ir60000004, inst_id: inst48000144}
// CHECK:STDOUT:     import_ir_inst59: {ir_id: import_ir60000004, inst_id: inst48000142}
// CHECK:STDOUT:     import_ir_inst5A: {ir_id: import_ir60000004, inst_id: inst48000140}
// CHECK:STDOUT:     import_ir_inst5B: {ir_id: import_ir60000004, inst_id: inst48000141}
// CHECK:STDOUT:     import_ir_inst5C: {ir_id: import_ir60000004, inst_id: inst48000176}
// CHECK:STDOUT:     import_ir_inst5D: {ir_id: import_ir60000004, inst_id: inst48000174}
// CHECK:STDOUT:     import_ir_inst5E: {ir_id: import_ir60000004, inst_id: inst4800015F}
// CHECK:STDOUT:     import_ir_inst5F: {ir_id: import_ir60000004, inst_id: inst48000159}
// CHECK:STDOUT:     import_ir_inst60: {ir_id: import_ir60000004, inst_id: inst48000157}
// CHECK:STDOUT:     import_ir_inst61: {ir_id: import_ir60000004, inst_id: inst4800015C}
// CHECK:STDOUT:     import_ir_inst62: {ir_id: import_ir60000004, inst_id: inst4800016F}
// CHECK:STDOUT:     import_ir_inst63: {ir_id: import_ir60000004, inst_id: inst48000171}
// CHECK:STDOUT:     import_ir_inst64: {ir_id: import_ir60000004, inst_id: inst4800018E}
// CHECK:STDOUT:     import_ir_inst65: {ir_id: import_ir60000004, inst_id: inst48000175}
// CHECK:STDOUT:     import_ir_inst66: {ir_id: import_ir60000004, inst_id: inst48000158}
// CHECK:STDOUT:     import_ir_inst67: {ir_id: import_ir60000004, inst_id: inst4800015E}
// CHECK:STDOUT:     import_ir_inst68: {ir_id: import_ir60000004, inst_id: inst48000165}
// CHECK:STDOUT:     import_ir_inst69: {ir_id: import_ir60000004, inst_id: inst48000168}
// CHECK:STDOUT:     import_ir_inst6A: {ir_id: import_ir60000004, inst_id: inst4800016C}
// CHECK:STDOUT:     import_ir_inst6B: {ir_id: import_ir60000004, inst_id: inst48000170}
// CHECK:STDOUT:     import_ir_inst6C: {ir_id: import_ir60000004, inst_id: inst48000178}
// CHECK:STDOUT:     import_ir_inst6D: {ir_id: import_ir60000004, inst_id: inst4800018E}
// CHECK:STDOUT:     import_ir_inst6E: {ir_id: import_ir60000004, inst_id: inst48000189}
// CHECK:STDOUT:     import_ir_inst6F: {ir_id: import_ir60000004, inst_id: inst4800018A}
// CHECK:STDOUT:     import_ir_inst70: {ir_id: import_ir60000004, inst_id: inst48000185}
// CHECK:STDOUT:     import_ir_inst71: {ir_id: import_ir60000004, inst_id: inst48000187}
// CHECK:STDOUT:     import_ir_inst72: {ir_id: import_ir60000004, inst_id: inst48000157}
// CHECK:STDOUT:     import_ir_inst73: {ir_id: import_ir60000004, inst_id: inst4800015C}
// CHECK:STDOUT:     import_ir_inst74: {ir_id: import_ir60000004, inst_id: inst48000191}
// CHECK:STDOUT:     import_ir_inst75: {ir_id: import_ir60000004, inst_id: inst48000192}
// CHECK:STDOUT:     import_ir_inst76: {ir_id: import_ir60000004, inst_id: inst4800017E}
// CHECK:STDOUT:     import_ir_inst77: {ir_id: import_ir60000004, inst_id: inst4800017F}
// CHECK:STDOUT:     import_ir_inst78: {ir_id: import_ir60000004, inst_id: inst48000180}
// CHECK:STDOUT:     import_ir_inst79: {ir_id: import_ir60000004, inst_id: inst48000181}
// CHECK:STDOUT:     import_ir_inst7A: {ir_id: import_ir60000004, inst_id: inst48000182}
// CHECK:STDOUT:     import_ir_inst7B: {ir_id: import_ir60000004, inst_id: inst48000186}
// CHECK:STDOUT:     import_ir_inst7C: {ir_id: import_ir60000004, inst_id: inst48000197}
// CHECK:STDOUT:     import_ir_inst7D: {ir_id: import_ir60000004, inst_id: inst4800019F}
// CHECK:STDOUT:     import_ir_inst7E: {ir_id: import_ir60000004, inst_id: inst480001A3}
// CHECK:STDOUT:     import_ir_inst7F: {ir_id: import_ir60000004, inst_id: inst480001A5}
// CHECK:STDOUT:     import_ir_inst80: {ir_id: import_ir60000004, inst_id: inst480001A6}
// CHECK:STDOUT:     import_ir_inst81: {ir_id: import_ir60000004, inst_id: inst480001A9}
// CHECK:STDOUT:     import_ir_inst82: {ir_id: import_ir60000004, inst_id: inst480001B5}
// CHECK:STDOUT:     import_ir_inst83: {ir_id: import_ir60000004, inst_id: inst480001BA}
// CHECK:STDOUT:     import_ir_inst84: {ir_id: import_ir60000004, inst_id: inst480001BE}
// CHECK:STDOUT:     import_ir_inst85: {ir_id: import_ir60000004, inst_id: inst480001BF}
// CHECK:STDOUT:     import_ir_inst86: {ir_id: import_ir60000004, inst_id: inst480001C4}
// CHECK:STDOUT:     import_ir_inst87: {ir_id: import_ir60000004, inst_id: inst480001FF}
// CHECK:STDOUT:     import_ir_inst88: {ir_id: import_ir60000004, inst_id: inst480001FD}
// CHECK:STDOUT:     import_ir_inst89: {ir_id: import_ir60000004, inst_id: inst480001E3}
// CHECK:STDOUT:     import_ir_inst8A: {ir_id: import_ir60000004, inst_id: inst480001DD}
// CHECK:STDOUT:     import_ir_inst8B: {ir_id: import_ir60000004, inst_id: inst480001D8}
// CHECK:STDOUT:     import_ir_inst8C: {ir_id: import_ir60000004, inst_id: inst480001D6}
// CHECK:STDOUT:     import_ir_inst8D: {ir_id: import_ir60000004, inst_id: inst480001DB}
// CHECK:STDOUT:     import_ir_inst8E: {ir_id: import_ir60000004, inst_id: inst480001E0}
// CHECK:STDOUT:     import_ir_inst8F: {ir_id: import_ir60000004, inst_id: inst480001F7}
// CHECK:STDOUT:     import_ir_inst90: {ir_id: import_ir60000004, inst_id: inst480001F9}
// CHECK:STDOUT:     import_ir_inst91: {ir_id: import_ir60000004, inst_id: inst4800021B}
// CHECK:STDOUT:     import_ir_inst92: {ir_id: import_ir60000004, inst_id: inst480001FE}
// CHECK:STDOUT:     import_ir_inst93: {ir_id: import_ir60000004, inst_id: inst480001D7}
// CHECK:STDOUT:     import_ir_inst94: {ir_id: import_ir60000004, inst_id: inst480001DC}
// CHECK:STDOUT:     import_ir_inst95: {ir_id: import_ir60000004, inst_id: inst480001E2}
// CHECK:STDOUT:     import_ir_inst96: {ir_id: import_ir60000004, inst_id: inst480001EA}
// CHECK:STDOUT:     import_ir_inst97: {ir_id: import_ir60000004, inst_id: inst480001ED}
// CHECK:STDOUT:     import_ir_inst98: {ir_id: import_ir60000004, inst_id: inst480001F0}
// CHECK:STDOUT:     import_ir_inst99: {ir_id: import_ir60000004, inst_id: inst480001F4}
// CHECK:STDOUT:     import_ir_inst9A: {ir_id: import_ir60000004, inst_id: inst480001F8}
// CHECK:STDOUT:     import_ir_inst9B: {ir_id: import_ir60000004, inst_id: inst48000201}
// CHECK:STDOUT:     import_ir_inst9C: {ir_id: import_ir60000004, inst_id: inst4800021B}
// CHECK:STDOUT:     import_ir_inst9D: {ir_id: import_ir60000004, inst_id: inst48000216}
// CHECK:STDOUT:     import_ir_inst9E: {ir_id: import_ir60000004, inst_id: inst48000217}
// CHECK:STDOUT:     import_ir_inst9F: {ir_id: import_ir60000004, inst_id: inst48000212}
// CHECK:STDOUT:     import_ir_instA0: {ir_id: import_ir60000004, inst_id: inst48000214}
// CHECK:STDOUT:     import_ir_instA1: {ir_id: import_ir60000004, inst_id: inst480001D6}
// CHECK:STDOUT:     import_ir_instA2: {ir_id: import_ir60000004, inst_id: inst480001DB}
// CHECK:STDOUT:     import_ir_instA3: {ir_id: import_ir60000004, inst_id: inst480001E0}
// CHECK:STDOUT:     import_ir_instA4: {ir_id: import_ir60000004, inst_id: inst4800021E}
// CHECK:STDOUT:     import_ir_instA5: {ir_id: import_ir60000004, inst_id: inst4800021F}
// CHECK:STDOUT:     import_ir_instA6: {ir_id: import_ir60000004, inst_id: inst48000209}
// CHECK:STDOUT:     import_ir_instA7: {ir_id: import_ir60000004, inst_id: inst4800020A}
// CHECK:STDOUT:     import_ir_instA8: {ir_id: import_ir60000004, inst_id: inst4800020B}
// CHECK:STDOUT:     import_ir_instA9: {ir_id: import_ir60000004, inst_id: inst4800020C}
// CHECK:STDOUT:     import_ir_instAA: {ir_id: import_ir60000004, inst_id: inst4800020D}
// CHECK:STDOUT:     import_ir_instAB: {ir_id: import_ir60000004, inst_id: inst4800020E}
// CHECK:STDOUT:     import_ir_instAC: {ir_id: import_ir60000004, inst_id: inst4800020F}
// CHECK:STDOUT:     import_ir_instAD: {ir_id: import_ir60000004, inst_id: inst48000213}
// CHECK:STDOUT:     import_ir_instAE: {ir_id: import_ir60000004, inst_id: inst48000224}
// CHECK:STDOUT:     import_ir_instAF: {ir_id: import_ir60000004, inst_id: inst4800022B}
// CHECK:STDOUT:     import_ir_instB0: {ir_id: import_ir60000004, inst_id: inst4800022F}
// CHECK:STDOUT:     import_ir_instB1: {ir_id: import_ir60000004, inst_id: inst48000231}
// CHECK:STDOUT:     import_ir_instB2: {ir_id: import_ir60000004, inst_id: inst48000232}
// CHECK:STDOUT:     import_ir_instB3: {ir_id: import_ir60000004, inst_id: inst48000235}
// CHECK:STDOUT:     import_ir_instB4: {ir_id: import_ir60000004, inst_id: inst4800023F}
// CHECK:STDOUT:     import_ir_instB5: {ir_id: import_ir60000004, inst_id: inst48000243}
// CHECK:STDOUT:     import_ir_instB6: {ir_id: import_ir60000004, inst_id: inst48000245}
// CHECK:STDOUT:     import_ir_instB7: {ir_id: import_ir60000004, inst_id: inst48000246}
// CHECK:STDOUT:     import_ir_instB8: {ir_id: import_ir60000004, inst_id: inst48000249}
// CHECK:STDOUT:     import_ir_instB9: {ir_id: import_ir60000004, inst_id: inst48000255}
// CHECK:STDOUT:     import_ir_instBA: {ir_id: import_ir60000004, inst_id: inst4800025A}
// CHECK:STDOUT:     import_ir_instBB: {ir_id: import_ir60000004, inst_id: inst4800025E}
// CHECK:STDOUT:     import_ir_instBC: {ir_id: import_ir60000004, inst_id: inst4800025F}
// CHECK:STDOUT:     import_ir_instBD: {ir_id: import_ir60000004, inst_id: inst48000264}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instE, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst60000010, name0: inst60000038}}
// CHECK:STDOUT:     name_scope60000001: {inst: inst60000010, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: inst60000048}}
// CHECK:STDOUT:     name_scope60000002: {inst: inst60000049, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {name(SelfType): inst6000004C, name4: inst6000004D}}
// CHECK:STDOUT:     name_scope60000003: {inst: inst60000061, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000004: {inst: inst6000008E, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000005: {inst: inst60000092, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000006: {inst: inst60000096, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000007: {inst: inst6000009A, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000008: {inst: inst6000009E, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000009: {inst: inst600000B8, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6000000A: {inst: inst600000BC, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6000000B: {inst: inst600000C0, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6000000C: {inst: inst600000FB, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name60000000: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000001: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000002: {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000003: {name: name3, parent_scope: name_scope60000001, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000004: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000005: {name: name4, parent_scope: name_scope60000002, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000006: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000007: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000008: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000009: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000A: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000B: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000C: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000D: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000E: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000F: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000010: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000011: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000012: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000013: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000014: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000015: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000016: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000017: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000018: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000019: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000001A: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000001B: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000001C: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000001D: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000001E: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000001F: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000020: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000021: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000022: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000023: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000024: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000025: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000026: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000027: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000028: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000029: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name6000002A: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name6000002B: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name6000002C: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000002D: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000002E: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000002F: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000030: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name60000031: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000032: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000033: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name60000034: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000035: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000036: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name60000037: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000038: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000039: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000003A: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000003B: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000003C: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name6000003D: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function60000000: {name: name0, parent_scope: name_scope0, call_params_id: inst_block6000000F, return_slot_pattern: inst60000033, body: [inst_block60000016]}
// CHECK:STDOUT:     function60000001: {name: name4, parent_scope: name_scope60000002, return_slot_pattern: inst60000058}
// CHECK:STDOUT:     function60000002: {name: name4, parent_scope: name_scope60000003, return_slot_pattern: inst60000076}
// CHECK:STDOUT:     function60000003: {name: name4, parent_scope: name_scope60000008, return_slot_pattern: inst600000AD}
// CHECK:STDOUT:     function60000004: {name: name4, parent_scope: name_scope6000000B, return_slot_pattern: inst600000DB}
// CHECK:STDOUT:     function60000005: {name: name4, parent_scope: name_scope6000000C, return_slot_pattern: inst6000011A}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic60000000: {decl: inst60000038, bindings: inst_block60000012}
// CHECK:STDOUT:     generic60000001: {decl: inst60000052, bindings: inst_block6000001A}
// CHECK:STDOUT:     generic60000002: {decl: inst60000061, bindings: inst_block6000001F}
// CHECK:STDOUT:     generic60000003: {decl: inst60000071, bindings: inst_block60000026}
// CHECK:STDOUT:     generic60000004: {decl: inst6000009E, bindings: inst_block60000030}
// CHECK:STDOUT:     generic60000005: {decl: inst600000A9, bindings: inst_block60000036}
// CHECK:STDOUT:     generic60000006: {decl: inst600000C0, bindings: inst_block6000003D}
// CHECK:STDOUT:     generic60000007: {decl: inst600000D6, bindings: inst_block60000047}
// CHECK:STDOUT:     generic60000008: {decl: inst600000FB, bindings: inst_block60000055}
// CHECK:STDOUT:     generic60000009: {decl: inst60000115, bindings: inst_block6000005F}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific60000000: {generic: generic60000000, args: inst_block60000013}
// CHECK:STDOUT:     specific60000001: {generic: generic60000001, args: inst_block6000001C}
// CHECK:STDOUT:     specific60000002: {generic: generic60000002, args: inst_block60000021}
// CHECK:STDOUT:     specific60000003: {generic: generic60000002, args: inst_block60000022}
// CHECK:STDOUT:     specific60000004: {generic: generic60000003, args: inst_block60000021}
// CHECK:STDOUT:     specific60000005: {generic: generic60000001, args: inst_block60000021}
// CHECK:STDOUT:     specific60000006: {generic: generic60000001, args: inst_block6000002B}
// CHECK:STDOUT:     specific60000007: {generic: generic60000004, args: inst_block60000013}
// CHECK:STDOUT:     specific60000008: {generic: generic60000004, args: inst_block60000032}
// CHECK:STDOUT:     specific60000009: {generic: generic60000005, args: inst_block60000013}
// CHECK:STDOUT:     specific6000000A: {generic: generic60000006, args: inst_block6000003F}
// CHECK:STDOUT:     specific6000000B: {generic: generic60000006, args: inst_block60000041}
// CHECK:STDOUT:     specific6000000C: {generic: generic60000007, args: inst_block6000003F}
// CHECK:STDOUT:     specific6000000D: {generic: generic60000001, args: inst_block6000004D}
// CHECK:STDOUT:     specific6000000E: {generic: generic60000001, args: inst_block6000004E}
// CHECK:STDOUT:     specific6000000F: {generic: generic60000001, args: inst_block6000004F}
// CHECK:STDOUT:     specific60000010: {generic: generic60000008, args: inst_block60000057}
// CHECK:STDOUT:     specific60000011: {generic: generic60000008, args: inst_block60000059}
// CHECK:STDOUT:     specific60000012: {generic: generic60000009, args: inst_block60000057}
// CHECK:STDOUT:     specific60000013: {generic: generic60000001, args: inst_block60000065}
// CHECK:STDOUT:     specific60000014: {generic: generic60000001, args: inst_block60000066}
// CHECK:STDOUT:     specific60000015: {generic: generic60000001, args: inst_block60000067}
// CHECK:STDOUT:     specific60000016: {generic: generic60000001, args: inst_block60000068}
// CHECK:STDOUT:     specific60000017: {generic: generic60000001, args: inst_block6000006F}
// CHECK:STDOUT:     specific60000018: {generic: generic60000001, args: inst_block60000071}
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
// CHECK:STDOUT:     'type(inst6000002B)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst6000002B)}
// CHECK:STDOUT:     'type(inst60000027)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst6000002B)}
// CHECK:STDOUT:     'type(inst60000039)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(symbolic_constant3)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant3)}
// CHECK:STDOUT:     'type(symbolic_constantD)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constantD)}
// CHECK:STDOUT:     'type(symbolic_constant9)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constantD)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(symbolic_constant4)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant4)}
// CHECK:STDOUT:     'type(symbolic_constantA)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constantD)}
// CHECK:STDOUT:     'type(inst(InstType))':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(inst6000004A)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst6000004A)}
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     'type(symbolic_constant60000109)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(symbolic_constant6000010C)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instE:           {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst6000000F:    {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst60000010:    {kind: Namespace, arg0: name_scope60000001, arg1: inst6000000F, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000011:    {kind: FacetType, arg0: facet_type60000000, type: type(TypeType)}
// CHECK:STDOUT:     inst60000012:    {kind: SymbolicBinding, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000013:    {kind: SymbolicBinding, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000014:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000015:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000016:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000017:    {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst60000018:    {kind: SymbolicBindingPattern, arg0: entity_name60000001, type: type(inst60000017)}
// CHECK:STDOUT:     inst60000019:    {kind: NameRef, arg0: name1, arg1: inst60000014, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001A:    {kind: PointerType, arg0: inst60000019, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001B:    {kind: PointerType, arg0: inst60000015, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001C:    {kind: PointerType, arg0: inst60000016, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001D:    {kind: ValueBinding, arg0: entity_name60000002, arg1: inst60000034, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst6000001E:    {kind: PatternType, arg0: inst6000001B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001F:    {kind: ValueBindingPattern, arg0: entity_name60000002, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst60000020:    {kind: PatternType, arg0: inst6000001C, type: type(TypeType)}
// CHECK:STDOUT:     inst60000021:    {kind: ValueParamPattern, arg0: inst6000001F, arg1: call_param0, type: type(symbolic_constant6)}
// CHECK:STDOUT:     inst60000022:    {kind: NameRef, arg0: name1, arg1: inst60000014, type: type(TypeType)}
// CHECK:STDOUT:     inst60000023:    {kind: PointerType, arg0: inst60000022, type: type(TypeType)}
// CHECK:STDOUT:     inst60000024:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000025:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000026:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000027:    {kind: TupleType, arg0: inst_block60000009, type: type(TypeType)}
// CHECK:STDOUT:     inst60000028:    {kind: TupleLiteral, arg0: inst_block60000008, type: type(inst60000027)}
// CHECK:STDOUT:     inst60000029:    {kind: TupleValue, arg0: inst_block6000000A, type: type(inst60000027)}
// CHECK:STDOUT:     inst6000002A:    {kind: TupleValue, arg0: inst_block6000000B, type: type(inst60000027)}
// CHECK:STDOUT:     inst6000002B:    {kind: PointerType, arg0: inst60000027, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002C:    {kind: Converted, arg0: inst60000026, arg1: inst60000024, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002D:    {kind: TupleType, arg0: inst_block6000000D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002E:    {kind: Converted, arg0: inst60000028, arg1: inst6000002D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002F:    {kind: TupleType, arg0: inst_block6000000E, type: type(TypeType)}
// CHECK:STDOUT:     inst60000030:    {kind: PatternType, arg0: inst6000002D, type: type(TypeType)}
// CHECK:STDOUT:     inst60000031:    {kind: ReturnSlotPattern, arg0: inst6000002E, type: type(symbolic_constantC)}
// CHECK:STDOUT:     inst60000032:    {kind: PatternType, arg0: inst6000002F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000033:    {kind: OutParamPattern, arg0: inst60000031, arg1: call_param1, type: type(symbolic_constantC)}
// CHECK:STDOUT:     inst60000034:    {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000035:    {kind: SpliceBlock, arg0: inst_block60000006, arg1: inst6000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000036:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constantA)}
// CHECK:STDOUT:     inst60000037:    {kind: ReturnSlot, arg0: inst6000002D, arg1: inst60000036, type: type(symbolic_constantA)}
// CHECK:STDOUT:     inst60000038:    {kind: FunctionDecl, arg0: function60000000, arg1: inst_block60000011, type: type(inst60000039)}
// CHECK:STDOUT:     inst60000039:    {kind: FunctionType, arg0: function60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000003A:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000039)}
// CHECK:STDOUT:     inst6000003B:    {kind: PointerType, arg0: inst6000002D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000003C:    {kind: RequireCompleteType, arg0: inst6000002D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003D:    {kind: RequireCompleteType, arg0: inst6000002D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003E:    {kind: RequireCompleteType, arg0: inst6000002F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003F:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000040:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000041:    {kind: RequireCompleteType, arg0: inst6000001C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000042:    {kind: NameRef, arg0: name2, arg1: inst6000001D, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000043:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000044:    {kind: TupleLiteral, arg0: inst_block60000017, type: type(symbolic_constantA)}
// CHECK:STDOUT:     inst60000045:    {kind: RequireCompleteType, arg0: inst6000002D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000046:    {kind: TupleAccess, arg0: inst60000037, arg1: element0, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000047:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000048:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name60000003, type: type(TypeType)}
// CHECK:STDOUT:     inst60000049:    {kind: InterfaceDecl, arg0: interface60000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst6000004A:    {kind: FacetType, arg0: facet_type60000001, type: type(TypeType)}
// CHECK:STDOUT:     inst6000004B:    {kind: SymbolicBinding, arg0: entity_name60000004, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000004C:    {kind: ImportRefUnloaded, arg0: import_ir_inst2, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000004D:    {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst6000004F)}
// CHECK:STDOUT:     inst6000004E:    {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name60000005}
// CHECK:STDOUT:     inst6000004F:    {kind: AssociatedEntityType, arg0: interface60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000050:    {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(inst60000053)}
// CHECK:STDOUT:     inst60000051:    {kind: AssociatedEntity, arg0: element0, arg1: inst60000050, type: type(inst6000004F)}
// CHECK:STDOUT:     inst60000052:    {kind: FunctionDecl, arg0: function60000001, arg1: inst_block_empty, type: type(inst60000053)}
// CHECK:STDOUT:     inst60000053:    {kind: FunctionType, arg0: function60000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000054:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000053)}
// CHECK:STDOUT:     inst60000055:    {kind: SymbolicBindingType, arg0: entity_name60000004, arg1: inst6000004B, type: type(TypeType)}
// CHECK:STDOUT:     inst60000056:    {kind: PatternType, arg0: inst60000055, type: type(TypeType)}
// CHECK:STDOUT:     inst60000057:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant60000015)}
// CHECK:STDOUT:     inst60000058:    {kind: OutParamPattern, arg0: inst60000057, arg1: call_param1, type: type(symbolic_constant60000015)}
// CHECK:STDOUT:     inst60000059:    {kind: ValueBindingPattern, arg0: entity_name60000008, type: type(symbolic_constant60000015)}
// CHECK:STDOUT:     inst6000005A:    {kind: ValueParamPattern, arg0: inst60000059, arg1: call_param0, type: type(symbolic_constant60000015)}
// CHECK:STDOUT:     inst6000005B:    {kind: ImportRefLoaded, arg0: import_ir_instB, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000005C:    {kind: SymbolicBinding, arg0: entity_name60000004, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000005D:    {kind: SymbolicBindingType, arg0: entity_name60000004, arg1: inst6000005C, type: type(TypeType)}
// CHECK:STDOUT:     inst6000005E:    {kind: PatternType, arg0: inst6000005D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000005F:    {kind: LookupImplWitness, arg0: inst6000001B, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000060:    {kind: ImportRefUnloaded, arg0: import_ir_instF, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000061:    {kind: ImplDecl, arg0: impl60000000, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000062:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000063:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst60000062, type: type(TypeType)}
// CHECK:STDOUT:     inst60000064:    {kind: ConstType, arg0: inst60000063, type: type(TypeType)}
// CHECK:STDOUT:     inst60000065:    {kind: PatternType, arg0: inst6000004A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000066:    {kind: SymbolicBindingPattern, arg0: entity_name6000000D, type: type(inst60000065)}
// CHECK:STDOUT:     inst60000067:    {kind: ImportRefLoaded, arg0: import_ir_inst12, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000068:    {kind: ImportRefLoaded, arg0: import_ir_inst13, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000069:    {kind: ImportRefLoaded, arg0: import_ir_inst14, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000006A:    {kind: ImportRefUnloaded, arg0: import_ir_inst15, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000006B:    {kind: ImplWitnessTable, arg0: inst_block60000020, arg1: impl60000000}
// CHECK:STDOUT:     inst6000006C:    {kind: ImplWitness, arg0: inst6000006B, arg1: specific60000002, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000006D:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000006E:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst6000006D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000006F:    {kind: ConstType, arg0: inst6000006E, type: type(TypeType)}
// CHECK:STDOUT:     inst60000070:    {kind: ImplWitness, arg0: inst6000006B, arg1: specific60000003, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000071:    {kind: FunctionDecl, arg0: function60000002, arg1: inst_block_empty, type: type(symbolic_constant60000029)}
// CHECK:STDOUT:     inst60000072:    {kind: FunctionType, arg0: function60000002, arg1: specific60000002, type: type(TypeType)}
// CHECK:STDOUT:     inst60000073:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000029)}
// CHECK:STDOUT:     inst60000074:    {kind: PatternType, arg0: inst60000064, type: type(TypeType)}
// CHECK:STDOUT:     inst60000075:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant6000002E)}
// CHECK:STDOUT:     inst60000076:    {kind: OutParamPattern, arg0: inst60000075, arg1: call_param1, type: type(symbolic_constant6000002E)}
// CHECK:STDOUT:     inst60000077:    {kind: ValueBindingPattern, arg0: entity_name60000011, type: type(symbolic_constant6000002E)}
// CHECK:STDOUT:     inst60000078:    {kind: ValueParamPattern, arg0: inst60000077, arg1: call_param0, type: type(symbolic_constant6000002E)}
// CHECK:STDOUT:     inst60000079:    {kind: ImportRefLoaded, arg0: import_ir_inst20, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000007A:    {kind: FunctionType, arg0: function60000002, arg1: specific60000003, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007B:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000030)}
// CHECK:STDOUT:     inst6000007C:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000007D:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst6000007C, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007E:    {kind: ConstType, arg0: inst6000007D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007F:    {kind: PatternType, arg0: inst6000007E, type: type(TypeType)}
// CHECK:STDOUT:     inst60000080:    {kind: RequireCompleteType, arg0: inst60000064, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000081:    {kind: RequireCompleteType, arg0: inst60000063, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000082:    {kind: LookupImplWitness, arg0: inst60000062, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000083:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst60000062, type: type(TypeType)}
// CHECK:STDOUT:     inst60000084:    {kind: ImplWitnessAccess, arg0: inst60000082, arg1: element0, type: type(symbolic_constant6000003F)}
// CHECK:STDOUT:     inst60000085:    {kind: SpecificImplFunction, arg0: inst60000084, arg1: specific60000005, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000086:    {kind: RequireCompleteType, arg0: inst6000007E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000087:    {kind: RequireCompleteType, arg0: inst6000007D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000088:    {kind: LookupImplWitness, arg0: inst6000007C, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000089:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst6000007C, type: type(TypeType)}
// CHECK:STDOUT:     inst6000008A:    {kind: ImplWitnessAccess, arg0: inst60000088, arg1: element0, type: type(symbolic_constant60000048)}
// CHECK:STDOUT:     inst6000008B:    {kind: SpecificImplFunction, arg0: inst6000008A, arg1: specific60000006, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst6000008C:    {kind: PatternType, arg0: inst60000063, type: type(TypeType)}
// CHECK:STDOUT:     inst6000008D:    {kind: ImportRefUnloaded, arg0: import_ir_inst2D, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000008E:    {kind: ImplDecl, arg0: impl60000001, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000008F:    {kind: ImportRefLoaded, arg0: import_ir_inst2F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000090:    {kind: ImportRefLoaded, arg0: import_ir_inst30, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000091:    {kind: ImportRefUnloaded, arg0: import_ir_inst31, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000092:    {kind: ImplDecl, arg0: impl60000002, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000093:    {kind: ImportRefLoaded, arg0: import_ir_inst33, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000094:    {kind: ImportRefLoaded, arg0: import_ir_inst34, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000095:    {kind: ImportRefUnloaded, arg0: import_ir_inst35, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000096:    {kind: ImplDecl, arg0: impl60000003, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000097:    {kind: ImportRefLoaded, arg0: import_ir_inst37, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000098:    {kind: ImportRefLoaded, arg0: import_ir_inst38, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000099:    {kind: ImportRefUnloaded, arg0: import_ir_inst39, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000009A:    {kind: ImplDecl, arg0: impl60000004, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000009B:    {kind: ImportRefLoaded, arg0: import_ir_inst3B, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000009C:    {kind: ImportRefLoaded, arg0: import_ir_inst3C, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000009D:    {kind: ImportRefLoaded, arg0: import_ir_inst3D, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000009E:    {kind: ImplDecl, arg0: impl60000005, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000009F:    {kind: SymbolicBindingPattern, arg0: entity_name60000015, type: type(inst60000017)}
// CHECK:STDOUT:     inst600000A0:    {kind: ImportRefLoaded, arg0: import_ir_inst40, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A1:    {kind: ImportRefLoaded, arg0: import_ir_inst41, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A2:    {kind: ImportRefLoaded, arg0: import_ir_inst42, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A3:    {kind: ImportRefUnloaded, arg0: import_ir_inst43, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000A4:    {kind: ImplWitnessTable, arg0: inst_block60000031, arg1: impl60000005}
// CHECK:STDOUT:     inst600000A5:    {kind: ImplWitness, arg0: inst600000A4, arg1: specific60000007, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000A6:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A7:    {kind: PointerType, arg0: inst600000A6, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A8:    {kind: ImplWitness, arg0: inst600000A4, arg1: specific60000008, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000A9:    {kind: FunctionDecl, arg0: function60000003, arg1: inst_block_empty, type: type(symbolic_constant60000055)}
// CHECK:STDOUT:     inst600000AA:    {kind: FunctionType, arg0: function60000003, arg1: specific60000007, type: type(TypeType)}
// CHECK:STDOUT:     inst600000AB:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000055)}
// CHECK:STDOUT:     inst600000AC:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant60000059)}
// CHECK:STDOUT:     inst600000AD:    {kind: OutParamPattern, arg0: inst600000AC, arg1: call_param1, type: type(symbolic_constant60000059)}
// CHECK:STDOUT:     inst600000AE:    {kind: ValueBindingPattern, arg0: entity_name60000018, type: type(symbolic_constant60000059)}
// CHECK:STDOUT:     inst600000AF:    {kind: ValueParamPattern, arg0: inst600000AE, arg1: call_param0, type: type(symbolic_constant60000059)}
// CHECK:STDOUT:     inst600000B0:    {kind: ImportRefLoaded, arg0: import_ir_inst4D, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B1:    {kind: FunctionType, arg0: function60000003, arg1: specific60000008, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B2:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant6000005C)}
// CHECK:STDOUT:     inst600000B3:    {kind: RequireCompleteType, arg0: inst600000A7, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000B4:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B5:    {kind: PointerType, arg0: inst600000B4, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B6:    {kind: PatternType, arg0: inst600000B5, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B7:    {kind: ImportRefUnloaded, arg0: import_ir_inst54, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000B8:    {kind: ImplDecl, arg0: impl60000006, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000B9:    {kind: ImportRefLoaded, arg0: import_ir_inst56, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000BA:    {kind: ImportRefLoaded, arg0: import_ir_inst57, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000BB:    {kind: ImportRefUnloaded, arg0: import_ir_inst58, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000BC:    {kind: ImplDecl, arg0: impl60000007, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000BD:    {kind: ImportRefLoaded, arg0: import_ir_inst5A, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000BE:    {kind: ImportRefLoaded, arg0: import_ir_inst5B, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000BF:    {kind: ImportRefUnloaded, arg0: import_ir_inst5C, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000C0:    {kind: ImplDecl, arg0: impl60000008, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000C1:    {kind: SymbolicBinding, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000C2:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst600000C1, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C3:    {kind: TupleType, arg0: inst_block6000003B, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C4:    {kind: SymbolicBindingPattern, arg0: entity_name6000001C, type: type(inst60000065)}
// CHECK:STDOUT:     inst600000C5:    {kind: SymbolicBindingPattern, arg0: entity_name6000001D, type: type(inst60000065)}
// CHECK:STDOUT:     inst600000C6:    {kind: ImportRefLoaded, arg0: import_ir_inst60, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000C7:    {kind: ImportRefLoaded, arg0: import_ir_inst61, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000C8:    {kind: ImportRefLoaded, arg0: import_ir_inst62, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C9:    {kind: ImportRefLoaded, arg0: import_ir_inst63, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000CA:    {kind: TupleType, arg0: inst_block6000003E, type: type(TypeType)}
// CHECK:STDOUT:     inst600000CB:    {kind: TupleValue, arg0: inst_block6000003F, type: type(inst600000CA)}
// CHECK:STDOUT:     inst600000CC:    {kind: ImportRefUnloaded, arg0: import_ir_inst64, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000CD:    {kind: ImplWitnessTable, arg0: inst_block60000040, arg1: impl60000008}
// CHECK:STDOUT:     inst600000CE:    {kind: ImplWitness, arg0: inst600000CD, arg1: specific6000000A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000CF:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000D0:    {kind: SymbolicBinding, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000D1:    {kind: TupleValue, arg0: inst_block60000041, type: type(inst600000CA)}
// CHECK:STDOUT:     inst600000D2:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst600000CF, type: type(TypeType)}
// CHECK:STDOUT:     inst600000D3:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst600000D0, type: type(TypeType)}
// CHECK:STDOUT:     inst600000D4:    {kind: TupleType, arg0: inst_block60000042, type: type(TypeType)}
// CHECK:STDOUT:     inst600000D5:    {kind: ImplWitness, arg0: inst600000CD, arg1: specific6000000B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000D6:    {kind: FunctionDecl, arg0: function60000004, arg1: inst_block_empty, type: type(symbolic_constant6000007A)}
// CHECK:STDOUT:     inst600000D7:    {kind: FunctionType, arg0: function60000004, arg1: specific6000000A, type: type(TypeType)}
// CHECK:STDOUT:     inst600000D8:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant6000007A)}
// CHECK:STDOUT:     inst600000D9:    {kind: PatternType, arg0: inst600000C3, type: type(TypeType)}
// CHECK:STDOUT:     inst600000DA:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant6000007F)}
// CHECK:STDOUT:     inst600000DB:    {kind: OutParamPattern, arg0: inst600000DA, arg1: call_param1, type: type(symbolic_constant6000007F)}
// CHECK:STDOUT:     inst600000DC:    {kind: ValueBindingPattern, arg0: entity_name60000024, type: type(symbolic_constant6000007F)}
// CHECK:STDOUT:     inst600000DD:    {kind: ValueParamPattern, arg0: inst600000DC, arg1: call_param0, type: type(symbolic_constant6000007F)}
// CHECK:STDOUT:     inst600000DE:    {kind: ImportRefLoaded, arg0: import_ir_inst72, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000DF:    {kind: ImportRefLoaded, arg0: import_ir_inst73, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000E0:    {kind: FunctionType, arg0: function60000004, arg1: specific6000000B, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E1:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000081)}
// CHECK:STDOUT:     inst600000E2:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000E3:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst600000E2, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E4:    {kind: SymbolicBinding, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000E5:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst600000E4, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E6:    {kind: TupleType, arg0: inst_block6000004A, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E7:    {kind: PatternType, arg0: inst600000E6, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E8:    {kind: RequireCompleteType, arg0: inst600000C3, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E9:    {kind: RequireCompleteType, arg0: inst600000C2, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EA:    {kind: LookupImplWitness, arg0: inst600000C1, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EB:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst600000C1, type: type(TypeType)}
// CHECK:STDOUT:     inst600000EC:    {kind: ImplWitnessAccess, arg0: inst600000EA, arg1: element0, type: type(symbolic_constant60000099)}
// CHECK:STDOUT:     inst600000ED:    {kind: SpecificImplFunction, arg0: inst600000EC, arg1: specific6000000D, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000EE:    {kind: RequireCompleteType, arg0: inst600000E6, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EF:    {kind: RequireCompleteType, arg0: inst600000E3, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000F0:    {kind: LookupImplWitness, arg0: inst600000E2, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000F1:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst600000E2, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F2:    {kind: ImplWitnessAccess, arg0: inst600000F0, arg1: element0, type: type(symbolic_constant600000A2)}
// CHECK:STDOUT:     inst600000F3:    {kind: SpecificImplFunction, arg0: inst600000F2, arg1: specific6000000E, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000F4:    {kind: RequireCompleteType, arg0: inst600000E5, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000F5:    {kind: LookupImplWitness, arg0: inst600000E4, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000F6:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst600000E4, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F7:    {kind: ImplWitnessAccess, arg0: inst600000F5, arg1: element0, type: type(symbolic_constant600000A7)}
// CHECK:STDOUT:     inst600000F8:    {kind: SpecificImplFunction, arg0: inst600000F7, arg1: specific6000000F, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000F9:    {kind: PatternType, arg0: inst600000C2, type: type(TypeType)}
// CHECK:STDOUT:     inst600000FA:    {kind: ImportRefUnloaded, arg0: import_ir_inst87, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000FB:    {kind: ImplDecl, arg0: impl60000009, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000FC:    {kind: SymbolicBinding, arg0: entity_name60000029, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst600000FD:    {kind: SymbolicBindingType, arg0: entity_name60000029, arg1: inst600000FC, type: type(TypeType)}
// CHECK:STDOUT:     inst600000FE:    {kind: TupleType, arg0: inst_block60000053, type: type(TypeType)}
// CHECK:STDOUT:     inst600000FF:    {kind: SymbolicBindingPattern, arg0: entity_name6000002B, type: type(inst60000065)}
// CHECK:STDOUT:     inst60000100:    {kind: SymbolicBindingPattern, arg0: entity_name6000002C, type: type(inst60000065)}
// CHECK:STDOUT:     inst60000101:    {kind: SymbolicBindingPattern, arg0: entity_name6000002D, type: type(inst60000065)}
// CHECK:STDOUT:     inst60000102:    {kind: ImportRefLoaded, arg0: import_ir_inst8C, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000103:    {kind: ImportRefLoaded, arg0: import_ir_inst8D, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000104:    {kind: ImportRefLoaded, arg0: import_ir_inst8E, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000105:    {kind: ImportRefLoaded, arg0: import_ir_inst8F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000106:    {kind: ImportRefLoaded, arg0: import_ir_inst90, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000107:    {kind: TupleType, arg0: inst_block60000056, type: type(TypeType)}
// CHECK:STDOUT:     inst60000108:    {kind: TupleValue, arg0: inst_block60000057, type: type(inst60000107)}
// CHECK:STDOUT:     inst60000109:    {kind: ImportRefUnloaded, arg0: import_ir_inst91, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000010A:    {kind: ImplWitnessTable, arg0: inst_block60000058, arg1: impl60000009}
// CHECK:STDOUT:     inst6000010B:    {kind: ImplWitness, arg0: inst6000010A, arg1: specific60000010, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000010C:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000010D:    {kind: SymbolicBinding, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000010E:    {kind: SymbolicBinding, arg0: entity_name60000029, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000010F:    {kind: TupleValue, arg0: inst_block60000059, type: type(inst60000107)}
// CHECK:STDOUT:     inst60000110:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst6000010C, type: type(TypeType)}
// CHECK:STDOUT:     inst60000111:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst6000010D, type: type(TypeType)}
// CHECK:STDOUT:     inst60000112:    {kind: SymbolicBindingType, arg0: entity_name60000029, arg1: inst6000010E, type: type(TypeType)}
// CHECK:STDOUT:     inst60000113:    {kind: TupleType, arg0: inst_block6000005A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000114:    {kind: ImplWitness, arg0: inst6000010A, arg1: specific60000011, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000115:    {kind: FunctionDecl, arg0: function60000005, arg1: inst_block_empty, type: type(symbolic_constant600000C6)}
// CHECK:STDOUT:     inst60000116:    {kind: FunctionType, arg0: function60000005, arg1: specific60000010, type: type(TypeType)}
// CHECK:STDOUT:     inst60000117:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant600000C6)}
// CHECK:STDOUT:     inst60000118:    {kind: PatternType, arg0: inst600000FE, type: type(TypeType)}
// CHECK:STDOUT:     inst60000119:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant600000CB)}
// CHECK:STDOUT:     inst6000011A:    {kind: OutParamPattern, arg0: inst60000119, arg1: call_param1, type: type(symbolic_constant600000CB)}
// CHECK:STDOUT:     inst6000011B:    {kind: ValueBindingPattern, arg0: entity_name60000037, type: type(symbolic_constant600000CB)}
// CHECK:STDOUT:     inst6000011C:    {kind: ValueParamPattern, arg0: inst6000011B, arg1: call_param0, type: type(symbolic_constant600000CB)}
// CHECK:STDOUT:     inst6000011D:    {kind: ImportRefLoaded, arg0: import_ir_instA1, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000011E:    {kind: ImportRefLoaded, arg0: import_ir_instA2, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst6000011F:    {kind: ImportRefLoaded, arg0: import_ir_instA3, arg1: entity_name<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000120:    {kind: FunctionType, arg0: function60000005, arg1: specific60000011, type: type(TypeType)}
// CHECK:STDOUT:     inst60000121:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant600000CD)}
// CHECK:STDOUT:     inst60000122:    {kind: SymbolicBinding, arg0: entity_name60000001, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000123:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst60000122, type: type(TypeType)}
// CHECK:STDOUT:     inst60000124:    {kind: SymbolicBinding, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000125:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst60000124, type: type(TypeType)}
// CHECK:STDOUT:     inst60000126:    {kind: SymbolicBinding, arg0: entity_name60000029, arg1: inst<none>, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000127:    {kind: SymbolicBindingType, arg0: entity_name60000029, arg1: inst60000126, type: type(TypeType)}
// CHECK:STDOUT:     inst60000128:    {kind: TupleType, arg0: inst_block60000062, type: type(TypeType)}
// CHECK:STDOUT:     inst60000129:    {kind: PatternType, arg0: inst60000128, type: type(TypeType)}
// CHECK:STDOUT:     inst6000012A:    {kind: RequireCompleteType, arg0: inst600000FE, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012B:    {kind: RequireCompleteType, arg0: inst600000FD, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012C:    {kind: LookupImplWitness, arg0: inst600000FC, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012D:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst600000FC, type: type(TypeType)}
// CHECK:STDOUT:     inst6000012E:    {kind: ImplWitnessAccess, arg0: inst6000012C, arg1: element0, type: type(symbolic_constant600000EE)}
// CHECK:STDOUT:     inst6000012F:    {kind: SpecificImplFunction, arg0: inst6000012E, arg1: specific60000013, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000130:    {kind: RequireCompleteType, arg0: inst60000128, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000131:    {kind: RequireCompleteType, arg0: inst60000123, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000132:    {kind: LookupImplWitness, arg0: inst60000122, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000133:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst60000122, type: type(TypeType)}
// CHECK:STDOUT:     inst60000134:    {kind: ImplWitnessAccess, arg0: inst60000132, arg1: element0, type: type(symbolic_constant600000F7)}
// CHECK:STDOUT:     inst60000135:    {kind: SpecificImplFunction, arg0: inst60000134, arg1: specific60000014, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000136:    {kind: RequireCompleteType, arg0: inst60000125, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000137:    {kind: LookupImplWitness, arg0: inst60000124, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000138:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst60000124, type: type(TypeType)}
// CHECK:STDOUT:     inst60000139:    {kind: ImplWitnessAccess, arg0: inst60000137, arg1: element0, type: type(symbolic_constant600000FC)}
// CHECK:STDOUT:     inst6000013A:    {kind: SpecificImplFunction, arg0: inst60000139, arg1: specific60000015, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst6000013B:    {kind: RequireCompleteType, arg0: inst60000127, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000013C:    {kind: LookupImplWitness, arg0: inst60000126, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000013D:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst60000126, type: type(TypeType)}
// CHECK:STDOUT:     inst6000013E:    {kind: ImplWitnessAccess, arg0: inst6000013C, arg1: element0, type: type(symbolic_constant60000101)}
// CHECK:STDOUT:     inst6000013F:    {kind: SpecificImplFunction, arg0: inst6000013E, arg1: specific60000016, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000140:    {kind: PatternType, arg0: inst600000FD, type: type(TypeType)}
// CHECK:STDOUT:     inst60000141:    {kind: LookupImplWitness, arg0: inst6000001B, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000142:    {kind: LookupImplWitness, arg0: inst6000001C, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000143:    {kind: FacetValue, arg0: inst6000001B, arg1: inst_block6000006D, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000144:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst60000143, type: type(TypeType)}
// CHECK:STDOUT:     inst60000145:    {kind: ImplWitnessAccess, arg0: inst60000141, arg1: element0, type: type(symbolic_constant6000010C)}
// CHECK:STDOUT:     inst60000146:    {kind: ImplWitnessAccess, arg0: inst60000141, arg1: element0, type: type(symbolic_constant60000109)}
// CHECK:STDOUT:     inst60000147:    {kind: FacetValue, arg0: inst6000001C, arg1: inst_block6000006E, type: type(inst6000004A)}
// CHECK:STDOUT:     inst60000148:    {kind: FunctionTypeWithSelfType, arg0: inst60000053, arg1: inst60000147, type: type(TypeType)}
// CHECK:STDOUT:     inst60000149:    {kind: ImplWitnessAccess, arg0: inst60000142, arg1: element0, type: type(symbolic_constant6000010C)}
// CHECK:STDOUT:     inst6000014A:    {kind: BoundMethod, arg0: inst60000042, arg1: inst60000145, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst6000014B:    {kind: SpecificImplFunction, arg0: inst60000145, arg1: specific60000017, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst6000014C:    {kind: SpecificImplFunction, arg0: inst60000146, arg1: specific60000017, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst6000014D:    {kind: SpecificImplFunction, arg0: inst60000149, arg1: specific60000018, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst6000014E:    {kind: BoundMethod, arg0: inst60000042, arg1: inst6000014B, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst6000014F:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000150:    {kind: Call, arg0: inst6000014E, arg1: inst_block60000072, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000151:    {kind: InitializeFrom, arg0: inst60000150, arg1: inst60000046, type: type(symbolic_constant4)}
// CHECK:STDOUT:     inst60000152:    {kind: TupleAccess, arg0: inst60000037, arg1: element1, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000153:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst60000152, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000154:    {kind: Converted, arg0: inst60000043, arg1: inst60000153, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000155:    {kind: TupleInit, arg0: inst_block60000073, arg1: inst60000037, type: type(symbolic_constantA)}
// CHECK:STDOUT:     inst60000156:    {kind: Converted, arg0: inst60000044, arg1: inst60000155, type: type(symbolic_constantA)}
// CHECK:STDOUT:     inst60000157:    {kind: ReturnExpr, arg0: inst60000156, arg1: inst60000037}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       instE:           concrete_constant(instE)
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
// CHECK:STDOUT:       inst60000025:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000026:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000027:    concrete_constant(inst60000027)
// CHECK:STDOUT:       inst60000028:    symbolic_constant8
// CHECK:STDOUT:       inst60000029:    symbolic_constant7
// CHECK:STDOUT:       inst6000002A:    symbolic_constant8
// CHECK:STDOUT:       inst6000002B:    concrete_constant(inst6000002B)
// CHECK:STDOUT:       inst6000002C:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst6000002D:    symbolic_constant9
// CHECK:STDOUT:       inst6000002E:    symbolic_constantA
// CHECK:STDOUT:       inst6000002F:    symbolic_constantA
// CHECK:STDOUT:       inst60000030:    symbolic_constantB
// CHECK:STDOUT:       inst60000031:    concrete_constant(inst60000031)
// CHECK:STDOUT:       inst60000032:    symbolic_constantC
// CHECK:STDOUT:       inst60000033:    concrete_constant(inst60000033)
// CHECK:STDOUT:       inst60000035:    symbolic_constant4
// CHECK:STDOUT:       inst60000038:    concrete_constant(inst6000003A)
// CHECK:STDOUT:       inst60000039:    concrete_constant(inst60000039)
// CHECK:STDOUT:       inst6000003A:    concrete_constant(inst6000003A)
// CHECK:STDOUT:       inst6000003B:    symbolic_constantD
// CHECK:STDOUT:       inst6000003C:    symbolic_constant6000000F
// CHECK:STDOUT:       inst6000003D:    symbolic_constantE
// CHECK:STDOUT:       inst6000003E:    symbolic_constant6000000F
// CHECK:STDOUT:       inst6000003F:    symbolic_constant60000011
// CHECK:STDOUT:       inst60000040:    symbolic_constant60000010
// CHECK:STDOUT:       inst60000041:    symbolic_constant60000011
// CHECK:STDOUT:       inst60000043:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000045:    symbolic_constant6000000F
// CHECK:STDOUT:       inst60000047:    symbolic_constant60000011
// CHECK:STDOUT:       inst60000048:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst60000049:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst6000004A:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst6000004B:    symbolic_constant60000012
// CHECK:STDOUT:       inst6000004C:    constant<none>
// CHECK:STDOUT:       inst6000004D:    concrete_constant(inst60000051)
// CHECK:STDOUT:       inst6000004E:    constant<none>
// CHECK:STDOUT:       inst6000004F:    concrete_constant(inst6000004F)
// CHECK:STDOUT:       inst60000050:    concrete_constant(inst60000054)
// CHECK:STDOUT:       inst60000051:    concrete_constant(inst60000051)
// CHECK:STDOUT:       inst60000052:    concrete_constant(inst60000054)
// CHECK:STDOUT:       inst60000053:    concrete_constant(inst60000053)
// CHECK:STDOUT:       inst60000054:    concrete_constant(inst60000054)
// CHECK:STDOUT:       inst60000055:    symbolic_constant60000013
// CHECK:STDOUT:       inst60000056:    symbolic_constant60000014
// CHECK:STDOUT:       inst60000057:    concrete_constant(inst60000057)
// CHECK:STDOUT:       inst60000058:    concrete_constant(inst60000058)
// CHECK:STDOUT:       inst60000059:    concrete_constant(inst60000059)
// CHECK:STDOUT:       inst6000005A:    concrete_constant(inst6000005A)
// CHECK:STDOUT:       inst6000005B:    symbolic_constant60000012
// CHECK:STDOUT:       inst6000005C:    symbolic_constant60000018
// CHECK:STDOUT:       inst6000005D:    symbolic_constant60000019
// CHECK:STDOUT:       inst6000005E:    symbolic_constant6000001A
// CHECK:STDOUT:       inst6000005F:    symbolic_constant60000107
// CHECK:STDOUT:       inst60000060:    constant<none>
// CHECK:STDOUT:       inst60000061:    concrete_constant(inst60000061)
// CHECK:STDOUT:       inst60000062:    symbolic_constant6000001B
// CHECK:STDOUT:       inst60000063:    symbolic_constant6000001C
// CHECK:STDOUT:       inst60000064:    symbolic_constant6000001D
// CHECK:STDOUT:       inst60000065:    concrete_constant(inst60000065)
// CHECK:STDOUT:       inst60000066:    concrete_constant(inst60000066)
// CHECK:STDOUT:       inst60000067:    symbolic_constant6000001F
// CHECK:STDOUT:       inst60000068:    symbolic_constant6000001E
// CHECK:STDOUT:       inst60000069:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst6000006A:    constant<none>
// CHECK:STDOUT:       inst6000006B:    concrete_constant(inst6000006B)
// CHECK:STDOUT:       inst6000006C:    symbolic_constant60000023
// CHECK:STDOUT:       inst6000006D:    symbolic_constant60000025
// CHECK:STDOUT:       inst6000006E:    symbolic_constant60000026
// CHECK:STDOUT:       inst6000006F:    symbolic_constant60000027
// CHECK:STDOUT:       inst60000070:    symbolic_constant60000028
// CHECK:STDOUT:       inst60000071:    symbolic_constant6000002B
// CHECK:STDOUT:       inst60000072:    symbolic_constant60000029
// CHECK:STDOUT:       inst60000073:    symbolic_constant6000002A
// CHECK:STDOUT:       inst60000074:    symbolic_constant6000002D
// CHECK:STDOUT:       inst60000075:    concrete_constant(inst60000075)
// CHECK:STDOUT:       inst60000076:    concrete_constant(inst60000076)
// CHECK:STDOUT:       inst60000077:    concrete_constant(inst60000077)
// CHECK:STDOUT:       inst60000078:    concrete_constant(inst60000078)
// CHECK:STDOUT:       inst60000079:    symbolic_constant6000001F
// CHECK:STDOUT:       inst6000007A:    symbolic_constant60000030
// CHECK:STDOUT:       inst6000007B:    symbolic_constant60000031
// CHECK:STDOUT:       inst6000007C:    symbolic_constant60000035
// CHECK:STDOUT:       inst6000007D:    symbolic_constant60000036
// CHECK:STDOUT:       inst6000007E:    symbolic_constant60000037
// CHECK:STDOUT:       inst6000007F:    symbolic_constant60000038
// CHECK:STDOUT:       inst60000080:    symbolic_constant60000039
// CHECK:STDOUT:       inst60000081:    symbolic_constant6000003B
// CHECK:STDOUT:       inst60000082:    symbolic_constant6000003D
// CHECK:STDOUT:       inst60000083:    symbolic_constant6000003F
// CHECK:STDOUT:       inst60000084:    symbolic_constant60000041
// CHECK:STDOUT:       inst60000085:    symbolic_constant60000043
// CHECK:STDOUT:       inst60000086:    symbolic_constant60000045
// CHECK:STDOUT:       inst60000087:    symbolic_constant60000046
// CHECK:STDOUT:       inst60000088:    symbolic_constant60000047
// CHECK:STDOUT:       inst60000089:    symbolic_constant60000048
// CHECK:STDOUT:       inst6000008A:    symbolic_constant60000049
// CHECK:STDOUT:       inst6000008B:    symbolic_constant6000004A
// CHECK:STDOUT:       inst6000008C:    symbolic_constant6000004B
// CHECK:STDOUT:       inst6000008D:    constant<none>
// CHECK:STDOUT:       inst6000008E:    concrete_constant(inst6000008E)
// CHECK:STDOUT:       inst6000008F:    concrete_constant(inst(BoolType))
// CHECK:STDOUT:       inst60000090:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst60000091:    constant<none>
// CHECK:STDOUT:       inst60000092:    concrete_constant(inst60000092)
// CHECK:STDOUT:       inst60000093:    concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       inst60000094:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst60000095:    constant<none>
// CHECK:STDOUT:       inst60000096:    concrete_constant(inst60000096)
// CHECK:STDOUT:       inst60000097:    concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       inst60000098:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst60000099:    constant<none>
// CHECK:STDOUT:       inst6000009A:    concrete_constant(inst6000009A)
// CHECK:STDOUT:       inst6000009B:    concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       inst6000009C:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst6000009D:    symbolic_constant60000105
// CHECK:STDOUT:       inst6000009E:    concrete_constant(inst6000009E)
// CHECK:STDOUT:       inst6000009F:    concrete_constant(inst6000009F)
// CHECK:STDOUT:       inst600000A0:    symbolic_constant6000004D
// CHECK:STDOUT:       inst600000A1:    symbolic_constant6000004C
// CHECK:STDOUT:       inst600000A2:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst600000A3:    constant<none>
// CHECK:STDOUT:       inst600000A4:    concrete_constant(inst600000A4)
// CHECK:STDOUT:       inst600000A5:    symbolic_constant60000050
// CHECK:STDOUT:       inst600000A6:    symbolic_constant60000052
// CHECK:STDOUT:       inst600000A7:    symbolic_constant60000053
// CHECK:STDOUT:       inst600000A8:    symbolic_constant60000054
// CHECK:STDOUT:       inst600000A9:    symbolic_constant60000057
// CHECK:STDOUT:       inst600000AA:    symbolic_constant60000055
// CHECK:STDOUT:       inst600000AB:    symbolic_constant60000056
// CHECK:STDOUT:       inst600000AC:    concrete_constant(inst600000AC)
// CHECK:STDOUT:       inst600000AD:    concrete_constant(inst600000AD)
// CHECK:STDOUT:       inst600000AE:    concrete_constant(inst600000AE)
// CHECK:STDOUT:       inst600000AF:    concrete_constant(inst600000AF)
// CHECK:STDOUT:       inst600000B0:    symbolic_constant6000004D
// CHECK:STDOUT:       inst600000B1:    symbolic_constant6000005C
// CHECK:STDOUT:       inst600000B2:    symbolic_constant6000005D
// CHECK:STDOUT:       inst600000B3:    symbolic_constant6000005E
// CHECK:STDOUT:       inst600000B4:    symbolic_constant60000061
// CHECK:STDOUT:       inst600000B5:    symbolic_constant60000062
// CHECK:STDOUT:       inst600000B6:    symbolic_constant60000063
// CHECK:STDOUT:       inst600000B7:    constant<none>
// CHECK:STDOUT:       inst600000B8:    concrete_constant(inst600000B8)
// CHECK:STDOUT:       inst600000B9:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst600000BA:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst600000BB:    constant<none>
// CHECK:STDOUT:       inst600000BC:    concrete_constant(inst600000BC)
// CHECK:STDOUT:       inst600000BD:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst600000BE:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst600000BF:    constant<none>
// CHECK:STDOUT:       inst600000C0:    concrete_constant(inst600000C0)
// CHECK:STDOUT:       inst600000C1:    symbolic_constant60000064
// CHECK:STDOUT:       inst600000C2:    symbolic_constant60000065
// CHECK:STDOUT:       inst600000C3:    symbolic_constant60000066
// CHECK:STDOUT:       inst600000C4:    concrete_constant(inst600000C4)
// CHECK:STDOUT:       inst600000C5:    concrete_constant(inst600000C5)
// CHECK:STDOUT:       inst600000C6:    symbolic_constant60000068
// CHECK:STDOUT:       inst600000C7:    symbolic_constant60000069
// CHECK:STDOUT:       inst600000C8:    symbolic_constant60000067
// CHECK:STDOUT:       inst600000C9:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst600000CA:    concrete_constant(inst600000CA)
// CHECK:STDOUT:       inst600000CB:    symbolic_constant6000006C
// CHECK:STDOUT:       inst600000CC:    constant<none>
// CHECK:STDOUT:       inst600000CD:    concrete_constant(inst600000CD)
// CHECK:STDOUT:       inst600000CE:    symbolic_constant60000071
// CHECK:STDOUT:       inst600000CF:    symbolic_constant60000073
// CHECK:STDOUT:       inst600000D0:    symbolic_constant60000074
// CHECK:STDOUT:       inst600000D1:    symbolic_constant60000075
// CHECK:STDOUT:       inst600000D2:    symbolic_constant60000076
// CHECK:STDOUT:       inst600000D3:    symbolic_constant60000077
// CHECK:STDOUT:       inst600000D4:    symbolic_constant60000078
// CHECK:STDOUT:       inst600000D5:    symbolic_constant60000079
// CHECK:STDOUT:       inst600000D6:    symbolic_constant6000007C
// CHECK:STDOUT:       inst600000D7:    symbolic_constant6000007A
// CHECK:STDOUT:       inst600000D8:    symbolic_constant6000007B
// CHECK:STDOUT:       inst600000D9:    symbolic_constant6000007E
// CHECK:STDOUT:       inst600000DA:    concrete_constant(inst600000DA)
// CHECK:STDOUT:       inst600000DB:    concrete_constant(inst600000DB)
// CHECK:STDOUT:       inst600000DC:    concrete_constant(inst600000DC)
// CHECK:STDOUT:       inst600000DD:    concrete_constant(inst600000DD)
// CHECK:STDOUT:       inst600000DE:    symbolic_constant60000068
// CHECK:STDOUT:       inst600000DF:    symbolic_constant60000069
// CHECK:STDOUT:       inst600000E0:    symbolic_constant60000081
// CHECK:STDOUT:       inst600000E1:    symbolic_constant60000082
// CHECK:STDOUT:       inst600000E2:    symbolic_constant60000088
// CHECK:STDOUT:       inst600000E3:    symbolic_constant60000089
// CHECK:STDOUT:       inst600000E4:    symbolic_constant6000008A
// CHECK:STDOUT:       inst600000E5:    symbolic_constant6000008B
// CHECK:STDOUT:       inst600000E6:    symbolic_constant6000008C
// CHECK:STDOUT:       inst600000E7:    symbolic_constant6000008D
// CHECK:STDOUT:       inst600000E8:    symbolic_constant6000008E
// CHECK:STDOUT:       inst600000E9:    symbolic_constant60000095
// CHECK:STDOUT:       inst600000EA:    symbolic_constant60000097
// CHECK:STDOUT:       inst600000EB:    symbolic_constant60000099
// CHECK:STDOUT:       inst600000EC:    symbolic_constant6000009B
// CHECK:STDOUT:       inst600000ED:    symbolic_constant6000009D
// CHECK:STDOUT:       inst600000EE:    symbolic_constant6000009F
// CHECK:STDOUT:       inst600000EF:    symbolic_constant600000A0
// CHECK:STDOUT:       inst600000F0:    symbolic_constant600000A1
// CHECK:STDOUT:       inst600000F1:    symbolic_constant600000A2
// CHECK:STDOUT:       inst600000F2:    symbolic_constant600000A3
// CHECK:STDOUT:       inst600000F3:    symbolic_constant600000A4
// CHECK:STDOUT:       inst600000F4:    symbolic_constant600000A5
// CHECK:STDOUT:       inst600000F5:    symbolic_constant600000A6
// CHECK:STDOUT:       inst600000F6:    symbolic_constant600000A7
// CHECK:STDOUT:       inst600000F7:    symbolic_constant600000A8
// CHECK:STDOUT:       inst600000F8:    symbolic_constant600000A9
// CHECK:STDOUT:       inst600000F9:    symbolic_constant600000AA
// CHECK:STDOUT:       inst600000FA:    constant<none>
// CHECK:STDOUT:       inst600000FB:    concrete_constant(inst600000FB)
// CHECK:STDOUT:       inst600000FC:    symbolic_constant600000AB
// CHECK:STDOUT:       inst600000FD:    symbolic_constant600000AC
// CHECK:STDOUT:       inst600000FE:    symbolic_constant600000AD
// CHECK:STDOUT:       inst600000FF:    concrete_constant(inst600000FF)
// CHECK:STDOUT:       inst60000100:    concrete_constant(inst60000100)
// CHECK:STDOUT:       inst60000101:    concrete_constant(inst60000101)
// CHECK:STDOUT:       inst60000102:    symbolic_constant600000AF
// CHECK:STDOUT:       inst60000103:    symbolic_constant600000B0
// CHECK:STDOUT:       inst60000104:    symbolic_constant600000B1
// CHECK:STDOUT:       inst60000105:    symbolic_constant600000AE
// CHECK:STDOUT:       inst60000106:    concrete_constant(inst6000004A)
// CHECK:STDOUT:       inst60000107:    concrete_constant(inst60000107)
// CHECK:STDOUT:       inst60000108:    symbolic_constant600000B5
// CHECK:STDOUT:       inst60000109:    constant<none>
// CHECK:STDOUT:       inst6000010A:    concrete_constant(inst6000010A)
// CHECK:STDOUT:       inst6000010B:    symbolic_constant600000BB
// CHECK:STDOUT:       inst6000010C:    symbolic_constant600000BD
// CHECK:STDOUT:       inst6000010D:    symbolic_constant600000BE
// CHECK:STDOUT:       inst6000010E:    symbolic_constant600000BF
// CHECK:STDOUT:       inst6000010F:    symbolic_constant600000C0
// CHECK:STDOUT:       inst60000110:    symbolic_constant600000C1
// CHECK:STDOUT:       inst60000111:    symbolic_constant600000C2
// CHECK:STDOUT:       inst60000112:    symbolic_constant600000C3
// CHECK:STDOUT:       inst60000113:    symbolic_constant600000C4
// CHECK:STDOUT:       inst60000114:    symbolic_constant600000C5
// CHECK:STDOUT:       inst60000115:    symbolic_constant600000C8
// CHECK:STDOUT:       inst60000116:    symbolic_constant600000C6
// CHECK:STDOUT:       inst60000117:    symbolic_constant600000C7
// CHECK:STDOUT:       inst60000118:    symbolic_constant600000CA
// CHECK:STDOUT:       inst60000119:    concrete_constant(inst60000119)
// CHECK:STDOUT:       inst6000011A:    concrete_constant(inst6000011A)
// CHECK:STDOUT:       inst6000011B:    concrete_constant(inst6000011B)
// CHECK:STDOUT:       inst6000011C:    concrete_constant(inst6000011C)
// CHECK:STDOUT:       inst6000011D:    symbolic_constant600000AF
// CHECK:STDOUT:       inst6000011E:    symbolic_constant600000B0
// CHECK:STDOUT:       inst6000011F:    symbolic_constant600000B1
// CHECK:STDOUT:       inst60000120:    symbolic_constant600000CD
// CHECK:STDOUT:       inst60000121:    symbolic_constant600000CE
// CHECK:STDOUT:       inst60000122:    symbolic_constant600000D6
// CHECK:STDOUT:       inst60000123:    symbolic_constant600000D7
// CHECK:STDOUT:       inst60000124:    symbolic_constant600000D8
// CHECK:STDOUT:       inst60000125:    symbolic_constant600000D9
// CHECK:STDOUT:       inst60000126:    symbolic_constant600000DA
// CHECK:STDOUT:       inst60000127:    symbolic_constant600000DB
// CHECK:STDOUT:       inst60000128:    symbolic_constant600000DC
// CHECK:STDOUT:       inst60000129:    symbolic_constant600000DD
// CHECK:STDOUT:       inst6000012A:    symbolic_constant600000DE
// CHECK:STDOUT:       inst6000012B:    symbolic_constant600000EA
// CHECK:STDOUT:       inst6000012C:    symbolic_constant600000EC
// CHECK:STDOUT:       inst6000012D:    symbolic_constant600000EE
// CHECK:STDOUT:       inst6000012E:    symbolic_constant600000F0
// CHECK:STDOUT:       inst6000012F:    symbolic_constant600000F2
// CHECK:STDOUT:       inst60000130:    symbolic_constant600000F4
// CHECK:STDOUT:       inst60000131:    symbolic_constant600000F5
// CHECK:STDOUT:       inst60000132:    symbolic_constant600000F6
// CHECK:STDOUT:       inst60000133:    symbolic_constant600000F7
// CHECK:STDOUT:       inst60000134:    symbolic_constant600000F8
// CHECK:STDOUT:       inst60000135:    symbolic_constant600000F9
// CHECK:STDOUT:       inst60000136:    symbolic_constant600000FA
// CHECK:STDOUT:       inst60000137:    symbolic_constant600000FB
// CHECK:STDOUT:       inst60000138:    symbolic_constant600000FC
// CHECK:STDOUT:       inst60000139:    symbolic_constant600000FD
// CHECK:STDOUT:       inst6000013A:    symbolic_constant600000FE
// CHECK:STDOUT:       inst6000013B:    symbolic_constant600000FF
// CHECK:STDOUT:       inst6000013C:    symbolic_constant60000100
// CHECK:STDOUT:       inst6000013D:    symbolic_constant60000101
// CHECK:STDOUT:       inst6000013E:    symbolic_constant60000102
// CHECK:STDOUT:       inst6000013F:    symbolic_constant60000103
// CHECK:STDOUT:       inst60000140:    symbolic_constant60000104
// CHECK:STDOUT:       inst60000141:    symbolic_constant60000106
// CHECK:STDOUT:       inst60000142:    symbolic_constant60000107
// CHECK:STDOUT:       inst60000143:    symbolic_constant60000108
// CHECK:STDOUT:       inst60000144:    symbolic_constant60000109
// CHECK:STDOUT:       inst60000145:    symbolic_constant6000010D
// CHECK:STDOUT:       inst60000146:    symbolic_constant6000010A
// CHECK:STDOUT:       inst60000147:    symbolic_constant6000010B
// CHECK:STDOUT:       inst60000148:    symbolic_constant6000010C
// CHECK:STDOUT:       inst60000149:    symbolic_constant6000010D
// CHECK:STDOUT:       inst6000014B:    symbolic_constant6000010F
// CHECK:STDOUT:       inst6000014C:    symbolic_constant6000010E
// CHECK:STDOUT:       inst6000014D:    symbolic_constant6000010F
// CHECK:STDOUT:       inst6000014F:    symbolic_constant60000011
// CHECK:STDOUT:       inst60000153:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000154:    concrete_constant(inst60000026)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant0: {inst: inst60000013, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant1: {inst: inst60000015, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant2: {inst: inst60000015, generic: generic60000000, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant3: {inst: inst6000001B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant4: {inst: inst6000001B, generic: generic60000000, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant5: {inst: inst6000001E, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6: {inst: inst6000001E, generic: generic60000000, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant7: {inst: inst60000029, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant8: {inst: inst60000029, generic: generic60000000, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant9: {inst: inst6000002D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constantA: {inst: inst6000002D, generic: generic60000000, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constantB: {inst: inst60000030, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constantC: {inst: inst60000030, generic: generic60000000, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constantD: {inst: inst6000003B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constantE: {inst: inst6000003D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000000F: {inst: inst6000003D, generic: generic60000000, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000010: {inst: inst60000040, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000011: {inst: inst60000040, generic: generic60000000, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000012: {inst: inst6000004B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000013: {inst: inst60000055, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000014: {inst: inst60000056, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000015: {inst: inst60000056, generic: generic60000001, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000016: {inst: inst6000004B, generic: generic60000001, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000017: {inst: inst60000055, generic: generic60000001, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000018: {inst: inst6000004B, generic: generic60000001, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000019: {inst: inst60000055, generic: generic60000001, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001A: {inst: inst60000056, generic: generic60000001, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001B: {inst: inst60000062, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001C: {inst: inst60000063, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001D: {inst: inst60000064, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001E: {inst: inst60000064, generic: generic60000002, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001F: {inst: inst60000062, generic: generic60000002, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000020: {inst: inst60000062, generic: generic60000002, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000021: {inst: inst60000063, generic: generic60000002, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000022: {inst: inst60000064, generic: generic60000002, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000023: {inst: inst6000006C, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000024: {inst: inst6000006C, generic: generic60000002, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000025: {inst: inst60000062, generic: generic60000002, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000026: {inst: inst60000063, generic: generic60000002, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000027: {inst: inst60000064, generic: generic60000002, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000028: {inst: inst6000006C, generic: generic60000002, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000029: {inst: inst60000072, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002A: {inst: inst60000073, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002B: {inst: inst60000073, generic: generic60000002, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002C: {inst: inst60000072, generic: generic60000002, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002D: {inst: inst60000074, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002E: {inst: inst60000074, generic: generic60000003, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002F: {inst: inst60000073, generic: generic60000002, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000030: {inst: inst60000072, generic: generic60000002, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000031: {inst: inst60000073, generic: generic60000002, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000032: {inst: inst60000062, generic: generic60000003, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000033: {inst: inst60000063, generic: generic60000003, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000034: {inst: inst60000064, generic: generic60000003, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000035: {inst: inst60000062, generic: generic60000003, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000036: {inst: inst60000063, generic: generic60000003, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000037: {inst: inst60000064, generic: generic60000003, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000038: {inst: inst60000074, generic: generic60000003, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000039: {inst: inst60000080, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003A: {inst: inst60000080, generic: generic60000003, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003B: {inst: inst60000081, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003C: {inst: inst60000081, generic: generic60000003, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003D: {inst: inst60000082, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003E: {inst: inst60000082, generic: generic60000003, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003F: {inst: inst60000083, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000040: {inst: inst60000083, generic: generic60000003, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000041: {inst: inst60000084, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000042: {inst: inst60000084, generic: generic60000003, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000043: {inst: inst60000085, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000044: {inst: inst60000085, generic: generic60000003, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000045: {inst: inst60000080, generic: generic60000003, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000046: {inst: inst60000081, generic: generic60000003, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000047: {inst: inst60000082, generic: generic60000003, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000048: {inst: inst60000083, generic: generic60000003, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000049: {inst: inst60000084, generic: generic60000003, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004A: {inst: inst60000085, generic: generic60000003, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004B: {inst: inst6000008C, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004C: {inst: inst6000001B, generic: generic60000004, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004D: {inst: inst60000015, generic: generic60000004, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004E: {inst: inst60000015, generic: generic60000004, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004F: {inst: inst6000001B, generic: generic60000004, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000050: {inst: inst600000A5, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000051: {inst: inst600000A5, generic: generic60000004, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000052: {inst: inst60000015, generic: generic60000004, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000053: {inst: inst6000001B, generic: generic60000004, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000054: {inst: inst600000A5, generic: generic60000004, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000055: {inst: inst600000AA, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000056: {inst: inst600000AB, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000057: {inst: inst600000AB, generic: generic60000004, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000058: {inst: inst600000AA, generic: generic60000004, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000059: {inst: inst6000001E, generic: generic60000005, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005A: {inst: inst600000AB, generic: generic60000004, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005B: {inst: inst60000040, generic: generic60000004, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005C: {inst: inst600000AA, generic: generic60000004, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005D: {inst: inst600000AB, generic: generic60000004, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005E: {inst: inst60000040, generic: generic60000004, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005F: {inst: inst60000015, generic: generic60000005, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000060: {inst: inst6000001B, generic: generic60000005, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000061: {inst: inst60000015, generic: generic60000005, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000062: {inst: inst6000001B, generic: generic60000005, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000063: {inst: inst6000001E, generic: generic60000005, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000064: {inst: inst600000C1, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000065: {inst: inst600000C2, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000066: {inst: inst600000C3, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000067: {inst: inst600000C3, generic: generic60000006, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000068: {inst: inst60000062, generic: generic60000006, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000069: {inst: inst600000C1, generic: generic60000006, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006A: {inst: inst60000062, generic: generic60000006, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006B: {inst: inst600000C1, generic: generic60000006, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006C: {inst: inst600000CB, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006D: {inst: inst600000CB, generic: generic60000006, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006E: {inst: inst60000063, generic: generic60000006, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006F: {inst: inst600000C2, generic: generic60000006, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000070: {inst: inst600000C3, generic: generic60000006, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000071: {inst: inst600000CE, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000072: {inst: inst600000CE, generic: generic60000006, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000073: {inst: inst60000062, generic: generic60000006, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000074: {inst: inst600000C1, generic: generic60000006, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000075: {inst: inst600000CB, generic: generic60000006, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000076: {inst: inst60000063, generic: generic60000006, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000077: {inst: inst600000C2, generic: generic60000006, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000078: {inst: inst600000C3, generic: generic60000006, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000079: {inst: inst600000CE, generic: generic60000006, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007A: {inst: inst600000D7, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007B: {inst: inst600000D8, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007C: {inst: inst600000D8, generic: generic60000006, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007D: {inst: inst600000D7, generic: generic60000006, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007E: {inst: inst600000D9, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007F: {inst: inst600000D9, generic: generic60000007, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000080: {inst: inst600000D8, generic: generic60000006, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000081: {inst: inst600000D7, generic: generic60000006, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000082: {inst: inst600000D8, generic: generic60000006, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000083: {inst: inst60000062, generic: generic60000007, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000084: {inst: inst60000063, generic: generic60000007, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000085: {inst: inst600000C1, generic: generic60000007, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000086: {inst: inst600000C2, generic: generic60000007, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000087: {inst: inst600000C3, generic: generic60000007, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000088: {inst: inst60000062, generic: generic60000007, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000089: {inst: inst60000063, generic: generic60000007, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008A: {inst: inst600000C1, generic: generic60000007, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008B: {inst: inst600000C2, generic: generic60000007, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008C: {inst: inst600000C3, generic: generic60000007, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008D: {inst: inst600000D9, generic: generic60000007, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008E: {inst: inst600000E8, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008F: {inst: inst600000E8, generic: generic60000007, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000090: {inst: inst60000081, generic: generic60000007, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000091: {inst: inst60000082, generic: generic60000007, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000092: {inst: inst60000083, generic: generic60000007, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000093: {inst: inst60000084, generic: generic60000007, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000094: {inst: inst60000085, generic: generic60000007, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000095: {inst: inst600000E9, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000096: {inst: inst600000E9, generic: generic60000007, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000097: {inst: inst600000EA, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000098: {inst: inst600000EA, generic: generic60000007, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000099: {inst: inst600000EB, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009A: {inst: inst600000EB, generic: generic60000007, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009B: {inst: inst600000EC, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009C: {inst: inst600000EC, generic: generic60000007, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009D: {inst: inst600000ED, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009E: {inst: inst600000ED, generic: generic60000007, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009F: {inst: inst600000E8, generic: generic60000007, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A0: {inst: inst60000081, generic: generic60000007, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A1: {inst: inst60000082, generic: generic60000007, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A2: {inst: inst60000083, generic: generic60000007, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A3: {inst: inst60000084, generic: generic60000007, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A4: {inst: inst60000085, generic: generic60000007, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A5: {inst: inst600000E9, generic: generic60000007, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A6: {inst: inst600000EA, generic: generic60000007, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A7: {inst: inst600000EB, generic: generic60000007, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A8: {inst: inst600000EC, generic: generic60000007, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A9: {inst: inst600000ED, generic: generic60000007, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AA: {inst: inst600000F9, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AB: {inst: inst600000FC, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AC: {inst: inst600000FD, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AD: {inst: inst600000FE, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AE: {inst: inst600000FE, generic: generic60000008, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AF: {inst: inst60000062, generic: generic60000008, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B0: {inst: inst600000C1, generic: generic60000008, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B1: {inst: inst600000FC, generic: generic60000008, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B2: {inst: inst60000062, generic: generic60000008, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B3: {inst: inst600000C1, generic: generic60000008, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B4: {inst: inst600000FC, generic: generic60000008, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B5: {inst: inst60000108, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B6: {inst: inst60000108, generic: generic60000008, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B7: {inst: inst60000063, generic: generic60000008, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B8: {inst: inst600000C2, generic: generic60000008, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B9: {inst: inst600000FD, generic: generic60000008, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BA: {inst: inst600000FE, generic: generic60000008, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BB: {inst: inst6000010B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BC: {inst: inst6000010B, generic: generic60000008, index: generic_inst_in_decl8, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BD: {inst: inst60000062, generic: generic60000008, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BE: {inst: inst600000C1, generic: generic60000008, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BF: {inst: inst600000FC, generic: generic60000008, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C0: {inst: inst60000108, generic: generic60000008, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C1: {inst: inst60000063, generic: generic60000008, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C2: {inst: inst600000C2, generic: generic60000008, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C3: {inst: inst600000FD, generic: generic60000008, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C4: {inst: inst600000FE, generic: generic60000008, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C5: {inst: inst6000010B, generic: generic60000008, index: generic_inst_in_decl8, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C6: {inst: inst60000116, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C7: {inst: inst60000117, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C8: {inst: inst60000117, generic: generic60000008, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C9: {inst: inst60000116, generic: generic60000008, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CA: {inst: inst60000118, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CB: {inst: inst60000118, generic: generic60000009, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CC: {inst: inst60000117, generic: generic60000008, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CD: {inst: inst60000116, generic: generic60000008, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CE: {inst: inst60000117, generic: generic60000008, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CF: {inst: inst60000062, generic: generic60000009, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D0: {inst: inst60000063, generic: generic60000009, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D1: {inst: inst600000C1, generic: generic60000009, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D2: {inst: inst600000C2, generic: generic60000009, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D3: {inst: inst600000FC, generic: generic60000009, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D4: {inst: inst600000FD, generic: generic60000009, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D5: {inst: inst600000FE, generic: generic60000009, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D6: {inst: inst60000062, generic: generic60000009, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D7: {inst: inst60000063, generic: generic60000009, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D8: {inst: inst600000C1, generic: generic60000009, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D9: {inst: inst600000C2, generic: generic60000009, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DA: {inst: inst600000FC, generic: generic60000009, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DB: {inst: inst600000FD, generic: generic60000009, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DC: {inst: inst600000FE, generic: generic60000009, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DD: {inst: inst60000118, generic: generic60000009, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DE: {inst: inst6000012A, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DF: {inst: inst6000012A, generic: generic60000009, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E0: {inst: inst60000081, generic: generic60000009, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E1: {inst: inst60000082, generic: generic60000009, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E2: {inst: inst60000083, generic: generic60000009, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E3: {inst: inst60000084, generic: generic60000009, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E4: {inst: inst60000085, generic: generic60000009, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E5: {inst: inst600000E9, generic: generic60000009, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E6: {inst: inst600000EA, generic: generic60000009, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E7: {inst: inst600000EB, generic: generic60000009, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E8: {inst: inst600000EC, generic: generic60000009, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E9: {inst: inst600000ED, generic: generic60000009, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EA: {inst: inst6000012B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EB: {inst: inst6000012B, generic: generic60000009, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EC: {inst: inst6000012C, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000ED: {inst: inst6000012C, generic: generic60000009, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EE: {inst: inst6000012D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EF: {inst: inst6000012D, generic: generic60000009, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F0: {inst: inst6000012E, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F1: {inst: inst6000012E, generic: generic60000009, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F2: {inst: inst6000012F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F3: {inst: inst6000012F, generic: generic60000009, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F4: {inst: inst6000012A, generic: generic60000009, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F5: {inst: inst60000081, generic: generic60000009, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F6: {inst: inst60000082, generic: generic60000009, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F7: {inst: inst60000083, generic: generic60000009, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F8: {inst: inst60000084, generic: generic60000009, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F9: {inst: inst60000085, generic: generic60000009, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FA: {inst: inst600000E9, generic: generic60000009, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FB: {inst: inst600000EA, generic: generic60000009, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FC: {inst: inst600000EB, generic: generic60000009, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FD: {inst: inst600000EC, generic: generic60000009, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FE: {inst: inst600000ED, generic: generic60000009, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FF: {inst: inst6000012B, generic: generic60000009, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000100: {inst: inst6000012C, generic: generic60000009, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000101: {inst: inst6000012D, generic: generic60000009, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000102: {inst: inst6000012E, generic: generic60000009, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000103: {inst: inst6000012F, generic: generic60000009, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000104: {inst: inst60000140, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000105: {inst: inst600000A5, generic: generic60000004, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000106: {inst: inst60000141, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000107: {inst: inst60000141, generic: generic60000000, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000108: {inst: inst60000143, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000109: {inst: inst60000144, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000010A: {inst: inst60000146, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000010B: {inst: inst60000143, generic: generic60000000, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000010C: {inst: inst60000144, generic: generic60000000, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000010D: {inst: inst60000146, generic: generic60000000, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000010E: {inst: inst6000014C, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000010F: {inst: inst6000014C, generic: generic60000000, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst60000038
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst60000010
// CHECK:STDOUT:       1:               inst60000048
// CHECK:STDOUT:       2:               inst60000049
// CHECK:STDOUT:       3:               inst6000004C
// CHECK:STDOUT:       4:               inst6000004D
// CHECK:STDOUT:       5:               inst6000004E
// CHECK:STDOUT:       6:               inst60000050
// CHECK:STDOUT:       7:               inst60000052
// CHECK:STDOUT:       8:               inst60000057
// CHECK:STDOUT:       9:               inst60000058
// CHECK:STDOUT:       10:              inst60000059
// CHECK:STDOUT:       11:              inst6000005A
// CHECK:STDOUT:       12:              inst6000005B
// CHECK:STDOUT:       13:              inst60000060
// CHECK:STDOUT:       14:              inst60000061
// CHECK:STDOUT:       15:              inst60000066
// CHECK:STDOUT:       16:              inst60000067
// CHECK:STDOUT:       17:              inst60000068
// CHECK:STDOUT:       18:              inst60000069
// CHECK:STDOUT:       19:              inst6000006A
// CHECK:STDOUT:       20:              inst6000006B
// CHECK:STDOUT:       21:              inst60000071
// CHECK:STDOUT:       22:              inst60000075
// CHECK:STDOUT:       23:              inst60000076
// CHECK:STDOUT:       24:              inst60000077
// CHECK:STDOUT:       25:              inst60000078
// CHECK:STDOUT:       26:              inst60000079
// CHECK:STDOUT:       27:              inst6000008D
// CHECK:STDOUT:       28:              inst6000008E
// CHECK:STDOUT:       29:              inst6000008F
// CHECK:STDOUT:       30:              inst60000090
// CHECK:STDOUT:       31:              inst60000091
// CHECK:STDOUT:       32:              inst60000092
// CHECK:STDOUT:       33:              inst60000093
// CHECK:STDOUT:       34:              inst60000094
// CHECK:STDOUT:       35:              inst60000095
// CHECK:STDOUT:       36:              inst60000096
// CHECK:STDOUT:       37:              inst60000097
// CHECK:STDOUT:       38:              inst60000098
// CHECK:STDOUT:       39:              inst60000099
// CHECK:STDOUT:       40:              inst6000009A
// CHECK:STDOUT:       41:              inst6000009B
// CHECK:STDOUT:       42:              inst6000009C
// CHECK:STDOUT:       43:              inst6000009D
// CHECK:STDOUT:       44:              inst6000009E
// CHECK:STDOUT:       45:              inst6000009F
// CHECK:STDOUT:       46:              inst600000A0
// CHECK:STDOUT:       47:              inst600000A1
// CHECK:STDOUT:       48:              inst600000A2
// CHECK:STDOUT:       49:              inst600000A3
// CHECK:STDOUT:       50:              inst600000A4
// CHECK:STDOUT:       51:              inst600000A9
// CHECK:STDOUT:       52:              inst600000AC
// CHECK:STDOUT:       53:              inst600000AD
// CHECK:STDOUT:       54:              inst600000AE
// CHECK:STDOUT:       55:              inst600000AF
// CHECK:STDOUT:       56:              inst600000B0
// CHECK:STDOUT:       57:              inst600000B7
// CHECK:STDOUT:       58:              inst600000B8
// CHECK:STDOUT:       59:              inst600000B9
// CHECK:STDOUT:       60:              inst600000BA
// CHECK:STDOUT:       61:              inst600000BB
// CHECK:STDOUT:       62:              inst600000BC
// CHECK:STDOUT:       63:              inst600000BD
// CHECK:STDOUT:       64:              inst600000BE
// CHECK:STDOUT:       65:              inst600000BF
// CHECK:STDOUT:       66:              inst600000C0
// CHECK:STDOUT:       67:              inst600000C4
// CHECK:STDOUT:       68:              inst600000C5
// CHECK:STDOUT:       69:              inst600000C6
// CHECK:STDOUT:       70:              inst600000C7
// CHECK:STDOUT:       71:              inst600000C8
// CHECK:STDOUT:       72:              inst600000C9
// CHECK:STDOUT:       73:              inst600000CC
// CHECK:STDOUT:       74:              inst600000CD
// CHECK:STDOUT:       75:              inst600000D6
// CHECK:STDOUT:       76:              inst600000DA
// CHECK:STDOUT:       77:              inst600000DB
// CHECK:STDOUT:       78:              inst600000DC
// CHECK:STDOUT:       79:              inst600000DD
// CHECK:STDOUT:       80:              inst600000DE
// CHECK:STDOUT:       81:              inst600000DF
// CHECK:STDOUT:       82:              inst600000FA
// CHECK:STDOUT:       83:              inst600000FB
// CHECK:STDOUT:       84:              inst600000FF
// CHECK:STDOUT:       85:              inst60000100
// CHECK:STDOUT:       86:              inst60000101
// CHECK:STDOUT:       87:              inst60000102
// CHECK:STDOUT:       88:              inst60000103
// CHECK:STDOUT:       89:              inst60000104
// CHECK:STDOUT:       90:              inst60000105
// CHECK:STDOUT:       91:              inst60000106
// CHECK:STDOUT:       92:              inst60000109
// CHECK:STDOUT:       93:              inst6000010A
// CHECK:STDOUT:       94:              inst60000115
// CHECK:STDOUT:       95:              inst60000119
// CHECK:STDOUT:       96:              inst6000011A
// CHECK:STDOUT:       97:              inst6000011B
// CHECK:STDOUT:       98:              inst6000011C
// CHECK:STDOUT:       99:              inst6000011D
// CHECK:STDOUT:       100:             inst6000011E
// CHECK:STDOUT:       101:             inst6000011F
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block60000004:
// CHECK:STDOUT:       0:               inst60000012
// CHECK:STDOUT:     inst_block60000005:
// CHECK:STDOUT:       0:               inst60000018
// CHECK:STDOUT:     inst_block60000006:
// CHECK:STDOUT:       0:               inst60000019
// CHECK:STDOUT:       1:               inst6000001A
// CHECK:STDOUT:     inst_block60000007:
// CHECK:STDOUT:       0:               inst60000021
// CHECK:STDOUT:     inst_block60000008:
// CHECK:STDOUT:       0:               inst60000023
// CHECK:STDOUT:       1:               inst60000025
// CHECK:STDOUT:     inst_block60000009:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst60000024
// CHECK:STDOUT:     inst_block6000000A:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:       1:               inst60000026
// CHECK:STDOUT:     inst_block6000000B:
// CHECK:STDOUT:       0:               inst6000001C
// CHECK:STDOUT:       1:               inst60000026
// CHECK:STDOUT:     inst_block6000000C:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:       1:               inst6000002C
// CHECK:STDOUT:     inst_block6000000D:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:       1:               inst60000024
// CHECK:STDOUT:     inst_block6000000E:
// CHECK:STDOUT:       0:               inst6000001C
// CHECK:STDOUT:       1:               inst60000024
// CHECK:STDOUT:     inst_block6000000F:
// CHECK:STDOUT:       0:               inst60000034
// CHECK:STDOUT:       1:               inst60000036
// CHECK:STDOUT:     inst_block60000010:
// CHECK:STDOUT:       0:               inst60000018
// CHECK:STDOUT:       1:               inst6000001F
// CHECK:STDOUT:       2:               inst60000021
// CHECK:STDOUT:       3:               inst60000031
// CHECK:STDOUT:       4:               inst60000033
// CHECK:STDOUT:     inst_block60000011:
// CHECK:STDOUT:       0:               inst60000022
// CHECK:STDOUT:       1:               inst60000023
// CHECK:STDOUT:       2:               inst60000025
// CHECK:STDOUT:       3:               inst60000028
// CHECK:STDOUT:       4:               inst6000002C
// CHECK:STDOUT:       5:               inst6000002E
// CHECK:STDOUT:       6:               inst60000012
// CHECK:STDOUT:       7:               inst60000014
// CHECK:STDOUT:       8:               inst60000034
// CHECK:STDOUT:       9:               inst60000035
// CHECK:STDOUT:       10:              inst6000001D
// CHECK:STDOUT:       11:              inst60000036
// CHECK:STDOUT:       12:              inst60000037
// CHECK:STDOUT:     inst_block60000012:
// CHECK:STDOUT:       0:               inst60000014
// CHECK:STDOUT:     inst_block60000013:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:     inst_block60000014:
// CHECK:STDOUT:       0:               inst60000016
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:       2:               inst60000020
// CHECK:STDOUT:       3:               inst6000002A
// CHECK:STDOUT:       4:               inst6000002F
// CHECK:STDOUT:       5:               inst60000032
// CHECK:STDOUT:     inst_block60000015:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:       3:               inst60000029
// CHECK:STDOUT:       4:               inst6000002D
// CHECK:STDOUT:       5:               inst60000030
// CHECK:STDOUT:     inst_block60000016:
// CHECK:STDOUT:       0:               inst60000042
// CHECK:STDOUT:       1:               inst60000043
// CHECK:STDOUT:       2:               inst60000044
// CHECK:STDOUT:       3:               inst60000145
// CHECK:STDOUT:       4:               inst6000014A
// CHECK:STDOUT:       5:               inst6000014B
// CHECK:STDOUT:       6:               inst6000014E
// CHECK:STDOUT:       7:               inst60000150
// CHECK:STDOUT:       8:               inst60000046
// CHECK:STDOUT:       9:               inst60000151
// CHECK:STDOUT:       10:              inst60000152
// CHECK:STDOUT:       11:              inst60000153
// CHECK:STDOUT:       12:              inst60000154
// CHECK:STDOUT:       13:              inst60000155
// CHECK:STDOUT:       14:              inst60000156
// CHECK:STDOUT:       15:              inst60000157
// CHECK:STDOUT:     inst_block60000017:
// CHECK:STDOUT:       0:               inst60000042
// CHECK:STDOUT:       1:               inst60000043
// CHECK:STDOUT:     inst_block60000018:
// CHECK:STDOUT:       0:               inst6000004E
// CHECK:STDOUT:     inst_block60000019:
// CHECK:STDOUT:       0:               inst6000005A
// CHECK:STDOUT:     inst_block6000001A:
// CHECK:STDOUT:       0:               inst6000005B
// CHECK:STDOUT:     inst_block6000001B:
// CHECK:STDOUT:       0:               inst6000005C
// CHECK:STDOUT:       1:               inst6000005D
// CHECK:STDOUT:       2:               inst6000005E
// CHECK:STDOUT:     inst_block6000001C:
// CHECK:STDOUT:       0:               inst6000004B
// CHECK:STDOUT:     inst_block6000001D:
// CHECK:STDOUT:       0:               inst6000004B
// CHECK:STDOUT:       1:               inst60000055
// CHECK:STDOUT:       2:               inst60000056
// CHECK:STDOUT:     inst_block6000001E:
// CHECK:STDOUT:       0:               inst60000066
// CHECK:STDOUT:     inst_block6000001F:
// CHECK:STDOUT:       0:               inst60000067
// CHECK:STDOUT:     inst_block60000020:
// CHECK:STDOUT:       0:               inst6000006A
// CHECK:STDOUT:     inst_block60000021:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:     inst_block60000022:
// CHECK:STDOUT:       0:               inst6000006D
// CHECK:STDOUT:     inst_block60000023:
// CHECK:STDOUT:       0:               inst6000006D
// CHECK:STDOUT:       1:               inst6000006E
// CHECK:STDOUT:       2:               inst6000006F
// CHECK:STDOUT:       3:               inst60000070
// CHECK:STDOUT:     inst_block60000024:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst60000063
// CHECK:STDOUT:       2:               inst60000064
// CHECK:STDOUT:       3:               inst6000006C
// CHECK:STDOUT:     inst_block60000025:
// CHECK:STDOUT:       0:               inst60000078
// CHECK:STDOUT:     inst_block60000026:
// CHECK:STDOUT:       0:               inst60000079
// CHECK:STDOUT:     inst_block60000027:
// CHECK:STDOUT:       0:               inst6000006D
// CHECK:STDOUT:     inst_block60000028:
// CHECK:STDOUT:       0:               inst6000007A
// CHECK:STDOUT:       1:               inst6000007B
// CHECK:STDOUT:     inst_block60000029:
// CHECK:STDOUT:       0:               inst6000007C
// CHECK:STDOUT:       1:               inst6000007D
// CHECK:STDOUT:       2:               inst6000007E
// CHECK:STDOUT:       3:               inst6000007F
// CHECK:STDOUT:     inst_block6000002A:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst60000063
// CHECK:STDOUT:       2:               inst60000064
// CHECK:STDOUT:       3:               inst60000074
// CHECK:STDOUT:     inst_block6000002B:
// CHECK:STDOUT:       0:               inst6000007C
// CHECK:STDOUT:     inst_block6000002C:
// CHECK:STDOUT:       0:               inst60000086
// CHECK:STDOUT:       1:               inst60000087
// CHECK:STDOUT:       2:               inst60000088
// CHECK:STDOUT:       3:               inst60000089
// CHECK:STDOUT:       4:               inst6000008A
// CHECK:STDOUT:       5:               inst6000008B
// CHECK:STDOUT:     inst_block6000002D:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst60000063
// CHECK:STDOUT:       2:               inst6000008C
// CHECK:STDOUT:     inst_block6000002E:
// CHECK:STDOUT:       0:               inst60000072
// CHECK:STDOUT:       1:               inst60000073
// CHECK:STDOUT:     inst_block6000002F:
// CHECK:STDOUT:       0:               inst6000009F
// CHECK:STDOUT:     inst_block60000030:
// CHECK:STDOUT:       0:               inst600000A0
// CHECK:STDOUT:     inst_block60000031:
// CHECK:STDOUT:       0:               inst600000A3
// CHECK:STDOUT:     inst_block60000032:
// CHECK:STDOUT:       0:               inst600000A6
// CHECK:STDOUT:     inst_block60000033:
// CHECK:STDOUT:       0:               inst600000A6
// CHECK:STDOUT:       1:               inst600000A7
// CHECK:STDOUT:       2:               inst600000A8
// CHECK:STDOUT:     inst_block60000034:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst600000A5
// CHECK:STDOUT:     inst_block60000035:
// CHECK:STDOUT:       0:               inst600000AF
// CHECK:STDOUT:     inst_block60000036:
// CHECK:STDOUT:       0:               inst600000B0
// CHECK:STDOUT:     inst_block60000037:
// CHECK:STDOUT:       0:               inst600000A6
// CHECK:STDOUT:     inst_block60000038:
// CHECK:STDOUT:       0:               inst600000B1
// CHECK:STDOUT:       1:               inst600000B2
// CHECK:STDOUT:       2:               inst600000B3
// CHECK:STDOUT:     inst_block60000039:
// CHECK:STDOUT:       0:               inst600000B4
// CHECK:STDOUT:       1:               inst600000B5
// CHECK:STDOUT:       2:               inst600000B6
// CHECK:STDOUT:     inst_block6000003A:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:     inst_block6000003B:
// CHECK:STDOUT:       0:               inst60000063
// CHECK:STDOUT:       1:               inst600000C2
// CHECK:STDOUT:     inst_block6000003C:
// CHECK:STDOUT:       0:               inst600000C5
// CHECK:STDOUT:       1:               inst600000C4
// CHECK:STDOUT:     inst_block6000003D:
// CHECK:STDOUT:       0:               inst600000C6
// CHECK:STDOUT:       1:               inst600000C7
// CHECK:STDOUT:     inst_block6000003E:
// CHECK:STDOUT:       0:               inst6000004A
// CHECK:STDOUT:       1:               inst6000004A
// CHECK:STDOUT:     inst_block6000003F:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst600000C1
// CHECK:STDOUT:     inst_block60000040:
// CHECK:STDOUT:       0:               inst600000CC
// CHECK:STDOUT:     inst_block60000041:
// CHECK:STDOUT:       0:               inst600000CF
// CHECK:STDOUT:       1:               inst600000D0
// CHECK:STDOUT:     inst_block60000042:
// CHECK:STDOUT:       0:               inst600000D2
// CHECK:STDOUT:       1:               inst600000D3
// CHECK:STDOUT:     inst_block60000043:
// CHECK:STDOUT:       0:               inst600000CF
// CHECK:STDOUT:       1:               inst600000D0
// CHECK:STDOUT:     inst_block60000044:
// CHECK:STDOUT:       0:               inst600000CF
// CHECK:STDOUT:       1:               inst600000D0
// CHECK:STDOUT:       2:               inst600000D1
// CHECK:STDOUT:       3:               inst600000D2
// CHECK:STDOUT:       4:               inst600000D3
// CHECK:STDOUT:       5:               inst600000D4
// CHECK:STDOUT:       6:               inst600000D5
// CHECK:STDOUT:     inst_block60000045:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst600000C1
// CHECK:STDOUT:       2:               inst600000CB
// CHECK:STDOUT:       3:               inst60000063
// CHECK:STDOUT:       4:               inst600000C2
// CHECK:STDOUT:       5:               inst600000C3
// CHECK:STDOUT:       6:               inst600000CE
// CHECK:STDOUT:     inst_block60000046:
// CHECK:STDOUT:       0:               inst600000DD
// CHECK:STDOUT:     inst_block60000047:
// CHECK:STDOUT:       0:               inst600000DE
// CHECK:STDOUT:       1:               inst600000DF
// CHECK:STDOUT:     inst_block60000048:
// CHECK:STDOUT:       0:               inst600000CF
// CHECK:STDOUT:       1:               inst600000D0
// CHECK:STDOUT:     inst_block60000049:
// CHECK:STDOUT:       0:               inst600000E0
// CHECK:STDOUT:       1:               inst600000E1
// CHECK:STDOUT:     inst_block6000004A:
// CHECK:STDOUT:       0:               inst600000E3
// CHECK:STDOUT:       1:               inst600000E5
// CHECK:STDOUT:     inst_block6000004B:
// CHECK:STDOUT:       0:               inst600000E2
// CHECK:STDOUT:       1:               inst600000E3
// CHECK:STDOUT:       2:               inst600000E4
// CHECK:STDOUT:       3:               inst600000E5
// CHECK:STDOUT:       4:               inst600000E6
// CHECK:STDOUT:       5:               inst600000E7
// CHECK:STDOUT:     inst_block6000004C:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst60000063
// CHECK:STDOUT:       2:               inst600000C1
// CHECK:STDOUT:       3:               inst600000C2
// CHECK:STDOUT:       4:               inst600000C3
// CHECK:STDOUT:       5:               inst600000D9
// CHECK:STDOUT:     inst_block6000004D:
// CHECK:STDOUT:       0:               inst600000C1
// CHECK:STDOUT:     inst_block6000004E:
// CHECK:STDOUT:       0:               inst600000E2
// CHECK:STDOUT:     inst_block6000004F:
// CHECK:STDOUT:       0:               inst600000E4
// CHECK:STDOUT:     inst_block60000050:
// CHECK:STDOUT:       0:               inst600000EE
// CHECK:STDOUT:       1:               inst600000EF
// CHECK:STDOUT:       2:               inst600000F0
// CHECK:STDOUT:       3:               inst600000F1
// CHECK:STDOUT:       4:               inst600000F2
// CHECK:STDOUT:       5:               inst600000F3
// CHECK:STDOUT:       6:               inst600000F4
// CHECK:STDOUT:       7:               inst600000F5
// CHECK:STDOUT:       8:               inst600000F6
// CHECK:STDOUT:       9:               inst600000F7
// CHECK:STDOUT:       10:              inst600000F8
// CHECK:STDOUT:     inst_block60000051:
// CHECK:STDOUT:       0:               inst600000C1
// CHECK:STDOUT:       1:               inst600000C2
// CHECK:STDOUT:       2:               inst600000F9
// CHECK:STDOUT:     inst_block60000052:
// CHECK:STDOUT:       0:               inst600000D7
// CHECK:STDOUT:       1:               inst600000D8
// CHECK:STDOUT:     inst_block60000053:
// CHECK:STDOUT:       0:               inst60000063
// CHECK:STDOUT:       1:               inst600000C2
// CHECK:STDOUT:       2:               inst600000FD
// CHECK:STDOUT:     inst_block60000054:
// CHECK:STDOUT:       0:               inst60000101
// CHECK:STDOUT:       1:               inst60000100
// CHECK:STDOUT:       2:               inst600000FF
// CHECK:STDOUT:     inst_block60000055:
// CHECK:STDOUT:       0:               inst60000102
// CHECK:STDOUT:       1:               inst60000103
// CHECK:STDOUT:       2:               inst60000104
// CHECK:STDOUT:     inst_block60000056:
// CHECK:STDOUT:       0:               inst6000004A
// CHECK:STDOUT:       1:               inst6000004A
// CHECK:STDOUT:       2:               inst6000004A
// CHECK:STDOUT:     inst_block60000057:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst600000C1
// CHECK:STDOUT:       2:               inst600000FC
// CHECK:STDOUT:     inst_block60000058:
// CHECK:STDOUT:       0:               inst60000109
// CHECK:STDOUT:     inst_block60000059:
// CHECK:STDOUT:       0:               inst6000010C
// CHECK:STDOUT:       1:               inst6000010D
// CHECK:STDOUT:       2:               inst6000010E
// CHECK:STDOUT:     inst_block6000005A:
// CHECK:STDOUT:       0:               inst60000110
// CHECK:STDOUT:       1:               inst60000111
// CHECK:STDOUT:       2:               inst60000112
// CHECK:STDOUT:     inst_block6000005B:
// CHECK:STDOUT:       0:               inst6000010C
// CHECK:STDOUT:       1:               inst6000010D
// CHECK:STDOUT:       2:               inst6000010E
// CHECK:STDOUT:     inst_block6000005C:
// CHECK:STDOUT:       0:               inst6000010C
// CHECK:STDOUT:       1:               inst6000010D
// CHECK:STDOUT:       2:               inst6000010E
// CHECK:STDOUT:       3:               inst6000010F
// CHECK:STDOUT:       4:               inst60000110
// CHECK:STDOUT:       5:               inst60000111
// CHECK:STDOUT:       6:               inst60000112
// CHECK:STDOUT:       7:               inst60000113
// CHECK:STDOUT:       8:               inst60000114
// CHECK:STDOUT:     inst_block6000005D:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst600000C1
// CHECK:STDOUT:       2:               inst600000FC
// CHECK:STDOUT:       3:               inst60000108
// CHECK:STDOUT:       4:               inst60000063
// CHECK:STDOUT:       5:               inst600000C2
// CHECK:STDOUT:       6:               inst600000FD
// CHECK:STDOUT:       7:               inst600000FE
// CHECK:STDOUT:       8:               inst6000010B
// CHECK:STDOUT:     inst_block6000005E:
// CHECK:STDOUT:       0:               inst6000011C
// CHECK:STDOUT:     inst_block6000005F:
// CHECK:STDOUT:       0:               inst6000011D
// CHECK:STDOUT:       1:               inst6000011E
// CHECK:STDOUT:       2:               inst6000011F
// CHECK:STDOUT:     inst_block60000060:
// CHECK:STDOUT:       0:               inst6000010C
// CHECK:STDOUT:       1:               inst6000010D
// CHECK:STDOUT:       2:               inst6000010E
// CHECK:STDOUT:     inst_block60000061:
// CHECK:STDOUT:       0:               inst60000120
// CHECK:STDOUT:       1:               inst60000121
// CHECK:STDOUT:     inst_block60000062:
// CHECK:STDOUT:       0:               inst60000123
// CHECK:STDOUT:       1:               inst60000125
// CHECK:STDOUT:       2:               inst60000127
// CHECK:STDOUT:     inst_block60000063:
// CHECK:STDOUT:       0:               inst60000122
// CHECK:STDOUT:       1:               inst60000123
// CHECK:STDOUT:       2:               inst60000124
// CHECK:STDOUT:       3:               inst60000125
// CHECK:STDOUT:       4:               inst60000126
// CHECK:STDOUT:       5:               inst60000127
// CHECK:STDOUT:       6:               inst60000128
// CHECK:STDOUT:       7:               inst60000129
// CHECK:STDOUT:     inst_block60000064:
// CHECK:STDOUT:       0:               inst60000062
// CHECK:STDOUT:       1:               inst60000063
// CHECK:STDOUT:       2:               inst600000C1
// CHECK:STDOUT:       3:               inst600000C2
// CHECK:STDOUT:       4:               inst600000FC
// CHECK:STDOUT:       5:               inst600000FD
// CHECK:STDOUT:       6:               inst600000FE
// CHECK:STDOUT:       7:               inst60000118
// CHECK:STDOUT:     inst_block60000065:
// CHECK:STDOUT:       0:               inst600000FC
// CHECK:STDOUT:     inst_block60000066:
// CHECK:STDOUT:       0:               inst60000122
// CHECK:STDOUT:     inst_block60000067:
// CHECK:STDOUT:       0:               inst60000124
// CHECK:STDOUT:     inst_block60000068:
// CHECK:STDOUT:       0:               inst60000126
// CHECK:STDOUT:     inst_block60000069:
// CHECK:STDOUT:       0:               inst60000130
// CHECK:STDOUT:       1:               inst60000131
// CHECK:STDOUT:       2:               inst60000132
// CHECK:STDOUT:       3:               inst60000133
// CHECK:STDOUT:       4:               inst60000134
// CHECK:STDOUT:       5:               inst60000135
// CHECK:STDOUT:       6:               inst60000136
// CHECK:STDOUT:       7:               inst60000137
// CHECK:STDOUT:       8:               inst60000138
// CHECK:STDOUT:       9:               inst60000139
// CHECK:STDOUT:       10:              inst6000013A
// CHECK:STDOUT:       11:              inst6000013B
// CHECK:STDOUT:       12:              inst6000013C
// CHECK:STDOUT:       13:              inst6000013D
// CHECK:STDOUT:       14:              inst6000013E
// CHECK:STDOUT:       15:              inst6000013F
// CHECK:STDOUT:     inst_block6000006A:
// CHECK:STDOUT:       0:               inst600000FC
// CHECK:STDOUT:       1:               inst600000FD
// CHECK:STDOUT:       2:               inst60000140
// CHECK:STDOUT:     inst_block6000006B:
// CHECK:STDOUT:       0:               inst60000116
// CHECK:STDOUT:       1:               inst60000117
// CHECK:STDOUT:     inst_block6000006C:
// CHECK:STDOUT:       0:               inst600000AA
// CHECK:STDOUT:       1:               inst600000AB
// CHECK:STDOUT:       2:               inst60000040
// CHECK:STDOUT:     inst_block6000006D:
// CHECK:STDOUT:       0:               inst60000141
// CHECK:STDOUT:     inst_block6000006E:
// CHECK:STDOUT:       0:               inst60000142
// CHECK:STDOUT:     inst_block6000006F:
// CHECK:STDOUT:       0:               inst60000143
// CHECK:STDOUT:     inst_block60000070:
// CHECK:STDOUT:       0:               inst60000143
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:     inst_block60000071:
// CHECK:STDOUT:       0:               inst60000147
// CHECK:STDOUT:     inst_block60000072:
// CHECK:STDOUT:       0:               inst60000042
// CHECK:STDOUT:     inst_block60000073:
// CHECK:STDOUT:       0:               inst60000151
// CHECK:STDOUT:       1:               inst60000154
// CHECK:STDOUT:     inst_block60000074:
// CHECK:STDOUT:       0:               inst6000003E
// CHECK:STDOUT:       1:               inst60000041
// CHECK:STDOUT:       2:               inst60000142
// CHECK:STDOUT:       3:               inst60000147
// CHECK:STDOUT:       4:               inst60000148
// CHECK:STDOUT:       5:               inst60000149
// CHECK:STDOUT:       6:               inst6000014D
// CHECK:STDOUT:     inst_block60000075:
// CHECK:STDOUT:       0:               instE
// CHECK:STDOUT:       1:               inst6000000F
// CHECK:STDOUT:       2:               inst60000038
// CHECK:STDOUT: ...
