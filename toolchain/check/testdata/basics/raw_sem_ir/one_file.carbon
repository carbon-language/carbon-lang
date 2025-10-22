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
// CHECK:STDOUT:     import_ir00000002: {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir00000003: {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir00000004: {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:     import_ir00000005: {decl_id: inst6000000F, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst00000000: {ir_id: import_ir00000004, inst_id: inst4800000F}
// CHECK:STDOUT:     import_ir_inst00000001: {ir_id: import_ir00000004, inst_id: inst4800000F}
// CHECK:STDOUT:     import_ir_inst00000002: {ir_id: import_ir00000004, inst_id: inst48000011}
// CHECK:STDOUT:     import_ir_inst00000003: {ir_id: import_ir00000004, inst_id: inst4800002D}
// CHECK:STDOUT:     import_ir_inst00000004: {ir_id: import_ir00000004, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst00000005: {ir_id: import_ir00000004, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst00000006: {ir_id: import_ir00000004, inst_id: inst48000028}
// CHECK:STDOUT:     import_ir_inst00000007: {ir_id: import_ir00000004, inst_id: inst48000022}
// CHECK:STDOUT:     import_ir_inst00000008: {ir_id: import_ir00000004, inst_id: inst48000023}
// CHECK:STDOUT:     import_ir_inst00000009: {ir_id: import_ir00000004, inst_id: inst4800001C}
// CHECK:STDOUT:     import_ir_inst0000000A: {ir_id: import_ir00000004, inst_id: inst4800001E}
// CHECK:STDOUT:     import_ir_inst0000000B: {ir_id: import_ir00000004, inst_id: inst48000011}
// CHECK:STDOUT:     import_ir_inst0000000C: {ir_id: import_ir00000004, inst_id: inst48000015}
// CHECK:STDOUT:     import_ir_inst0000000D: {ir_id: import_ir00000004, inst_id: inst48000018}
// CHECK:STDOUT:     import_ir_inst0000000E: {ir_id: import_ir00000004, inst_id: inst4800001D}
// CHECK:STDOUT:     import_ir_inst0000000F: {ir_id: import_ir00000004, inst_id: inst48000061}
// CHECK:STDOUT:     import_ir_inst00000010: {ir_id: import_ir00000004, inst_id: inst4800005F}
// CHECK:STDOUT:     import_ir_inst00000011: {ir_id: import_ir00000004, inst_id: inst48000054}
// CHECK:STDOUT:     import_ir_inst00000012: {ir_id: import_ir00000004, inst_id: inst48000050}
// CHECK:STDOUT:     import_ir_inst00000013: {ir_id: import_ir00000004, inst_id: inst4800005A}
// CHECK:STDOUT:     import_ir_inst00000014: {ir_id: import_ir00000004, inst_id: inst4800005D}
// CHECK:STDOUT:     import_ir_inst00000015: {ir_id: import_ir00000004, inst_id: inst48000075}
// CHECK:STDOUT:     import_ir_inst00000016: {ir_id: import_ir00000004, inst_id: inst48000060}
// CHECK:STDOUT:     import_ir_inst00000017: {ir_id: import_ir00000004, inst_id: inst48000052}
// CHECK:STDOUT:     import_ir_inst00000018: {ir_id: import_ir00000004, inst_id: inst48000058}
// CHECK:STDOUT:     import_ir_inst00000019: {ir_id: import_ir00000004, inst_id: inst4800005C}
// CHECK:STDOUT:     import_ir_inst0000001A: {ir_id: import_ir00000004, inst_id: inst48000063}
// CHECK:STDOUT:     import_ir_inst0000001B: {ir_id: import_ir00000004, inst_id: inst48000075}
// CHECK:STDOUT:     import_ir_inst0000001C: {ir_id: import_ir00000004, inst_id: inst48000070}
// CHECK:STDOUT:     import_ir_inst0000001D: {ir_id: import_ir00000004, inst_id: inst48000071}
// CHECK:STDOUT:     import_ir_inst0000001E: {ir_id: import_ir00000004, inst_id: inst4800006C}
// CHECK:STDOUT:     import_ir_inst0000001F: {ir_id: import_ir00000004, inst_id: inst4800006E}
// CHECK:STDOUT:     import_ir_inst00000020: {ir_id: import_ir00000004, inst_id: inst48000050}
// CHECK:STDOUT:     import_ir_inst00000021: {ir_id: import_ir00000004, inst_id: inst48000078}
// CHECK:STDOUT:     import_ir_inst00000022: {ir_id: import_ir00000004, inst_id: inst48000079}
// CHECK:STDOUT:     import_ir_inst00000023: {ir_id: import_ir00000004, inst_id: inst48000067}
// CHECK:STDOUT:     import_ir_inst00000024: {ir_id: import_ir00000004, inst_id: inst48000068}
// CHECK:STDOUT:     import_ir_inst00000025: {ir_id: import_ir00000004, inst_id: inst48000069}
// CHECK:STDOUT:     import_ir_inst00000026: {ir_id: import_ir00000004, inst_id: inst4800006D}
// CHECK:STDOUT:     import_ir_inst00000027: {ir_id: import_ir00000004, inst_id: inst4800007D}
// CHECK:STDOUT:     import_ir_inst00000028: {ir_id: import_ir00000004, inst_id: inst48000085}
// CHECK:STDOUT:     import_ir_inst00000029: {ir_id: import_ir00000004, inst_id: inst4800008C}
// CHECK:STDOUT:     import_ir_inst0000002A: {ir_id: import_ir00000004, inst_id: inst48000090}
// CHECK:STDOUT:     import_ir_inst0000002B: {ir_id: import_ir00000004, inst_id: inst48000091}
// CHECK:STDOUT:     import_ir_inst0000002C: {ir_id: import_ir00000004, inst_id: inst48000096}
// CHECK:STDOUT:     import_ir_inst0000002D: {ir_id: import_ir00000004, inst_id: inst480000AB}
// CHECK:STDOUT:     import_ir_inst0000002E: {ir_id: import_ir00000004, inst_id: inst480000A9}
// CHECK:STDOUT:     import_ir_inst0000002F: {ir_id: import_ir00000004, inst_id: inst480000A7}
// CHECK:STDOUT:     import_ir_inst00000030: {ir_id: import_ir00000004, inst_id: inst480000A8}
// CHECK:STDOUT:     import_ir_inst00000031: {ir_id: import_ir00000004, inst_id: inst480000C3}
// CHECK:STDOUT:     import_ir_inst00000032: {ir_id: import_ir00000004, inst_id: inst480000C1}
// CHECK:STDOUT:     import_ir_inst00000033: {ir_id: import_ir00000004, inst_id: inst480000BF}
// CHECK:STDOUT:     import_ir_inst00000034: {ir_id: import_ir00000004, inst_id: inst480000C0}
// CHECK:STDOUT:     import_ir_inst00000035: {ir_id: import_ir00000004, inst_id: inst480000DB}
// CHECK:STDOUT:     import_ir_inst00000036: {ir_id: import_ir00000004, inst_id: inst480000D9}
// CHECK:STDOUT:     import_ir_inst00000037: {ir_id: import_ir00000004, inst_id: inst480000D7}
// CHECK:STDOUT:     import_ir_inst00000038: {ir_id: import_ir00000004, inst_id: inst480000D8}
// CHECK:STDOUT:     import_ir_inst00000039: {ir_id: import_ir00000004, inst_id: inst480000F3}
// CHECK:STDOUT:     import_ir_inst0000003A: {ir_id: import_ir00000004, inst_id: inst480000F1}
// CHECK:STDOUT:     import_ir_inst0000003B: {ir_id: import_ir00000004, inst_id: inst480000EF}
// CHECK:STDOUT:     import_ir_inst0000003C: {ir_id: import_ir00000004, inst_id: inst480000F0}
// CHECK:STDOUT:     import_ir_inst0000003D: {ir_id: import_ir00000004, inst_id: inst48000110}
// CHECK:STDOUT:     import_ir_inst0000003E: {ir_id: import_ir00000004, inst_id: inst4800010E}
// CHECK:STDOUT:     import_ir_inst0000003F: {ir_id: import_ir00000004, inst_id: inst48000108}
// CHECK:STDOUT:     import_ir_inst00000040: {ir_id: import_ir00000004, inst_id: inst48000105}
// CHECK:STDOUT:     import_ir_inst00000041: {ir_id: import_ir00000004, inst_id: inst4800010A}
// CHECK:STDOUT:     import_ir_inst00000042: {ir_id: import_ir00000004, inst_id: inst4800010D}
// CHECK:STDOUT:     import_ir_inst00000043: {ir_id: import_ir00000004, inst_id: inst48000121}
// CHECK:STDOUT:     import_ir_inst00000044: {ir_id: import_ir00000004, inst_id: inst4800010F}
// CHECK:STDOUT:     import_ir_inst00000045: {ir_id: import_ir00000004, inst_id: inst48000107}
// CHECK:STDOUT:     import_ir_inst00000046: {ir_id: import_ir00000004, inst_id: inst4800010C}
// CHECK:STDOUT:     import_ir_inst00000047: {ir_id: import_ir00000004, inst_id: inst48000112}
// CHECK:STDOUT:     import_ir_inst00000048: {ir_id: import_ir00000004, inst_id: inst48000121}
// CHECK:STDOUT:     import_ir_inst00000049: {ir_id: import_ir00000004, inst_id: inst4800011C}
// CHECK:STDOUT:     import_ir_inst0000004A: {ir_id: import_ir00000004, inst_id: inst4800011D}
// CHECK:STDOUT:     import_ir_inst0000004B: {ir_id: import_ir00000004, inst_id: inst48000118}
// CHECK:STDOUT:     import_ir_inst0000004C: {ir_id: import_ir00000004, inst_id: inst4800011A}
// CHECK:STDOUT:     import_ir_inst0000004D: {ir_id: import_ir00000004, inst_id: inst48000105}
// CHECK:STDOUT:     import_ir_inst0000004E: {ir_id: import_ir00000004, inst_id: inst48000124}
// CHECK:STDOUT:     import_ir_inst0000004F: {ir_id: import_ir00000004, inst_id: inst48000125}
// CHECK:STDOUT:     import_ir_inst00000050: {ir_id: import_ir00000004, inst_id: inst48000128}
// CHECK:STDOUT:     import_ir_inst00000051: {ir_id: import_ir00000004, inst_id: inst48000114}
// CHECK:STDOUT:     import_ir_inst00000052: {ir_id: import_ir00000004, inst_id: inst48000115}
// CHECK:STDOUT:     import_ir_inst00000053: {ir_id: import_ir00000004, inst_id: inst48000119}
// CHECK:STDOUT:     import_ir_inst00000054: {ir_id: import_ir00000004, inst_id: inst4800012E}
// CHECK:STDOUT:     import_ir_inst00000055: {ir_id: import_ir00000004, inst_id: inst4800012C}
// CHECK:STDOUT:     import_ir_inst00000056: {ir_id: import_ir00000004, inst_id: inst(TypeType)}
// CHECK:STDOUT:     import_ir_inst00000057: {ir_id: import_ir00000004, inst_id: inst4800012B}
// CHECK:STDOUT:     import_ir_inst00000058: {ir_id: import_ir00000004, inst_id: inst48000143}
// CHECK:STDOUT:     import_ir_inst00000059: {ir_id: import_ir00000004, inst_id: inst48000141}
// CHECK:STDOUT:     import_ir_inst0000005A: {ir_id: import_ir00000004, inst_id: inst4800013F}
// CHECK:STDOUT:     import_ir_inst0000005B: {ir_id: import_ir00000004, inst_id: inst48000140}
// CHECK:STDOUT:     import_ir_inst0000005C: {ir_id: import_ir00000004, inst_id: inst48000173}
// CHECK:STDOUT:     import_ir_inst0000005D: {ir_id: import_ir00000004, inst_id: inst48000171}
// CHECK:STDOUT:     import_ir_inst0000005E: {ir_id: import_ir00000004, inst_id: inst4800015E}
// CHECK:STDOUT:     import_ir_inst0000005F: {ir_id: import_ir00000004, inst_id: inst48000158}
// CHECK:STDOUT:     import_ir_inst00000060: {ir_id: import_ir00000004, inst_id: inst48000156}
// CHECK:STDOUT:     import_ir_inst00000061: {ir_id: import_ir00000004, inst_id: inst4800015B}
// CHECK:STDOUT:     import_ir_inst00000062: {ir_id: import_ir00000004, inst_id: inst4800016C}
// CHECK:STDOUT:     import_ir_inst00000063: {ir_id: import_ir00000004, inst_id: inst4800016E}
// CHECK:STDOUT:     import_ir_inst00000064: {ir_id: import_ir00000004, inst_id: inst4800018B}
// CHECK:STDOUT:     import_ir_inst00000065: {ir_id: import_ir00000004, inst_id: inst48000172}
// CHECK:STDOUT:     import_ir_inst00000066: {ir_id: import_ir00000004, inst_id: inst48000157}
// CHECK:STDOUT:     import_ir_inst00000067: {ir_id: import_ir00000004, inst_id: inst4800015D}
// CHECK:STDOUT:     import_ir_inst00000068: {ir_id: import_ir00000004, inst_id: inst48000165}
// CHECK:STDOUT:     import_ir_inst00000069: {ir_id: import_ir00000004, inst_id: inst48000169}
// CHECK:STDOUT:     import_ir_inst0000006A: {ir_id: import_ir00000004, inst_id: inst4800016D}
// CHECK:STDOUT:     import_ir_inst0000006B: {ir_id: import_ir00000004, inst_id: inst48000175}
// CHECK:STDOUT:     import_ir_inst0000006C: {ir_id: import_ir00000004, inst_id: inst4800018B}
// CHECK:STDOUT:     import_ir_inst0000006D: {ir_id: import_ir00000004, inst_id: inst48000186}
// CHECK:STDOUT:     import_ir_inst0000006E: {ir_id: import_ir00000004, inst_id: inst48000187}
// CHECK:STDOUT:     import_ir_inst0000006F: {ir_id: import_ir00000004, inst_id: inst48000182}
// CHECK:STDOUT:     import_ir_inst00000070: {ir_id: import_ir00000004, inst_id: inst48000184}
// CHECK:STDOUT:     import_ir_inst00000071: {ir_id: import_ir00000004, inst_id: inst48000156}
// CHECK:STDOUT:     import_ir_inst00000072: {ir_id: import_ir00000004, inst_id: inst4800015B}
// CHECK:STDOUT:     import_ir_inst00000073: {ir_id: import_ir00000004, inst_id: inst4800018E}
// CHECK:STDOUT:     import_ir_inst00000074: {ir_id: import_ir00000004, inst_id: inst4800018F}
// CHECK:STDOUT:     import_ir_inst00000075: {ir_id: import_ir00000004, inst_id: inst4800017B}
// CHECK:STDOUT:     import_ir_inst00000076: {ir_id: import_ir00000004, inst_id: inst4800017C}
// CHECK:STDOUT:     import_ir_inst00000077: {ir_id: import_ir00000004, inst_id: inst4800017D}
// CHECK:STDOUT:     import_ir_inst00000078: {ir_id: import_ir00000004, inst_id: inst4800017E}
// CHECK:STDOUT:     import_ir_inst00000079: {ir_id: import_ir00000004, inst_id: inst4800017F}
// CHECK:STDOUT:     import_ir_inst0000007A: {ir_id: import_ir00000004, inst_id: inst48000183}
// CHECK:STDOUT:     import_ir_inst0000007B: {ir_id: import_ir00000004, inst_id: inst48000194}
// CHECK:STDOUT:     import_ir_inst0000007C: {ir_id: import_ir00000004, inst_id: inst4800019C}
// CHECK:STDOUT:     import_ir_inst0000007D: {ir_id: import_ir00000004, inst_id: inst480001A0}
// CHECK:STDOUT:     import_ir_inst0000007E: {ir_id: import_ir00000004, inst_id: inst480001A2}
// CHECK:STDOUT:     import_ir_inst0000007F: {ir_id: import_ir00000004, inst_id: inst480001A3}
// CHECK:STDOUT:     import_ir_inst00000080: {ir_id: import_ir00000004, inst_id: inst480001A6}
// CHECK:STDOUT:     import_ir_inst00000081: {ir_id: import_ir00000004, inst_id: inst480001B2}
// CHECK:STDOUT:     import_ir_inst00000082: {ir_id: import_ir00000004, inst_id: inst480001B7}
// CHECK:STDOUT:     import_ir_inst00000083: {ir_id: import_ir00000004, inst_id: inst480001BB}
// CHECK:STDOUT:     import_ir_inst00000084: {ir_id: import_ir00000004, inst_id: inst480001BC}
// CHECK:STDOUT:     import_ir_inst00000085: {ir_id: import_ir00000004, inst_id: inst480001C1}
// CHECK:STDOUT:     import_ir_inst00000086: {ir_id: import_ir00000004, inst_id: inst480001FA}
// CHECK:STDOUT:     import_ir_inst00000087: {ir_id: import_ir00000004, inst_id: inst480001F8}
// CHECK:STDOUT:     import_ir_inst00000088: {ir_id: import_ir00000004, inst_id: inst480001E0}
// CHECK:STDOUT:     import_ir_inst00000089: {ir_id: import_ir00000004, inst_id: inst480001DA}
// CHECK:STDOUT:     import_ir_inst0000008A: {ir_id: import_ir00000004, inst_id: inst480001D5}
// CHECK:STDOUT:     import_ir_inst0000008B: {ir_id: import_ir00000004, inst_id: inst480001D3}
// CHECK:STDOUT:     import_ir_inst0000008C: {ir_id: import_ir00000004, inst_id: inst480001D8}
// CHECK:STDOUT:     import_ir_inst0000008D: {ir_id: import_ir00000004, inst_id: inst480001DD}
// CHECK:STDOUT:     import_ir_inst0000008E: {ir_id: import_ir00000004, inst_id: inst480001F2}
// CHECK:STDOUT:     import_ir_inst0000008F: {ir_id: import_ir00000004, inst_id: inst480001F4}
// CHECK:STDOUT:     import_ir_inst00000090: {ir_id: import_ir00000004, inst_id: inst48000216}
// CHECK:STDOUT:     import_ir_inst00000091: {ir_id: import_ir00000004, inst_id: inst480001F9}
// CHECK:STDOUT:     import_ir_inst00000092: {ir_id: import_ir00000004, inst_id: inst480001D4}
// CHECK:STDOUT:     import_ir_inst00000093: {ir_id: import_ir00000004, inst_id: inst480001D9}
// CHECK:STDOUT:     import_ir_inst00000094: {ir_id: import_ir00000004, inst_id: inst480001DF}
// CHECK:STDOUT:     import_ir_inst00000095: {ir_id: import_ir00000004, inst_id: inst480001E8}
// CHECK:STDOUT:     import_ir_inst00000096: {ir_id: import_ir00000004, inst_id: inst480001EB}
// CHECK:STDOUT:     import_ir_inst00000097: {ir_id: import_ir00000004, inst_id: inst480001EF}
// CHECK:STDOUT:     import_ir_inst00000098: {ir_id: import_ir00000004, inst_id: inst480001F3}
// CHECK:STDOUT:     import_ir_inst00000099: {ir_id: import_ir00000004, inst_id: inst480001FC}
// CHECK:STDOUT:     import_ir_inst0000009A: {ir_id: import_ir00000004, inst_id: inst48000216}
// CHECK:STDOUT:     import_ir_inst0000009B: {ir_id: import_ir00000004, inst_id: inst48000211}
// CHECK:STDOUT:     import_ir_inst0000009C: {ir_id: import_ir00000004, inst_id: inst48000212}
// CHECK:STDOUT:     import_ir_inst0000009D: {ir_id: import_ir00000004, inst_id: inst4800020D}
// CHECK:STDOUT:     import_ir_inst0000009E: {ir_id: import_ir00000004, inst_id: inst4800020F}
// CHECK:STDOUT:     import_ir_inst0000009F: {ir_id: import_ir00000004, inst_id: inst480001D3}
// CHECK:STDOUT:     import_ir_inst000000A0: {ir_id: import_ir00000004, inst_id: inst480001D8}
// CHECK:STDOUT:     import_ir_inst000000A1: {ir_id: import_ir00000004, inst_id: inst480001DD}
// CHECK:STDOUT:     import_ir_inst000000A2: {ir_id: import_ir00000004, inst_id: inst48000219}
// CHECK:STDOUT:     import_ir_inst000000A3: {ir_id: import_ir00000004, inst_id: inst4800021A}
// CHECK:STDOUT:     import_ir_inst000000A4: {ir_id: import_ir00000004, inst_id: inst48000204}
// CHECK:STDOUT:     import_ir_inst000000A5: {ir_id: import_ir00000004, inst_id: inst48000205}
// CHECK:STDOUT:     import_ir_inst000000A6: {ir_id: import_ir00000004, inst_id: inst48000206}
// CHECK:STDOUT:     import_ir_inst000000A7: {ir_id: import_ir00000004, inst_id: inst48000207}
// CHECK:STDOUT:     import_ir_inst000000A8: {ir_id: import_ir00000004, inst_id: inst48000208}
// CHECK:STDOUT:     import_ir_inst000000A9: {ir_id: import_ir00000004, inst_id: inst48000209}
// CHECK:STDOUT:     import_ir_inst000000AA: {ir_id: import_ir00000004, inst_id: inst4800020A}
// CHECK:STDOUT:     import_ir_inst000000AB: {ir_id: import_ir00000004, inst_id: inst4800020E}
// CHECK:STDOUT:     import_ir_inst000000AC: {ir_id: import_ir00000004, inst_id: inst4800021F}
// CHECK:STDOUT:     import_ir_inst000000AD: {ir_id: import_ir00000004, inst_id: inst48000226}
// CHECK:STDOUT:     import_ir_inst000000AE: {ir_id: import_ir00000004, inst_id: inst4800022A}
// CHECK:STDOUT:     import_ir_inst000000AF: {ir_id: import_ir00000004, inst_id: inst4800022C}
// CHECK:STDOUT:     import_ir_inst000000B0: {ir_id: import_ir00000004, inst_id: inst4800022D}
// CHECK:STDOUT:     import_ir_inst000000B1: {ir_id: import_ir00000004, inst_id: inst48000230}
// CHECK:STDOUT:     import_ir_inst000000B2: {ir_id: import_ir00000004, inst_id: inst4800023A}
// CHECK:STDOUT:     import_ir_inst000000B3: {ir_id: import_ir00000004, inst_id: inst4800023E}
// CHECK:STDOUT:     import_ir_inst000000B4: {ir_id: import_ir00000004, inst_id: inst48000240}
// CHECK:STDOUT:     import_ir_inst000000B5: {ir_id: import_ir00000004, inst_id: inst48000241}
// CHECK:STDOUT:     import_ir_inst000000B6: {ir_id: import_ir00000004, inst_id: inst48000244}
// CHECK:STDOUT:     import_ir_inst000000B7: {ir_id: import_ir00000004, inst_id: inst48000250}
// CHECK:STDOUT:     import_ir_inst000000B8: {ir_id: import_ir00000004, inst_id: inst48000255}
// CHECK:STDOUT:     import_ir_inst000000B9: {ir_id: import_ir00000004, inst_id: inst48000259}
// CHECK:STDOUT:     import_ir_inst000000BA: {ir_id: import_ir00000004, inst_id: inst4800025A}
// CHECK:STDOUT:     import_ir_inst000000BB: {ir_id: import_ir00000004, inst_id: inst4800025F}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope00000000: {inst: inst0000000E, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst60000010, name00000000: inst60000035}}
// CHECK:STDOUT:     name_scope60000001: {inst: inst60000010, parent_scope: name_scope00000000, has_error: false, extended_scopes: [], names: {name00000003: inst60000045}}
// CHECK:STDOUT:     name_scope60000002: {inst: inst60000046, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {name(SelfType): inst60000049, name00000004: inst6000004A}}
// CHECK:STDOUT:     name_scope60000003: {inst: inst6000005E, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000004: {inst: inst6000008B, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000005: {inst: inst6000008F, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000006: {inst: inst60000093, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000007: {inst: inst60000097, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000008: {inst: inst6000009B, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope60000009: {inst: inst600000B5, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6000000A: {inst: inst600000B9, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6000000B: {inst: inst600000BD, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope6000000C: {inst: inst600000F5, parent_scope: name_scope60000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name60000000: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000001: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000002: {name: name00000002, parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000003: {name: name00000003, parent_scope: name_scope60000001, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000004: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000005: {name: name00000004, parent_scope: name_scope60000002, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000006: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000007: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000008: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000009: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000A: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000B: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000C: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000D: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000E: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000000F: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000010: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000011: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000012: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000013: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000014: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000015: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000016: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000017: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000018: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000019: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000001A: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000001B: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000001C: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000001D: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000001E: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000001F: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000020: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000021: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000022: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000023: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000024: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000025: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000026: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000027: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000028: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000029: {name: name00000006, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name6000002A: {name: name00000006, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name6000002B: {name: name00000006, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name6000002C: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000002D: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000002E: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000002F: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000030: {name: name00000006, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name60000031: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000032: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000033: {name: name00000006, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name60000034: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000035: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name60000036: {name: name00000006, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name60000037: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0}
// CHECK:STDOUT:     entity_name60000038: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name60000039: {name: name00000001, parent_scope: name_scope<none>, index: 0, is_template: 0}
// CHECK:STDOUT:     entity_name6000003A: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000003B: {name: name00000005, parent_scope: name_scope<none>, index: 1, is_template: 0}
// CHECK:STDOUT:     entity_name6000003C: {name: name00000006, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:     entity_name6000003D: {name: name00000006, parent_scope: name_scope<none>, index: 2, is_template: 0}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function60000000: {name: name00000000, parent_scope: name_scope00000000, call_params_id: inst_block6000000D, return_slot_pattern: inst60000030, body: [inst_block60000014]}
// CHECK:STDOUT:     function60000001: {name: name00000004, parent_scope: name_scope60000002, return_slot_pattern: inst60000055}
// CHECK:STDOUT:     function60000002: {name: name00000004, parent_scope: name_scope60000003, return_slot_pattern: inst60000073}
// CHECK:STDOUT:     function60000003: {name: name00000004, parent_scope: name_scope60000008, return_slot_pattern: inst600000AA}
// CHECK:STDOUT:     function60000004: {name: name00000004, parent_scope: name_scope6000000B, return_slot_pattern: inst600000D5}
// CHECK:STDOUT:     function60000005: {name: name00000004, parent_scope: name_scope6000000C, return_slot_pattern: inst60000111}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic60000000: {decl: inst60000035, bindings: inst_block60000010}
// CHECK:STDOUT:     generic60000001: {decl: inst6000004F, bindings: inst_block60000018}
// CHECK:STDOUT:     generic60000002: {decl: inst6000005E, bindings: inst_block6000001D}
// CHECK:STDOUT:     generic60000003: {decl: inst6000006E, bindings: inst_block60000024}
// CHECK:STDOUT:     generic60000004: {decl: inst6000009B, bindings: inst_block6000002E}
// CHECK:STDOUT:     generic60000005: {decl: inst600000A6, bindings: inst_block60000034}
// CHECK:STDOUT:     generic60000006: {decl: inst600000BD, bindings: inst_block6000003B}
// CHECK:STDOUT:     generic60000007: {decl: inst600000D0, bindings: inst_block60000043}
// CHECK:STDOUT:     generic60000008: {decl: inst600000F5, bindings: inst_block60000051}
// CHECK:STDOUT:     generic60000009: {decl: inst6000010C, bindings: inst_block60000059}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific60000000: {generic: generic60000000, args: inst_block60000011}
// CHECK:STDOUT:     specific60000001: {generic: generic60000001, args: inst_block6000001A}
// CHECK:STDOUT:     specific60000002: {generic: generic60000002, args: inst_block6000001F}
// CHECK:STDOUT:     specific60000003: {generic: generic60000002, args: inst_block60000020}
// CHECK:STDOUT:     specific60000004: {generic: generic60000003, args: inst_block6000001F}
// CHECK:STDOUT:     specific60000005: {generic: generic60000001, args: inst_block6000001F}
// CHECK:STDOUT:     specific60000006: {generic: generic60000001, args: inst_block60000029}
// CHECK:STDOUT:     specific60000007: {generic: generic60000004, args: inst_block60000011}
// CHECK:STDOUT:     specific60000008: {generic: generic60000004, args: inst_block60000030}
// CHECK:STDOUT:     specific60000009: {generic: generic60000005, args: inst_block60000011}
// CHECK:STDOUT:     specific6000000A: {generic: generic60000006, args: inst_block6000003D}
// CHECK:STDOUT:     specific6000000B: {generic: generic60000006, args: inst_block6000003F}
// CHECK:STDOUT:     specific6000000C: {generic: generic60000007, args: inst_block6000003D}
// CHECK:STDOUT:     specific6000000D: {generic: generic60000001, args: inst_block60000049}
// CHECK:STDOUT:     specific6000000E: {generic: generic60000001, args: inst_block6000004A}
// CHECK:STDOUT:     specific6000000F: {generic: generic60000001, args: inst_block6000004B}
// CHECK:STDOUT:     specific60000010: {generic: generic60000008, args: inst_block60000053}
// CHECK:STDOUT:     specific60000011: {generic: generic60000008, args: inst_block60000055}
// CHECK:STDOUT:     specific60000012: {generic: generic60000009, args: inst_block60000053}
// CHECK:STDOUT:     specific60000013: {generic: generic60000001, args: inst_block6000005F}
// CHECK:STDOUT:     specific60000014: {generic: generic60000001, args: inst_block60000060}
// CHECK:STDOUT:     specific60000015: {generic: generic60000001, args: inst_block60000061}
// CHECK:STDOUT:     specific60000016: {generic: generic60000001, args: inst_block60000062}
// CHECK:STDOUT:     specific60000017: {generic: generic60000001, args: inst_block60000069}
// CHECK:STDOUT:     specific60000018: {generic: generic60000001, args: inst_block6000006B}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields00000000: {}
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
// CHECK:STDOUT:     'type(symbolic_constant00000003)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant00000003)}
// CHECK:STDOUT:     'type(symbolic_constant00000007)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant0000000B)}
// CHECK:STDOUT:     'type(symbolic_constant0000000B)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant0000000B)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(symbolic_constant00000004)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant00000004)}
// CHECK:STDOUT:     'type(symbolic_constant00000008)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant0000000B)}
// CHECK:STDOUT:     'type(inst60000047)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst60000047)}
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     'type(symbolic_constant600000FD)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(symbolic_constant60000100)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst60000024)}
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     inst0000000E:    {kind: Namespace, arg0: name_scope00000000, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst6000000F:    {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst60000010:    {kind: Namespace, arg0: name_scope60000001, arg1: inst6000000F, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst60000011:    {kind: FacetType, arg0: facet_type60000000, type: type(TypeType)}
// CHECK:STDOUT:     inst60000012:    {kind: BindSymbolicName, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000013:    {kind: BindSymbolicName, arg0: entity_name60000000, arg1: inst<none>, type: type(inst60000011)}
// CHECK:STDOUT:     inst60000014:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000015:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000016:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000017:    {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst60000018:    {kind: SymbolicBindingPattern, arg0: entity_name60000001, type: type(inst60000017)}
// CHECK:STDOUT:     inst60000019:    {kind: NameRef, arg0: name00000001, arg1: inst60000014, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001A:    {kind: PointerType, arg0: inst60000019, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001B:    {kind: PointerType, arg0: inst60000015, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001C:    {kind: PointerType, arg0: inst60000016, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001D:    {kind: BindName, arg0: entity_name60000002, arg1: inst60000031, type: type(symbolic_constant00000004)}
// CHECK:STDOUT:     inst6000001E:    {kind: PatternType, arg0: inst6000001B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000001F:    {kind: BindingPattern, arg0: entity_name60000002, type: type(symbolic_constant00000006)}
// CHECK:STDOUT:     inst60000020:    {kind: PatternType, arg0: inst6000001C, type: type(TypeType)}
// CHECK:STDOUT:     inst60000021:    {kind: ValueParamPattern, arg0: inst6000001F, arg1: call_param0, type: type(symbolic_constant00000006)}
// CHECK:STDOUT:     inst60000022:    {kind: NameRef, arg0: name00000001, arg1: inst60000014, type: type(TypeType)}
// CHECK:STDOUT:     inst60000023:    {kind: PointerType, arg0: inst60000022, type: type(TypeType)}
// CHECK:STDOUT:     inst60000024:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000025:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000026:    {kind: TupleType, arg0: inst_block60000009, type: type(TypeType)}
// CHECK:STDOUT:     inst60000027:    {kind: TupleLiteral, arg0: inst_block60000008, type: type(inst60000026)}
// CHECK:STDOUT:     inst60000028:    {kind: PointerType, arg0: inst60000026, type: type(TypeType)}
// CHECK:STDOUT:     inst60000029:    {kind: Converted, arg0: inst60000025, arg1: inst60000024, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002A:    {kind: TupleType, arg0: inst_block6000000B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002B:    {kind: Converted, arg0: inst60000027, arg1: inst6000002A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002C:    {kind: TupleType, arg0: inst_block6000000C, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002D:    {kind: PatternType, arg0: inst6000002A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000002E:    {kind: ReturnSlotPattern, arg0: inst6000002B, type: type(symbolic_constant0000000A)}
// CHECK:STDOUT:     inst6000002F:    {kind: PatternType, arg0: inst6000002C, type: type(TypeType)}
// CHECK:STDOUT:     inst60000030:    {kind: OutParamPattern, arg0: inst6000002E, arg1: call_param1, type: type(symbolic_constant0000000A)}
// CHECK:STDOUT:     inst60000031:    {kind: ValueParam, arg0: call_param0, arg1: name00000002, type: type(symbolic_constant00000004)}
// CHECK:STDOUT:     inst60000032:    {kind: SpliceBlock, arg0: inst_block60000006, arg1: inst6000001A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000033:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant00000008)}
// CHECK:STDOUT:     inst60000034:    {kind: ReturnSlot, arg0: inst6000002A, arg1: inst60000033, type: type(symbolic_constant00000008)}
// CHECK:STDOUT:     inst60000035:    {kind: FunctionDecl, arg0: function60000000, arg1: inst_block6000000F, type: type(inst60000036)}
// CHECK:STDOUT:     inst60000036:    {kind: FunctionType, arg0: function60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000037:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000036)}
// CHECK:STDOUT:     inst60000038:    {kind: PointerType, arg0: inst6000002A, type: type(TypeType)}
// CHECK:STDOUT:     inst60000039:    {kind: RequireCompleteType, arg0: inst6000002A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003A:    {kind: RequireCompleteType, arg0: inst6000002A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003B:    {kind: RequireCompleteType, arg0: inst6000002C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003C:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003D:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003E:    {kind: RequireCompleteType, arg0: inst6000001C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000003F:    {kind: NameRef, arg0: name00000002, arg1: inst6000001D, type: type(symbolic_constant00000004)}
// CHECK:STDOUT:     inst60000040:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst60000041:    {kind: TupleLiteral, arg0: inst_block60000015, type: type(symbolic_constant00000008)}
// CHECK:STDOUT:     inst60000042:    {kind: RequireCompleteType, arg0: inst6000002A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000043:    {kind: TupleAccess, arg0: inst60000034, arg1: element0, type: type(symbolic_constant00000004)}
// CHECK:STDOUT:     inst60000044:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000045:    {kind: ImportRefLoaded, arg0: import_ir_inst00000000, arg1: entity_name60000003, type: type(TypeType)}
// CHECK:STDOUT:     inst60000046:    {kind: InterfaceDecl, arg0: interface60000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst60000047:    {kind: FacetType, arg0: facet_type60000001, type: type(TypeType)}
// CHECK:STDOUT:     inst60000048:    {kind: BindSymbolicName, arg0: entity_name60000004, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000049:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000002, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000004A:    {kind: ImportRefLoaded, arg0: import_ir_inst00000003, arg1: entity_name<none>, type: type(inst6000004C)}
// CHECK:STDOUT:     inst6000004B:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000004, arg1: entity_name60000005}
// CHECK:STDOUT:     inst6000004C:    {kind: AssociatedEntityType, arg0: interface60000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000004D:    {kind: ImportRefLoaded, arg0: import_ir_inst00000005, arg1: entity_name<none>, type: type(inst60000050)}
// CHECK:STDOUT:     inst6000004E:    {kind: AssociatedEntity, arg0: element0, arg1: inst6000004D, type: type(inst6000004C)}
// CHECK:STDOUT:     inst6000004F:    {kind: FunctionDecl, arg0: function60000001, arg1: inst_block_empty, type: type(inst60000050)}
// CHECK:STDOUT:     inst60000050:    {kind: FunctionType, arg0: function60000001, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000051:    {kind: StructValue, arg0: inst_block_empty, type: type(inst60000050)}
// CHECK:STDOUT:     inst60000052:    {kind: SymbolicBindingType, arg0: entity_name60000004, arg1: inst60000048, type: type(TypeType)}
// CHECK:STDOUT:     inst60000053:    {kind: PatternType, arg0: inst60000052, type: type(TypeType)}
// CHECK:STDOUT:     inst60000054:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant60000013)}
// CHECK:STDOUT:     inst60000055:    {kind: OutParamPattern, arg0: inst60000054, arg1: call_param1, type: type(symbolic_constant60000013)}
// CHECK:STDOUT:     inst60000056:    {kind: BindingPattern, arg0: entity_name60000008, type: type(symbolic_constant60000013)}
// CHECK:STDOUT:     inst60000057:    {kind: ValueParamPattern, arg0: inst60000056, arg1: call_param0, type: type(symbolic_constant60000013)}
// CHECK:STDOUT:     inst60000058:    {kind: ImportRefLoaded, arg0: import_ir_inst0000000B, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000059:    {kind: BindSymbolicName, arg0: entity_name60000004, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000005A:    {kind: SymbolicBindingType, arg0: entity_name60000004, arg1: inst60000059, type: type(TypeType)}
// CHECK:STDOUT:     inst6000005B:    {kind: PatternType, arg0: inst6000005A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000005C:    {kind: LookupImplWitness, arg0: inst6000001B, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000005D:    {kind: ImportRefUnloaded, arg0: import_ir_inst0000000F, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000005E:    {kind: ImplDecl, arg0: impl60000000, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000005F:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000060:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst6000005F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000061:    {kind: ConstType, arg0: inst60000060, type: type(TypeType)}
// CHECK:STDOUT:     inst60000062:    {kind: PatternType, arg0: inst60000047, type: type(TypeType)}
// CHECK:STDOUT:     inst60000063:    {kind: SymbolicBindingPattern, arg0: entity_name6000000D, type: type(inst60000062)}
// CHECK:STDOUT:     inst60000064:    {kind: ImportRefLoaded, arg0: import_ir_inst00000012, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000065:    {kind: ImportRefLoaded, arg0: import_ir_inst00000013, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000066:    {kind: ImportRefLoaded, arg0: import_ir_inst00000014, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000067:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000015, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000068:    {kind: ImplWitnessTable, arg0: inst_block6000001E, arg1: impl60000000}
// CHECK:STDOUT:     inst60000069:    {kind: ImplWitness, arg0: inst60000068, arg1: specific60000002, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000006A:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000006B:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst6000006A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000006C:    {kind: ConstType, arg0: inst6000006B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000006D:    {kind: ImplWitness, arg0: inst60000068, arg1: specific60000003, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000006E:    {kind: FunctionDecl, arg0: function60000002, arg1: inst_block_empty, type: type(symbolic_constant60000026)}
// CHECK:STDOUT:     inst6000006F:    {kind: FunctionType, arg0: function60000002, arg1: specific60000002, type: type(TypeType)}
// CHECK:STDOUT:     inst60000070:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000026)}
// CHECK:STDOUT:     inst60000071:    {kind: PatternType, arg0: inst60000061, type: type(TypeType)}
// CHECK:STDOUT:     inst60000072:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant6000002B)}
// CHECK:STDOUT:     inst60000073:    {kind: OutParamPattern, arg0: inst60000072, arg1: call_param1, type: type(symbolic_constant6000002B)}
// CHECK:STDOUT:     inst60000074:    {kind: BindingPattern, arg0: entity_name60000011, type: type(symbolic_constant6000002B)}
// CHECK:STDOUT:     inst60000075:    {kind: ValueParamPattern, arg0: inst60000074, arg1: call_param0, type: type(symbolic_constant6000002B)}
// CHECK:STDOUT:     inst60000076:    {kind: ImportRefLoaded, arg0: import_ir_inst00000020, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000077:    {kind: FunctionType, arg0: function60000002, arg1: specific60000003, type: type(TypeType)}
// CHECK:STDOUT:     inst60000078:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant6000002D)}
// CHECK:STDOUT:     inst60000079:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000007A:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst60000079, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007B:    {kind: ConstType, arg0: inst6000007A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007C:    {kind: PatternType, arg0: inst6000007B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000007D:    {kind: RequireCompleteType, arg0: inst60000061, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000007E:    {kind: RequireCompleteType, arg0: inst60000060, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000007F:    {kind: LookupImplWitness, arg0: inst6000005F, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000080:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000005F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000081:    {kind: ImplWitnessAccess, arg0: inst6000007F, arg1: element0, type: type(symbolic_constant6000003C)}
// CHECK:STDOUT:     inst60000082:    {kind: SpecificImplFunction, arg0: inst60000081, arg1: specific60000005, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000083:    {kind: RequireCompleteType, arg0: inst6000007B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000084:    {kind: RequireCompleteType, arg0: inst6000007A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000085:    {kind: LookupImplWitness, arg0: inst60000079, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000086:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst60000079, type: type(TypeType)}
// CHECK:STDOUT:     inst60000087:    {kind: ImplWitnessAccess, arg0: inst60000085, arg1: element0, type: type(symbolic_constant60000045)}
// CHECK:STDOUT:     inst60000088:    {kind: SpecificImplFunction, arg0: inst60000087, arg1: specific60000006, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000089:    {kind: PatternType, arg0: inst60000060, type: type(TypeType)}
// CHECK:STDOUT:     inst6000008A:    {kind: ImportRefUnloaded, arg0: import_ir_inst0000002D, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000008B:    {kind: ImplDecl, arg0: impl60000001, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000008C:    {kind: ImportRefLoaded, arg0: import_ir_inst0000002F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000008D:    {kind: ImportRefLoaded, arg0: import_ir_inst00000030, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000008E:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000031, arg1: entity_name<none>}
// CHECK:STDOUT:     inst6000008F:    {kind: ImplDecl, arg0: impl60000002, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000090:    {kind: ImportRefLoaded, arg0: import_ir_inst00000033, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000091:    {kind: ImportRefLoaded, arg0: import_ir_inst00000034, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000092:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000035, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000093:    {kind: ImplDecl, arg0: impl60000003, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000094:    {kind: ImportRefLoaded, arg0: import_ir_inst00000037, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000095:    {kind: ImportRefLoaded, arg0: import_ir_inst00000038, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000096:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000039, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000097:    {kind: ImplDecl, arg0: impl60000004, arg1: inst_block_empty}
// CHECK:STDOUT:     inst60000098:    {kind: ImportRefLoaded, arg0: import_ir_inst0000003B, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000099:    {kind: ImportRefLoaded, arg0: import_ir_inst0000003C, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000009A:    {kind: ImportRefLoaded, arg0: import_ir_inst0000003D, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000009B:    {kind: ImplDecl, arg0: impl60000005, arg1: inst_block_empty}
// CHECK:STDOUT:     inst6000009C:    {kind: SymbolicBindingPattern, arg0: entity_name60000015, type: type(inst60000017)}
// CHECK:STDOUT:     inst6000009D:    {kind: ImportRefLoaded, arg0: import_ir_inst00000040, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000009E:    {kind: ImportRefLoaded, arg0: import_ir_inst00000041, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst6000009F:    {kind: ImportRefLoaded, arg0: import_ir_inst00000042, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A0:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000043, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000A1:    {kind: ImplWitnessTable, arg0: inst_block6000002F, arg1: impl60000005}
// CHECK:STDOUT:     inst600000A2:    {kind: ImplWitness, arg0: inst600000A1, arg1: specific60000007, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000A3:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A4:    {kind: PointerType, arg0: inst600000A3, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A5:    {kind: ImplWitness, arg0: inst600000A1, arg1: specific60000008, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000A6:    {kind: FunctionDecl, arg0: function60000003, arg1: inst_block_empty, type: type(symbolic_constant60000051)}
// CHECK:STDOUT:     inst600000A7:    {kind: FunctionType, arg0: function60000003, arg1: specific60000007, type: type(TypeType)}
// CHECK:STDOUT:     inst600000A8:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000051)}
// CHECK:STDOUT:     inst600000A9:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant60000055)}
// CHECK:STDOUT:     inst600000AA:    {kind: OutParamPattern, arg0: inst600000A9, arg1: call_param1, type: type(symbolic_constant60000055)}
// CHECK:STDOUT:     inst600000AB:    {kind: BindingPattern, arg0: entity_name60000018, type: type(symbolic_constant60000055)}
// CHECK:STDOUT:     inst600000AC:    {kind: ValueParamPattern, arg0: inst600000AB, arg1: call_param0, type: type(symbolic_constant60000055)}
// CHECK:STDOUT:     inst600000AD:    {kind: ImportRefLoaded, arg0: import_ir_inst0000004D, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000AE:    {kind: FunctionType, arg0: function60000003, arg1: specific60000008, type: type(TypeType)}
// CHECK:STDOUT:     inst600000AF:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000058)}
// CHECK:STDOUT:     inst600000B0:    {kind: RequireCompleteType, arg0: inst600000A4, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000B1:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B2:    {kind: PointerType, arg0: inst600000B1, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B3:    {kind: PatternType, arg0: inst600000B2, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B4:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000054, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000B5:    {kind: ImplDecl, arg0: impl60000006, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000B6:    {kind: ImportRefLoaded, arg0: import_ir_inst00000056, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B7:    {kind: ImportRefLoaded, arg0: import_ir_inst00000057, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000B8:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000058, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000B9:    {kind: ImplDecl, arg0: impl60000007, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000BA:    {kind: ImportRefLoaded, arg0: import_ir_inst0000005A, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000BB:    {kind: ImportRefLoaded, arg0: import_ir_inst0000005B, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000BC:    {kind: ImportRefUnloaded, arg0: import_ir_inst0000005C, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000BD:    {kind: ImplDecl, arg0: impl60000008, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000BE:    {kind: BindSymbolicName, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000BF:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst600000BE, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C0:    {kind: TupleType, arg0: inst_block60000039, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C1:    {kind: SymbolicBindingPattern, arg0: entity_name6000001C, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000C2:    {kind: SymbolicBindingPattern, arg0: entity_name6000001D, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000C3:    {kind: ImportRefLoaded, arg0: import_ir_inst00000060, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000C4:    {kind: ImportRefLoaded, arg0: import_ir_inst00000061, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000C5:    {kind: ImportRefLoaded, arg0: import_ir_inst00000062, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C6:    {kind: ImportRefLoaded, arg0: import_ir_inst00000063, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst600000C7:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000064, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000C8:    {kind: ImplWitnessTable, arg0: inst_block6000003C, arg1: impl60000008}
// CHECK:STDOUT:     inst600000C9:    {kind: ImplWitness, arg0: inst600000C8, arg1: specific6000000A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000CA:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000CB:    {kind: BindSymbolicName, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000CC:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst600000CA, type: type(TypeType)}
// CHECK:STDOUT:     inst600000CD:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst600000CB, type: type(TypeType)}
// CHECK:STDOUT:     inst600000CE:    {kind: TupleType, arg0: inst_block6000003E, type: type(TypeType)}
// CHECK:STDOUT:     inst600000CF:    {kind: ImplWitness, arg0: inst600000C8, arg1: specific6000000B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000D0:    {kind: FunctionDecl, arg0: function60000004, arg1: inst_block_empty, type: type(symbolic_constant60000072)}
// CHECK:STDOUT:     inst600000D1:    {kind: FunctionType, arg0: function60000004, arg1: specific6000000A, type: type(TypeType)}
// CHECK:STDOUT:     inst600000D2:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000072)}
// CHECK:STDOUT:     inst600000D3:    {kind: PatternType, arg0: inst600000C0, type: type(TypeType)}
// CHECK:STDOUT:     inst600000D4:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant60000077)}
// CHECK:STDOUT:     inst600000D5:    {kind: OutParamPattern, arg0: inst600000D4, arg1: call_param1, type: type(symbolic_constant60000077)}
// CHECK:STDOUT:     inst600000D6:    {kind: BindingPattern, arg0: entity_name60000024, type: type(symbolic_constant60000077)}
// CHECK:STDOUT:     inst600000D7:    {kind: ValueParamPattern, arg0: inst600000D6, arg1: call_param0, type: type(symbolic_constant60000077)}
// CHECK:STDOUT:     inst600000D8:    {kind: ImportRefLoaded, arg0: import_ir_inst00000071, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000D9:    {kind: ImportRefLoaded, arg0: import_ir_inst00000072, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000DA:    {kind: FunctionType, arg0: function60000004, arg1: specific6000000B, type: type(TypeType)}
// CHECK:STDOUT:     inst600000DB:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant60000079)}
// CHECK:STDOUT:     inst600000DC:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000DD:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst600000DC, type: type(TypeType)}
// CHECK:STDOUT:     inst600000DE:    {kind: BindSymbolicName, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000DF:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst600000DE, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E0:    {kind: TupleType, arg0: inst_block60000046, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E1:    {kind: PatternType, arg0: inst600000E0, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E2:    {kind: RequireCompleteType, arg0: inst600000C0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E3:    {kind: RequireCompleteType, arg0: inst600000BF, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E4:    {kind: LookupImplWitness, arg0: inst600000BE, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E5:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst600000BE, type: type(TypeType)}
// CHECK:STDOUT:     inst600000E6:    {kind: ImplWitnessAccess, arg0: inst600000E4, arg1: element0, type: type(symbolic_constant60000091)}
// CHECK:STDOUT:     inst600000E7:    {kind: SpecificImplFunction, arg0: inst600000E6, arg1: specific6000000D, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000E8:    {kind: RequireCompleteType, arg0: inst600000E0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000E9:    {kind: RequireCompleteType, arg0: inst600000DD, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EA:    {kind: LookupImplWitness, arg0: inst600000DC, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EB:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst600000DC, type: type(TypeType)}
// CHECK:STDOUT:     inst600000EC:    {kind: ImplWitnessAccess, arg0: inst600000EA, arg1: element0, type: type(symbolic_constant6000009A)}
// CHECK:STDOUT:     inst600000ED:    {kind: SpecificImplFunction, arg0: inst600000EC, arg1: specific6000000E, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000EE:    {kind: RequireCompleteType, arg0: inst600000DF, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000EF:    {kind: LookupImplWitness, arg0: inst600000DE, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst600000F0:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst600000DE, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F1:    {kind: ImplWitnessAccess, arg0: inst600000EF, arg1: element0, type: type(symbolic_constant6000009F)}
// CHECK:STDOUT:     inst600000F2:    {kind: SpecificImplFunction, arg0: inst600000F1, arg1: specific6000000F, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst600000F3:    {kind: PatternType, arg0: inst600000BF, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F4:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000086, arg1: entity_name<none>}
// CHECK:STDOUT:     inst600000F5:    {kind: ImplDecl, arg0: impl60000009, arg1: inst_block_empty}
// CHECK:STDOUT:     inst600000F6:    {kind: BindSymbolicName, arg0: entity_name60000029, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000F7:    {kind: SymbolicBindingType, arg0: entity_name60000029, arg1: inst600000F6, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F8:    {kind: TupleType, arg0: inst_block6000004F, type: type(TypeType)}
// CHECK:STDOUT:     inst600000F9:    {kind: SymbolicBindingPattern, arg0: entity_name6000002B, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000FA:    {kind: SymbolicBindingPattern, arg0: entity_name6000002C, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000FB:    {kind: SymbolicBindingPattern, arg0: entity_name6000002D, type: type(inst60000062)}
// CHECK:STDOUT:     inst600000FC:    {kind: ImportRefLoaded, arg0: import_ir_inst0000008B, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000FD:    {kind: ImportRefLoaded, arg0: import_ir_inst0000008C, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000FE:    {kind: ImportRefLoaded, arg0: import_ir_inst0000008D, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst600000FF:    {kind: ImportRefLoaded, arg0: import_ir_inst0000008E, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000100:    {kind: ImportRefLoaded, arg0: import_ir_inst0000008F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst60000101:    {kind: ImportRefUnloaded, arg0: import_ir_inst00000090, arg1: entity_name<none>}
// CHECK:STDOUT:     inst60000102:    {kind: ImplWitnessTable, arg0: inst_block60000052, arg1: impl60000009}
// CHECK:STDOUT:     inst60000103:    {kind: ImplWitness, arg0: inst60000102, arg1: specific60000010, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000104:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000105:    {kind: BindSymbolicName, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000106:    {kind: BindSymbolicName, arg0: entity_name60000029, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000107:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst60000104, type: type(TypeType)}
// CHECK:STDOUT:     inst60000108:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst60000105, type: type(TypeType)}
// CHECK:STDOUT:     inst60000109:    {kind: SymbolicBindingType, arg0: entity_name60000029, arg1: inst60000106, type: type(TypeType)}
// CHECK:STDOUT:     inst6000010A:    {kind: TupleType, arg0: inst_block60000054, type: type(TypeType)}
// CHECK:STDOUT:     inst6000010B:    {kind: ImplWitness, arg0: inst60000102, arg1: specific60000011, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000010C:    {kind: FunctionDecl, arg0: function60000005, arg1: inst_block_empty, type: type(symbolic_constant600000BA)}
// CHECK:STDOUT:     inst6000010D:    {kind: FunctionType, arg0: function60000005, arg1: specific60000010, type: type(TypeType)}
// CHECK:STDOUT:     inst6000010E:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant600000BA)}
// CHECK:STDOUT:     inst6000010F:    {kind: PatternType, arg0: inst600000F8, type: type(TypeType)}
// CHECK:STDOUT:     inst60000110:    {kind: ReturnSlotPattern, arg0: inst<none>, type: type(symbolic_constant600000BF)}
// CHECK:STDOUT:     inst60000111:    {kind: OutParamPattern, arg0: inst60000110, arg1: call_param1, type: type(symbolic_constant600000BF)}
// CHECK:STDOUT:     inst60000112:    {kind: BindingPattern, arg0: entity_name60000037, type: type(symbolic_constant600000BF)}
// CHECK:STDOUT:     inst60000113:    {kind: ValueParamPattern, arg0: inst60000112, arg1: call_param0, type: type(symbolic_constant600000BF)}
// CHECK:STDOUT:     inst60000114:    {kind: ImportRefLoaded, arg0: import_ir_inst0000009F, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000115:    {kind: ImportRefLoaded, arg0: import_ir_inst000000A0, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000116:    {kind: ImportRefLoaded, arg0: import_ir_inst000000A1, arg1: entity_name<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst60000117:    {kind: FunctionType, arg0: function60000005, arg1: specific60000011, type: type(TypeType)}
// CHECK:STDOUT:     inst60000118:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant600000C1)}
// CHECK:STDOUT:     inst60000119:    {kind: BindSymbolicName, arg0: entity_name60000001, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000011A:    {kind: SymbolicBindingType, arg0: entity_name60000001, arg1: inst60000119, type: type(TypeType)}
// CHECK:STDOUT:     inst6000011B:    {kind: BindSymbolicName, arg0: entity_name6000001A, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000011C:    {kind: SymbolicBindingType, arg0: entity_name6000001A, arg1: inst6000011B, type: type(TypeType)}
// CHECK:STDOUT:     inst6000011D:    {kind: BindSymbolicName, arg0: entity_name60000029, arg1: inst<none>, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000011E:    {kind: SymbolicBindingType, arg0: entity_name60000029, arg1: inst6000011D, type: type(TypeType)}
// CHECK:STDOUT:     inst6000011F:    {kind: TupleType, arg0: inst_block6000005C, type: type(TypeType)}
// CHECK:STDOUT:     inst60000120:    {kind: PatternType, arg0: inst6000011F, type: type(TypeType)}
// CHECK:STDOUT:     inst60000121:    {kind: RequireCompleteType, arg0: inst600000F8, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000122:    {kind: RequireCompleteType, arg0: inst600000F7, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000123:    {kind: LookupImplWitness, arg0: inst600000F6, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000124:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst600000F6, type: type(TypeType)}
// CHECK:STDOUT:     inst60000125:    {kind: ImplWitnessAccess, arg0: inst60000123, arg1: element0, type: type(symbolic_constant600000E2)}
// CHECK:STDOUT:     inst60000126:    {kind: SpecificImplFunction, arg0: inst60000125, arg1: specific60000013, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000127:    {kind: RequireCompleteType, arg0: inst6000011F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000128:    {kind: RequireCompleteType, arg0: inst6000011A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000129:    {kind: LookupImplWitness, arg0: inst60000119, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012A:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst60000119, type: type(TypeType)}
// CHECK:STDOUT:     inst6000012B:    {kind: ImplWitnessAccess, arg0: inst60000129, arg1: element0, type: type(symbolic_constant600000EB)}
// CHECK:STDOUT:     inst6000012C:    {kind: SpecificImplFunction, arg0: inst6000012B, arg1: specific60000014, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst6000012D:    {kind: RequireCompleteType, arg0: inst6000011C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012E:    {kind: LookupImplWitness, arg0: inst6000011B, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000012F:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000011B, type: type(TypeType)}
// CHECK:STDOUT:     inst60000130:    {kind: ImplWitnessAccess, arg0: inst6000012E, arg1: element0, type: type(symbolic_constant600000F0)}
// CHECK:STDOUT:     inst60000131:    {kind: SpecificImplFunction, arg0: inst60000130, arg1: specific60000015, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000132:    {kind: RequireCompleteType, arg0: inst6000011E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000133:    {kind: LookupImplWitness, arg0: inst6000011D, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000134:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000011D, type: type(TypeType)}
// CHECK:STDOUT:     inst60000135:    {kind: ImplWitnessAccess, arg0: inst60000133, arg1: element0, type: type(symbolic_constant600000F5)}
// CHECK:STDOUT:     inst60000136:    {kind: SpecificImplFunction, arg0: inst60000135, arg1: specific60000016, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000137:    {kind: PatternType, arg0: inst600000F7, type: type(TypeType)}
// CHECK:STDOUT:     inst60000138:    {kind: LookupImplWitness, arg0: inst6000001B, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000139:    {kind: LookupImplWitness, arg0: inst6000001C, arg1: specific_interface60000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst6000013A:    {kind: FacetValue, arg0: inst6000001B, arg1: inst_block60000067, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000013B:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000013A, type: type(TypeType)}
// CHECK:STDOUT:     inst6000013C:    {kind: ImplWitnessAccess, arg0: inst60000138, arg1: element0, type: type(symbolic_constant60000100)}
// CHECK:STDOUT:     inst6000013D:    {kind: ImplWitnessAccess, arg0: inst60000138, arg1: element0, type: type(symbolic_constant600000FD)}
// CHECK:STDOUT:     inst6000013E:    {kind: FacetValue, arg0: inst6000001C, arg1: inst_block60000068, type: type(inst60000047)}
// CHECK:STDOUT:     inst6000013F:    {kind: FunctionTypeWithSelfType, arg0: inst60000050, arg1: inst6000013E, type: type(TypeType)}
// CHECK:STDOUT:     inst60000140:    {kind: ImplWitnessAccess, arg0: inst60000139, arg1: element0, type: type(symbolic_constant60000100)}
// CHECK:STDOUT:     inst60000141:    {kind: BoundMethod, arg0: inst6000003F, arg1: inst6000013C, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst60000142:    {kind: SpecificImplFunction, arg0: inst6000013C, arg1: specific60000017, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000143:    {kind: SpecificImplFunction, arg0: inst6000013D, arg1: specific60000017, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000144:    {kind: SpecificImplFunction, arg0: inst60000140, arg1: specific60000018, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst60000145:    {kind: BoundMethod, arg0: inst6000003F, arg1: inst60000142, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst60000146:    {kind: RequireCompleteType, arg0: inst6000001B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst60000147:    {kind: Call, arg0: inst60000145, arg1: inst_block6000006C, type: type(symbolic_constant00000004)}
// CHECK:STDOUT:     inst60000148:    {kind: InitializeFrom, arg0: inst60000147, arg1: inst60000043, type: type(symbolic_constant00000004)}
// CHECK:STDOUT:     inst60000149:    {kind: TupleAccess, arg0: inst60000034, arg1: element1, type: type(inst60000024)}
// CHECK:STDOUT:     inst6000014A:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst60000149, type: type(inst60000024)}
// CHECK:STDOUT:     inst6000014B:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst60000024)}
// CHECK:STDOUT:     inst6000014C:    {kind: Converted, arg0: inst60000040, arg1: inst6000014A, type: type(inst60000024)}
// CHECK:STDOUT:     inst6000014D:    {kind: TupleInit, arg0: inst_block6000006D, arg1: inst60000034, type: type(symbolic_constant00000008)}
// CHECK:STDOUT:     inst6000014E:    {kind: Converted, arg0: inst60000041, arg1: inst6000014D, type: type(symbolic_constant00000008)}
// CHECK:STDOUT:     inst6000014F:    {kind: ReturnExpr, arg0: inst6000014E, arg1: inst60000034}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       inst0000000E:    concrete_constant(inst0000000E)
// CHECK:STDOUT:       inst60000010:    concrete_constant(inst60000010)
// CHECK:STDOUT:       inst60000011:    concrete_constant(inst60000011)
// CHECK:STDOUT:       inst60000012:    symbolic_constant00000000
// CHECK:STDOUT:       inst60000013:    symbolic_constant00000000
// CHECK:STDOUT:       inst60000014:    symbolic_constant00000002
// CHECK:STDOUT:       inst60000015:    symbolic_constant00000001
// CHECK:STDOUT:       inst60000016:    symbolic_constant00000002
// CHECK:STDOUT:       inst60000017:    concrete_constant(inst60000017)
// CHECK:STDOUT:       inst60000018:    concrete_constant(inst60000018)
// CHECK:STDOUT:       inst60000019:    symbolic_constant00000002
// CHECK:STDOUT:       inst6000001A:    symbolic_constant00000004
// CHECK:STDOUT:       inst6000001B:    symbolic_constant00000003
// CHECK:STDOUT:       inst6000001C:    symbolic_constant00000004
// CHECK:STDOUT:       inst6000001E:    symbolic_constant00000005
// CHECK:STDOUT:       inst6000001F:    concrete_constant(inst6000001F)
// CHECK:STDOUT:       inst60000020:    symbolic_constant00000006
// CHECK:STDOUT:       inst60000021:    concrete_constant(inst60000021)
// CHECK:STDOUT:       inst60000022:    symbolic_constant00000002
// CHECK:STDOUT:       inst60000023:    symbolic_constant00000004
// CHECK:STDOUT:       inst60000024:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst60000026:    concrete_constant(inst60000026)
// CHECK:STDOUT:       inst60000028:    concrete_constant(inst60000028)
// CHECK:STDOUT:       inst60000029:    concrete_constant(inst60000024)
// CHECK:STDOUT:       inst6000002A:    symbolic_constant00000007
// CHECK:STDOUT:       inst6000002B:    symbolic_constant00000008
// CHECK:STDOUT:       inst6000002C:    symbolic_constant00000008
// CHECK:STDOUT:       inst6000002D:    symbolic_constant00000009
// CHECK:STDOUT:       inst6000002E:    concrete_constant(inst6000002E)
// CHECK:STDOUT:       inst6000002F:    symbolic_constant0000000A
// CHECK:STDOUT:       inst60000030:    concrete_constant(inst60000030)
// CHECK:STDOUT:       inst60000032:    symbolic_constant00000004
// CHECK:STDOUT:       inst60000035:    concrete_constant(inst60000037)
// CHECK:STDOUT:       inst60000036:    concrete_constant(inst60000036)
// CHECK:STDOUT:       inst60000037:    concrete_constant(inst60000037)
// CHECK:STDOUT:       inst60000038:    symbolic_constant0000000B
// CHECK:STDOUT:       inst60000039:    symbolic_constant0000000D
// CHECK:STDOUT:       inst6000003A:    symbolic_constant0000000C
// CHECK:STDOUT:       inst6000003B:    symbolic_constant0000000D
// CHECK:STDOUT:       inst6000003C:    symbolic_constant6000000F
// CHECK:STDOUT:       inst6000003D:    symbolic_constant0000000E
// CHECK:STDOUT:       inst6000003E:    symbolic_constant6000000F
// CHECK:STDOUT:       inst60000042:    symbolic_constant0000000D
// CHECK:STDOUT:       inst60000044:    symbolic_constant6000000F
// CHECK:STDOUT:       inst60000045:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000046:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000047:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000048:    symbolic_constant60000010
// CHECK:STDOUT:       inst60000049:    constant<none>
// CHECK:STDOUT:       inst6000004A:    concrete_constant(inst6000004E)
// CHECK:STDOUT:       inst6000004B:    constant<none>
// CHECK:STDOUT:       inst6000004C:    concrete_constant(inst6000004C)
// CHECK:STDOUT:       inst6000004D:    concrete_constant(inst60000051)
// CHECK:STDOUT:       inst6000004E:    concrete_constant(inst6000004E)
// CHECK:STDOUT:       inst6000004F:    concrete_constant(inst60000051)
// CHECK:STDOUT:       inst60000050:    concrete_constant(inst60000050)
// CHECK:STDOUT:       inst60000051:    concrete_constant(inst60000051)
// CHECK:STDOUT:       inst60000052:    symbolic_constant60000011
// CHECK:STDOUT:       inst60000053:    symbolic_constant60000012
// CHECK:STDOUT:       inst60000054:    concrete_constant(inst60000054)
// CHECK:STDOUT:       inst60000055:    concrete_constant(inst60000055)
// CHECK:STDOUT:       inst60000056:    concrete_constant(inst60000056)
// CHECK:STDOUT:       inst60000057:    concrete_constant(inst60000057)
// CHECK:STDOUT:       inst60000058:    symbolic_constant60000010
// CHECK:STDOUT:       inst60000059:    symbolic_constant60000016
// CHECK:STDOUT:       inst6000005A:    symbolic_constant60000017
// CHECK:STDOUT:       inst6000005B:    symbolic_constant60000018
// CHECK:STDOUT:       inst6000005C:    symbolic_constant600000FB
// CHECK:STDOUT:       inst6000005D:    constant<none>
// CHECK:STDOUT:       inst6000005E:    concrete_constant(inst6000005E)
// CHECK:STDOUT:       inst6000005F:    symbolic_constant60000019
// CHECK:STDOUT:       inst60000060:    symbolic_constant6000001A
// CHECK:STDOUT:       inst60000061:    symbolic_constant6000001B
// CHECK:STDOUT:       inst60000062:    concrete_constant(inst60000062)
// CHECK:STDOUT:       inst60000063:    concrete_constant(inst60000063)
// CHECK:STDOUT:       inst60000064:    symbolic_constant6000001D
// CHECK:STDOUT:       inst60000065:    symbolic_constant6000001C
// CHECK:STDOUT:       inst60000066:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000067:    constant<none>
// CHECK:STDOUT:       inst60000068:    concrete_constant(inst60000068)
// CHECK:STDOUT:       inst60000069:    symbolic_constant60000020
// CHECK:STDOUT:       inst6000006A:    symbolic_constant60000022
// CHECK:STDOUT:       inst6000006B:    symbolic_constant60000023
// CHECK:STDOUT:       inst6000006C:    symbolic_constant60000024
// CHECK:STDOUT:       inst6000006D:    symbolic_constant60000025
// CHECK:STDOUT:       inst6000006E:    symbolic_constant60000028
// CHECK:STDOUT:       inst6000006F:    symbolic_constant60000026
// CHECK:STDOUT:       inst60000070:    symbolic_constant60000027
// CHECK:STDOUT:       inst60000071:    symbolic_constant6000002A
// CHECK:STDOUT:       inst60000072:    concrete_constant(inst60000072)
// CHECK:STDOUT:       inst60000073:    concrete_constant(inst60000073)
// CHECK:STDOUT:       inst60000074:    concrete_constant(inst60000074)
// CHECK:STDOUT:       inst60000075:    concrete_constant(inst60000075)
// CHECK:STDOUT:       inst60000076:    symbolic_constant6000001D
// CHECK:STDOUT:       inst60000077:    symbolic_constant6000002D
// CHECK:STDOUT:       inst60000078:    symbolic_constant6000002E
// CHECK:STDOUT:       inst60000079:    symbolic_constant60000032
// CHECK:STDOUT:       inst6000007A:    symbolic_constant60000033
// CHECK:STDOUT:       inst6000007B:    symbolic_constant60000034
// CHECK:STDOUT:       inst6000007C:    symbolic_constant60000035
// CHECK:STDOUT:       inst6000007D:    symbolic_constant60000036
// CHECK:STDOUT:       inst6000007E:    symbolic_constant60000038
// CHECK:STDOUT:       inst6000007F:    symbolic_constant6000003A
// CHECK:STDOUT:       inst60000080:    symbolic_constant6000003C
// CHECK:STDOUT:       inst60000081:    symbolic_constant6000003E
// CHECK:STDOUT:       inst60000082:    symbolic_constant60000040
// CHECK:STDOUT:       inst60000083:    symbolic_constant60000042
// CHECK:STDOUT:       inst60000084:    symbolic_constant60000043
// CHECK:STDOUT:       inst60000085:    symbolic_constant60000044
// CHECK:STDOUT:       inst60000086:    symbolic_constant60000045
// CHECK:STDOUT:       inst60000087:    symbolic_constant60000046
// CHECK:STDOUT:       inst60000088:    symbolic_constant60000047
// CHECK:STDOUT:       inst60000089:    symbolic_constant60000048
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
// CHECK:STDOUT:       inst6000009A:    symbolic_constant600000F9
// CHECK:STDOUT:       inst6000009B:    concrete_constant(inst6000009B)
// CHECK:STDOUT:       inst6000009C:    concrete_constant(inst6000009C)
// CHECK:STDOUT:       inst6000009D:    symbolic_constant6000004A
// CHECK:STDOUT:       inst6000009E:    symbolic_constant60000049
// CHECK:STDOUT:       inst6000009F:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst600000A0:    constant<none>
// CHECK:STDOUT:       inst600000A1:    concrete_constant(inst600000A1)
// CHECK:STDOUT:       inst600000A2:    symbolic_constant6000004C
// CHECK:STDOUT:       inst600000A3:    symbolic_constant6000004E
// CHECK:STDOUT:       inst600000A4:    symbolic_constant6000004F
// CHECK:STDOUT:       inst600000A5:    symbolic_constant60000050
// CHECK:STDOUT:       inst600000A6:    symbolic_constant60000053
// CHECK:STDOUT:       inst600000A7:    symbolic_constant60000051
// CHECK:STDOUT:       inst600000A8:    symbolic_constant60000052
// CHECK:STDOUT:       inst600000A9:    concrete_constant(inst600000A9)
// CHECK:STDOUT:       inst600000AA:    concrete_constant(inst600000AA)
// CHECK:STDOUT:       inst600000AB:    concrete_constant(inst600000AB)
// CHECK:STDOUT:       inst600000AC:    concrete_constant(inst600000AC)
// CHECK:STDOUT:       inst600000AD:    symbolic_constant6000004A
// CHECK:STDOUT:       inst600000AE:    symbolic_constant60000058
// CHECK:STDOUT:       inst600000AF:    symbolic_constant60000059
// CHECK:STDOUT:       inst600000B0:    symbolic_constant6000005A
// CHECK:STDOUT:       inst600000B1:    symbolic_constant6000005D
// CHECK:STDOUT:       inst600000B2:    symbolic_constant6000005E
// CHECK:STDOUT:       inst600000B3:    symbolic_constant6000005F
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
// CHECK:STDOUT:       inst600000BE:    symbolic_constant60000060
// CHECK:STDOUT:       inst600000BF:    symbolic_constant60000061
// CHECK:STDOUT:       inst600000C0:    symbolic_constant60000062
// CHECK:STDOUT:       inst600000C1:    concrete_constant(inst600000C1)
// CHECK:STDOUT:       inst600000C2:    concrete_constant(inst600000C2)
// CHECK:STDOUT:       inst600000C3:    symbolic_constant60000064
// CHECK:STDOUT:       inst600000C4:    symbolic_constant60000065
// CHECK:STDOUT:       inst600000C5:    symbolic_constant60000063
// CHECK:STDOUT:       inst600000C6:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst600000C7:    constant<none>
// CHECK:STDOUT:       inst600000C8:    concrete_constant(inst600000C8)
// CHECK:STDOUT:       inst600000C9:    symbolic_constant6000006A
// CHECK:STDOUT:       inst600000CA:    symbolic_constant6000006C
// CHECK:STDOUT:       inst600000CB:    symbolic_constant6000006D
// CHECK:STDOUT:       inst600000CC:    symbolic_constant6000006E
// CHECK:STDOUT:       inst600000CD:    symbolic_constant6000006F
// CHECK:STDOUT:       inst600000CE:    symbolic_constant60000070
// CHECK:STDOUT:       inst600000CF:    symbolic_constant60000071
// CHECK:STDOUT:       inst600000D0:    symbolic_constant60000074
// CHECK:STDOUT:       inst600000D1:    symbolic_constant60000072
// CHECK:STDOUT:       inst600000D2:    symbolic_constant60000073
// CHECK:STDOUT:       inst600000D3:    symbolic_constant60000076
// CHECK:STDOUT:       inst600000D4:    concrete_constant(inst600000D4)
// CHECK:STDOUT:       inst600000D5:    concrete_constant(inst600000D5)
// CHECK:STDOUT:       inst600000D6:    concrete_constant(inst600000D6)
// CHECK:STDOUT:       inst600000D7:    concrete_constant(inst600000D7)
// CHECK:STDOUT:       inst600000D8:    symbolic_constant60000064
// CHECK:STDOUT:       inst600000D9:    symbolic_constant60000065
// CHECK:STDOUT:       inst600000DA:    symbolic_constant60000079
// CHECK:STDOUT:       inst600000DB:    symbolic_constant6000007A
// CHECK:STDOUT:       inst600000DC:    symbolic_constant60000080
// CHECK:STDOUT:       inst600000DD:    symbolic_constant60000081
// CHECK:STDOUT:       inst600000DE:    symbolic_constant60000082
// CHECK:STDOUT:       inst600000DF:    symbolic_constant60000083
// CHECK:STDOUT:       inst600000E0:    symbolic_constant60000084
// CHECK:STDOUT:       inst600000E1:    symbolic_constant60000085
// CHECK:STDOUT:       inst600000E2:    symbolic_constant60000086
// CHECK:STDOUT:       inst600000E3:    symbolic_constant6000008D
// CHECK:STDOUT:       inst600000E4:    symbolic_constant6000008F
// CHECK:STDOUT:       inst600000E5:    symbolic_constant60000091
// CHECK:STDOUT:       inst600000E6:    symbolic_constant60000093
// CHECK:STDOUT:       inst600000E7:    symbolic_constant60000095
// CHECK:STDOUT:       inst600000E8:    symbolic_constant60000097
// CHECK:STDOUT:       inst600000E9:    symbolic_constant60000098
// CHECK:STDOUT:       inst600000EA:    symbolic_constant60000099
// CHECK:STDOUT:       inst600000EB:    symbolic_constant6000009A
// CHECK:STDOUT:       inst600000EC:    symbolic_constant6000009B
// CHECK:STDOUT:       inst600000ED:    symbolic_constant6000009C
// CHECK:STDOUT:       inst600000EE:    symbolic_constant6000009D
// CHECK:STDOUT:       inst600000EF:    symbolic_constant6000009E
// CHECK:STDOUT:       inst600000F0:    symbolic_constant6000009F
// CHECK:STDOUT:       inst600000F1:    symbolic_constant600000A0
// CHECK:STDOUT:       inst600000F2:    symbolic_constant600000A1
// CHECK:STDOUT:       inst600000F3:    symbolic_constant600000A2
// CHECK:STDOUT:       inst600000F4:    constant<none>
// CHECK:STDOUT:       inst600000F5:    concrete_constant(inst600000F5)
// CHECK:STDOUT:       inst600000F6:    symbolic_constant600000A3
// CHECK:STDOUT:       inst600000F7:    symbolic_constant600000A4
// CHECK:STDOUT:       inst600000F8:    symbolic_constant600000A5
// CHECK:STDOUT:       inst600000F9:    concrete_constant(inst600000F9)
// CHECK:STDOUT:       inst600000FA:    concrete_constant(inst600000FA)
// CHECK:STDOUT:       inst600000FB:    concrete_constant(inst600000FB)
// CHECK:STDOUT:       inst600000FC:    symbolic_constant600000A7
// CHECK:STDOUT:       inst600000FD:    symbolic_constant600000A8
// CHECK:STDOUT:       inst600000FE:    symbolic_constant600000A9
// CHECK:STDOUT:       inst600000FF:    symbolic_constant600000A6
// CHECK:STDOUT:       inst60000100:    concrete_constant(inst60000047)
// CHECK:STDOUT:       inst60000101:    constant<none>
// CHECK:STDOUT:       inst60000102:    concrete_constant(inst60000102)
// CHECK:STDOUT:       inst60000103:    symbolic_constant600000B0
// CHECK:STDOUT:       inst60000104:    symbolic_constant600000B2
// CHECK:STDOUT:       inst60000105:    symbolic_constant600000B3
// CHECK:STDOUT:       inst60000106:    symbolic_constant600000B4
// CHECK:STDOUT:       inst60000107:    symbolic_constant600000B5
// CHECK:STDOUT:       inst60000108:    symbolic_constant600000B6
// CHECK:STDOUT:       inst60000109:    symbolic_constant600000B7
// CHECK:STDOUT:       inst6000010A:    symbolic_constant600000B8
// CHECK:STDOUT:       inst6000010B:    symbolic_constant600000B9
// CHECK:STDOUT:       inst6000010C:    symbolic_constant600000BC
// CHECK:STDOUT:       inst6000010D:    symbolic_constant600000BA
// CHECK:STDOUT:       inst6000010E:    symbolic_constant600000BB
// CHECK:STDOUT:       inst6000010F:    symbolic_constant600000BE
// CHECK:STDOUT:       inst60000110:    concrete_constant(inst60000110)
// CHECK:STDOUT:       inst60000111:    concrete_constant(inst60000111)
// CHECK:STDOUT:       inst60000112:    concrete_constant(inst60000112)
// CHECK:STDOUT:       inst60000113:    concrete_constant(inst60000113)
// CHECK:STDOUT:       inst60000114:    symbolic_constant600000A7
// CHECK:STDOUT:       inst60000115:    symbolic_constant600000A8
// CHECK:STDOUT:       inst60000116:    symbolic_constant600000A9
// CHECK:STDOUT:       inst60000117:    symbolic_constant600000C1
// CHECK:STDOUT:       inst60000118:    symbolic_constant600000C2
// CHECK:STDOUT:       inst60000119:    symbolic_constant600000CA
// CHECK:STDOUT:       inst6000011A:    symbolic_constant600000CB
// CHECK:STDOUT:       inst6000011B:    symbolic_constant600000CC
// CHECK:STDOUT:       inst6000011C:    symbolic_constant600000CD
// CHECK:STDOUT:       inst6000011D:    symbolic_constant600000CE
// CHECK:STDOUT:       inst6000011E:    symbolic_constant600000CF
// CHECK:STDOUT:       inst6000011F:    symbolic_constant600000D0
// CHECK:STDOUT:       inst60000120:    symbolic_constant600000D1
// CHECK:STDOUT:       inst60000121:    symbolic_constant600000D2
// CHECK:STDOUT:       inst60000122:    symbolic_constant600000DE
// CHECK:STDOUT:       inst60000123:    symbolic_constant600000E0
// CHECK:STDOUT:       inst60000124:    symbolic_constant600000E2
// CHECK:STDOUT:       inst60000125:    symbolic_constant600000E4
// CHECK:STDOUT:       inst60000126:    symbolic_constant600000E6
// CHECK:STDOUT:       inst60000127:    symbolic_constant600000E8
// CHECK:STDOUT:       inst60000128:    symbolic_constant600000E9
// CHECK:STDOUT:       inst60000129:    symbolic_constant600000EA
// CHECK:STDOUT:       inst6000012A:    symbolic_constant600000EB
// CHECK:STDOUT:       inst6000012B:    symbolic_constant600000EC
// CHECK:STDOUT:       inst6000012C:    symbolic_constant600000ED
// CHECK:STDOUT:       inst6000012D:    symbolic_constant600000EE
// CHECK:STDOUT:       inst6000012E:    symbolic_constant600000EF
// CHECK:STDOUT:       inst6000012F:    symbolic_constant600000F0
// CHECK:STDOUT:       inst60000130:    symbolic_constant600000F1
// CHECK:STDOUT:       inst60000131:    symbolic_constant600000F2
// CHECK:STDOUT:       inst60000132:    symbolic_constant600000F3
// CHECK:STDOUT:       inst60000133:    symbolic_constant600000F4
// CHECK:STDOUT:       inst60000134:    symbolic_constant600000F5
// CHECK:STDOUT:       inst60000135:    symbolic_constant600000F6
// CHECK:STDOUT:       inst60000136:    symbolic_constant600000F7
// CHECK:STDOUT:       inst60000137:    symbolic_constant600000F8
// CHECK:STDOUT:       inst60000138:    symbolic_constant600000FA
// CHECK:STDOUT:       inst60000139:    symbolic_constant600000FB
// CHECK:STDOUT:       inst6000013A:    symbolic_constant600000FC
// CHECK:STDOUT:       inst6000013B:    symbolic_constant600000FD
// CHECK:STDOUT:       inst6000013C:    symbolic_constant60000101
// CHECK:STDOUT:       inst6000013D:    symbolic_constant600000FE
// CHECK:STDOUT:       inst6000013E:    symbolic_constant600000FF
// CHECK:STDOUT:       inst6000013F:    symbolic_constant60000100
// CHECK:STDOUT:       inst60000140:    symbolic_constant60000101
// CHECK:STDOUT:       inst60000142:    symbolic_constant60000103
// CHECK:STDOUT:       inst60000143:    symbolic_constant60000102
// CHECK:STDOUT:       inst60000144:    symbolic_constant60000103
// CHECK:STDOUT:       inst60000146:    symbolic_constant6000000F
// CHECK:STDOUT:       inst6000014A:    concrete_constant(inst6000014B)
// CHECK:STDOUT:       inst6000014B:    concrete_constant(inst6000014B)
// CHECK:STDOUT:       inst6000014C:    concrete_constant(inst6000014B)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant00000000: {inst: inst60000013, generic: generic<none>, index: generic_inst<none>, kind: self}
// CHECK:STDOUT:       symbolic_constant00000001: {inst: inst60000015, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant00000002: {inst: inst60000015, generic: generic60000000, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant00000003: {inst: inst6000001B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant00000004: {inst: inst6000001B, generic: generic60000000, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant00000005: {inst: inst6000001E, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant00000006: {inst: inst6000001E, generic: generic60000000, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant00000007: {inst: inst6000002A, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant00000008: {inst: inst6000002A, generic: generic60000000, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant00000009: {inst: inst6000002D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant0000000A: {inst: inst6000002D, generic: generic60000000, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant0000000B: {inst: inst60000038, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant0000000C: {inst: inst6000003A, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant0000000D: {inst: inst6000003A, generic: generic60000000, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant0000000E: {inst: inst6000003D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000000F: {inst: inst6000003D, generic: generic60000000, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000010: {inst: inst60000048, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000011: {inst: inst60000052, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000012: {inst: inst60000053, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000013: {inst: inst60000053, generic: generic60000001, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000014: {inst: inst60000048, generic: generic60000001, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000015: {inst: inst60000052, generic: generic60000001, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000016: {inst: inst60000048, generic: generic60000001, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000017: {inst: inst60000052, generic: generic60000001, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000018: {inst: inst60000053, generic: generic60000001, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000019: {inst: inst6000005F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001A: {inst: inst60000060, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001B: {inst: inst60000061, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001C: {inst: inst60000061, generic: generic60000002, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001D: {inst: inst6000005F, generic: generic60000002, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001E: {inst: inst6000005F, generic: generic60000002, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000001F: {inst: inst60000060, generic: generic60000002, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000020: {inst: inst60000069, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000021: {inst: inst60000069, generic: generic60000002, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000022: {inst: inst6000005F, generic: generic60000002, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000023: {inst: inst60000060, generic: generic60000002, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000024: {inst: inst60000061, generic: generic60000002, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000025: {inst: inst60000069, generic: generic60000002, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000026: {inst: inst6000006F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000027: {inst: inst60000070, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000028: {inst: inst60000070, generic: generic60000002, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000029: {inst: inst6000006F, generic: generic60000002, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002A: {inst: inst60000071, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002B: {inst: inst60000071, generic: generic60000003, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002C: {inst: inst60000070, generic: generic60000002, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002D: {inst: inst6000006F, generic: generic60000002, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002E: {inst: inst60000070, generic: generic60000002, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000002F: {inst: inst6000005F, generic: generic60000003, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000030: {inst: inst60000060, generic: generic60000003, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000031: {inst: inst60000061, generic: generic60000003, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000032: {inst: inst6000005F, generic: generic60000003, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000033: {inst: inst60000060, generic: generic60000003, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000034: {inst: inst60000061, generic: generic60000003, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000035: {inst: inst60000071, generic: generic60000003, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000036: {inst: inst6000007D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000037: {inst: inst6000007D, generic: generic60000003, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000038: {inst: inst6000007E, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000039: {inst: inst6000007E, generic: generic60000003, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003A: {inst: inst6000007F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003B: {inst: inst6000007F, generic: generic60000003, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003C: {inst: inst60000080, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003D: {inst: inst60000080, generic: generic60000003, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003E: {inst: inst60000081, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000003F: {inst: inst60000081, generic: generic60000003, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000040: {inst: inst60000082, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000041: {inst: inst60000082, generic: generic60000003, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000042: {inst: inst6000007D, generic: generic60000003, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000043: {inst: inst6000007E, generic: generic60000003, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000044: {inst: inst6000007F, generic: generic60000003, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000045: {inst: inst60000080, generic: generic60000003, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000046: {inst: inst60000081, generic: generic60000003, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000047: {inst: inst60000082, generic: generic60000003, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000048: {inst: inst60000089, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000049: {inst: inst6000001B, generic: generic60000004, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004A: {inst: inst60000015, generic: generic60000004, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004B: {inst: inst60000015, generic: generic60000004, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004C: {inst: inst600000A2, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004D: {inst: inst600000A2, generic: generic60000004, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004E: {inst: inst60000015, generic: generic60000004, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000004F: {inst: inst6000001B, generic: generic60000004, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000050: {inst: inst600000A2, generic: generic60000004, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000051: {inst: inst600000A7, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000052: {inst: inst600000A8, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000053: {inst: inst600000A8, generic: generic60000004, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000054: {inst: inst600000A7, generic: generic60000004, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000055: {inst: inst6000001E, generic: generic60000005, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000056: {inst: inst600000A8, generic: generic60000004, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000057: {inst: inst6000003D, generic: generic60000004, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000058: {inst: inst600000A7, generic: generic60000004, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000059: {inst: inst600000A8, generic: generic60000004, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005A: {inst: inst6000003D, generic: generic60000004, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005B: {inst: inst60000015, generic: generic60000005, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005C: {inst: inst6000001B, generic: generic60000005, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005D: {inst: inst60000015, generic: generic60000005, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005E: {inst: inst6000001B, generic: generic60000005, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000005F: {inst: inst6000001E, generic: generic60000005, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000060: {inst: inst600000BE, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000061: {inst: inst600000BF, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000062: {inst: inst600000C0, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000063: {inst: inst600000C0, generic: generic60000006, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000064: {inst: inst6000005F, generic: generic60000006, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000065: {inst: inst600000BE, generic: generic60000006, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000066: {inst: inst6000005F, generic: generic60000006, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000067: {inst: inst600000BE, generic: generic60000006, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000068: {inst: inst60000060, generic: generic60000006, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000069: {inst: inst600000BF, generic: generic60000006, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006A: {inst: inst600000C9, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006B: {inst: inst600000C9, generic: generic60000006, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006C: {inst: inst6000005F, generic: generic60000006, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006D: {inst: inst600000BE, generic: generic60000006, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006E: {inst: inst60000060, generic: generic60000006, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000006F: {inst: inst600000BF, generic: generic60000006, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000070: {inst: inst600000C0, generic: generic60000006, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000071: {inst: inst600000C9, generic: generic60000006, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000072: {inst: inst600000D1, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000073: {inst: inst600000D2, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000074: {inst: inst600000D2, generic: generic60000006, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000075: {inst: inst600000D1, generic: generic60000006, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000076: {inst: inst600000D3, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000077: {inst: inst600000D3, generic: generic60000007, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000078: {inst: inst600000D2, generic: generic60000006, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000079: {inst: inst600000D1, generic: generic60000006, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007A: {inst: inst600000D2, generic: generic60000006, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007B: {inst: inst6000005F, generic: generic60000007, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007C: {inst: inst60000060, generic: generic60000007, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007D: {inst: inst600000BE, generic: generic60000007, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007E: {inst: inst600000BF, generic: generic60000007, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000007F: {inst: inst600000C0, generic: generic60000007, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000080: {inst: inst6000005F, generic: generic60000007, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000081: {inst: inst60000060, generic: generic60000007, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000082: {inst: inst600000BE, generic: generic60000007, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000083: {inst: inst600000BF, generic: generic60000007, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000084: {inst: inst600000C0, generic: generic60000007, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000085: {inst: inst600000D3, generic: generic60000007, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000086: {inst: inst600000E2, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000087: {inst: inst600000E2, generic: generic60000007, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000088: {inst: inst6000007E, generic: generic60000007, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000089: {inst: inst6000007F, generic: generic60000007, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008A: {inst: inst60000080, generic: generic60000007, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008B: {inst: inst60000081, generic: generic60000007, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008C: {inst: inst60000082, generic: generic60000007, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008D: {inst: inst600000E3, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008E: {inst: inst600000E3, generic: generic60000007, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000008F: {inst: inst600000E4, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000090: {inst: inst600000E4, generic: generic60000007, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000091: {inst: inst600000E5, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000092: {inst: inst600000E5, generic: generic60000007, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000093: {inst: inst600000E6, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000094: {inst: inst600000E6, generic: generic60000007, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000095: {inst: inst600000E7, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000096: {inst: inst600000E7, generic: generic60000007, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000097: {inst: inst600000E2, generic: generic60000007, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000098: {inst: inst6000007E, generic: generic60000007, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000099: {inst: inst6000007F, generic: generic60000007, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009A: {inst: inst60000080, generic: generic60000007, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009B: {inst: inst60000081, generic: generic60000007, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009C: {inst: inst60000082, generic: generic60000007, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009D: {inst: inst600000E3, generic: generic60000007, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009E: {inst: inst600000E4, generic: generic60000007, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant6000009F: {inst: inst600000E5, generic: generic60000007, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A0: {inst: inst600000E6, generic: generic60000007, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A1: {inst: inst600000E7, generic: generic60000007, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A2: {inst: inst600000F3, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A3: {inst: inst600000F6, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A4: {inst: inst600000F7, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A5: {inst: inst600000F8, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A6: {inst: inst600000F8, generic: generic60000008, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A7: {inst: inst6000005F, generic: generic60000008, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A8: {inst: inst600000BE, generic: generic60000008, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000A9: {inst: inst600000F6, generic: generic60000008, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AA: {inst: inst6000005F, generic: generic60000008, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AB: {inst: inst600000BE, generic: generic60000008, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AC: {inst: inst600000F6, generic: generic60000008, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AD: {inst: inst60000060, generic: generic60000008, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AE: {inst: inst600000BF, generic: generic60000008, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000AF: {inst: inst600000F7, generic: generic60000008, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B0: {inst: inst60000103, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B1: {inst: inst60000103, generic: generic60000008, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B2: {inst: inst6000005F, generic: generic60000008, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B3: {inst: inst600000BE, generic: generic60000008, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B4: {inst: inst600000F6, generic: generic60000008, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B5: {inst: inst60000060, generic: generic60000008, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B6: {inst: inst600000BF, generic: generic60000008, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B7: {inst: inst600000F7, generic: generic60000008, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B8: {inst: inst600000F8, generic: generic60000008, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000B9: {inst: inst60000103, generic: generic60000008, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BA: {inst: inst6000010D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BB: {inst: inst6000010E, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BC: {inst: inst6000010E, generic: generic60000008, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BD: {inst: inst6000010D, generic: generic60000008, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BE: {inst: inst6000010F, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000BF: {inst: inst6000010F, generic: generic60000009, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C0: {inst: inst6000010E, generic: generic60000008, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C1: {inst: inst6000010D, generic: generic60000008, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C2: {inst: inst6000010E, generic: generic60000008, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C3: {inst: inst6000005F, generic: generic60000009, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C4: {inst: inst60000060, generic: generic60000009, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C5: {inst: inst600000BE, generic: generic60000009, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C6: {inst: inst600000BF, generic: generic60000009, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C7: {inst: inst600000F6, generic: generic60000009, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C8: {inst: inst600000F7, generic: generic60000009, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000C9: {inst: inst600000F8, generic: generic60000009, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CA: {inst: inst6000005F, generic: generic60000009, index: generic_inst_in_decl0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CB: {inst: inst60000060, generic: generic60000009, index: generic_inst_in_decl1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CC: {inst: inst600000BE, generic: generic60000009, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CD: {inst: inst600000BF, generic: generic60000009, index: generic_inst_in_decl3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CE: {inst: inst600000F6, generic: generic60000009, index: generic_inst_in_decl4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000CF: {inst: inst600000F7, generic: generic60000009, index: generic_inst_in_decl5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D0: {inst: inst600000F8, generic: generic60000009, index: generic_inst_in_decl6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D1: {inst: inst6000010F, generic: generic60000009, index: generic_inst_in_decl7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D2: {inst: inst60000121, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D3: {inst: inst60000121, generic: generic60000009, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D4: {inst: inst6000007E, generic: generic60000009, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D5: {inst: inst6000007F, generic: generic60000009, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D6: {inst: inst60000080, generic: generic60000009, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D7: {inst: inst60000081, generic: generic60000009, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D8: {inst: inst60000082, generic: generic60000009, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000D9: {inst: inst600000E3, generic: generic60000009, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DA: {inst: inst600000E4, generic: generic60000009, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DB: {inst: inst600000E5, generic: generic60000009, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DC: {inst: inst600000E6, generic: generic60000009, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DD: {inst: inst600000E7, generic: generic60000009, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DE: {inst: inst60000122, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000DF: {inst: inst60000122, generic: generic60000009, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E0: {inst: inst60000123, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E1: {inst: inst60000123, generic: generic60000009, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E2: {inst: inst60000124, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E3: {inst: inst60000124, generic: generic60000009, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E4: {inst: inst60000125, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E5: {inst: inst60000125, generic: generic60000009, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E6: {inst: inst60000126, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E7: {inst: inst60000126, generic: generic60000009, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E8: {inst: inst60000121, generic: generic60000009, index: generic_inst_in_def0, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000E9: {inst: inst6000007E, generic: generic60000009, index: generic_inst_in_def1, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EA: {inst: inst6000007F, generic: generic60000009, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EB: {inst: inst60000080, generic: generic60000009, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EC: {inst: inst60000081, generic: generic60000009, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000ED: {inst: inst60000082, generic: generic60000009, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EE: {inst: inst600000E3, generic: generic60000009, index: generic_inst_in_def6, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000EF: {inst: inst600000E4, generic: generic60000009, index: generic_inst_in_def7, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F0: {inst: inst600000E5, generic: generic60000009, index: generic_inst_in_def8, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F1: {inst: inst600000E6, generic: generic60000009, index: generic_inst_in_def9, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F2: {inst: inst600000E7, generic: generic60000009, index: generic_inst_in_def10, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F3: {inst: inst60000122, generic: generic60000009, index: generic_inst_in_def11, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F4: {inst: inst60000123, generic: generic60000009, index: generic_inst_in_def12, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F5: {inst: inst60000124, generic: generic60000009, index: generic_inst_in_def13, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F6: {inst: inst60000125, generic: generic60000009, index: generic_inst_in_def14, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F7: {inst: inst60000126, generic: generic60000009, index: generic_inst_in_def15, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F8: {inst: inst60000137, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000F9: {inst: inst600000A2, generic: generic60000004, index: generic_inst_in_decl2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FA: {inst: inst60000138, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FB: {inst: inst60000138, generic: generic60000000, index: generic_inst_in_def2, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FC: {inst: inst6000013A, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FD: {inst: inst6000013B, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FE: {inst: inst6000013D, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant600000FF: {inst: inst6000013A, generic: generic60000000, index: generic_inst_in_def3, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000100: {inst: inst6000013B, generic: generic60000000, index: generic_inst_in_def4, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000101: {inst: inst6000013D, generic: generic60000000, index: generic_inst_in_def5, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000102: {inst: inst60000143, generic: generic<none>, index: generic_inst<none>, kind: checked}
// CHECK:STDOUT:       symbolic_constant60000103: {inst: inst60000143, generic: generic60000000, index: generic_inst_in_def6, kind: checked}
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
// CHECK:STDOUT:       0:               inst60000023
// CHECK:STDOUT:       1:               inst60000029
// CHECK:STDOUT:     inst_block6000000B:
// CHECK:STDOUT:       0:               inst6000001B
// CHECK:STDOUT:       1:               inst60000024
// CHECK:STDOUT:     inst_block6000000C:
// CHECK:STDOUT:       0:               inst6000001C
// CHECK:STDOUT:       1:               inst60000024
// CHECK:STDOUT:     inst_block6000000D:
// CHECK:STDOUT:       0:               inst60000031
// CHECK:STDOUT:       1:               inst60000033
// CHECK:STDOUT:     inst_block6000000E:
// CHECK:STDOUT:       0:               inst60000018
// CHECK:STDOUT:       1:               inst6000001F
// CHECK:STDOUT:       2:               inst60000021
// CHECK:STDOUT:       3:               inst6000002E
// CHECK:STDOUT:       4:               inst60000030
// CHECK:STDOUT:     inst_block6000000F:
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
// CHECK:STDOUT:     inst_block60000010:
// CHECK:STDOUT:       0:               inst60000014
// CHECK:STDOUT:     inst_block60000011:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:     inst_block60000012:
// CHECK:STDOUT:       0:               inst60000016
// CHECK:STDOUT:       1:               inst6000001C
// CHECK:STDOUT:       2:               inst60000020
// CHECK:STDOUT:       3:               inst6000002C
// CHECK:STDOUT:       4:               inst6000002F
// CHECK:STDOUT:     inst_block60000013:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:       3:               inst6000002A
// CHECK:STDOUT:       4:               inst6000002D
// CHECK:STDOUT:     inst_block60000014:
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
// CHECK:STDOUT:     inst_block60000015:
// CHECK:STDOUT:       0:               inst6000003F
// CHECK:STDOUT:       1:               inst60000040
// CHECK:STDOUT:     inst_block60000016:
// CHECK:STDOUT:       0:               inst6000004B
// CHECK:STDOUT:     inst_block60000017:
// CHECK:STDOUT:       0:               inst60000057
// CHECK:STDOUT:     inst_block60000018:
// CHECK:STDOUT:       0:               inst60000058
// CHECK:STDOUT:     inst_block60000019:
// CHECK:STDOUT:       0:               inst60000059
// CHECK:STDOUT:       1:               inst6000005A
// CHECK:STDOUT:       2:               inst6000005B
// CHECK:STDOUT:     inst_block6000001A:
// CHECK:STDOUT:       0:               inst60000048
// CHECK:STDOUT:     inst_block6000001B:
// CHECK:STDOUT:       0:               inst60000048
// CHECK:STDOUT:       1:               inst60000052
// CHECK:STDOUT:       2:               inst60000053
// CHECK:STDOUT:     inst_block6000001C:
// CHECK:STDOUT:       0:               inst60000063
// CHECK:STDOUT:     inst_block6000001D:
// CHECK:STDOUT:       0:               inst60000064
// CHECK:STDOUT:     inst_block6000001E:
// CHECK:STDOUT:       0:               inst60000067
// CHECK:STDOUT:     inst_block6000001F:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:     inst_block60000020:
// CHECK:STDOUT:       0:               inst6000006A
// CHECK:STDOUT:     inst_block60000021:
// CHECK:STDOUT:       0:               inst6000006A
// CHECK:STDOUT:       1:               inst6000006B
// CHECK:STDOUT:       2:               inst6000006C
// CHECK:STDOUT:       3:               inst6000006D
// CHECK:STDOUT:     inst_block60000022:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst60000061
// CHECK:STDOUT:       3:               inst60000069
// CHECK:STDOUT:     inst_block60000023:
// CHECK:STDOUT:       0:               inst60000075
// CHECK:STDOUT:     inst_block60000024:
// CHECK:STDOUT:       0:               inst60000076
// CHECK:STDOUT:     inst_block60000025:
// CHECK:STDOUT:       0:               inst6000006A
// CHECK:STDOUT:     inst_block60000026:
// CHECK:STDOUT:       0:               inst60000077
// CHECK:STDOUT:       1:               inst60000078
// CHECK:STDOUT:     inst_block60000027:
// CHECK:STDOUT:       0:               inst60000079
// CHECK:STDOUT:       1:               inst6000007A
// CHECK:STDOUT:       2:               inst6000007B
// CHECK:STDOUT:       3:               inst6000007C
// CHECK:STDOUT:     inst_block60000028:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst60000061
// CHECK:STDOUT:       3:               inst60000071
// CHECK:STDOUT:     inst_block60000029:
// CHECK:STDOUT:       0:               inst60000079
// CHECK:STDOUT:     inst_block6000002A:
// CHECK:STDOUT:       0:               inst60000083
// CHECK:STDOUT:       1:               inst60000084
// CHECK:STDOUT:       2:               inst60000085
// CHECK:STDOUT:       3:               inst60000086
// CHECK:STDOUT:       4:               inst60000087
// CHECK:STDOUT:       5:               inst60000088
// CHECK:STDOUT:     inst_block6000002B:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst60000089
// CHECK:STDOUT:     inst_block6000002C:
// CHECK:STDOUT:       0:               inst6000006F
// CHECK:STDOUT:       1:               inst60000070
// CHECK:STDOUT:     inst_block6000002D:
// CHECK:STDOUT:       0:               inst6000009C
// CHECK:STDOUT:     inst_block6000002E:
// CHECK:STDOUT:       0:               inst6000009D
// CHECK:STDOUT:     inst_block6000002F:
// CHECK:STDOUT:       0:               inst600000A0
// CHECK:STDOUT:     inst_block60000030:
// CHECK:STDOUT:       0:               inst600000A3
// CHECK:STDOUT:     inst_block60000031:
// CHECK:STDOUT:       0:               inst600000A3
// CHECK:STDOUT:       1:               inst600000A4
// CHECK:STDOUT:       2:               inst600000A5
// CHECK:STDOUT:     inst_block60000032:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst600000A2
// CHECK:STDOUT:     inst_block60000033:
// CHECK:STDOUT:       0:               inst600000AC
// CHECK:STDOUT:     inst_block60000034:
// CHECK:STDOUT:       0:               inst600000AD
// CHECK:STDOUT:     inst_block60000035:
// CHECK:STDOUT:       0:               inst600000A3
// CHECK:STDOUT:     inst_block60000036:
// CHECK:STDOUT:       0:               inst600000AE
// CHECK:STDOUT:       1:               inst600000AF
// CHECK:STDOUT:       2:               inst600000B0
// CHECK:STDOUT:     inst_block60000037:
// CHECK:STDOUT:       0:               inst600000B1
// CHECK:STDOUT:       1:               inst600000B2
// CHECK:STDOUT:       2:               inst600000B3
// CHECK:STDOUT:     inst_block60000038:
// CHECK:STDOUT:       0:               inst60000015
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:     inst_block60000039:
// CHECK:STDOUT:       0:               inst60000060
// CHECK:STDOUT:       1:               inst600000BF
// CHECK:STDOUT:     inst_block6000003A:
// CHECK:STDOUT:       0:               inst600000C2
// CHECK:STDOUT:       1:               inst600000C1
// CHECK:STDOUT:     inst_block6000003B:
// CHECK:STDOUT:       0:               inst600000C3
// CHECK:STDOUT:       1:               inst600000C4
// CHECK:STDOUT:     inst_block6000003C:
// CHECK:STDOUT:       0:               inst600000C7
// CHECK:STDOUT:     inst_block6000003D:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst600000BE
// CHECK:STDOUT:     inst_block6000003E:
// CHECK:STDOUT:       0:               inst600000CC
// CHECK:STDOUT:       1:               inst600000CD
// CHECK:STDOUT:     inst_block6000003F:
// CHECK:STDOUT:       0:               inst600000CA
// CHECK:STDOUT:       1:               inst600000CB
// CHECK:STDOUT:     inst_block60000040:
// CHECK:STDOUT:       0:               inst600000CA
// CHECK:STDOUT:       1:               inst600000CB
// CHECK:STDOUT:       2:               inst600000CC
// CHECK:STDOUT:       3:               inst600000CD
// CHECK:STDOUT:       4:               inst600000CE
// CHECK:STDOUT:       5:               inst600000CF
// CHECK:STDOUT:     inst_block60000041:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst600000BE
// CHECK:STDOUT:       2:               inst60000060
// CHECK:STDOUT:       3:               inst600000BF
// CHECK:STDOUT:       4:               inst600000C0
// CHECK:STDOUT:       5:               inst600000C9
// CHECK:STDOUT:     inst_block60000042:
// CHECK:STDOUT:       0:               inst600000D7
// CHECK:STDOUT:     inst_block60000043:
// CHECK:STDOUT:       0:               inst600000D8
// CHECK:STDOUT:       1:               inst600000D9
// CHECK:STDOUT:     inst_block60000044:
// CHECK:STDOUT:       0:               inst600000CA
// CHECK:STDOUT:       1:               inst600000CB
// CHECK:STDOUT:     inst_block60000045:
// CHECK:STDOUT:       0:               inst600000DA
// CHECK:STDOUT:       1:               inst600000DB
// CHECK:STDOUT:     inst_block60000046:
// CHECK:STDOUT:       0:               inst600000DD
// CHECK:STDOUT:       1:               inst600000DF
// CHECK:STDOUT:     inst_block60000047:
// CHECK:STDOUT:       0:               inst600000DC
// CHECK:STDOUT:       1:               inst600000DD
// CHECK:STDOUT:       2:               inst600000DE
// CHECK:STDOUT:       3:               inst600000DF
// CHECK:STDOUT:       4:               inst600000E0
// CHECK:STDOUT:       5:               inst600000E1
// CHECK:STDOUT:     inst_block60000048:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst600000BE
// CHECK:STDOUT:       3:               inst600000BF
// CHECK:STDOUT:       4:               inst600000C0
// CHECK:STDOUT:       5:               inst600000D3
// CHECK:STDOUT:     inst_block60000049:
// CHECK:STDOUT:       0:               inst600000BE
// CHECK:STDOUT:     inst_block6000004A:
// CHECK:STDOUT:       0:               inst600000DC
// CHECK:STDOUT:     inst_block6000004B:
// CHECK:STDOUT:       0:               inst600000DE
// CHECK:STDOUT:     inst_block6000004C:
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
// CHECK:STDOUT:     inst_block6000004D:
// CHECK:STDOUT:       0:               inst600000BE
// CHECK:STDOUT:       1:               inst600000BF
// CHECK:STDOUT:       2:               inst600000F3
// CHECK:STDOUT:     inst_block6000004E:
// CHECK:STDOUT:       0:               inst600000D1
// CHECK:STDOUT:       1:               inst600000D2
// CHECK:STDOUT:     inst_block6000004F:
// CHECK:STDOUT:       0:               inst60000060
// CHECK:STDOUT:       1:               inst600000BF
// CHECK:STDOUT:       2:               inst600000F7
// CHECK:STDOUT:     inst_block60000050:
// CHECK:STDOUT:       0:               inst600000FB
// CHECK:STDOUT:       1:               inst600000FA
// CHECK:STDOUT:       2:               inst600000F9
// CHECK:STDOUT:     inst_block60000051:
// CHECK:STDOUT:       0:               inst600000FC
// CHECK:STDOUT:       1:               inst600000FD
// CHECK:STDOUT:       2:               inst600000FE
// CHECK:STDOUT:     inst_block60000052:
// CHECK:STDOUT:       0:               inst60000101
// CHECK:STDOUT:     inst_block60000053:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst600000BE
// CHECK:STDOUT:       2:               inst600000F6
// CHECK:STDOUT:     inst_block60000054:
// CHECK:STDOUT:       0:               inst60000107
// CHECK:STDOUT:       1:               inst60000108
// CHECK:STDOUT:       2:               inst60000109
// CHECK:STDOUT:     inst_block60000055:
// CHECK:STDOUT:       0:               inst60000104
// CHECK:STDOUT:       1:               inst60000105
// CHECK:STDOUT:       2:               inst60000106
// CHECK:STDOUT:     inst_block60000056:
// CHECK:STDOUT:       0:               inst60000104
// CHECK:STDOUT:       1:               inst60000105
// CHECK:STDOUT:       2:               inst60000106
// CHECK:STDOUT:       3:               inst60000107
// CHECK:STDOUT:       4:               inst60000108
// CHECK:STDOUT:       5:               inst60000109
// CHECK:STDOUT:       6:               inst6000010A
// CHECK:STDOUT:       7:               inst6000010B
// CHECK:STDOUT:     inst_block60000057:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst600000BE
// CHECK:STDOUT:       2:               inst600000F6
// CHECK:STDOUT:       3:               inst60000060
// CHECK:STDOUT:       4:               inst600000BF
// CHECK:STDOUT:       5:               inst600000F7
// CHECK:STDOUT:       6:               inst600000F8
// CHECK:STDOUT:       7:               inst60000103
// CHECK:STDOUT:     inst_block60000058:
// CHECK:STDOUT:       0:               inst60000113
// CHECK:STDOUT:     inst_block60000059:
// CHECK:STDOUT:       0:               inst60000114
// CHECK:STDOUT:       1:               inst60000115
// CHECK:STDOUT:       2:               inst60000116
// CHECK:STDOUT:     inst_block6000005A:
// CHECK:STDOUT:       0:               inst60000104
// CHECK:STDOUT:       1:               inst60000105
// CHECK:STDOUT:       2:               inst60000106
// CHECK:STDOUT:     inst_block6000005B:
// CHECK:STDOUT:       0:               inst60000117
// CHECK:STDOUT:       1:               inst60000118
// CHECK:STDOUT:     inst_block6000005C:
// CHECK:STDOUT:       0:               inst6000011A
// CHECK:STDOUT:       1:               inst6000011C
// CHECK:STDOUT:       2:               inst6000011E
// CHECK:STDOUT:     inst_block6000005D:
// CHECK:STDOUT:       0:               inst60000119
// CHECK:STDOUT:       1:               inst6000011A
// CHECK:STDOUT:       2:               inst6000011B
// CHECK:STDOUT:       3:               inst6000011C
// CHECK:STDOUT:       4:               inst6000011D
// CHECK:STDOUT:       5:               inst6000011E
// CHECK:STDOUT:       6:               inst6000011F
// CHECK:STDOUT:       7:               inst60000120
// CHECK:STDOUT:     inst_block6000005E:
// CHECK:STDOUT:       0:               inst6000005F
// CHECK:STDOUT:       1:               inst60000060
// CHECK:STDOUT:       2:               inst600000BE
// CHECK:STDOUT:       3:               inst600000BF
// CHECK:STDOUT:       4:               inst600000F6
// CHECK:STDOUT:       5:               inst600000F7
// CHECK:STDOUT:       6:               inst600000F8
// CHECK:STDOUT:       7:               inst6000010F
// CHECK:STDOUT:     inst_block6000005F:
// CHECK:STDOUT:       0:               inst600000F6
// CHECK:STDOUT:     inst_block60000060:
// CHECK:STDOUT:       0:               inst60000119
// CHECK:STDOUT:     inst_block60000061:
// CHECK:STDOUT:       0:               inst6000011B
// CHECK:STDOUT:     inst_block60000062:
// CHECK:STDOUT:       0:               inst6000011D
// CHECK:STDOUT:     inst_block60000063:
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
// CHECK:STDOUT:     inst_block60000064:
// CHECK:STDOUT:       0:               inst600000F6
// CHECK:STDOUT:       1:               inst600000F7
// CHECK:STDOUT:       2:               inst60000137
// CHECK:STDOUT:     inst_block60000065:
// CHECK:STDOUT:       0:               inst6000010D
// CHECK:STDOUT:       1:               inst6000010E
// CHECK:STDOUT:     inst_block60000066:
// CHECK:STDOUT:       0:               inst600000A7
// CHECK:STDOUT:       1:               inst600000A8
// CHECK:STDOUT:       2:               inst6000003D
// CHECK:STDOUT:     inst_block60000067:
// CHECK:STDOUT:       0:               inst60000138
// CHECK:STDOUT:     inst_block60000068:
// CHECK:STDOUT:       0:               inst60000139
// CHECK:STDOUT:     inst_block60000069:
// CHECK:STDOUT:       0:               inst6000013A
// CHECK:STDOUT:     inst_block6000006A:
// CHECK:STDOUT:       0:               inst6000013A
// CHECK:STDOUT:       1:               inst6000001B
// CHECK:STDOUT:       2:               inst6000001E
// CHECK:STDOUT:     inst_block6000006B:
// CHECK:STDOUT:       0:               inst6000013E
// CHECK:STDOUT:     inst_block6000006C:
// CHECK:STDOUT:       0:               inst6000003F
// CHECK:STDOUT:     inst_block6000006D:
// CHECK:STDOUT:       0:               inst60000148
// CHECK:STDOUT:       1:               inst6000014C
// CHECK:STDOUT:     inst_block6000006E:
// CHECK:STDOUT:       0:               inst6000003B
// CHECK:STDOUT:       1:               inst6000003E
// CHECK:STDOUT:       2:               inst60000139
// CHECK:STDOUT:       3:               inst6000013E
// CHECK:STDOUT:       4:               inst6000013F
// CHECK:STDOUT:       5:               inst60000140
// CHECK:STDOUT:       6:               inst60000144
// CHECK:STDOUT:     inst_block6000006F:
// CHECK:STDOUT:       0:               inst0000000E
// CHECK:STDOUT:       1:               inst6000000F
// CHECK:STDOUT:       2:               inst60000035
// CHECK:STDOUT: ...
