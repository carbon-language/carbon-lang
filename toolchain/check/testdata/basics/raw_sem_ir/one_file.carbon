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
// CHECK:STDOUT:     'import_ir(ApiForImpl)': {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     'import_ir(Cpp)':  {decl_id: inst<none>, is_export: false}
// CHECK:STDOUT:     import_ir78000002: {decl_id: inst78000010, is_export: false}
// CHECK:STDOUT:     import_ir78000003: {decl_id: inst78000010, is_export: false}
// CHECK:STDOUT:     import_ir78000004: {decl_id: inst78000010, is_export: false}
// CHECK:STDOUT:     import_ir78000005: {decl_id: inst78000010, is_export: false}
// CHECK:STDOUT:     import_ir78000006: {decl_id: inst78000010, is_export: false}
// CHECK:STDOUT:     import_ir78000007: {decl_id: inst78000010, is_export: false}
// CHECK:STDOUT:   import_ir_insts:
// CHECK:STDOUT:     import_ir_inst0: {ir_id: import_ir78000004, inst_id: inst70000010}
// CHECK:STDOUT:     import_ir_inst1: {ir_id: import_ir78000004, inst_id: inst70000010}
// CHECK:STDOUT:     import_ir_inst2: {ir_id: import_ir78000004, inst_id: inst70000015}
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir78000004, inst_id: inst7000003D}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir78000004, inst_id: inst70000035}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: import_ir78000004, inst_id: inst70000012}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: import_ir78000004, inst_id: inst70000035}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: import_ir78000004, inst_id: inst7000002C}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: import_ir78000004, inst_id: inst7000001E}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: import_ir78000004, inst_id: inst7000002C}
// CHECK:STDOUT:     import_ir_instA: {ir_id: import_ir78000004, inst_id: inst7000001E}
// CHECK:STDOUT:     import_ir_instB: {ir_id: import_ir78000004, inst_id: inst7000001D}
// CHECK:STDOUT:     import_ir_instC: {ir_id: import_ir78000004, inst_id: inst7000002B}
// CHECK:STDOUT:     import_ir_instD: {ir_id: import_ir78000004, inst_id: inst70000021}
// CHECK:STDOUT:     import_ir_instE: {ir_id: import_ir78000004, inst_id: inst70000021}
// CHECK:STDOUT:     import_ir_instF: {ir_id: import_ir78000004, inst_id: inst70000027}
// CHECK:STDOUT:     import_ir_inst10: {ir_id: import_ir78000004, inst_id: inst70000028}
// CHECK:STDOUT:     import_ir_inst11: {ir_id: import_ir78000004, inst_id: inst7000002E}
// CHECK:STDOUT:     import_ir_inst12: {ir_id: import_ir78000004, inst_id: inst70000012}
// CHECK:STDOUT:     import_ir_inst13: {ir_id: import_ir78000004, inst_id: inst70000017}
// CHECK:STDOUT:     import_ir_inst14: {ir_id: import_ir78000004, inst_id: inst7000001A}
// CHECK:STDOUT:     import_ir_inst15: {ir_id: import_ir78000004, inst_id: inst7000001F}
// CHECK:STDOUT:     import_ir_inst16: {ir_id: import_ir78000004, inst_id: inst70000020}
// CHECK:STDOUT:     import_ir_inst17: {ir_id: import_ir78000004, inst_id: inst70000023}
// CHECK:STDOUT:     import_ir_inst18: {ir_id: import_ir78000004, inst_id: inst7000002A}
// CHECK:STDOUT:     import_ir_inst19: {ir_id: import_ir78000004, inst_id: inst7000002D}
// CHECK:STDOUT:     import_ir_inst1A: {ir_id: import_ir78000004, inst_id: inst70000030}
// CHECK:STDOUT:     import_ir_inst1B: {ir_id: import_ir78000004, inst_id: inst7000002C}
// CHECK:STDOUT:     import_ir_inst1C: {ir_id: import_ir78000004, inst_id: inst7000001E}
// CHECK:STDOUT:     import_ir_inst1D: {ir_id: import_ir78000004, inst_id: inst70000039}
// CHECK:STDOUT:     import_ir_inst1E: {ir_id: import_ir78000004, inst_id: inst7000003A}
// CHECK:STDOUT:     import_ir_inst1F: {ir_id: import_ir78000004, inst_id: inst7000003B}
// CHECK:STDOUT:     import_ir_inst20: {ir_id: import_ir78000004, inst_id: inst70000012}
// CHECK:STDOUT:     import_ir_inst21: {ir_id: import_ir78000004, inst_id: inst70000035}
// CHECK:STDOUT:     import_ir_inst22: {ir_id: import_ir78000004, inst_id: inst70000094}
// CHECK:STDOUT:     import_ir_inst23: {ir_id: import_ir78000004, inst_id: inst7000008F}
// CHECK:STDOUT:     import_ir_inst24: {ir_id: import_ir78000004, inst_id: inst700000B3}
// CHECK:STDOUT:     import_ir_inst25: {ir_id: import_ir78000004, inst_id: inst70000093}
// CHECK:STDOUT:     import_ir_inst26: {ir_id: import_ir78000004, inst_id: inst700000B3}
// CHECK:STDOUT:     import_ir_inst27: {ir_id: import_ir78000004, inst_id: inst700000AB}
// CHECK:STDOUT:     import_ir_inst28: {ir_id: import_ir78000004, inst_id: inst7000009F}
// CHECK:STDOUT:     import_ir_inst29: {ir_id: import_ir78000004, inst_id: inst700000AB}
// CHECK:STDOUT:     import_ir_inst2A: {ir_id: import_ir78000004, inst_id: inst7000009F}
// CHECK:STDOUT:     import_ir_inst2B: {ir_id: import_ir78000004, inst_id: inst7000009E}
// CHECK:STDOUT:     import_ir_inst2C: {ir_id: import_ir78000004, inst_id: inst700000AA}
// CHECK:STDOUT:     import_ir_inst2D: {ir_id: import_ir78000004, inst_id: inst700000A2}
// CHECK:STDOUT:     import_ir_inst2E: {ir_id: import_ir78000004, inst_id: inst700000A2}
// CHECK:STDOUT:     import_ir_inst2F: {ir_id: import_ir78000004, inst_id: inst700000A6}
// CHECK:STDOUT:     import_ir_inst30: {ir_id: import_ir78000004, inst_id: inst700000A7}
// CHECK:STDOUT:     import_ir_inst31: {ir_id: import_ir78000004, inst_id: inst700000AD}
// CHECK:STDOUT:     import_ir_inst32: {ir_id: import_ir78000004, inst_id: inst70000082}
// CHECK:STDOUT:     import_ir_inst33: {ir_id: import_ir78000004, inst_id: inst7000009A}
// CHECK:STDOUT:     import_ir_inst34: {ir_id: import_ir78000004, inst_id: inst7000009B}
// CHECK:STDOUT:     import_ir_inst35: {ir_id: import_ir78000004, inst_id: inst7000009C}
// CHECK:STDOUT:     import_ir_inst36: {ir_id: import_ir78000004, inst_id: inst700000A0}
// CHECK:STDOUT:     import_ir_inst37: {ir_id: import_ir78000004, inst_id: inst700000A1}
// CHECK:STDOUT:     import_ir_inst38: {ir_id: import_ir78000004, inst_id: inst700000A4}
// CHECK:STDOUT:     import_ir_inst39: {ir_id: import_ir78000004, inst_id: inst700000A9}
// CHECK:STDOUT:     import_ir_inst3A: {ir_id: import_ir78000004, inst_id: inst700000AC}
// CHECK:STDOUT:     import_ir_inst3B: {ir_id: import_ir78000004, inst_id: inst700000AF}
// CHECK:STDOUT:     import_ir_inst3C: {ir_id: import_ir78000004, inst_id: inst700000DB}
// CHECK:STDOUT:     import_ir_inst3D: {ir_id: import_ir78000004, inst_id: inst700000D8}
// CHECK:STDOUT:     import_ir_inst3E: {ir_id: import_ir78000004, inst_id: inst700000BD}
// CHECK:STDOUT:     import_ir_inst3F: {ir_id: import_ir78000004, inst_id: inst700000C4}
// CHECK:STDOUT:     import_ir_inst40: {ir_id: import_ir78000004, inst_id: inst700000D0}
// CHECK:STDOUT:     import_ir_inst41: {ir_id: import_ir78000004, inst_id: inst700000D1}
// CHECK:STDOUT:     import_ir_inst42: {ir_id: import_ir78000004, inst_id: inst700000D2}
// CHECK:STDOUT:     import_ir_inst43: {ir_id: import_ir78000004, inst_id: inst700000D3}
// CHECK:STDOUT:     import_ir_inst44: {ir_id: import_ir78000004, inst_id: inst700000DF}
// CHECK:STDOUT:     import_ir_inst45: {ir_id: import_ir78000004, inst_id: inst700000AB}
// CHECK:STDOUT:     import_ir_inst46: {ir_id: import_ir78000004, inst_id: inst7000009F}
// CHECK:STDOUT:     import_ir_inst47: {ir_id: import_ir78000004, inst_id: inst7000007F}
// CHECK:STDOUT:     import_ir_inst48: {ir_id: import_ir78000004, inst_id: inst7000008A}
// CHECK:STDOUT:     import_ir_inst49: {ir_id: import_ir78000004, inst_id: inst7000008D}
// CHECK:STDOUT:     import_ir_inst4A: {ir_id: import_ir78000004, inst_id: inst70000090}
// CHECK:STDOUT:     import_ir_inst4B: {ir_id: import_ir78000004, inst_id: inst70000082}
// CHECK:STDOUT:     import_ir_inst4C: {ir_id: import_ir78000004, inst_id: inst70000081}
// CHECK:STDOUT:     import_ir_inst4D: {ir_id: import_ir78000004, inst_id: inst70000084}
// CHECK:STDOUT:     import_ir_inst4E: {ir_id: import_ir78000004, inst_id: inst70000088}
// CHECK:STDOUT:     import_ir_inst4F: {ir_id: import_ir78000004, inst_id: inst7000008C}
// CHECK:STDOUT:     import_ir_inst50: {ir_id: import_ir78000004, inst_id: inst70000092}
// CHECK:STDOUT:     import_ir_inst51: {ir_id: import_ir78000004, inst_id: inst70000096}
// CHECK:STDOUT:     import_ir_inst52: {ir_id: import_ir78000004, inst_id: inst700000B6}
// CHECK:STDOUT:     import_ir_inst53: {ir_id: import_ir78000004, inst_id: inst700000B7}
// CHECK:STDOUT:     import_ir_inst54: {ir_id: import_ir78000004, inst_id: inst700000F1}
// CHECK:STDOUT:     import_ir_inst55: {ir_id: import_ir78000004, inst_id: inst700000ED}
// CHECK:STDOUT:     import_ir_inst56: {ir_id: import_ir78000004, inst_id: inst700000EB}
// CHECK:STDOUT:     import_ir_inst57: {ir_id: import_ir78000004, inst_id: inst700000EC}
// CHECK:STDOUT:     import_ir_inst58: {ir_id: import_ir78000004, inst_id: inst700000EE}
// CHECK:STDOUT:     import_ir_inst59: {ir_id: import_ir78000004, inst_id: inst70000110}
// CHECK:STDOUT:     import_ir_inst5A: {ir_id: import_ir78000004, inst_id: inst7000010C}
// CHECK:STDOUT:     import_ir_inst5B: {ir_id: import_ir78000004, inst_id: inst7000010A}
// CHECK:STDOUT:     import_ir_inst5C: {ir_id: import_ir78000004, inst_id: inst7000010B}
// CHECK:STDOUT:     import_ir_inst5D: {ir_id: import_ir78000004, inst_id: inst7000010D}
// CHECK:STDOUT:     import_ir_inst5E: {ir_id: import_ir78000004, inst_id: inst7000012F}
// CHECK:STDOUT:     import_ir_inst5F: {ir_id: import_ir78000004, inst_id: inst7000012B}
// CHECK:STDOUT:     import_ir_inst60: {ir_id: import_ir78000004, inst_id: inst70000129}
// CHECK:STDOUT:     import_ir_inst61: {ir_id: import_ir78000004, inst_id: inst7000012A}
// CHECK:STDOUT:     import_ir_inst62: {ir_id: import_ir78000004, inst_id: inst7000012C}
// CHECK:STDOUT:     import_ir_inst63: {ir_id: import_ir78000004, inst_id: inst7000014E}
// CHECK:STDOUT:     import_ir_inst64: {ir_id: import_ir78000004, inst_id: inst7000014A}
// CHECK:STDOUT:     import_ir_inst65: {ir_id: import_ir78000004, inst_id: inst70000148}
// CHECK:STDOUT:     import_ir_inst66: {ir_id: import_ir78000004, inst_id: inst70000149}
// CHECK:STDOUT:     import_ir_inst67: {ir_id: import_ir78000004, inst_id: inst7000014B}
// CHECK:STDOUT:     import_ir_inst68: {ir_id: import_ir78000004, inst_id: inst7000017A}
// CHECK:STDOUT:     import_ir_inst69: {ir_id: import_ir78000004, inst_id: inst70000175}
// CHECK:STDOUT:     import_ir_inst6A: {ir_id: import_ir78000004, inst_id: inst70000196}
// CHECK:STDOUT:     import_ir_inst6B: {ir_id: import_ir78000004, inst_id: inst70000179}
// CHECK:STDOUT:     import_ir_inst6C: {ir_id: import_ir78000004, inst_id: inst70000169}
// CHECK:STDOUT:     import_ir_inst6D: {ir_id: import_ir78000004, inst_id: inst70000170}
// CHECK:STDOUT:     import_ir_inst6E: {ir_id: import_ir78000004, inst_id: inst70000173}
// CHECK:STDOUT:     import_ir_inst6F: {ir_id: import_ir78000004, inst_id: inst70000176}
// CHECK:STDOUT:     import_ir_inst70: {ir_id: import_ir78000004, inst_id: inst7000016C}
// CHECK:STDOUT:     import_ir_inst71: {ir_id: import_ir78000004, inst_id: inst7000016B}
// CHECK:STDOUT:     import_ir_inst72: {ir_id: import_ir78000004, inst_id: inst7000016E}
// CHECK:STDOUT:     import_ir_inst73: {ir_id: import_ir78000004, inst_id: inst70000172}
// CHECK:STDOUT:     import_ir_inst74: {ir_id: import_ir78000004, inst_id: inst70000178}
// CHECK:STDOUT:     import_ir_inst75: {ir_id: import_ir78000004, inst_id: inst7000017C}
// CHECK:STDOUT:     import_ir_inst76: {ir_id: import_ir78000004, inst_id: inst70000196}
// CHECK:STDOUT:     import_ir_inst77: {ir_id: import_ir78000004, inst_id: inst7000018E}
// CHECK:STDOUT:     import_ir_inst78: {ir_id: import_ir78000004, inst_id: inst70000182}
// CHECK:STDOUT:     import_ir_inst79: {ir_id: import_ir78000004, inst_id: inst7000018E}
// CHECK:STDOUT:     import_ir_inst7A: {ir_id: import_ir78000004, inst_id: inst70000182}
// CHECK:STDOUT:     import_ir_inst7B: {ir_id: import_ir78000004, inst_id: inst70000181}
// CHECK:STDOUT:     import_ir_inst7C: {ir_id: import_ir78000004, inst_id: inst7000018D}
// CHECK:STDOUT:     import_ir_inst7D: {ir_id: import_ir78000004, inst_id: inst70000185}
// CHECK:STDOUT:     import_ir_inst7E: {ir_id: import_ir78000004, inst_id: inst70000185}
// CHECK:STDOUT:     import_ir_inst7F: {ir_id: import_ir78000004, inst_id: inst70000189}
// CHECK:STDOUT:     import_ir_inst80: {ir_id: import_ir78000004, inst_id: inst7000018A}
// CHECK:STDOUT:     import_ir_inst81: {ir_id: import_ir78000004, inst_id: inst70000190}
// CHECK:STDOUT:     import_ir_inst82: {ir_id: import_ir78000004, inst_id: inst7000016C}
// CHECK:STDOUT:     import_ir_inst83: {ir_id: import_ir78000004, inst_id: inst7000017E}
// CHECK:STDOUT:     import_ir_inst84: {ir_id: import_ir78000004, inst_id: inst7000017F}
// CHECK:STDOUT:     import_ir_inst85: {ir_id: import_ir78000004, inst_id: inst70000183}
// CHECK:STDOUT:     import_ir_inst86: {ir_id: import_ir78000004, inst_id: inst70000184}
// CHECK:STDOUT:     import_ir_inst87: {ir_id: import_ir78000004, inst_id: inst70000187}
// CHECK:STDOUT:     import_ir_inst88: {ir_id: import_ir78000004, inst_id: inst7000018C}
// CHECK:STDOUT:     import_ir_inst89: {ir_id: import_ir78000004, inst_id: inst7000018F}
// CHECK:STDOUT:     import_ir_inst8A: {ir_id: import_ir78000004, inst_id: inst70000192}
// CHECK:STDOUT:     import_ir_inst8B: {ir_id: import_ir78000004, inst_id: inst7000018E}
// CHECK:STDOUT:     import_ir_inst8C: {ir_id: import_ir78000004, inst_id: inst70000182}
// CHECK:STDOUT:     import_ir_inst8D: {ir_id: import_ir78000004, inst_id: inst70000199}
// CHECK:STDOUT:     import_ir_inst8E: {ir_id: import_ir78000004, inst_id: inst7000019A}
// CHECK:STDOUT:     import_ir_inst8F: {ir_id: import_ir78000004, inst_id: inst7000019D}
// CHECK:STDOUT:     import_ir_inst90: {ir_id: import_ir78000004, inst_id: inst700001A7}
// CHECK:STDOUT:     import_ir_inst91: {ir_id: import_ir78000004, inst_id: inst700001A3}
// CHECK:STDOUT:     import_ir_inst92: {ir_id: import_ir78000004, inst_id: inst700001A1}
// CHECK:STDOUT:     import_ir_inst93: {ir_id: import_ir78000004, inst_id: inst700001A2}
// CHECK:STDOUT:     import_ir_inst94: {ir_id: import_ir78000004, inst_id: inst700001A4}
// CHECK:STDOUT:     import_ir_inst95: {ir_id: import_ir78000004, inst_id: inst700001C4}
// CHECK:STDOUT:     import_ir_inst96: {ir_id: import_ir78000004, inst_id: inst700001C0}
// CHECK:STDOUT:     import_ir_inst97: {ir_id: import_ir78000004, inst_id: inst700001BE}
// CHECK:STDOUT:     import_ir_inst98: {ir_id: import_ir78000004, inst_id: inst700001BF}
// CHECK:STDOUT:     import_ir_inst99: {ir_id: import_ir78000004, inst_id: inst700001C1}
// CHECK:STDOUT:     import_ir_inst9A: {ir_id: import_ir78000004, inst_id: inst70000204}
// CHECK:STDOUT:     import_ir_inst9B: {ir_id: import_ir78000004, inst_id: inst700001FF}
// CHECK:STDOUT:     import_ir_inst9C: {ir_id: import_ir78000004, inst_id: inst70000227}
// CHECK:STDOUT:     import_ir_inst9D: {ir_id: import_ir78000004, inst_id: inst70000203}
// CHECK:STDOUT:     import_ir_inst9E: {ir_id: import_ir78000004, inst_id: inst70000227}
// CHECK:STDOUT:     import_ir_inst9F: {ir_id: import_ir78000004, inst_id: inst7000021F}
// CHECK:STDOUT:     import_ir_instA0: {ir_id: import_ir78000004, inst_id: inst70000213}
// CHECK:STDOUT:     import_ir_instA1: {ir_id: import_ir78000004, inst_id: inst7000021F}
// CHECK:STDOUT:     import_ir_instA2: {ir_id: import_ir78000004, inst_id: inst70000213}
// CHECK:STDOUT:     import_ir_instA3: {ir_id: import_ir78000004, inst_id: inst70000212}
// CHECK:STDOUT:     import_ir_instA4: {ir_id: import_ir78000004, inst_id: inst7000021E}
// CHECK:STDOUT:     import_ir_instA5: {ir_id: import_ir78000004, inst_id: inst70000216}
// CHECK:STDOUT:     import_ir_instA6: {ir_id: import_ir78000004, inst_id: inst70000216}
// CHECK:STDOUT:     import_ir_instA7: {ir_id: import_ir78000004, inst_id: inst7000021A}
// CHECK:STDOUT:     import_ir_instA8: {ir_id: import_ir78000004, inst_id: inst7000021B}
// CHECK:STDOUT:     import_ir_instA9: {ir_id: import_ir78000004, inst_id: inst70000221}
// CHECK:STDOUT:     import_ir_instAA: {ir_id: import_ir78000004, inst_id: inst700001E1}
// CHECK:STDOUT:     import_ir_instAB: {ir_id: import_ir78000004, inst_id: inst700001E8}
// CHECK:STDOUT:     import_ir_instAC: {ir_id: import_ir78000004, inst_id: inst7000020C}
// CHECK:STDOUT:     import_ir_instAD: {ir_id: import_ir78000004, inst_id: inst7000020D}
// CHECK:STDOUT:     import_ir_instAE: {ir_id: import_ir78000004, inst_id: inst7000020E}
// CHECK:STDOUT:     import_ir_instAF: {ir_id: import_ir78000004, inst_id: inst7000020F}
// CHECK:STDOUT:     import_ir_instB0: {ir_id: import_ir78000004, inst_id: inst70000210}
// CHECK:STDOUT:     import_ir_instB1: {ir_id: import_ir78000004, inst_id: inst70000214}
// CHECK:STDOUT:     import_ir_instB2: {ir_id: import_ir78000004, inst_id: inst70000215}
// CHECK:STDOUT:     import_ir_instB3: {ir_id: import_ir78000004, inst_id: inst70000218}
// CHECK:STDOUT:     import_ir_instB4: {ir_id: import_ir78000004, inst_id: inst7000021D}
// CHECK:STDOUT:     import_ir_instB5: {ir_id: import_ir78000004, inst_id: inst70000220}
// CHECK:STDOUT:     import_ir_instB6: {ir_id: import_ir78000004, inst_id: inst70000223}
// CHECK:STDOUT:     import_ir_instB7: {ir_id: import_ir78000004, inst_id: inst70000262}
// CHECK:STDOUT:     import_ir_instB8: {ir_id: import_ir78000004, inst_id: inst7000025F}
// CHECK:STDOUT:     import_ir_instB9: {ir_id: import_ir78000004, inst_id: inst70000232}
// CHECK:STDOUT:     import_ir_instBA: {ir_id: import_ir78000004, inst_id: inst70000238}
// CHECK:STDOUT:     import_ir_instBB: {ir_id: import_ir78000004, inst_id: inst7000023C}
// CHECK:STDOUT:     import_ir_instBC: {ir_id: import_ir78000004, inst_id: inst7000023D}
// CHECK:STDOUT:     import_ir_instBD: {ir_id: import_ir78000004, inst_id: inst7000023E}
// CHECK:STDOUT:     import_ir_instBE: {ir_id: import_ir78000004, inst_id: inst7000023F}
// CHECK:STDOUT:     import_ir_instBF: {ir_id: import_ir78000004, inst_id: inst70000244}
// CHECK:STDOUT:     import_ir_instC0: {ir_id: import_ir78000004, inst_id: inst7000024E}
// CHECK:STDOUT:     import_ir_instC1: {ir_id: import_ir78000004, inst_id: inst70000257}
// CHECK:STDOUT:     import_ir_instC2: {ir_id: import_ir78000004, inst_id: inst70000258}
// CHECK:STDOUT:     import_ir_instC3: {ir_id: import_ir78000004, inst_id: inst70000259}
// CHECK:STDOUT:     import_ir_instC4: {ir_id: import_ir78000004, inst_id: inst7000025A}
// CHECK:STDOUT:     import_ir_instC5: {ir_id: import_ir78000004, inst_id: inst70000266}
// CHECK:STDOUT:     import_ir_instC6: {ir_id: import_ir78000004, inst_id: inst7000021F}
// CHECK:STDOUT:     import_ir_instC7: {ir_id: import_ir78000004, inst_id: inst70000213}
// CHECK:STDOUT:     import_ir_instC8: {ir_id: import_ir78000004, inst_id: inst700001DF}
// CHECK:STDOUT:     import_ir_instC9: {ir_id: import_ir78000004, inst_id: inst700001E5}
// CHECK:STDOUT:     import_ir_instCA: {ir_id: import_ir78000004, inst_id: inst700001FA}
// CHECK:STDOUT:     import_ir_instCB: {ir_id: import_ir78000004, inst_id: inst700001FC}
// CHECK:STDOUT:     import_ir_instCC: {ir_id: import_ir78000004, inst_id: inst70000200}
// CHECK:STDOUT:     import_ir_instCD: {ir_id: import_ir78000004, inst_id: inst700001E1}
// CHECK:STDOUT:     import_ir_instCE: {ir_id: import_ir78000004, inst_id: inst700001E8}
// CHECK:STDOUT:     import_ir_instCF: {ir_id: import_ir78000004, inst_id: inst700001E0}
// CHECK:STDOUT:     import_ir_instD0: {ir_id: import_ir78000004, inst_id: inst700001E2}
// CHECK:STDOUT:     import_ir_instD1: {ir_id: import_ir78000004, inst_id: inst700001E7}
// CHECK:STDOUT:     import_ir_instD2: {ir_id: import_ir78000004, inst_id: inst700001EA}
// CHECK:STDOUT:     import_ir_instD3: {ir_id: import_ir78000004, inst_id: inst700001F0}
// CHECK:STDOUT:     import_ir_instD4: {ir_id: import_ir78000004, inst_id: inst700001F3}
// CHECK:STDOUT:     import_ir_instD5: {ir_id: import_ir78000004, inst_id: inst700001F7}
// CHECK:STDOUT:     import_ir_instD6: {ir_id: import_ir78000004, inst_id: inst700001FB}
// CHECK:STDOUT:     import_ir_instD7: {ir_id: import_ir78000004, inst_id: inst70000202}
// CHECK:STDOUT:     import_ir_instD8: {ir_id: import_ir78000004, inst_id: inst70000206}
// CHECK:STDOUT:     import_ir_instD9: {ir_id: import_ir78000004, inst_id: inst7000022A}
// CHECK:STDOUT:     import_ir_instDA: {ir_id: import_ir78000004, inst_id: inst7000022B}
// CHECK:STDOUT:     import_ir_instDB: {ir_id: import_ir78000004, inst_id: inst700002A7}
// CHECK:STDOUT:     import_ir_instDC: {ir_id: import_ir78000004, inst_id: inst700002A2}
// CHECK:STDOUT:     import_ir_instDD: {ir_id: import_ir78000004, inst_id: inst700002CE}
// CHECK:STDOUT:     import_ir_instDE: {ir_id: import_ir78000004, inst_id: inst700002A6}
// CHECK:STDOUT:     import_ir_instDF: {ir_id: import_ir78000004, inst_id: inst700002CE}
// CHECK:STDOUT:     import_ir_instE0: {ir_id: import_ir78000004, inst_id: inst700002C6}
// CHECK:STDOUT:     import_ir_instE1: {ir_id: import_ir78000004, inst_id: inst700002BA}
// CHECK:STDOUT:     import_ir_instE2: {ir_id: import_ir78000004, inst_id: inst700002C6}
// CHECK:STDOUT:     import_ir_instE3: {ir_id: import_ir78000004, inst_id: inst700002BA}
// CHECK:STDOUT:     import_ir_instE4: {ir_id: import_ir78000004, inst_id: inst700002B9}
// CHECK:STDOUT:     import_ir_instE5: {ir_id: import_ir78000004, inst_id: inst700002C5}
// CHECK:STDOUT:     import_ir_instE6: {ir_id: import_ir78000004, inst_id: inst700002BD}
// CHECK:STDOUT:     import_ir_instE7: {ir_id: import_ir78000004, inst_id: inst700002BD}
// CHECK:STDOUT:     import_ir_instE8: {ir_id: import_ir78000004, inst_id: inst700002C1}
// CHECK:STDOUT:     import_ir_instE9: {ir_id: import_ir78000004, inst_id: inst700002C2}
// CHECK:STDOUT:     import_ir_instEA: {ir_id: import_ir78000004, inst_id: inst700002C8}
// CHECK:STDOUT:     import_ir_instEB: {ir_id: import_ir78000004, inst_id: inst70000279}
// CHECK:STDOUT:     import_ir_instEC: {ir_id: import_ir78000004, inst_id: inst7000027F}
// CHECK:STDOUT:     import_ir_instED: {ir_id: import_ir78000004, inst_id: inst70000286}
// CHECK:STDOUT:     import_ir_instEE: {ir_id: import_ir78000004, inst_id: inst700002B1}
// CHECK:STDOUT:     import_ir_instEF: {ir_id: import_ir78000004, inst_id: inst700002B2}
// CHECK:STDOUT:     import_ir_instF0: {ir_id: import_ir78000004, inst_id: inst700002B3}
// CHECK:STDOUT:     import_ir_instF1: {ir_id: import_ir78000004, inst_id: inst700002B4}
// CHECK:STDOUT:     import_ir_instF2: {ir_id: import_ir78000004, inst_id: inst700002B5}
// CHECK:STDOUT:     import_ir_instF3: {ir_id: import_ir78000004, inst_id: inst700002B6}
// CHECK:STDOUT:     import_ir_instF4: {ir_id: import_ir78000004, inst_id: inst700002B7}
// CHECK:STDOUT:     import_ir_instF5: {ir_id: import_ir78000004, inst_id: inst700002BB}
// CHECK:STDOUT:     import_ir_instF6: {ir_id: import_ir78000004, inst_id: inst700002BC}
// CHECK:STDOUT:     import_ir_instF7: {ir_id: import_ir78000004, inst_id: inst700002BF}
// CHECK:STDOUT:     import_ir_instF8: {ir_id: import_ir78000004, inst_id: inst700002C4}
// CHECK:STDOUT:     import_ir_instF9: {ir_id: import_ir78000004, inst_id: inst700002C7}
// CHECK:STDOUT:     import_ir_instFA: {ir_id: import_ir78000004, inst_id: inst700002CA}
// CHECK:STDOUT:     import_ir_instFB: {ir_id: import_ir78000004, inst_id: inst7000031C}
// CHECK:STDOUT:     import_ir_instFC: {ir_id: import_ir78000004, inst_id: inst70000319}
// CHECK:STDOUT:     import_ir_instFD: {ir_id: import_ir78000004, inst_id: inst700002D9}
// CHECK:STDOUT:     import_ir_instFE: {ir_id: import_ir78000004, inst_id: inst700002DE}
// CHECK:STDOUT:     import_ir_instFF: {ir_id: import_ir78000004, inst_id: inst700002E2}
// CHECK:STDOUT:     import_ir_inst100: {ir_id: import_ir78000004, inst_id: inst700002E3}
// CHECK:STDOUT:     import_ir_inst101: {ir_id: import_ir78000004, inst_id: inst700002E4}
// CHECK:STDOUT:     import_ir_inst102: {ir_id: import_ir78000004, inst_id: inst700002E5}
// CHECK:STDOUT:     import_ir_inst103: {ir_id: import_ir78000004, inst_id: inst700002EA}
// CHECK:STDOUT:     import_ir_inst104: {ir_id: import_ir78000004, inst_id: inst700002F2}
// CHECK:STDOUT:     import_ir_inst105: {ir_id: import_ir78000004, inst_id: inst700002F6}
// CHECK:STDOUT:     import_ir_inst106: {ir_id: import_ir78000004, inst_id: inst700002F7}
// CHECK:STDOUT:     import_ir_inst107: {ir_id: import_ir78000004, inst_id: inst700002F8}
// CHECK:STDOUT:     import_ir_inst108: {ir_id: import_ir78000004, inst_id: inst700002F9}
// CHECK:STDOUT:     import_ir_inst109: {ir_id: import_ir78000004, inst_id: inst700002FE}
// CHECK:STDOUT:     import_ir_inst10A: {ir_id: import_ir78000004, inst_id: inst70000308}
// CHECK:STDOUT:     import_ir_inst10B: {ir_id: import_ir78000004, inst_id: inst70000311}
// CHECK:STDOUT:     import_ir_inst10C: {ir_id: import_ir78000004, inst_id: inst70000312}
// CHECK:STDOUT:     import_ir_inst10D: {ir_id: import_ir78000004, inst_id: inst70000313}
// CHECK:STDOUT:     import_ir_inst10E: {ir_id: import_ir78000004, inst_id: inst70000314}
// CHECK:STDOUT:     import_ir_inst10F: {ir_id: import_ir78000004, inst_id: inst70000320}
// CHECK:STDOUT:     import_ir_inst110: {ir_id: import_ir78000004, inst_id: inst700002C6}
// CHECK:STDOUT:     import_ir_inst111: {ir_id: import_ir78000004, inst_id: inst700002BA}
// CHECK:STDOUT:     import_ir_inst112: {ir_id: import_ir78000004, inst_id: inst70000277}
// CHECK:STDOUT:     import_ir_inst113: {ir_id: import_ir78000004, inst_id: inst7000027D}
// CHECK:STDOUT:     import_ir_inst114: {ir_id: import_ir78000004, inst_id: inst70000283}
// CHECK:STDOUT:     import_ir_inst115: {ir_id: import_ir78000004, inst_id: inst7000029C}
// CHECK:STDOUT:     import_ir_inst116: {ir_id: import_ir78000004, inst_id: inst7000029E}
// CHECK:STDOUT:     import_ir_inst117: {ir_id: import_ir78000004, inst_id: inst700002A3}
// CHECK:STDOUT:     import_ir_inst118: {ir_id: import_ir78000004, inst_id: inst70000279}
// CHECK:STDOUT:     import_ir_inst119: {ir_id: import_ir78000004, inst_id: inst7000027F}
// CHECK:STDOUT:     import_ir_inst11A: {ir_id: import_ir78000004, inst_id: inst70000286}
// CHECK:STDOUT:     import_ir_inst11B: {ir_id: import_ir78000004, inst_id: inst70000278}
// CHECK:STDOUT:     import_ir_inst11C: {ir_id: import_ir78000004, inst_id: inst7000027A}
// CHECK:STDOUT:     import_ir_inst11D: {ir_id: import_ir78000004, inst_id: inst7000027E}
// CHECK:STDOUT:     import_ir_inst11E: {ir_id: import_ir78000004, inst_id: inst70000280}
// CHECK:STDOUT:     import_ir_inst11F: {ir_id: import_ir78000004, inst_id: inst70000285}
// CHECK:STDOUT:     import_ir_inst120: {ir_id: import_ir78000004, inst_id: inst70000288}
// CHECK:STDOUT:     import_ir_inst121: {ir_id: import_ir78000004, inst_id: inst7000028F}
// CHECK:STDOUT:     import_ir_inst122: {ir_id: import_ir78000004, inst_id: inst70000292}
// CHECK:STDOUT:     import_ir_inst123: {ir_id: import_ir78000004, inst_id: inst70000295}
// CHECK:STDOUT:     import_ir_inst124: {ir_id: import_ir78000004, inst_id: inst70000299}
// CHECK:STDOUT:     import_ir_inst125: {ir_id: import_ir78000004, inst_id: inst7000029D}
// CHECK:STDOUT:     import_ir_inst126: {ir_id: import_ir78000004, inst_id: inst700002A5}
// CHECK:STDOUT:     import_ir_inst127: {ir_id: import_ir78000004, inst_id: inst700002A9}
// CHECK:STDOUT:     import_ir_inst128: {ir_id: import_ir78000004, inst_id: inst700002D1}
// CHECK:STDOUT:     import_ir_inst129: {ir_id: import_ir78000004, inst_id: inst700002D2}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   clang_decl_signatures: {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: inst(Package), parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst78000011, name0: inst78000047}}
// CHECK:STDOUT:     name_scope78000001: {inst: inst78000011, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: inst78000057}}
// CHECK:STDOUT:     name_scope78000002: {inst: inst78000058, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {name(SelfType): inst78000082}}
// CHECK:STDOUT:     name_scope78000003: {inst: inst7800005A, parent_scope: name_scope78000002, has_error: false, extended_scopes: [], names: {name4: inst7800005C}}
// CHECK:STDOUT:     name_scope78000004: {inst: inst78000088, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000005: {inst: inst780000D7, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000006: {inst: inst780000DD, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000007: {inst: inst780000E3, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000008: {inst: inst780000E9, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000009: {inst: inst780000EF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000A: {inst: inst78000120, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000B: {inst: inst78000126, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000C: {inst: inst7800012C, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000D: {inst: inst7800018B, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name78000000: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, is_frozen_period_self: 1}, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000001: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst78000014}
// CHECK:STDOUT:     entity_name78000002: {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst7800001D}
// CHECK:STDOUT:     entity_name78000003: {name: name3, parent_scope: name_scope78000001, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000004: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000005: {name: name4, parent_scope: name_scope78000003, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000006: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000007: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000008: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000009: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000A: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000B: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000C: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000D: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000E: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800000F: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000010: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000011: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000012: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000013: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000014: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000015: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000016: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000017: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000018: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000019: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001A: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001B: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001C: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001D: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001E: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800001F: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000020: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000021: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000022: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000023: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000024: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000025: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000026: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000027: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000028: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000029: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002A: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002B: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002C: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002D: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002E: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800002F: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000030: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000031: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000032: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000033: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000034: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000035: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000036: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000037: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000038: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000039: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003A: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003B: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003C: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003D: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003E: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name7800003F: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000040: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000041: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000042: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000043: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000044: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:     entity_name78000045: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: inst<none>, type: inst<none>}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function78000000: {name: name0, parent_scope: name_scope0, call_param_patterns_id: inst_block78000010, call_params_id: inst_block78000011, return_type_inst_id: inst78000035, return_form_inst_id: inst78000037, return_pattern_id: inst7800003F, body: [inst_block78000018]}
// CHECK:STDOUT:     function78000001: {name: name4, parent_scope: name_scope78000003, call_param_patterns_id: inst_block7800001E, return_type_inst_id: inst78000071, return_form_inst_id: inst78000072, return_pattern_id: inst78000073}
// CHECK:STDOUT:     function78000002: {name: name4, parent_scope: name_scope78000004, call_param_patterns_id: inst_block78000029, return_type_inst_id: inst780000A3, return_form_inst_id: inst780000A4, return_pattern_id: inst780000A5}
// CHECK:STDOUT:     function78000003: {name: name4, parent_scope: name_scope78000009, call_param_patterns_id: inst_block7800003F, return_type_inst_id: inst7800010E, return_form_inst_id: inst7800010F, return_pattern_id: inst78000110, builtin: primitive_copy}
// CHECK:STDOUT:     function78000004: {name: name4, parent_scope: name_scope7800000C, call_param_patterns_id: inst_block7800004C, return_type_inst_id: inst78000148, return_form_inst_id: inst78000149, return_pattern_id: inst7800014A}
// CHECK:STDOUT:     function78000005: {name: name4, parent_scope: name_scope7800000D, call_param_patterns_id: inst_block78000068, return_type_inst_id: inst780001A7, return_form_inst_id: inst780001A8, return_pattern_id: inst780001A9}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:
// CHECK:STDOUT:     interface78000000: {name: name3, parent_scope: name_scope78000001, require_impls_block_id: require_block_empty, core_interface: Copy}
// CHECK:STDOUT:   associated_constants: {}
// CHECK:STDOUT:   impls:
// CHECK:STDOUT:     impl78000000:    {self: inst780000CA, constraint: inst780000CB, witness: inst78000087}
// CHECK:STDOUT:     impl78000001:    {self: inst780000D9, constraint: inst780000DA, witness: inst780000D6}
// CHECK:STDOUT:     impl78000002:    {self: inst780000DF, constraint: inst780000E0, witness: inst780000DC}
// CHECK:STDOUT:     impl78000003:    {self: inst780000E5, constraint: inst780000E6, witness: inst780000E2}
// CHECK:STDOUT:     impl78000004:    {self: inst780000EB, constraint: inst780000EC, witness: inst780000E8}
// CHECK:STDOUT:     impl78000005:    {self: inst780000F5, constraint: inst780000F6, witness: inst780000EE}
// CHECK:STDOUT:     impl78000006:    {self: inst78000122, constraint: inst78000123, witness: inst7800011F}
// CHECK:STDOUT:     impl78000007:    {self: inst78000128, constraint: inst78000129, witness: inst78000125}
// CHECK:STDOUT:     impl78000008:    {self: inst78000179, constraint: inst7800017A, witness: inst7800012B}
// CHECK:STDOUT:     impl78000009:    {self: inst780001E2, constraint: inst780001E3, witness: inst7800018A}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic78000000: {decl: inst78000047, bindings: inst_block78000014, self_specific_id: specific78000000, decl_block_id: inst_block78000016, definition_block_id: inst_block7800008B}
// CHECK:STDOUT:     generic78000001: {decl: inst7800005A, bindings: inst_block7800001B, self_specific_id: specific78000001, decl_block_id: inst_block_empty, definition_block_id: inst_block78000024}
// CHECK:STDOUT:     generic78000002: {decl: inst7800005F, bindings: inst_block78000020, self_specific_id: specific78000002, decl_block_id: inst_block78000021, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     generic78000003: {decl: inst78000088, bindings: inst_block78000034, self_specific_id: specific78000004, decl_block_id: inst_block78000036, definition_block_id: inst_block78000038}
// CHECK:STDOUT:     generic78000004: {decl: inst78000090, bindings: inst_block7800002B, self_specific_id: specific78000007, decl_block_id: inst_block7800002C, definition_block_id: inst_block78000031}
// CHECK:STDOUT:     generic78000005: {decl: inst780000EF, bindings: inst_block7800003C, self_specific_id: specific7800000B, decl_block_id: inst_block7800003E, definition_block_id: inst_block78000045}
// CHECK:STDOUT:     generic78000006: {decl: inst780000FE, bindings: inst_block78000041, self_specific_id: specific7800000D, decl_block_id: inst_block78000042, definition_block_id: inst_block_empty}
// CHECK:STDOUT:     generic78000007: {decl: inst7800012C, bindings: inst_block7800005B, self_specific_id: specific7800000E, decl_block_id: inst_block7800005F, definition_block_id: inst_block78000061}
// CHECK:STDOUT:     generic78000008: {decl: inst78000134, bindings: inst_block7800004E, self_specific_id: specific78000011, decl_block_id: inst_block78000050, definition_block_id: inst_block78000058}
// CHECK:STDOUT:     generic78000009: {decl: inst7800018B, bindings: inst_block78000079, self_specific_id: specific78000017, decl_block_id: inst_block7800007D, definition_block_id: inst_block7800007F}
// CHECK:STDOUT:     generic7800000A: {decl: inst78000193, bindings: inst_block7800006A, self_specific_id: specific7800001A, decl_block_id: inst_block7800006C, definition_block_id: inst_block78000076}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific78000000: {generic: generic78000000, args: inst_block78000015, decl_block_id: inst_block78000017, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000001: {generic: generic78000001, args: inst_block7800001C, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block7800001D, definition_has_error: 0}
// CHECK:STDOUT:     specific78000002: {generic: generic78000002, args: inst_block7800001C, decl_block_id: inst_block78000022, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000003: {generic: generic78000001, args: inst_block78000023, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000004: {generic: generic78000003, args: inst_block78000026, decl_block_id: inst_block78000027, decl_has_error: 0, definition_block_id: inst_block78000028, definition_has_error: 0}
// CHECK:STDOUT:     specific78000005: {generic: generic78000001, args: inst_block78000026, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block7800002D, definition_has_error: 0}
// CHECK:STDOUT:     specific78000006: {generic: generic78000002, args: inst_block78000026, decl_block_id: inst_block7800002E, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000007: {generic: generic78000004, args: inst_block78000026, decl_block_id: inst_block78000032, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000008: {generic: generic78000001, args: inst_block7800002F, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000009: {generic: generic78000002, args: inst_block7800002F, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000A: {generic: generic78000003, args: inst_block78000035, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000B: {generic: generic78000005, args: inst_block78000015, decl_block_id: inst_block7800003A, decl_has_error: 0, definition_block_id: inst_block78000080, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000C: {generic: generic78000005, args: inst_block7800003D, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000D: {generic: generic78000006, args: inst_block78000015, decl_block_id: inst_block78000043, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000E: {generic: generic78000007, args: inst_block78000048, decl_block_id: inst_block7800004A, decl_has_error: 0, definition_block_id: inst_block7800004B, definition_has_error: 0}
// CHECK:STDOUT:     specific7800000F: {generic: generic78000001, args: inst_block78000051, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block78000052, definition_has_error: 0}
// CHECK:STDOUT:     specific78000010: {generic: generic78000002, args: inst_block78000051, decl_block_id: inst_block78000053, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000011: {generic: generic78000008, args: inst_block78000048, decl_block_id: inst_block78000059, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000012: {generic: generic78000001, args: inst_block78000054, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000013: {generic: generic78000002, args: inst_block78000054, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000014: {generic: generic78000001, args: inst_block78000056, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000015: {generic: generic78000002, args: inst_block78000056, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000016: {generic: generic78000007, args: inst_block7800005C, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000017: {generic: generic78000009, args: inst_block78000064, decl_block_id: inst_block78000066, decl_has_error: 0, definition_block_id: inst_block78000067, definition_has_error: 0}
// CHECK:STDOUT:     specific78000018: {generic: generic78000001, args: inst_block7800006D, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block7800006E, definition_has_error: 0}
// CHECK:STDOUT:     specific78000019: {generic: generic78000002, args: inst_block7800006D, decl_block_id: inst_block7800006F, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800001A: {generic: generic7800000A, args: inst_block78000064, decl_block_id: inst_block78000077, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800001B: {generic: generic78000001, args: inst_block78000070, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800001C: {generic: generic78000002, args: inst_block78000070, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800001D: {generic: generic78000001, args: inst_block78000072, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800001E: {generic: generic78000002, args: inst_block78000072, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific7800001F: {generic: generic78000001, args: inst_block78000074, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000020: {generic: generic78000002, args: inst_block78000074, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000021: {generic: generic78000009, args: inst_block7800007A, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000022: {generic: generic78000005, args: inst_block78000081, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000023: {generic: generic78000001, args: inst_block78000083, decl_block_id: inst_block_empty, decl_has_error: 0, definition_block_id: inst_block78000084, definition_has_error: 0}
// CHECK:STDOUT:     specific78000024: {generic: generic78000001, args: inst_block78000086, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000025: {generic: generic78000002, args: inst_block78000083, decl_block_id: inst_block78000087, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
// CHECK:STDOUT:     specific78000026: {generic: generic78000002, args: inst_block78000086, decl_block_id: inst_block<none>, decl_has_error: 0, definition_block_id: inst_block<none>, definition_has_error: 0}
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
// CHECK:STDOUT:     'type(inst7800002B)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst7800002B)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst78000032)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst78000032)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            8
// CHECK:STDOUT:         alignment:       8
// CHECK:STDOUT:     'type(inst7800002E)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst78000032)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst78000048)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst7800002B)}
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
// CHECK:STDOUT:     'type(inst78000059)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst78000059)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(RequireSpecificDefinitionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant780001D3)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst7800002B)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(symbolic_constant780001D8)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst7800002B)}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:       object_layout:
// CHECK:STDOUT:         size:            0
// CHECK:STDOUT:         alignment:       1
// CHECK:STDOUT:   declared_facet_types:
// CHECK:STDOUT:     'declared_facet_type(Empty)': {}
// CHECK:STDOUT:     declared_facet_type78000001: {extends interface: interface78000000}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     'inst(TypeType)':  {kind: FacetType, arg0: declared_facet_type(Empty), type: type(TypeType)}
// CHECK:STDOUT:     'inst(Package)':   {kind: Namespace, arg0: name_scope0, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst78000010:    {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst78000011:    {kind: Namespace, arg0: name_scope78000001, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst78000012:    {kind: SymbolicBinding, arg0: entity_name78000000, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000013:    {kind: SymbolicBinding, arg0: entity_name78000000, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000014:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst78000015:    {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst78000016:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000015)}
// CHECK:STDOUT:     inst78000017:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000015)}
// CHECK:STDOUT:     inst78000018:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000015)}
// CHECK:STDOUT:     inst78000019:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001A:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001B:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001C:    {kind: NameRef, arg0: name1, arg1: inst78000019, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001D:    {kind: PointerType, arg0: inst7800001C, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001E:    {kind: PointerType, arg0: inst7800001A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001F:    {kind: PointerType, arg0: inst7800001B, type: type(TypeType)}
// CHECK:STDOUT:     inst78000020:    {kind: PatternType, arg0: inst7800001E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000021:    {kind: ValueParamPattern, arg0: name2, type: type(symbolic_constant78000009)}
// CHECK:STDOUT:     inst78000022:    {kind: ValueParamPattern, arg0: name2, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000023:    {kind: PatternType, arg0: inst7800001F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000024:    {kind: ValueParamPattern, arg0: name2, type: type(symbolic_constant78000009)}
// CHECK:STDOUT:     inst78000025:    {kind: WrapperBindingPattern, arg0: entity_name78000002, arg1: inst78000021, type: type(symbolic_constant78000009)}
// CHECK:STDOUT:     inst78000026:    {kind: WrapperBindingPattern, arg0: entity_name78000002, arg1: inst78000022, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000027:    {kind: WrapperBindingPattern, arg0: entity_name78000002, arg1: inst78000024, type: type(symbolic_constant78000009)}
// CHECK:STDOUT:     inst78000028:    {kind: WrapperBinding, arg0: entity_name78000002, arg1: inst78000043, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000029:    {kind: NameRef, arg0: name1, arg1: inst78000019, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002A:    {kind: PointerType, arg0: inst78000029, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002B:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002C:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst7800002B)}
// CHECK:STDOUT:     inst7800002D:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst7800002B)}
// CHECK:STDOUT:     inst7800002E:    {kind: TupleType, arg0: inst_block7800000A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002F:    {kind: TupleLiteral, arg0: inst_block78000009, type: type(inst7800002E)}
// CHECK:STDOUT:     inst78000030:    {kind: TupleValue, arg0: inst_block7800000B, type: type(inst7800002E)}
// CHECK:STDOUT:     inst78000031:    {kind: TupleValue, arg0: inst_block7800000C, type: type(inst7800002E)}
// CHECK:STDOUT:     inst78000032:    {kind: PointerType, arg0: inst7800002E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000033:    {kind: Converted, arg0: inst7800002D, arg1: inst7800002B, type: type(TypeType)}
// CHECK:STDOUT:     inst78000034:    {kind: TupleType, arg0: inst_block7800000E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000035:    {kind: Converted, arg0: inst7800002F, arg1: inst78000034, type: type(TypeType)}
// CHECK:STDOUT:     inst78000036:    {kind: TupleType, arg0: inst_block7800000F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000037:    {kind: InitForm, arg0: inst78000035, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000038:    {kind: InitForm, arg0: inst78000034, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000039:    {kind: InitForm, arg0: inst78000036, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800003A:    {kind: PatternType, arg0: inst78000034, type: type(TypeType)}
// CHECK:STDOUT:     inst7800003B:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst7800003C:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000013)}
// CHECK:STDOUT:     inst7800003D:    {kind: PatternType, arg0: inst78000036, type: type(TypeType)}
// CHECK:STDOUT:     inst7800003E:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst7800003F:    {kind: ReturnSlotPattern, arg0: inst7800003B, arg1: inst78000035, type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst78000040:    {kind: ReturnSlotPattern, arg0: inst7800003C, arg1: inst78000034, type: type(symbolic_constant78000013)}
// CHECK:STDOUT:     inst78000041:    {kind: ReturnSlotPattern, arg0: inst7800003E, arg1: inst78000036, type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst78000042:    {kind: SpliceBlock, arg0: inst_block78000005, arg1: inst78000014, type: type(TypeType)}
// CHECK:STDOUT:     inst78000043:    {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000044:    {kind: SpliceBlock, arg0: inst_block78000007, arg1: inst7800001D, type: type(TypeType)}
// CHECK:STDOUT:     inst78000045:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst78000046:    {kind: ReturnSlot, arg0: inst78000034, arg1: inst78000045, type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst78000047:    {kind: FunctionDecl, arg0: function78000000, arg1: inst_block78000013, type: type(inst78000048)}
// CHECK:STDOUT:     inst78000048:    {kind: FunctionType, arg0: function78000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000049:    {kind: StructValue, arg0: inst_block_empty, type: type(inst78000048)}
// CHECK:STDOUT:     inst7800004A:    {kind: RequireCompleteType, arg0: inst7800001E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004B:    {kind: RequireCompleteType, arg0: inst7800001E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004C:    {kind: RequireCompleteType, arg0: inst7800001F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004D:    {kind: PointerType, arg0: inst78000034, type: type(TypeType)}
// CHECK:STDOUT:     inst7800004E:    {kind: RequireCompleteType, arg0: inst78000034, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004F:    {kind: RequireCompleteType, arg0: inst78000034, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000050:    {kind: RequireCompleteType, arg0: inst78000036, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000051:    {kind: NameRef, arg0: name2, arg1: inst78000028, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000052:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst7800002B)}
// CHECK:STDOUT:     inst78000053:    {kind: TupleLiteral, arg0: inst_block78000019, type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst78000054:    {kind: RequireCompleteType, arg0: inst78000034, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000055:    {kind: TupleAccess, arg0: inst78000045, arg1: element0, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000056:    {kind: RequireCompleteType, arg0: inst7800001E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000057:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name78000003, type: type(TypeType)}
// CHECK:STDOUT:     inst78000058:    {kind: InterfaceDecl, arg0: interface78000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst78000059:    {kind: FacetType, arg0: declared_facet_type78000001, type: type(TypeType)}
// CHECK:STDOUT:     inst7800005A:    {kind: InterfaceWithSelfDecl, arg0: interface78000000}
// CHECK:STDOUT:     inst7800005B:    {kind: SymbolicBinding, arg0: entity_name78000004, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800005C:    {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst78000083)}
// CHECK:STDOUT:     inst7800005D:    {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name78000005}
// CHECK:STDOUT:     inst7800005E:    {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800005F:    {kind: FunctionDecl, arg0: function78000001, arg1: inst_block_empty, type: type(symbolic_constant7800001F)}
// CHECK:STDOUT:     inst78000060:    {kind: FunctionType, arg0: function78000001, arg1: specific78000001, type: type(TypeType)}
// CHECK:STDOUT:     inst78000061:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant7800001F)}
// CHECK:STDOUT:     inst78000062:    {kind: FacetAccessType, arg0: inst7800005B, type: type(TypeType)}
// CHECK:STDOUT:     inst78000063:    {kind: PatternType, arg0: inst78000062, type: type(TypeType)}
// CHECK:STDOUT:     inst78000064:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000065:    {kind: ImportRefLoaded, arg0: import_ir_inst7, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000066:    {kind: ReturnSlotPattern, arg0: inst78000064, arg1: inst<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000067:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000068:    {kind: ImportRefLoaded, arg0: import_ir_inst8, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst78000069:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000067, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800006A:    {kind: ImportRefLoaded, arg0: import_ir_inst9, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800006B:    {kind: InitForm, arg0: inst78000062, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800006C:    {kind: ImportRefLoaded, arg0: import_ir_instA, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800006D:    {kind: ImportRefLoaded, arg0: import_ir_instB, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst7800006E:    {kind: ImportRefLoaded, arg0: import_ir_instC, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst7800006F:    {kind: ImportRefLoaded, arg0: import_ir_instD, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst78000070:    {kind: ImportRefLoaded, arg0: import_ir_instE, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst78000071:    {kind: ImportRefLoaded, arg0: import_ir_instF, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000072:    {kind: ImportRefLoaded, arg0: import_ir_inst10, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000073:    {kind: ImportRefLoaded, arg0: import_ir_inst11, arg1: entity_name<none>, type: type(symbolic_constant78000028)}
// CHECK:STDOUT:     inst78000074:    {kind: ImportRefLoaded, arg0: import_ir_inst12, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst78000075:    {kind: SymbolicBinding, arg0: entity_name78000004, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst78000076:    {kind: FacetAccessType, arg0: inst78000075, type: type(TypeType)}
// CHECK:STDOUT:     inst78000077:    {kind: PatternType, arg0: inst78000076, type: type(TypeType)}
// CHECK:STDOUT:     inst78000078:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant7800003A)}
// CHECK:STDOUT:     inst78000079:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000078, type: type(symbolic_constant7800003A)}
// CHECK:STDOUT:     inst7800007A:    {kind: InitForm, arg0: inst78000076, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800007B:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant7800003A)}
// CHECK:STDOUT:     inst7800007C:    {kind: ReturnSlotPattern, arg0: inst7800007B, arg1: inst<none>, type: type(symbolic_constant7800003A)}
// CHECK:STDOUT:     inst7800007D:    {kind: ImportRefLoaded, arg0: import_ir_inst1B, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800007E:    {kind: ImportRefLoaded, arg0: import_ir_inst1C, arg1: entity_name<none>, type: type(symbolic_constant78000023)}
// CHECK:STDOUT:     inst7800007F:    {kind: SymbolicBinding, arg0: entity_name78000004, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst78000080:    {kind: FunctionType, arg0: function78000001, arg1: specific78000003, type: type(TypeType)}
// CHECK:STDOUT:     inst78000081:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000043)}
// CHECK:STDOUT:     inst78000082:    {kind: ImportRefUnloaded, arg0: import_ir_inst20, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000083:    {kind: AssociatedEntityType, arg0: interface78000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000084:    {kind: ImportRefLoaded, arg0: import_ir_inst21, arg1: entity_name<none>, type: type(symbolic_constant78000025)}
// CHECK:STDOUT:     inst78000085:    {kind: AssociatedEntity, arg0: element0, arg1: inst78000084, type: type(inst78000083)}
// CHECK:STDOUT:     inst78000086:    {kind: LookupImplWitness, arg0: inst7800001E, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000087:    {kind: ImportRefUnloaded, arg0: import_ir_inst22, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000088:    {kind: ImplDecl, arg0: impl78000000, arg1: inst_block_empty}
// CHECK:STDOUT:     inst78000089:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800008A:    {kind: FacetAccessType, arg0: inst78000089, type: type(TypeType)}
// CHECK:STDOUT:     inst7800008B:    {kind: ConstType, arg0: inst7800008A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800008C:    {kind: ImplSelfWitness, arg0: inst7800008B, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800008D:    {kind: ImportRefUnloaded, arg0: import_ir_inst24, arg1: entity_name<none>}
// CHECK:STDOUT:     inst7800008E:    {kind: ImplWitnessTable, arg0: inst_block78000025, arg1: impl78000000}
// CHECK:STDOUT:     inst7800008F:    {kind: ImplWitness, arg0: inst7800008E, arg1: specific78000004, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000090:    {kind: FunctionDecl, arg0: function78000002, arg1: inst_block_empty, type: type(symbolic_constant7800004D)}
// CHECK:STDOUT:     inst78000091:    {kind: FunctionType, arg0: function78000002, arg1: specific78000004, type: type(TypeType)}
// CHECK:STDOUT:     inst78000092:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant7800004D)}
// CHECK:STDOUT:     inst78000093:    {kind: PatternType, arg0: inst7800008B, type: type(TypeType)}
// CHECK:STDOUT:     inst78000094:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst78000095:    {kind: PatternType, arg0: inst78000059, type: type(TypeType)}
// CHECK:STDOUT:     inst78000096:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000095)}
// CHECK:STDOUT:     inst78000097:    {kind: ImportRefLoaded, arg0: import_ir_inst27, arg1: entity_name<none>, type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst78000098:    {kind: ReturnSlotPattern, arg0: inst78000094, arg1: inst<none>, type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst78000099:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst7800009A:    {kind: ImportRefLoaded, arg0: import_ir_inst28, arg1: entity_name<none>, type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst7800009B:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000099, type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst7800009C:    {kind: ImportRefLoaded, arg0: import_ir_inst29, arg1: entity_name<none>, type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst7800009D:    {kind: InitForm, arg0: inst7800008B, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800009E:    {kind: ImportRefLoaded, arg0: import_ir_inst2A, arg1: entity_name<none>, type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst7800009F:    {kind: ImportRefLoaded, arg0: import_ir_inst2B, arg1: entity_name<none>, type: type(symbolic_constant78000056)}
// CHECK:STDOUT:     inst780000A0:    {kind: ImportRefLoaded, arg0: import_ir_inst2C, arg1: entity_name<none>, type: type(symbolic_constant78000056)}
// CHECK:STDOUT:     inst780000A1:    {kind: ImportRefLoaded, arg0: import_ir_inst2D, arg1: entity_name<none>, type: type(symbolic_constant78000056)}
// CHECK:STDOUT:     inst780000A2:    {kind: ImportRefLoaded, arg0: import_ir_inst2E, arg1: entity_name<none>, type: type(symbolic_constant78000056)}
// CHECK:STDOUT:     inst780000A3:    {kind: ImportRefLoaded, arg0: import_ir_inst2F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A4:    {kind: ImportRefLoaded, arg0: import_ir_inst30, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000A5:    {kind: ImportRefLoaded, arg0: import_ir_inst31, arg1: entity_name<none>, type: type(symbolic_constant78000056)}
// CHECK:STDOUT:     inst780000A6:    {kind: ImportRefLoaded, arg0: import_ir_inst32, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780000A7:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780000A8:    {kind: FacetAccessType, arg0: inst780000A7, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A9:    {kind: ConstType, arg0: inst780000A8, type: type(TypeType)}
// CHECK:STDOUT:     inst780000AA:    {kind: PatternType, arg0: inst780000A9, type: type(TypeType)}
// CHECK:STDOUT:     inst780000AB:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant7800006B)}
// CHECK:STDOUT:     inst780000AC:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst780000AB, type: type(symbolic_constant7800006B)}
// CHECK:STDOUT:     inst780000AD:    {kind: InitForm, arg0: inst780000A9, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000AE:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant7800006B)}
// CHECK:STDOUT:     inst780000AF:    {kind: ReturnSlotPattern, arg0: inst780000AE, arg1: inst<none>, type: type(symbolic_constant7800006B)}
// CHECK:STDOUT:     inst780000B0:    {kind: LookupImplWitness, arg0: inst78000089, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000B1:    {kind: FunctionType, arg0: function78000001, arg1: specific78000005, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B2:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000072)}
// CHECK:STDOUT:     inst780000B3:    {kind: FunctionTypeWithSelfType, arg0: inst780000B1, arg1: inst78000089, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B4:    {kind: ImplWitnessAccess, arg0: inst780000B0, arg1: element0, type: type(symbolic_constant78000074)}
// CHECK:STDOUT:     inst780000B5:    {kind: SpecificImplFunction, arg0: inst780000B4, arg1: specific78000006, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst780000B6:    {kind: PatternType, arg0: inst7800008A, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B7:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000078)}
// CHECK:STDOUT:     inst780000B8:    {kind: ImportRefLoaded, arg0: import_ir_inst3C, arg1: entity_name<none>, type: type(symbolic_constant78000078)}
// CHECK:STDOUT:     inst780000B9:    {kind: ReturnSlotPattern, arg0: inst780000B7, arg1: inst<none>, type: type(symbolic_constant78000078)}
// CHECK:STDOUT:     inst780000BA:    {kind: InitForm, arg0: inst7800008A, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000BB:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000078)}
// CHECK:STDOUT:     inst780000BC:    {kind: ImportRefLoaded, arg0: import_ir_inst3D, arg1: entity_name<none>, type: type(symbolic_constant78000078)}
// CHECK:STDOUT:     inst780000BD:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst780000BB, type: type(symbolic_constant78000078)}
// CHECK:STDOUT:     inst780000BE:    {kind: RequireCompleteType, arg0: inst7800008A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000BF:    {kind: RequireCompleteType, arg0: inst7800008B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000C0:    {kind: RequireCompleteType, arg0: inst780000A9, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000C1:    {kind: RequireCompleteType, arg0: inst780000A8, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000C2:    {kind: FunctionType, arg0: function78000001, arg1: specific78000008, type: type(TypeType)}
// CHECK:STDOUT:     inst780000C3:    {kind: FunctionTypeWithSelfType, arg0: inst780000C2, arg1: inst780000A7, type: type(TypeType)}
// CHECK:STDOUT:     inst780000C4:    {kind: LookupImplWitness, arg0: inst780000A7, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000C5:    {kind: ImplWitnessAccess, arg0: inst780000C4, arg1: element0, type: type(symbolic_constant78000089)}
// CHECK:STDOUT:     inst780000C6:    {kind: SpecificImplFunction, arg0: inst780000C5, arg1: specific78000009, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst780000C7:    {kind: ImportRefLoaded, arg0: import_ir_inst45, arg1: entity_name<none>, type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst780000C8:    {kind: ImportRefLoaded, arg0: import_ir_inst46, arg1: entity_name<none>, type: type(symbolic_constant78000050)}
// CHECK:STDOUT:     inst780000C9:    {kind: ImportRefLoaded, arg0: import_ir_inst47, arg1: entity_name<none>, type: type(inst78000095)}
// CHECK:STDOUT:     inst780000CA:    {kind: ImportRefLoaded, arg0: import_ir_inst48, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000CB:    {kind: ImportRefLoaded, arg0: import_ir_inst49, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000CC:    {kind: ImportRefLoaded, arg0: import_ir_inst4A, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000CD:    {kind: ImportRefLoaded, arg0: import_ir_inst4B, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780000CE:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000095)}
// CHECK:STDOUT:     inst780000CF:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780000D0:    {kind: FacetAccessType, arg0: inst780000CF, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D1:    {kind: ConstType, arg0: inst780000D0, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D2:    {kind: ImplSelfWitness, arg0: inst780000D1, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000D3:    {kind: ImplWitness, arg0: inst7800008E, arg1: specific7800000A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000D4:    {kind: FunctionType, arg0: function78000002, arg1: specific7800000A, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D5:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant7800009A)}
// CHECK:STDOUT:     inst780000D6:    {kind: ImportRefUnloaded, arg0: import_ir_inst54, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000D7:    {kind: ImplDecl, arg0: impl78000001, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000D8:    {kind: ImplSelfWitness, arg0: inst(BoolType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000D9:    {kind: ImportRefLoaded, arg0: import_ir_inst56, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DA:    {kind: ImportRefLoaded, arg0: import_ir_inst57, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DB:    {kind: ImportRefLoaded, arg0: import_ir_inst58, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000DC:    {kind: ImportRefUnloaded, arg0: import_ir_inst59, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000DD:    {kind: ImplDecl, arg0: impl78000002, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000DE:    {kind: ImplSelfWitness, arg0: inst(CharLiteralType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000DF:    {kind: ImportRefLoaded, arg0: import_ir_inst5B, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E0:    {kind: ImportRefLoaded, arg0: import_ir_inst5C, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E1:    {kind: ImportRefLoaded, arg0: import_ir_inst5D, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000E2:    {kind: ImportRefUnloaded, arg0: import_ir_inst5E, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000E3:    {kind: ImplDecl, arg0: impl78000003, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000E4:    {kind: ImplSelfWitness, arg0: inst(FloatLiteralType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000E5:    {kind: ImportRefLoaded, arg0: import_ir_inst60, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E6:    {kind: ImportRefLoaded, arg0: import_ir_inst61, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E7:    {kind: ImportRefLoaded, arg0: import_ir_inst62, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000E8:    {kind: ImportRefUnloaded, arg0: import_ir_inst63, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000E9:    {kind: ImplDecl, arg0: impl78000004, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000EA:    {kind: ImplSelfWitness, arg0: inst(IntLiteralType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000EB:    {kind: ImportRefLoaded, arg0: import_ir_inst65, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000EC:    {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000ED:    {kind: ImportRefLoaded, arg0: import_ir_inst67, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000EE:    {kind: ImportRefLoaded, arg0: import_ir_inst68, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000EF:    {kind: ImplDecl, arg0: impl78000005, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000F0:    {kind: ImplSelfWitness, arg0: inst7800001E, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000F1:    {kind: ImportRefUnloaded, arg0: import_ir_inst6A, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000F2:    {kind: ImplWitnessTable, arg0: inst_block78000039, arg1: impl78000005}
// CHECK:STDOUT:     inst780000F3:    {kind: ImplWitness, arg0: inst780000F2, arg1: specific7800000B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000F4:    {kind: ImportRefLoaded, arg0: import_ir_inst6C, arg1: entity_name<none>, type: type(inst78000015)}
// CHECK:STDOUT:     inst780000F5:    {kind: ImportRefLoaded, arg0: import_ir_inst6D, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F6:    {kind: ImportRefLoaded, arg0: import_ir_inst6E, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F7:    {kind: ImportRefLoaded, arg0: import_ir_inst6F, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000F8:    {kind: ImportRefLoaded, arg0: import_ir_inst70, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F9:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000015)}
// CHECK:STDOUT:     inst780000FA:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000FB:    {kind: PointerType, arg0: inst780000FA, type: type(TypeType)}
// CHECK:STDOUT:     inst780000FC:    {kind: ImplSelfWitness, arg0: inst780000FB, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000FD:    {kind: ImplWitness, arg0: inst780000F2, arg1: specific7800000C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000FE:    {kind: FunctionDecl, arg0: function78000003, arg1: inst_block_empty, type: type(symbolic_constant780000AD)}
// CHECK:STDOUT:     inst780000FF:    {kind: FunctionType, arg0: function78000003, arg1: specific7800000B, type: type(TypeType)}
// CHECK:STDOUT:     inst78000100:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780000AD)}
// CHECK:STDOUT:     inst78000101:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000102:    {kind: ImportRefLoaded, arg0: import_ir_inst77, arg1: entity_name<none>, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000103:    {kind: ReturnSlotPattern, arg0: inst78000101, arg1: inst<none>, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000104:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000105:    {kind: ImportRefLoaded, arg0: import_ir_inst78, arg1: entity_name<none>, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000106:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000104, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000107:    {kind: ImportRefLoaded, arg0: import_ir_inst79, arg1: entity_name<none>, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst78000108:    {kind: InitForm, arg0: inst7800001E, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000109:    {kind: ImportRefLoaded, arg0: import_ir_inst7A, arg1: entity_name<none>, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst7800010A:    {kind: ImportRefLoaded, arg0: import_ir_inst7B, arg1: entity_name<none>, type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst7800010B:    {kind: ImportRefLoaded, arg0: import_ir_inst7C, arg1: entity_name<none>, type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst7800010C:    {kind: ImportRefLoaded, arg0: import_ir_inst7D, arg1: entity_name<none>, type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst7800010D:    {kind: ImportRefLoaded, arg0: import_ir_inst7E, arg1: entity_name<none>, type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst7800010E:    {kind: ImportRefLoaded, arg0: import_ir_inst7F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800010F:    {kind: ImportRefLoaded, arg0: import_ir_inst80, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000110:    {kind: ImportRefLoaded, arg0: import_ir_inst81, arg1: entity_name<none>, type: type(symbolic_constant780000B4)}
// CHECK:STDOUT:     inst78000111:    {kind: ImportRefLoaded, arg0: import_ir_inst82, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000112:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000113:    {kind: PointerType, arg0: inst78000112, type: type(TypeType)}
// CHECK:STDOUT:     inst78000114:    {kind: PatternType, arg0: inst78000113, type: type(TypeType)}
// CHECK:STDOUT:     inst78000115:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant780000C6)}
// CHECK:STDOUT:     inst78000116:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000115, type: type(symbolic_constant780000C6)}
// CHECK:STDOUT:     inst78000117:    {kind: InitForm, arg0: inst78000113, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000118:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant780000C6)}
// CHECK:STDOUT:     inst78000119:    {kind: ReturnSlotPattern, arg0: inst78000118, arg1: inst<none>, type: type(symbolic_constant780000C6)}
// CHECK:STDOUT:     inst7800011A:    {kind: ImportRefLoaded, arg0: import_ir_inst8B, arg1: entity_name<none>, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst7800011B:    {kind: ImportRefLoaded, arg0: import_ir_inst8C, arg1: entity_name<none>, type: type(symbolic_constant78000007)}
// CHECK:STDOUT:     inst7800011C:    {kind: FunctionType, arg0: function78000003, arg1: specific7800000C, type: type(TypeType)}
// CHECK:STDOUT:     inst7800011D:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780000CD)}
// CHECK:STDOUT:     inst7800011E:    {kind: RequireCompleteType, arg0: inst780000FB, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800011F:    {kind: ImportRefUnloaded, arg0: import_ir_inst90, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000120:    {kind: ImplDecl, arg0: impl78000006, arg1: inst_block_empty}
// CHECK:STDOUT:     inst78000121:    {kind: ImplSelfWitness, arg0: inst(TypeType), arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000122:    {kind: ImportRefLoaded, arg0: import_ir_inst92, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000123:    {kind: ImportRefLoaded, arg0: import_ir_inst93, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000124:    {kind: ImportRefLoaded, arg0: import_ir_inst94, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000125:    {kind: ImportRefUnloaded, arg0: import_ir_inst95, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000126:    {kind: ImplDecl, arg0: impl78000007, arg1: inst_block_empty}
// CHECK:STDOUT:     inst78000127:    {kind: ImplSelfWitness, arg0: inst7800002B, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000128:    {kind: ImportRefLoaded, arg0: import_ir_inst97, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000129:    {kind: ImportRefLoaded, arg0: import_ir_inst98, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800012A:    {kind: ImportRefLoaded, arg0: import_ir_inst99, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800012B:    {kind: ImportRefUnloaded, arg0: import_ir_inst9A, arg1: entity_name<none>}
// CHECK:STDOUT:     inst7800012C:    {kind: ImplDecl, arg0: impl78000008, arg1: inst_block_empty}
// CHECK:STDOUT:     inst7800012D:    {kind: SymbolicBinding, arg0: entity_name78000021, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800012E:    {kind: FacetAccessType, arg0: inst7800012D, type: type(TypeType)}
// CHECK:STDOUT:     inst7800012F:    {kind: TupleType, arg0: inst_block78000046, type: type(TypeType)}
// CHECK:STDOUT:     inst78000130:    {kind: ImplSelfWitness, arg0: inst7800012F, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000131:    {kind: ImportRefUnloaded, arg0: import_ir_inst9C, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000132:    {kind: ImplWitnessTable, arg0: inst_block78000047, arg1: impl78000008}
// CHECK:STDOUT:     inst78000133:    {kind: ImplWitness, arg0: inst78000132, arg1: specific7800000E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000134:    {kind: FunctionDecl, arg0: function78000004, arg1: inst_block_empty, type: type(symbolic_constant780000D8)}
// CHECK:STDOUT:     inst78000135:    {kind: FunctionType, arg0: function78000004, arg1: specific7800000E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000136:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780000D8)}
// CHECK:STDOUT:     inst78000137:    {kind: PatternType, arg0: inst7800012F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000138:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst78000139:    {kind: TupleType, arg0: inst_block78000049, type: type(TypeType)}
// CHECK:STDOUT:     inst7800013A:    {kind: TupleValue, arg0: inst_block78000048, type: type(inst78000139)}
// CHECK:STDOUT:     inst7800013B:    {kind: SymbolicBindingPattern, arg0: entity_name78000021, type: type(inst78000095)}
// CHECK:STDOUT:     inst7800013C:    {kind: ImportRefLoaded, arg0: import_ir_inst9F, arg1: entity_name<none>, type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst7800013D:    {kind: ReturnSlotPattern, arg0: inst78000138, arg1: inst<none>, type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst7800013E:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst7800013F:    {kind: ImportRefLoaded, arg0: import_ir_instA0, arg1: entity_name<none>, type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst78000140:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst7800013E, type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst78000141:    {kind: ImportRefLoaded, arg0: import_ir_instA1, arg1: entity_name<none>, type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst78000142:    {kind: InitForm, arg0: inst7800012F, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000143:    {kind: ImportRefLoaded, arg0: import_ir_instA2, arg1: entity_name<none>, type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst78000144:    {kind: ImportRefLoaded, arg0: import_ir_instA3, arg1: entity_name<none>, type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst78000145:    {kind: ImportRefLoaded, arg0: import_ir_instA4, arg1: entity_name<none>, type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst78000146:    {kind: ImportRefLoaded, arg0: import_ir_instA5, arg1: entity_name<none>, type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst78000147:    {kind: ImportRefLoaded, arg0: import_ir_instA6, arg1: entity_name<none>, type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst78000148:    {kind: ImportRefLoaded, arg0: import_ir_instA7, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000149:    {kind: ImportRefLoaded, arg0: import_ir_instA8, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800014A:    {kind: ImportRefLoaded, arg0: import_ir_instA9, arg1: entity_name<none>, type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst7800014B:    {kind: ImportRefLoaded, arg0: import_ir_instAA, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800014C:    {kind: ImportRefLoaded, arg0: import_ir_instAB, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800014D:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800014E:    {kind: FacetAccessType, arg0: inst7800014D, type: type(TypeType)}
// CHECK:STDOUT:     inst7800014F:    {kind: SymbolicBinding, arg0: entity_name78000021, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst78000150:    {kind: FacetAccessType, arg0: inst7800014F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000151:    {kind: TupleType, arg0: inst_block7800004F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000152:    {kind: PatternType, arg0: inst78000151, type: type(TypeType)}
// CHECK:STDOUT:     inst78000153:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant780000FC)}
// CHECK:STDOUT:     inst78000154:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000153, type: type(symbolic_constant780000FC)}
// CHECK:STDOUT:     inst78000155:    {kind: InitForm, arg0: inst78000151, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000156:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant780000FC)}
// CHECK:STDOUT:     inst78000157:    {kind: ReturnSlotPattern, arg0: inst78000156, arg1: inst<none>, type: type(symbolic_constant780000FC)}
// CHECK:STDOUT:     inst78000158:    {kind: LookupImplWitness, arg0: inst7800012D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000159:    {kind: FunctionType, arg0: function78000001, arg1: specific7800000F, type: type(TypeType)}
// CHECK:STDOUT:     inst7800015A:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000103)}
// CHECK:STDOUT:     inst7800015B:    {kind: FunctionTypeWithSelfType, arg0: inst78000159, arg1: inst7800012D, type: type(TypeType)}
// CHECK:STDOUT:     inst7800015C:    {kind: ImplWitnessAccess, arg0: inst78000158, arg1: element0, type: type(symbolic_constant78000105)}
// CHECK:STDOUT:     inst7800015D:    {kind: SpecificImplFunction, arg0: inst7800015C, arg1: specific78000010, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst7800015E:    {kind: PatternType, arg0: inst7800012E, type: type(TypeType)}
// CHECK:STDOUT:     inst7800015F:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000109)}
// CHECK:STDOUT:     inst78000160:    {kind: ImportRefLoaded, arg0: import_ir_instB7, arg1: entity_name<none>, type: type(symbolic_constant78000109)}
// CHECK:STDOUT:     inst78000161:    {kind: ReturnSlotPattern, arg0: inst7800015F, arg1: inst<none>, type: type(symbolic_constant78000109)}
// CHECK:STDOUT:     inst78000162:    {kind: InitForm, arg0: inst7800012E, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000163:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000109)}
// CHECK:STDOUT:     inst78000164:    {kind: ImportRefLoaded, arg0: import_ir_instB8, arg1: entity_name<none>, type: type(symbolic_constant78000109)}
// CHECK:STDOUT:     inst78000165:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst78000163, type: type(symbolic_constant78000109)}
// CHECK:STDOUT:     inst78000166:    {kind: RequireCompleteType, arg0: inst7800012E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000167:    {kind: RequireCompleteType, arg0: inst7800012F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000168:    {kind: RequireCompleteType, arg0: inst78000151, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000169:    {kind: RequireCompleteType, arg0: inst7800014E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800016A:    {kind: FunctionType, arg0: function78000001, arg1: specific78000012, type: type(TypeType)}
// CHECK:STDOUT:     inst7800016B:    {kind: FunctionTypeWithSelfType, arg0: inst7800016A, arg1: inst7800014D, type: type(TypeType)}
// CHECK:STDOUT:     inst7800016C:    {kind: LookupImplWitness, arg0: inst7800014D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800016D:    {kind: ImplWitnessAccess, arg0: inst7800016C, arg1: element0, type: type(symbolic_constant78000120)}
// CHECK:STDOUT:     inst7800016E:    {kind: SpecificImplFunction, arg0: inst7800016D, arg1: specific78000013, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst7800016F:    {kind: RequireCompleteType, arg0: inst78000150, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000170:    {kind: FunctionType, arg0: function78000001, arg1: specific78000014, type: type(TypeType)}
// CHECK:STDOUT:     inst78000171:    {kind: FunctionTypeWithSelfType, arg0: inst78000170, arg1: inst7800014F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000172:    {kind: LookupImplWitness, arg0: inst7800014F, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000173:    {kind: ImplWitnessAccess, arg0: inst78000172, arg1: element0, type: type(symbolic_constant78000126)}
// CHECK:STDOUT:     inst78000174:    {kind: SpecificImplFunction, arg0: inst78000173, arg1: specific78000015, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000175:    {kind: ImportRefLoaded, arg0: import_ir_instC6, arg1: entity_name<none>, type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst78000176:    {kind: ImportRefLoaded, arg0: import_ir_instC7, arg1: entity_name<none>, type: type(symbolic_constant780000DB)}
// CHECK:STDOUT:     inst78000177:    {kind: ImportRefLoaded, arg0: import_ir_instC8, arg1: entity_name<none>, type: type(inst78000095)}
// CHECK:STDOUT:     inst78000178:    {kind: ImportRefLoaded, arg0: import_ir_instC9, arg1: entity_name<none>, type: type(inst78000095)}
// CHECK:STDOUT:     inst78000179:    {kind: ImportRefLoaded, arg0: import_ir_instCA, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800017A:    {kind: ImportRefLoaded, arg0: import_ir_instCB, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800017B:    {kind: ImportRefLoaded, arg0: import_ir_instCC, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800017C:    {kind: ImportRefLoaded, arg0: import_ir_instCD, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800017D:    {kind: ImportRefLoaded, arg0: import_ir_instCE, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800017E:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000095)}
// CHECK:STDOUT:     inst7800017F:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst78000180:    {kind: SymbolicBindingPattern, arg0: entity_name78000021, type: type(inst78000095)}
// CHECK:STDOUT:     inst78000181:    {kind: SymbolicBinding, arg0: entity_name78000021, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst78000182:    {kind: TupleValue, arg0: inst_block7800005C, type: type(inst78000139)}
// CHECK:STDOUT:     inst78000183:    {kind: FacetAccessType, arg0: inst7800017F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000184:    {kind: FacetAccessType, arg0: inst78000181, type: type(TypeType)}
// CHECK:STDOUT:     inst78000185:    {kind: TupleType, arg0: inst_block7800005D, type: type(TypeType)}
// CHECK:STDOUT:     inst78000186:    {kind: ImplSelfWitness, arg0: inst78000185, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000187:    {kind: ImplWitness, arg0: inst78000132, arg1: specific78000016, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000188:    {kind: FunctionType, arg0: function78000004, arg1: specific78000016, type: type(TypeType)}
// CHECK:STDOUT:     inst78000189:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000140)}
// CHECK:STDOUT:     inst7800018A:    {kind: ImportRefUnloaded, arg0: import_ir_instDB, arg1: entity_name<none>}
// CHECK:STDOUT:     inst7800018B:    {kind: ImplDecl, arg0: impl78000009, arg1: inst_block_empty}
// CHECK:STDOUT:     inst7800018C:    {kind: SymbolicBinding, arg0: entity_name78000031, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800018D:    {kind: FacetAccessType, arg0: inst7800018C, type: type(TypeType)}
// CHECK:STDOUT:     inst7800018E:    {kind: TupleType, arg0: inst_block78000062, type: type(TypeType)}
// CHECK:STDOUT:     inst7800018F:    {kind: ImplSelfWitness, arg0: inst7800018E, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000190:    {kind: ImportRefUnloaded, arg0: import_ir_instDD, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000191:    {kind: ImplWitnessTable, arg0: inst_block78000063, arg1: impl78000009}
// CHECK:STDOUT:     inst78000192:    {kind: ImplWitness, arg0: inst78000191, arg1: specific78000017, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000193:    {kind: FunctionDecl, arg0: function78000005, arg1: inst_block_empty, type: type(symbolic_constant7800014A)}
// CHECK:STDOUT:     inst78000194:    {kind: FunctionType, arg0: function78000005, arg1: specific78000017, type: type(TypeType)}
// CHECK:STDOUT:     inst78000195:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant7800014A)}
// CHECK:STDOUT:     inst78000196:    {kind: PatternType, arg0: inst7800018E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000197:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst78000198:    {kind: TupleType, arg0: inst_block78000065, type: type(TypeType)}
// CHECK:STDOUT:     inst78000199:    {kind: TupleValue, arg0: inst_block78000064, type: type(inst78000198)}
// CHECK:STDOUT:     inst7800019A:    {kind: SymbolicBindingPattern, arg0: entity_name78000031, type: type(inst78000095)}
// CHECK:STDOUT:     inst7800019B:    {kind: ImportRefLoaded, arg0: import_ir_instE0, arg1: entity_name<none>, type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst7800019C:    {kind: ReturnSlotPattern, arg0: inst78000197, arg1: inst<none>, type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst7800019D:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst7800019E:    {kind: ImportRefLoaded, arg0: import_ir_instE1, arg1: entity_name<none>, type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst7800019F:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst7800019D, type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst780001A0:    {kind: ImportRefLoaded, arg0: import_ir_instE2, arg1: entity_name<none>, type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst780001A1:    {kind: InitForm, arg0: inst7800018E, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780001A2:    {kind: ImportRefLoaded, arg0: import_ir_instE3, arg1: entity_name<none>, type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst780001A3:    {kind: ImportRefLoaded, arg0: import_ir_instE4, arg1: entity_name<none>, type: type(symbolic_constant78000154)}
// CHECK:STDOUT:     inst780001A4:    {kind: ImportRefLoaded, arg0: import_ir_instE5, arg1: entity_name<none>, type: type(symbolic_constant78000154)}
// CHECK:STDOUT:     inst780001A5:    {kind: ImportRefLoaded, arg0: import_ir_instE6, arg1: entity_name<none>, type: type(symbolic_constant78000154)}
// CHECK:STDOUT:     inst780001A6:    {kind: ImportRefLoaded, arg0: import_ir_instE7, arg1: entity_name<none>, type: type(symbolic_constant78000154)}
// CHECK:STDOUT:     inst780001A7:    {kind: ImportRefLoaded, arg0: import_ir_instE8, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780001A8:    {kind: ImportRefLoaded, arg0: import_ir_instE9, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780001A9:    {kind: ImportRefLoaded, arg0: import_ir_instEA, arg1: entity_name<none>, type: type(symbolic_constant78000154)}
// CHECK:STDOUT:     inst780001AA:    {kind: ImportRefLoaded, arg0: import_ir_instEB, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001AB:    {kind: ImportRefLoaded, arg0: import_ir_instEC, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001AC:    {kind: ImportRefLoaded, arg0: import_ir_instED, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001AD:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001AE:    {kind: FacetAccessType, arg0: inst780001AD, type: type(TypeType)}
// CHECK:STDOUT:     inst780001AF:    {kind: SymbolicBinding, arg0: entity_name78000021, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001B0:    {kind: FacetAccessType, arg0: inst780001AF, type: type(TypeType)}
// CHECK:STDOUT:     inst780001B1:    {kind: SymbolicBinding, arg0: entity_name78000031, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001B2:    {kind: FacetAccessType, arg0: inst780001B1, type: type(TypeType)}
// CHECK:STDOUT:     inst780001B3:    {kind: TupleType, arg0: inst_block7800006B, type: type(TypeType)}
// CHECK:STDOUT:     inst780001B4:    {kind: PatternType, arg0: inst780001B3, type: type(TypeType)}
// CHECK:STDOUT:     inst780001B5:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000173)}
// CHECK:STDOUT:     inst780001B6:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst780001B5, type: type(symbolic_constant78000173)}
// CHECK:STDOUT:     inst780001B7:    {kind: InitForm, arg0: inst780001B3, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780001B8:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000173)}
// CHECK:STDOUT:     inst780001B9:    {kind: ReturnSlotPattern, arg0: inst780001B8, arg1: inst<none>, type: type(symbolic_constant78000173)}
// CHECK:STDOUT:     inst780001BA:    {kind: LookupImplWitness, arg0: inst7800018C, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001BB:    {kind: FunctionType, arg0: function78000001, arg1: specific78000018, type: type(TypeType)}
// CHECK:STDOUT:     inst780001BC:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant7800017A)}
// CHECK:STDOUT:     inst780001BD:    {kind: FunctionTypeWithSelfType, arg0: inst780001BB, arg1: inst7800018C, type: type(TypeType)}
// CHECK:STDOUT:     inst780001BE:    {kind: ImplWitnessAccess, arg0: inst780001BA, arg1: element0, type: type(symbolic_constant7800017C)}
// CHECK:STDOUT:     inst780001BF:    {kind: SpecificImplFunction, arg0: inst780001BE, arg1: specific78000019, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst780001C0:    {kind: PatternType, arg0: inst7800018D, type: type(TypeType)}
// CHECK:STDOUT:     inst780001C1:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000180)}
// CHECK:STDOUT:     inst780001C2:    {kind: ImportRefLoaded, arg0: import_ir_instFB, arg1: entity_name<none>, type: type(symbolic_constant78000180)}
// CHECK:STDOUT:     inst780001C3:    {kind: ReturnSlotPattern, arg0: inst780001C1, arg1: inst<none>, type: type(symbolic_constant78000180)}
// CHECK:STDOUT:     inst780001C4:    {kind: InitForm, arg0: inst7800018D, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780001C5:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000180)}
// CHECK:STDOUT:     inst780001C6:    {kind: ImportRefLoaded, arg0: import_ir_instFC, arg1: entity_name<none>, type: type(symbolic_constant78000180)}
// CHECK:STDOUT:     inst780001C7:    {kind: WrapperBindingPattern, arg0: entity_name78000007, arg1: inst780001C5, type: type(symbolic_constant78000180)}
// CHECK:STDOUT:     inst780001C8:    {kind: RequireCompleteType, arg0: inst7800018D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001C9:    {kind: RequireCompleteType, arg0: inst7800018E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001CA:    {kind: RequireCompleteType, arg0: inst780001B3, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001CB:    {kind: RequireCompleteType, arg0: inst780001AE, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001CC:    {kind: FunctionType, arg0: function78000001, arg1: specific7800001B, type: type(TypeType)}
// CHECK:STDOUT:     inst780001CD:    {kind: FunctionTypeWithSelfType, arg0: inst780001CC, arg1: inst780001AD, type: type(TypeType)}
// CHECK:STDOUT:     inst780001CE:    {kind: LookupImplWitness, arg0: inst780001AD, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001CF:    {kind: ImplWitnessAccess, arg0: inst780001CE, arg1: element0, type: type(symbolic_constant7800019D)}
// CHECK:STDOUT:     inst780001D0:    {kind: SpecificImplFunction, arg0: inst780001CF, arg1: specific7800001C, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst780001D1:    {kind: RequireCompleteType, arg0: inst780001B0, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001D2:    {kind: FunctionType, arg0: function78000001, arg1: specific7800001D, type: type(TypeType)}
// CHECK:STDOUT:     inst780001D3:    {kind: FunctionTypeWithSelfType, arg0: inst780001D2, arg1: inst780001AF, type: type(TypeType)}
// CHECK:STDOUT:     inst780001D4:    {kind: LookupImplWitness, arg0: inst780001AF, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001D5:    {kind: ImplWitnessAccess, arg0: inst780001D4, arg1: element0, type: type(symbolic_constant780001A3)}
// CHECK:STDOUT:     inst780001D6:    {kind: SpecificImplFunction, arg0: inst780001D5, arg1: specific7800001E, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst780001D7:    {kind: RequireCompleteType, arg0: inst780001B2, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001D8:    {kind: FunctionType, arg0: function78000001, arg1: specific7800001F, type: type(TypeType)}
// CHECK:STDOUT:     inst780001D9:    {kind: FunctionTypeWithSelfType, arg0: inst780001D8, arg1: inst780001B1, type: type(TypeType)}
// CHECK:STDOUT:     inst780001DA:    {kind: LookupImplWitness, arg0: inst780001B1, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001DB:    {kind: ImplWitnessAccess, arg0: inst780001DA, arg1: element0, type: type(symbolic_constant780001A9)}
// CHECK:STDOUT:     inst780001DC:    {kind: SpecificImplFunction, arg0: inst780001DB, arg1: specific78000020, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst780001DD:    {kind: ImportRefLoaded, arg0: import_ir_inst110, arg1: entity_name<none>, type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst780001DE:    {kind: ImportRefLoaded, arg0: import_ir_inst111, arg1: entity_name<none>, type: type(symbolic_constant7800014D)}
// CHECK:STDOUT:     inst780001DF:    {kind: ImportRefLoaded, arg0: import_ir_inst112, arg1: entity_name<none>, type: type(inst78000095)}
// CHECK:STDOUT:     inst780001E0:    {kind: ImportRefLoaded, arg0: import_ir_inst113, arg1: entity_name<none>, type: type(inst78000095)}
// CHECK:STDOUT:     inst780001E1:    {kind: ImportRefLoaded, arg0: import_ir_inst114, arg1: entity_name<none>, type: type(inst78000095)}
// CHECK:STDOUT:     inst780001E2:    {kind: ImportRefLoaded, arg0: import_ir_inst115, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780001E3:    {kind: ImportRefLoaded, arg0: import_ir_inst116, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780001E4:    {kind: ImportRefLoaded, arg0: import_ir_inst117, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001E5:    {kind: ImportRefLoaded, arg0: import_ir_inst118, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001E6:    {kind: ImportRefLoaded, arg0: import_ir_inst119, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001E7:    {kind: ImportRefLoaded, arg0: import_ir_inst11A, arg1: entity_name<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001E8:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000095)}
// CHECK:STDOUT:     inst780001E9:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001EA:    {kind: SymbolicBindingPattern, arg0: entity_name78000021, type: type(inst78000095)}
// CHECK:STDOUT:     inst780001EB:    {kind: SymbolicBinding, arg0: entity_name78000021, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001EC:    {kind: SymbolicBindingPattern, arg0: entity_name78000031, type: type(inst78000095)}
// CHECK:STDOUT:     inst780001ED:    {kind: SymbolicBinding, arg0: entity_name78000031, arg1: inst<none>, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001EE:    {kind: TupleValue, arg0: inst_block7800007A, type: type(inst78000198)}
// CHECK:STDOUT:     inst780001EF:    {kind: FacetAccessType, arg0: inst780001E9, type: type(TypeType)}
// CHECK:STDOUT:     inst780001F0:    {kind: FacetAccessType, arg0: inst780001EB, type: type(TypeType)}
// CHECK:STDOUT:     inst780001F1:    {kind: FacetAccessType, arg0: inst780001ED, type: type(TypeType)}
// CHECK:STDOUT:     inst780001F2:    {kind: TupleType, arg0: inst_block7800007B, type: type(TypeType)}
// CHECK:STDOUT:     inst780001F3:    {kind: ImplSelfWitness, arg0: inst780001F2, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001F4:    {kind: ImplWitness, arg0: inst78000191, arg1: specific78000021, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001F5:    {kind: FunctionType, arg0: function78000005, arg1: specific78000021, type: type(TypeType)}
// CHECK:STDOUT:     inst780001F6:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780001CA)}
// CHECK:STDOUT:     inst780001F7:    {kind: LookupImplWitness, arg0: inst7800001E, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780001F8:    {kind: RequireSpecificDefinition, arg0: specific7800000B, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     inst780001F9:    {kind: RequireSpecificDefinition, arg0: specific7800000B, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     inst780001FA:    {kind: RequireSpecificDefinition, arg0: specific78000022, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     inst780001FB:    {kind: FacetValue, arg0: inst7800001E, arg1: inst_block78000082, type: type(inst78000059)}
// CHECK:STDOUT:     inst780001FC:    {kind: FunctionType, arg0: function78000001, arg1: specific78000023, type: type(TypeType)}
// CHECK:STDOUT:     inst780001FD:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780001D1)}
// CHECK:STDOUT:     inst780001FE:    {kind: FunctionTypeWithSelfType, arg0: inst780001FC, arg1: inst780001FB, type: type(TypeType)}
// CHECK:STDOUT:     inst780001FF:    {kind: ImplWitnessAccess, arg0: inst780001F7, arg1: element0, type: type(symbolic_constant780001D8)}
// CHECK:STDOUT:     inst78000200:    {kind: ImplWitnessAccess, arg0: inst780001F7, arg1: element0, type: type(symbolic_constant780001D3)}
// CHECK:STDOUT:     inst78000201:    {kind: LookupImplWitness, arg0: inst7800001F, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000202:    {kind: FacetValue, arg0: inst7800001F, arg1: inst_block78000085, type: type(inst78000059)}
// CHECK:STDOUT:     inst78000203:    {kind: FunctionType, arg0: function78000001, arg1: specific78000024, type: type(TypeType)}
// CHECK:STDOUT:     inst78000204:    {kind: FunctionTypeWithSelfType, arg0: inst78000203, arg1: inst78000202, type: type(TypeType)}
// CHECK:STDOUT:     inst78000205:    {kind: ImplWitnessAccess, arg0: inst78000201, arg1: element0, type: type(symbolic_constant780001D8)}
// CHECK:STDOUT:     inst78000206:    {kind: BoundMethod, arg0: inst78000051, arg1: inst780001FF, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst78000207:    {kind: LookupImplWitness, arg0: inst7800001E, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000208:    {kind: RequireSpecificDefinition, arg0: specific7800000B, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     inst78000209:    {kind: FacetValue, arg0: inst7800001E, arg1: inst_block78000082, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800020A:    {kind: Converted, arg0: inst7800001E, arg1: inst78000209, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800020B:    {kind: LookupImplWitness, arg0: inst7800001E, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800020C:    {kind: RequireSpecificDefinition, arg0: specific7800000B, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     inst7800020D:    {kind: FacetValue, arg0: inst7800001E, arg1: inst_block78000082, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800020E:    {kind: Converted, arg0: inst7800001E, arg1: inst7800020D, type: type(inst78000059)}
// CHECK:STDOUT:     inst7800020F:    {kind: SpecificImplFunction, arg0: inst780001FF, arg1: specific78000025, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000210:    {kind: SpecificImplFunction, arg0: inst78000200, arg1: specific78000025, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000211:    {kind: SpecificImplFunction, arg0: inst78000205, arg1: specific78000026, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000212:    {kind: BoundMethod, arg0: inst78000051, arg1: inst7800020F, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst78000213:    {kind: Call, arg0: inst78000212, arg1: inst_block78000089, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000214:    {kind: InPlaceInit, arg0: inst78000213, arg1: inst78000055, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000215:    {kind: TupleAccess, arg0: inst78000045, arg1: element1, type: type(inst7800002B)}
// CHECK:STDOUT:     inst78000216:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst78000215, type: type(inst7800002B)}
// CHECK:STDOUT:     inst78000217:    {kind: Converted, arg0: inst78000052, arg1: inst78000216, type: type(inst7800002B)}
// CHECK:STDOUT:     inst78000218:    {kind: InPlaceInit, arg0: inst78000217, arg1: inst78000215, type: type(inst7800002B)}
// CHECK:STDOUT:     inst78000219:    {kind: TupleInit, arg0: inst_block7800008A, arg1: inst78000045, type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst7800021A:    {kind: Converted, arg0: inst78000053, arg1: inst78000219, type: type(symbolic_constant78000010)}
// CHECK:STDOUT:     inst7800021B:    {kind: ReturnExpr, arg0: inst7800021A, arg1: inst78000045}
// CHECK:STDOUT:   bundles:         {}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       'inst(TypeType)':  concrete_constant(inst(TypeType))
// CHECK:STDOUT:       'inst(Package)':   concrete_constant(inst(Package))
// CHECK:STDOUT:       inst78000011:    concrete_constant(inst78000011)
// CHECK:STDOUT:       inst78000012:    symbolic_constant78000000
// CHECK:STDOUT:       inst78000013:    symbolic_constant78000000
// CHECK:STDOUT:       inst78000014:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst78000015:    concrete_constant(inst78000015)
// CHECK:STDOUT:       inst78000016:    symbolic_constant78000002
// CHECK:STDOUT:       inst78000017:    symbolic_constant78000001
// CHECK:STDOUT:       inst78000018:    symbolic_constant78000002
// CHECK:STDOUT:       inst78000019:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800001A:    symbolic_constant78000003
// CHECK:STDOUT:       inst7800001B:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800001C:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800001D:    symbolic_constant78000006
// CHECK:STDOUT:       inst7800001E:    symbolic_constant78000005
// CHECK:STDOUT:       inst7800001F:    symbolic_constant78000006
// CHECK:STDOUT:       inst78000020:    symbolic_constant78000007
// CHECK:STDOUT:       inst78000021:    symbolic_constant7800000A
// CHECK:STDOUT:       inst78000022:    symbolic_constant78000008
// CHECK:STDOUT:       inst78000023:    symbolic_constant78000009
// CHECK:STDOUT:       inst78000024:    symbolic_constant7800000A
// CHECK:STDOUT:       inst78000025:    symbolic_constant7800000C
// CHECK:STDOUT:       inst78000026:    symbolic_constant7800000B
// CHECK:STDOUT:       inst78000027:    symbolic_constant7800000C
// CHECK:STDOUT:       inst78000029:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800002A:    symbolic_constant78000006
// CHECK:STDOUT:       inst7800002B:    concrete_constant(inst7800002B)
// CHECK:STDOUT:       inst7800002C:    concrete_constant(inst7800002D)
// CHECK:STDOUT:       inst7800002D:    concrete_constant(inst7800002D)
// CHECK:STDOUT:       inst7800002E:    concrete_constant(inst7800002E)
// CHECK:STDOUT:       inst7800002F:    symbolic_constant7800000E
// CHECK:STDOUT:       inst78000030:    symbolic_constant7800000D
// CHECK:STDOUT:       inst78000031:    symbolic_constant7800000E
// CHECK:STDOUT:       inst78000032:    concrete_constant(inst78000032)
// CHECK:STDOUT:       inst78000033:    concrete_constant(inst7800002B)
// CHECK:STDOUT:       inst78000034:    symbolic_constant7800000F
// CHECK:STDOUT:       inst78000035:    symbolic_constant78000010
// CHECK:STDOUT:       inst78000036:    symbolic_constant78000010
// CHECK:STDOUT:       inst78000037:    symbolic_constant78000012
// CHECK:STDOUT:       inst78000038:    symbolic_constant78000011
// CHECK:STDOUT:       inst78000039:    symbolic_constant78000012
// CHECK:STDOUT:       inst7800003A:    symbolic_constant78000013
// CHECK:STDOUT:       inst7800003B:    symbolic_constant78000016
// CHECK:STDOUT:       inst7800003C:    symbolic_constant78000014
// CHECK:STDOUT:       inst7800003D:    symbolic_constant78000015
// CHECK:STDOUT:       inst7800003E:    symbolic_constant78000016
// CHECK:STDOUT:       inst7800003F:    symbolic_constant78000018
// CHECK:STDOUT:       inst78000040:    symbolic_constant78000017
// CHECK:STDOUT:       inst78000041:    symbolic_constant78000018
// CHECK:STDOUT:       inst78000042:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst78000044:    symbolic_constant78000006
// CHECK:STDOUT:       inst78000047:    concrete_constant(inst78000049)
// CHECK:STDOUT:       inst78000048:    concrete_constant(inst78000048)
// CHECK:STDOUT:       inst78000049:    concrete_constant(inst78000049)
// CHECK:STDOUT:       inst7800004A:    symbolic_constant7800001A
// CHECK:STDOUT:       inst7800004B:    symbolic_constant78000019
// CHECK:STDOUT:       inst7800004C:    symbolic_constant7800001A
// CHECK:STDOUT:       inst7800004D:    symbolic_constant7800001B
// CHECK:STDOUT:       inst7800004E:    symbolic_constant7800001D
// CHECK:STDOUT:       inst7800004F:    symbolic_constant7800001C
// CHECK:STDOUT:       inst78000050:    symbolic_constant7800001D
// CHECK:STDOUT:       inst78000052:    concrete_constant(inst7800002D)
// CHECK:STDOUT:       inst78000054:    symbolic_constant7800001D
// CHECK:STDOUT:       inst78000056:    symbolic_constant7800001A
// CHECK:STDOUT:       inst78000057:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst78000058:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst78000059:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst7800005A:    concrete_constant(inst7800005A)
// CHECK:STDOUT:       inst7800005B:    symbolic_constant7800001E
// CHECK:STDOUT:       inst7800005C:    concrete_constant(inst78000085)
// CHECK:STDOUT:       inst7800005D:    constant<none>
// CHECK:STDOUT:       inst7800005E:    symbolic_constant7800001E
// CHECK:STDOUT:       inst7800005F:    symbolic_constant78000021
// CHECK:STDOUT:       inst78000060:    symbolic_constant7800001F
// CHECK:STDOUT:       inst78000061:    symbolic_constant78000020
// CHECK:STDOUT:       inst78000062:    symbolic_constant78000022
// CHECK:STDOUT:       inst78000063:    symbolic_constant78000023
// CHECK:STDOUT:       inst78000064:    symbolic_constant78000024
// CHECK:STDOUT:       inst78000065:    symbolic_constant78000024
// CHECK:STDOUT:       inst78000066:    symbolic_constant78000026
// CHECK:STDOUT:       inst78000067:    symbolic_constant78000029
// CHECK:STDOUT:       inst78000068:    symbolic_constant78000029
// CHECK:STDOUT:       inst78000069:    symbolic_constant7800002A
// CHECK:STDOUT:       inst7800006A:    symbolic_constant78000024
// CHECK:STDOUT:       inst7800006B:    symbolic_constant7800002E
// CHECK:STDOUT:       inst7800006C:    symbolic_constant78000029
// CHECK:STDOUT:       inst7800006D:    symbolic_constant78000037
// CHECK:STDOUT:       inst7800006E:    symbolic_constant78000036
// CHECK:STDOUT:       inst7800006F:    symbolic_constant7800002B
// CHECK:STDOUT:       inst78000070:    symbolic_constant7800002B
// CHECK:STDOUT:       inst78000071:    symbolic_constant78000035
// CHECK:STDOUT:       inst78000072:    symbolic_constant78000034
// CHECK:STDOUT:       inst78000073:    symbolic_constant78000027
// CHECK:STDOUT:       inst78000074:    symbolic_constant7800001E
// CHECK:STDOUT:       inst78000075:    symbolic_constant78000038
// CHECK:STDOUT:       inst78000076:    symbolic_constant78000039
// CHECK:STDOUT:       inst78000077:    symbolic_constant7800003A
// CHECK:STDOUT:       inst78000078:    symbolic_constant7800003B
// CHECK:STDOUT:       inst78000079:    symbolic_constant7800003C
// CHECK:STDOUT:       inst7800007A:    symbolic_constant7800003D
// CHECK:STDOUT:       inst7800007B:    symbolic_constant7800003E
// CHECK:STDOUT:       inst7800007C:    symbolic_constant7800003F
// CHECK:STDOUT:       inst7800007D:    symbolic_constant78000024
// CHECK:STDOUT:       inst7800007E:    symbolic_constant78000029
// CHECK:STDOUT:       inst7800007F:    symbolic_constant78000042
// CHECK:STDOUT:       inst78000080:    symbolic_constant78000043
// CHECK:STDOUT:       inst78000081:    symbolic_constant78000044
// CHECK:STDOUT:       inst78000082:    constant<none>
// CHECK:STDOUT:       inst78000083:    concrete_constant(inst78000083)
// CHECK:STDOUT:       inst78000084:    symbolic_constant78000021
// CHECK:STDOUT:       inst78000085:    concrete_constant(inst78000085)
// CHECK:STDOUT:       inst78000087:    constant<none>
// CHECK:STDOUT:       inst78000088:    concrete_constant(inst78000088)
// CHECK:STDOUT:       inst78000089:    symbolic_constant78000045
// CHECK:STDOUT:       inst7800008A:    symbolic_constant78000046
// CHECK:STDOUT:       inst7800008B:    symbolic_constant78000047
// CHECK:STDOUT:       inst7800008C:    symbolic_constant78000048
// CHECK:STDOUT:       inst7800008D:    constant<none>
// CHECK:STDOUT:       inst7800008E:    concrete_constant(inst7800008E)
// CHECK:STDOUT:       inst7800008F:    symbolic_constant7800004B
// CHECK:STDOUT:       inst78000090:    symbolic_constant7800004F
// CHECK:STDOUT:       inst78000091:    symbolic_constant7800004D
// CHECK:STDOUT:       inst78000092:    symbolic_constant7800004E
// CHECK:STDOUT:       inst78000093:    symbolic_constant78000050
// CHECK:STDOUT:       inst78000094:    symbolic_constant78000051
// CHECK:STDOUT:       inst78000095:    concrete_constant(inst78000095)
// CHECK:STDOUT:       inst78000096:    symbolic_constant78000053
// CHECK:STDOUT:       inst78000097:    symbolic_constant78000051
// CHECK:STDOUT:       inst78000098:    symbolic_constant78000054
// CHECK:STDOUT:       inst78000099:    symbolic_constant78000057
// CHECK:STDOUT:       inst7800009A:    symbolic_constant78000057
// CHECK:STDOUT:       inst7800009B:    symbolic_constant78000058
// CHECK:STDOUT:       inst7800009C:    symbolic_constant78000051
// CHECK:STDOUT:       inst7800009D:    symbolic_constant7800005C
// CHECK:STDOUT:       inst7800009E:    symbolic_constant78000057
// CHECK:STDOUT:       inst7800009F:    symbolic_constant78000067
// CHECK:STDOUT:       inst780000A0:    symbolic_constant78000066
// CHECK:STDOUT:       inst780000A1:    symbolic_constant78000059
// CHECK:STDOUT:       inst780000A2:    symbolic_constant78000059
// CHECK:STDOUT:       inst780000A3:    symbolic_constant78000065
// CHECK:STDOUT:       inst780000A4:    symbolic_constant78000064
// CHECK:STDOUT:       inst780000A5:    symbolic_constant78000055
// CHECK:STDOUT:       inst780000A6:    symbolic_constant78000063
// CHECK:STDOUT:       inst780000A7:    symbolic_constant78000068
// CHECK:STDOUT:       inst780000A8:    symbolic_constant78000069
// CHECK:STDOUT:       inst780000A9:    symbolic_constant7800006A
// CHECK:STDOUT:       inst780000AA:    symbolic_constant7800006B
// CHECK:STDOUT:       inst780000AB:    symbolic_constant7800006C
// CHECK:STDOUT:       inst780000AC:    symbolic_constant7800006D
// CHECK:STDOUT:       inst780000AD:    symbolic_constant7800006E
// CHECK:STDOUT:       inst780000AE:    symbolic_constant7800006F
// CHECK:STDOUT:       inst780000AF:    symbolic_constant78000070
// CHECK:STDOUT:       inst780000B0:    symbolic_constant78000071
// CHECK:STDOUT:       inst780000B1:    symbolic_constant78000072
// CHECK:STDOUT:       inst780000B2:    symbolic_constant78000073
// CHECK:STDOUT:       inst780000B3:    symbolic_constant78000074
// CHECK:STDOUT:       inst780000B4:    symbolic_constant78000075
// CHECK:STDOUT:       inst780000B5:    symbolic_constant78000076
// CHECK:STDOUT:       inst780000B6:    symbolic_constant78000078
// CHECK:STDOUT:       inst780000B7:    symbolic_constant78000079
// CHECK:STDOUT:       inst780000B8:    symbolic_constant78000079
// CHECK:STDOUT:       inst780000B9:    symbolic_constant7800007A
// CHECK:STDOUT:       inst780000BA:    symbolic_constant7800007B
// CHECK:STDOUT:       inst780000BB:    symbolic_constant7800007C
// CHECK:STDOUT:       inst780000BC:    symbolic_constant7800007C
// CHECK:STDOUT:       inst780000BD:    symbolic_constant7800007D
// CHECK:STDOUT:       inst780000BE:    symbolic_constant78000082
// CHECK:STDOUT:       inst780000BF:    symbolic_constant78000084
// CHECK:STDOUT:       inst780000C0:    symbolic_constant78000086
// CHECK:STDOUT:       inst780000C1:    symbolic_constant78000087
// CHECK:STDOUT:       inst780000C2:    symbolic_constant78000088
// CHECK:STDOUT:       inst780000C3:    symbolic_constant78000089
// CHECK:STDOUT:       inst780000C4:    symbolic_constant7800008A
// CHECK:STDOUT:       inst780000C5:    symbolic_constant7800008B
// CHECK:STDOUT:       inst780000C6:    symbolic_constant7800008C
// CHECK:STDOUT:       inst780000C7:    symbolic_constant78000051
// CHECK:STDOUT:       inst780000C8:    symbolic_constant78000057
// CHECK:STDOUT:       inst780000C9:    symbolic_constant78000092
// CHECK:STDOUT:       inst780000CA:    symbolic_constant7800004A
// CHECK:STDOUT:       inst780000CB:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst780000CC:    symbolic_constant78000049
// CHECK:STDOUT:       inst780000CD:    symbolic_constant78000063
// CHECK:STDOUT:       inst780000CE:    symbolic_constant78000093
// CHECK:STDOUT:       inst780000CF:    symbolic_constant78000094
// CHECK:STDOUT:       inst780000D0:    symbolic_constant78000095
// CHECK:STDOUT:       inst780000D1:    symbolic_constant78000096
// CHECK:STDOUT:       inst780000D2:    symbolic_constant78000097
// CHECK:STDOUT:       inst780000D3:    symbolic_constant78000098
// CHECK:STDOUT:       inst780000D4:    symbolic_constant7800009A
// CHECK:STDOUT:       inst780000D5:    symbolic_constant7800009B
// CHECK:STDOUT:       inst780000D6:    constant<none>
// CHECK:STDOUT:       inst780000D7:    concrete_constant(inst780000D7)
// CHECK:STDOUT:       inst780000D8:    concrete_constant(inst780000D8)
// CHECK:STDOUT:       inst780000D9:    concrete_constant(inst(BoolType))
// CHECK:STDOUT:       inst780000DA:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst780000DB:    concrete_constant(inst780000D8)
// CHECK:STDOUT:       inst780000DC:    constant<none>
// CHECK:STDOUT:       inst780000DD:    concrete_constant(inst780000DD)
// CHECK:STDOUT:       inst780000DE:    concrete_constant(inst780000DE)
// CHECK:STDOUT:       inst780000DF:    concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       inst780000E0:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst780000E1:    concrete_constant(inst780000DE)
// CHECK:STDOUT:       inst780000E2:    constant<none>
// CHECK:STDOUT:       inst780000E3:    concrete_constant(inst780000E3)
// CHECK:STDOUT:       inst780000E4:    concrete_constant(inst780000E4)
// CHECK:STDOUT:       inst780000E5:    concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       inst780000E6:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst780000E7:    concrete_constant(inst780000E4)
// CHECK:STDOUT:       inst780000E8:    constant<none>
// CHECK:STDOUT:       inst780000E9:    concrete_constant(inst780000E9)
// CHECK:STDOUT:       inst780000EA:    concrete_constant(inst780000EA)
// CHECK:STDOUT:       inst780000EB:    concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       inst780000EC:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst780000ED:    concrete_constant(inst780000EA)
// CHECK:STDOUT:       inst780000EE:    symbolic_constant780001CD
// CHECK:STDOUT:       inst780000EF:    concrete_constant(inst780000EF)
// CHECK:STDOUT:       inst780000F0:    symbolic_constant7800009C
// CHECK:STDOUT:       inst780000F1:    constant<none>
// CHECK:STDOUT:       inst780000F2:    concrete_constant(inst780000F2)
// CHECK:STDOUT:       inst780000F3:    symbolic_constant7800009F
// CHECK:STDOUT:       inst780000F4:    symbolic_constant780000A6
// CHECK:STDOUT:       inst780000F5:    symbolic_constant7800009E
// CHECK:STDOUT:       inst780000F6:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst780000F7:    symbolic_constant7800009D
// CHECK:STDOUT:       inst780000F8:    symbolic_constant780000A5
// CHECK:STDOUT:       inst780000F9:    symbolic_constant780000A7
// CHECK:STDOUT:       inst780000FA:    symbolic_constant780000A8
// CHECK:STDOUT:       inst780000FB:    symbolic_constant780000A9
// CHECK:STDOUT:       inst780000FC:    symbolic_constant780000AA
// CHECK:STDOUT:       inst780000FD:    symbolic_constant780000AB
// CHECK:STDOUT:       inst780000FE:    symbolic_constant780000AF
// CHECK:STDOUT:       inst780000FF:    symbolic_constant780000AD
// CHECK:STDOUT:       inst78000100:    symbolic_constant780000AE
// CHECK:STDOUT:       inst78000101:    symbolic_constant780000B0
// CHECK:STDOUT:       inst78000102:    symbolic_constant780000B0
// CHECK:STDOUT:       inst78000103:    symbolic_constant780000B2
// CHECK:STDOUT:       inst78000104:    symbolic_constant780000B5
// CHECK:STDOUT:       inst78000105:    symbolic_constant780000B5
// CHECK:STDOUT:       inst78000106:    symbolic_constant780000B6
// CHECK:STDOUT:       inst78000107:    symbolic_constant780000B0
// CHECK:STDOUT:       inst78000108:    symbolic_constant780000BA
// CHECK:STDOUT:       inst78000109:    symbolic_constant780000B5
// CHECK:STDOUT:       inst7800010A:    symbolic_constant780000C3
// CHECK:STDOUT:       inst7800010B:    symbolic_constant780000C2
// CHECK:STDOUT:       inst7800010C:    symbolic_constant780000B7
// CHECK:STDOUT:       inst7800010D:    symbolic_constant780000B7
// CHECK:STDOUT:       inst7800010E:    symbolic_constant780000C1
// CHECK:STDOUT:       inst7800010F:    symbolic_constant780000C0
// CHECK:STDOUT:       inst78000110:    symbolic_constant780000B3
// CHECK:STDOUT:       inst78000111:    symbolic_constant780000A5
// CHECK:STDOUT:       inst78000112:    symbolic_constant780000C4
// CHECK:STDOUT:       inst78000113:    symbolic_constant780000C5
// CHECK:STDOUT:       inst78000114:    symbolic_constant780000C6
// CHECK:STDOUT:       inst78000115:    symbolic_constant780000C7
// CHECK:STDOUT:       inst78000116:    symbolic_constant780000C8
// CHECK:STDOUT:       inst78000117:    symbolic_constant780000C9
// CHECK:STDOUT:       inst78000118:    symbolic_constant780000CA
// CHECK:STDOUT:       inst78000119:    symbolic_constant780000CB
// CHECK:STDOUT:       inst7800011A:    symbolic_constant780000B0
// CHECK:STDOUT:       inst7800011B:    symbolic_constant780000B5
// CHECK:STDOUT:       inst7800011C:    symbolic_constant780000CD
// CHECK:STDOUT:       inst7800011D:    symbolic_constant780000CE
// CHECK:STDOUT:       inst7800011E:    symbolic_constant780000CF
// CHECK:STDOUT:       inst7800011F:    constant<none>
// CHECK:STDOUT:       inst78000120:    concrete_constant(inst78000120)
// CHECK:STDOUT:       inst78000121:    concrete_constant(inst78000121)
// CHECK:STDOUT:       inst78000122:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst78000123:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst78000124:    concrete_constant(inst78000121)
// CHECK:STDOUT:       inst78000125:    constant<none>
// CHECK:STDOUT:       inst78000126:    concrete_constant(inst78000126)
// CHECK:STDOUT:       inst78000127:    concrete_constant(inst78000127)
// CHECK:STDOUT:       inst78000128:    concrete_constant(inst7800002B)
// CHECK:STDOUT:       inst78000129:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst7800012A:    concrete_constant(inst78000127)
// CHECK:STDOUT:       inst7800012B:    constant<none>
// CHECK:STDOUT:       inst7800012C:    concrete_constant(inst7800012C)
// CHECK:STDOUT:       inst7800012D:    symbolic_constant780000D0
// CHECK:STDOUT:       inst7800012E:    symbolic_constant780000D1
// CHECK:STDOUT:       inst7800012F:    symbolic_constant780000D2
// CHECK:STDOUT:       inst78000130:    symbolic_constant780000D3
// CHECK:STDOUT:       inst78000131:    constant<none>
// CHECK:STDOUT:       inst78000132:    concrete_constant(inst78000132)
// CHECK:STDOUT:       inst78000133:    symbolic_constant780000D6
// CHECK:STDOUT:       inst78000134:    symbolic_constant780000DA
// CHECK:STDOUT:       inst78000135:    symbolic_constant780000D8
// CHECK:STDOUT:       inst78000136:    symbolic_constant780000D9
// CHECK:STDOUT:       inst78000137:    symbolic_constant780000DB
// CHECK:STDOUT:       inst78000138:    symbolic_constant780000DC
// CHECK:STDOUT:       inst78000139:    concrete_constant(inst78000139)
// CHECK:STDOUT:       inst7800013A:    symbolic_constant780000DE
// CHECK:STDOUT:       inst7800013B:    symbolic_constant780000DF
// CHECK:STDOUT:       inst7800013C:    symbolic_constant780000DC
// CHECK:STDOUT:       inst7800013D:    symbolic_constant780000E0
// CHECK:STDOUT:       inst7800013E:    symbolic_constant780000E3
// CHECK:STDOUT:       inst7800013F:    symbolic_constant780000E3
// CHECK:STDOUT:       inst78000140:    symbolic_constant780000E4
// CHECK:STDOUT:       inst78000141:    symbolic_constant780000DC
// CHECK:STDOUT:       inst78000142:    symbolic_constant780000E8
// CHECK:STDOUT:       inst78000143:    symbolic_constant780000E3
// CHECK:STDOUT:       inst78000144:    symbolic_constant780000F6
// CHECK:STDOUT:       inst78000145:    symbolic_constant780000F5
// CHECK:STDOUT:       inst78000146:    symbolic_constant780000E5
// CHECK:STDOUT:       inst78000147:    symbolic_constant780000E5
// CHECK:STDOUT:       inst78000148:    symbolic_constant780000F4
// CHECK:STDOUT:       inst78000149:    symbolic_constant780000F3
// CHECK:STDOUT:       inst7800014A:    symbolic_constant780000E1
// CHECK:STDOUT:       inst7800014B:    symbolic_constant780000F2
// CHECK:STDOUT:       inst7800014C:    symbolic_constant780000F1
// CHECK:STDOUT:       inst7800014D:    symbolic_constant780000F7
// CHECK:STDOUT:       inst7800014E:    symbolic_constant780000F8
// CHECK:STDOUT:       inst7800014F:    symbolic_constant780000F9
// CHECK:STDOUT:       inst78000150:    symbolic_constant780000FA
// CHECK:STDOUT:       inst78000151:    symbolic_constant780000FB
// CHECK:STDOUT:       inst78000152:    symbolic_constant780000FC
// CHECK:STDOUT:       inst78000153:    symbolic_constant780000FD
// CHECK:STDOUT:       inst78000154:    symbolic_constant780000FE
// CHECK:STDOUT:       inst78000155:    symbolic_constant780000FF
// CHECK:STDOUT:       inst78000156:    symbolic_constant78000100
// CHECK:STDOUT:       inst78000157:    symbolic_constant78000101
// CHECK:STDOUT:       inst78000158:    symbolic_constant78000102
// CHECK:STDOUT:       inst78000159:    symbolic_constant78000103
// CHECK:STDOUT:       inst7800015A:    symbolic_constant78000104
// CHECK:STDOUT:       inst7800015B:    symbolic_constant78000105
// CHECK:STDOUT:       inst7800015C:    symbolic_constant78000106
// CHECK:STDOUT:       inst7800015D:    symbolic_constant78000107
// CHECK:STDOUT:       inst7800015E:    symbolic_constant78000109
// CHECK:STDOUT:       inst7800015F:    symbolic_constant7800010A
// CHECK:STDOUT:       inst78000160:    symbolic_constant7800010A
// CHECK:STDOUT:       inst78000161:    symbolic_constant7800010B
// CHECK:STDOUT:       inst78000162:    symbolic_constant7800010C
// CHECK:STDOUT:       inst78000163:    symbolic_constant7800010D
// CHECK:STDOUT:       inst78000164:    symbolic_constant7800010D
// CHECK:STDOUT:       inst78000165:    symbolic_constant7800010E
// CHECK:STDOUT:       inst78000166:    symbolic_constant78000113
// CHECK:STDOUT:       inst78000167:    symbolic_constant7800011B
// CHECK:STDOUT:       inst78000168:    symbolic_constant7800011D
// CHECK:STDOUT:       inst78000169:    symbolic_constant7800011E
// CHECK:STDOUT:       inst7800016A:    symbolic_constant7800011F
// CHECK:STDOUT:       inst7800016B:    symbolic_constant78000120
// CHECK:STDOUT:       inst7800016C:    symbolic_constant78000121
// CHECK:STDOUT:       inst7800016D:    symbolic_constant78000122
// CHECK:STDOUT:       inst7800016E:    symbolic_constant78000123
// CHECK:STDOUT:       inst7800016F:    symbolic_constant78000124
// CHECK:STDOUT:       inst78000170:    symbolic_constant78000125
// CHECK:STDOUT:       inst78000171:    symbolic_constant78000126
// CHECK:STDOUT:       inst78000172:    symbolic_constant78000127
// CHECK:STDOUT:       inst78000173:    symbolic_constant78000128
// CHECK:STDOUT:       inst78000174:    symbolic_constant78000129
// CHECK:STDOUT:       inst78000175:    symbolic_constant780000DC
// CHECK:STDOUT:       inst78000176:    symbolic_constant780000E3
// CHECK:STDOUT:       inst78000177:    symbolic_constant78000134
// CHECK:STDOUT:       inst78000178:    symbolic_constant78000133
// CHECK:STDOUT:       inst78000179:    symbolic_constant780000D5
// CHECK:STDOUT:       inst7800017A:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst7800017B:    symbolic_constant780000D4
// CHECK:STDOUT:       inst7800017C:    symbolic_constant780000F2
// CHECK:STDOUT:       inst7800017D:    symbolic_constant780000F1
// CHECK:STDOUT:       inst7800017E:    symbolic_constant78000135
// CHECK:STDOUT:       inst7800017F:    symbolic_constant78000136
// CHECK:STDOUT:       inst78000180:    symbolic_constant78000137
// CHECK:STDOUT:       inst78000181:    symbolic_constant78000138
// CHECK:STDOUT:       inst78000182:    symbolic_constant78000139
// CHECK:STDOUT:       inst78000183:    symbolic_constant7800013A
// CHECK:STDOUT:       inst78000184:    symbolic_constant7800013B
// CHECK:STDOUT:       inst78000185:    symbolic_constant7800013C
// CHECK:STDOUT:       inst78000186:    symbolic_constant7800013D
// CHECK:STDOUT:       inst78000187:    symbolic_constant7800013E
// CHECK:STDOUT:       inst78000188:    symbolic_constant78000140
// CHECK:STDOUT:       inst78000189:    symbolic_constant78000141
// CHECK:STDOUT:       inst7800018A:    constant<none>
// CHECK:STDOUT:       inst7800018B:    concrete_constant(inst7800018B)
// CHECK:STDOUT:       inst7800018C:    symbolic_constant78000142
// CHECK:STDOUT:       inst7800018D:    symbolic_constant78000143
// CHECK:STDOUT:       inst7800018E:    symbolic_constant78000144
// CHECK:STDOUT:       inst7800018F:    symbolic_constant78000145
// CHECK:STDOUT:       inst78000190:    constant<none>
// CHECK:STDOUT:       inst78000191:    concrete_constant(inst78000191)
// CHECK:STDOUT:       inst78000192:    symbolic_constant78000148
// CHECK:STDOUT:       inst78000193:    symbolic_constant7800014C
// CHECK:STDOUT:       inst78000194:    symbolic_constant7800014A
// CHECK:STDOUT:       inst78000195:    symbolic_constant7800014B
// CHECK:STDOUT:       inst78000196:    symbolic_constant7800014D
// CHECK:STDOUT:       inst78000197:    symbolic_constant7800014E
// CHECK:STDOUT:       inst78000198:    concrete_constant(inst78000198)
// CHECK:STDOUT:       inst78000199:    symbolic_constant78000150
// CHECK:STDOUT:       inst7800019A:    symbolic_constant78000151
// CHECK:STDOUT:       inst7800019B:    symbolic_constant7800014E
// CHECK:STDOUT:       inst7800019C:    symbolic_constant78000152
// CHECK:STDOUT:       inst7800019D:    symbolic_constant78000155
// CHECK:STDOUT:       inst7800019E:    symbolic_constant78000155
// CHECK:STDOUT:       inst7800019F:    symbolic_constant78000156
// CHECK:STDOUT:       inst780001A0:    symbolic_constant7800014E
// CHECK:STDOUT:       inst780001A1:    symbolic_constant7800015A
// CHECK:STDOUT:       inst780001A2:    symbolic_constant78000155
// CHECK:STDOUT:       inst780001A3:    symbolic_constant7800016B
// CHECK:STDOUT:       inst780001A4:    symbolic_constant7800016A
// CHECK:STDOUT:       inst780001A5:    symbolic_constant78000157
// CHECK:STDOUT:       inst780001A6:    symbolic_constant78000157
// CHECK:STDOUT:       inst780001A7:    symbolic_constant78000169
// CHECK:STDOUT:       inst780001A8:    symbolic_constant78000168
// CHECK:STDOUT:       inst780001A9:    symbolic_constant78000153
// CHECK:STDOUT:       inst780001AA:    symbolic_constant78000167
// CHECK:STDOUT:       inst780001AB:    symbolic_constant78000166
// CHECK:STDOUT:       inst780001AC:    symbolic_constant78000165
// CHECK:STDOUT:       inst780001AD:    symbolic_constant7800016C
// CHECK:STDOUT:       inst780001AE:    symbolic_constant7800016D
// CHECK:STDOUT:       inst780001AF:    symbolic_constant7800016E
// CHECK:STDOUT:       inst780001B0:    symbolic_constant7800016F
// CHECK:STDOUT:       inst780001B1:    symbolic_constant78000170
// CHECK:STDOUT:       inst780001B2:    symbolic_constant78000171
// CHECK:STDOUT:       inst780001B3:    symbolic_constant78000172
// CHECK:STDOUT:       inst780001B4:    symbolic_constant78000173
// CHECK:STDOUT:       inst780001B5:    symbolic_constant78000174
// CHECK:STDOUT:       inst780001B6:    symbolic_constant78000175
// CHECK:STDOUT:       inst780001B7:    symbolic_constant78000176
// CHECK:STDOUT:       inst780001B8:    symbolic_constant78000177
// CHECK:STDOUT:       inst780001B9:    symbolic_constant78000178
// CHECK:STDOUT:       inst780001BA:    symbolic_constant78000179
// CHECK:STDOUT:       inst780001BB:    symbolic_constant7800017A
// CHECK:STDOUT:       inst780001BC:    symbolic_constant7800017B
// CHECK:STDOUT:       inst780001BD:    symbolic_constant7800017C
// CHECK:STDOUT:       inst780001BE:    symbolic_constant7800017D
// CHECK:STDOUT:       inst780001BF:    symbolic_constant7800017E
// CHECK:STDOUT:       inst780001C0:    symbolic_constant78000180
// CHECK:STDOUT:       inst780001C1:    symbolic_constant78000181
// CHECK:STDOUT:       inst780001C2:    symbolic_constant78000181
// CHECK:STDOUT:       inst780001C3:    symbolic_constant78000182
// CHECK:STDOUT:       inst780001C4:    symbolic_constant78000183
// CHECK:STDOUT:       inst780001C5:    symbolic_constant78000184
// CHECK:STDOUT:       inst780001C6:    symbolic_constant78000184
// CHECK:STDOUT:       inst780001C7:    symbolic_constant78000185
// CHECK:STDOUT:       inst780001C8:    symbolic_constant7800018A
// CHECK:STDOUT:       inst780001C9:    symbolic_constant78000198
// CHECK:STDOUT:       inst780001CA:    symbolic_constant7800019A
// CHECK:STDOUT:       inst780001CB:    symbolic_constant7800019B
// CHECK:STDOUT:       inst780001CC:    symbolic_constant7800019C
// CHECK:STDOUT:       inst780001CD:    symbolic_constant7800019D
// CHECK:STDOUT:       inst780001CE:    symbolic_constant7800019E
// CHECK:STDOUT:       inst780001CF:    symbolic_constant7800019F
// CHECK:STDOUT:       inst780001D0:    symbolic_constant780001A0
// CHECK:STDOUT:       inst780001D1:    symbolic_constant780001A1
// CHECK:STDOUT:       inst780001D2:    symbolic_constant780001A2
// CHECK:STDOUT:       inst780001D3:    symbolic_constant780001A3
// CHECK:STDOUT:       inst780001D4:    symbolic_constant780001A4
// CHECK:STDOUT:       inst780001D5:    symbolic_constant780001A5
// CHECK:STDOUT:       inst780001D6:    symbolic_constant780001A6
// CHECK:STDOUT:       inst780001D7:    symbolic_constant780001A7
// CHECK:STDOUT:       inst780001D8:    symbolic_constant780001A8
// CHECK:STDOUT:       inst780001D9:    symbolic_constant780001A9
// CHECK:STDOUT:       inst780001DA:    symbolic_constant780001AA
// CHECK:STDOUT:       inst780001DB:    symbolic_constant780001AB
// CHECK:STDOUT:       inst780001DC:    symbolic_constant780001AC
// CHECK:STDOUT:       inst780001DD:    symbolic_constant7800014E
// CHECK:STDOUT:       inst780001DE:    symbolic_constant78000155
// CHECK:STDOUT:       inst780001DF:    symbolic_constant780001BB
// CHECK:STDOUT:       inst780001E0:    symbolic_constant780001BA
// CHECK:STDOUT:       inst780001E1:    symbolic_constant780001B9
// CHECK:STDOUT:       inst780001E2:    symbolic_constant78000147
// CHECK:STDOUT:       inst780001E3:    concrete_constant(inst78000059)
// CHECK:STDOUT:       inst780001E4:    symbolic_constant78000146
// CHECK:STDOUT:       inst780001E5:    symbolic_constant78000167
// CHECK:STDOUT:       inst780001E6:    symbolic_constant78000166
// CHECK:STDOUT:       inst780001E7:    symbolic_constant78000165
// CHECK:STDOUT:       inst780001E8:    symbolic_constant780001BC
// CHECK:STDOUT:       inst780001E9:    symbolic_constant780001BD
// CHECK:STDOUT:       inst780001EA:    symbolic_constant780001BE
// CHECK:STDOUT:       inst780001EB:    symbolic_constant780001BF
// CHECK:STDOUT:       inst780001EC:    symbolic_constant780001C0
// CHECK:STDOUT:       inst780001ED:    symbolic_constant780001C1
// CHECK:STDOUT:       inst780001EE:    symbolic_constant780001C2
// CHECK:STDOUT:       inst780001EF:    symbolic_constant780001C3
// CHECK:STDOUT:       inst780001F0:    symbolic_constant780001C4
// CHECK:STDOUT:       inst780001F1:    symbolic_constant780001C5
// CHECK:STDOUT:       inst780001F2:    symbolic_constant780001C6
// CHECK:STDOUT:       inst780001F3:    symbolic_constant780001C7
// CHECK:STDOUT:       inst780001F4:    symbolic_constant780001C8
// CHECK:STDOUT:       inst780001F5:    symbolic_constant780001CA
// CHECK:STDOUT:       inst780001F6:    symbolic_constant780001CB
// CHECK:STDOUT:       inst780001F7:    symbolic_constant780001CC
// CHECK:STDOUT:       inst780001F8:    symbolic_constant780001CF
// CHECK:STDOUT:       inst780001F9:    symbolic_constant780001CE
// CHECK:STDOUT:       inst780001FA:    symbolic_constant780001CF
// CHECK:STDOUT:       inst780001FB:    symbolic_constant780001D0
// CHECK:STDOUT:       inst780001FC:    symbolic_constant780001D1
// CHECK:STDOUT:       inst780001FD:    symbolic_constant780001D2
// CHECK:STDOUT:       inst780001FE:    symbolic_constant780001D3
// CHECK:STDOUT:       inst780001FF:    symbolic_constant780001D9
// CHECK:STDOUT:       inst78000200:    symbolic_constant780001D4
// CHECK:STDOUT:       inst78000201:    symbolic_constant780001D5
// CHECK:STDOUT:       inst78000202:    symbolic_constant780001D6
// CHECK:STDOUT:       inst78000203:    symbolic_constant780001D7
// CHECK:STDOUT:       inst78000204:    symbolic_constant780001D8
// CHECK:STDOUT:       inst78000205:    symbolic_constant780001D9
// CHECK:STDOUT:       inst78000208:    symbolic_constant780001CF
// CHECK:STDOUT:       inst78000209:    symbolic_constant780001D6
// CHECK:STDOUT:       inst7800020A:    symbolic_constant780001D6
// CHECK:STDOUT:       inst7800020C:    symbolic_constant780001CF
// CHECK:STDOUT:       inst7800020D:    symbolic_constant780001D6
// CHECK:STDOUT:       inst7800020E:    symbolic_constant780001D6
// CHECK:STDOUT:       inst7800020F:    symbolic_constant780001DB
// CHECK:STDOUT:       inst78000210:    symbolic_constant780001DA
// CHECK:STDOUT:       inst78000211:    symbolic_constant780001DB
// CHECK:STDOUT:       inst78000216:    concrete_constant(inst7800002D)
// CHECK:STDOUT:       inst78000217:    concrete_constant(inst7800002D)
// CHECK:STDOUT:       inst78000218:    concrete_constant(inst7800002D)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant78000000: {inst: inst78000013, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant78000001: {inst: inst78000017, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000002: {inst: inst78000017, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000003: {inst: inst7800001A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000004: {inst: inst7800001A, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000005: {inst: inst7800001E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000006: {inst: inst7800001E, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000007: {inst: inst78000020, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000008: {inst: inst78000022, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000009: {inst: inst78000020, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800000A: {inst: inst78000022, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800000B: {inst: inst78000026, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800000C: {inst: inst78000026, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800000D: {inst: inst78000030, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800000E: {inst: inst78000030, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800000F: {inst: inst78000034, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000010: {inst: inst78000034, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000011: {inst: inst78000038, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000012: {inst: inst78000038, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000013: {inst: inst7800003A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000014: {inst: inst7800003C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000015: {inst: inst7800003A, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant78000016: {inst: inst7800003C, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000017: {inst: inst78000040, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000018: {inst: inst78000040, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant78000019: {inst: inst7800004B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001A: {inst: inst7800004B, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800001B: {inst: inst7800004D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001C: {inst: inst7800004F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001D: {inst: inst7800004F, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800001E: {inst: inst7800005B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001F: {inst: inst78000060, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000020: {inst: inst78000061, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000021: {inst: inst78000061, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000022: {inst: inst78000062, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000023: {inst: inst78000063, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000024: {inst: inst78000064, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000025: {inst: inst78000060, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000026: {inst: inst78000066, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000027: {inst: inst78000066, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000028: {inst: inst78000063, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000029: {inst: inst78000067, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002A: {inst: inst78000069, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002B: {inst: inst78000069, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800002C: {inst: inst78000066, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800002D: {inst: inst78000064, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800002E: {inst: inst7800006B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002F: {inst: inst7800006B, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000030: {inst: inst78000069, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000031: {inst: inst78000067, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000032: {inst: inst78000062, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000033: {inst: inst7800005B, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000034: {inst: inst7800006B, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000035: {inst: inst78000062, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000036: {inst: inst78000064, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000037: {inst: inst78000067, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000038: {inst: inst7800005B, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000039: {inst: inst78000062, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800003A: {inst: inst78000063, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800003B: {inst: inst78000067, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800003C: {inst: inst78000069, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800003D: {inst: inst7800006B, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800003E: {inst: inst78000064, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800003F: {inst: inst78000066, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000040: {inst: inst78000061, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000041: {inst: inst7800005B, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000042: {inst: inst7800005B, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000043: {inst: inst78000060, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000044: {inst: inst78000061, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000045: {inst: inst78000089, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000046: {inst: inst7800008A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000047: {inst: inst7800008B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000048: {inst: inst7800008C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000049: {inst: inst7800008C, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800004A: {inst: inst7800008B, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800004B: {inst: inst7800008F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004C: {inst: inst7800008F, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800004D: {inst: inst78000091, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004E: {inst: inst78000092, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004F: {inst: inst78000092, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000050: {inst: inst78000093, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000051: {inst: inst78000094, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000052: {inst: inst78000091, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000053: {inst: inst78000096, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000054: {inst: inst78000098, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000055: {inst: inst78000098, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000056: {inst: inst78000093, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000057: {inst: inst78000099, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000058: {inst: inst7800009B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000059: {inst: inst7800009B, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800005A: {inst: inst78000098, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800005B: {inst: inst78000094, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800005C: {inst: inst7800009D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800005D: {inst: inst7800009D, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800005E: {inst: inst7800009B, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800005F: {inst: inst78000099, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000060: {inst: inst7800008B, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000061: {inst: inst7800008A, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000062: {inst: inst78000089, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000063: {inst: inst78000089, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000064: {inst: inst7800009D, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000065: {inst: inst7800008B, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000066: {inst: inst78000094, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000067: {inst: inst78000099, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000068: {inst: inst78000089, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000069: {inst: inst7800008A, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800006A: {inst: inst7800008B, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800006B: {inst: inst78000093, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800006C: {inst: inst78000099, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800006D: {inst: inst7800009B, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800006E: {inst: inst7800009D, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800006F: {inst: inst78000094, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000070: {inst: inst78000098, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000071: {inst: inst780000B0, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000072: {inst: inst780000B1, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000073: {inst: inst780000B2, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000074: {inst: inst780000B3, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000075: {inst: inst780000B4, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000076: {inst: inst780000B5, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000077: {inst: inst780000B5, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant78000078: {inst: inst780000B6, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000079: {inst: inst780000B7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800007A: {inst: inst780000B9, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800007B: {inst: inst780000BA, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800007C: {inst: inst780000BB, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800007D: {inst: inst780000BD, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800007E: {inst: inst780000B4, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant7800007F: {inst: inst780000B0, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant78000080: {inst: inst780000B3, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant78000081: {inst: inst780000B1, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000082: {inst: inst780000BE, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000083: {inst: inst780000BE, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000084: {inst: inst780000BF, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000085: {inst: inst780000BF, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000086: {inst: inst780000BF, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000087: {inst: inst780000BE, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000088: {inst: inst780000B1, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000089: {inst: inst780000B3, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant7800008A: {inst: inst780000B0, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant7800008B: {inst: inst780000B4, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant7800008C: {inst: inst780000B5, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant7800008D: {inst: inst7800008C, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800008E: {inst: inst7800008B, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800008F: {inst: inst7800008A, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000090: {inst: inst78000089, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000091: {inst: inst78000096, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000092: {inst: inst78000096, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000093: {inst: inst78000096, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000094: {inst: inst78000089, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000095: {inst: inst7800008A, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000096: {inst: inst7800008B, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000097: {inst: inst7800008C, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000098: {inst: inst7800008F, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000099: {inst: inst78000092, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800009A: {inst: inst78000091, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800009B: {inst: inst78000092, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800009C: {inst: inst780000F0, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800009D: {inst: inst780000F0, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800009E: {inst: inst7800001E, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800009F: {inst: inst780000F3, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000A0: {inst: inst780000F3, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000A1: {inst: inst780000F0, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000A2: {inst: inst7800001E, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000A3: {inst: inst7800001A, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000A4: {inst: inst78000017, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000A5: {inst: inst7800001A, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000A6: {inst: inst78000017, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000A7: {inst: inst78000017, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000A8: {inst: inst7800001A, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000A9: {inst: inst7800001E, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000AA: {inst: inst780000F0, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000AB: {inst: inst780000F3, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000AC: {inst: inst7800004B, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant780000AD: {inst: inst780000FF, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000AE: {inst: inst78000100, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000AF: {inst: inst78000100, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000B0: {inst: inst78000101, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000B1: {inst: inst780000FF, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780000B2: {inst: inst78000103, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000B3: {inst: inst78000103, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000B4: {inst: inst78000020, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000B5: {inst: inst78000104, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000B6: {inst: inst78000106, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000B7: {inst: inst78000106, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000B8: {inst: inst78000103, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000B9: {inst: inst78000101, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000BA: {inst: inst78000108, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000BB: {inst: inst78000108, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000BC: {inst: inst78000106, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000BD: {inst: inst78000104, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000BE: {inst: inst7800001E, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000BF: {inst: inst7800001A, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000C0: {inst: inst78000108, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000C1: {inst: inst7800001E, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000C2: {inst: inst78000101, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000C3: {inst: inst78000104, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000C4: {inst: inst7800001A, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000C5: {inst: inst7800001E, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000C6: {inst: inst78000020, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000C7: {inst: inst78000104, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000C8: {inst: inst78000106, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000C9: {inst: inst78000108, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000CA: {inst: inst78000101, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000CB: {inst: inst78000103, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000CC: {inst: inst78000100, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000CD: {inst: inst780000FF, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780000CE: {inst: inst78000100, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000CF: {inst: inst7800004B, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant780000D0: {inst: inst7800012D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D1: {inst: inst7800012E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D2: {inst: inst7800012F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D3: {inst: inst78000130, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D4: {inst: inst78000130, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000D5: {inst: inst7800012F, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000D6: {inst: inst78000133, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D7: {inst: inst78000133, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000D8: {inst: inst78000135, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D9: {inst: inst78000136, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DA: {inst: inst78000136, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000DB: {inst: inst78000137, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DC: {inst: inst78000138, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DD: {inst: inst78000135, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780000DE: {inst: inst7800013A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DF: {inst: inst7800013B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E0: {inst: inst7800013D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E1: {inst: inst7800013D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant780000E2: {inst: inst78000137, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000E3: {inst: inst7800013E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E4: {inst: inst78000140, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E5: {inst: inst78000140, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000E6: {inst: inst7800013D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant780000E7: {inst: inst78000138, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000E8: {inst: inst78000142, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E9: {inst: inst78000142, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000EA: {inst: inst78000140, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000EB: {inst: inst7800013E, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000EC: {inst: inst7800012F, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000ED: {inst: inst7800012E, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000EE: {inst: inst7800012D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000EF: {inst: inst7800008A, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000F0: {inst: inst78000089, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000F1: {inst: inst7800012D, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000F2: {inst: inst78000089, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000F3: {inst: inst78000142, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000F4: {inst: inst7800012F, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000F5: {inst: inst78000138, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780000F6: {inst: inst7800013E, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000F7: {inst: inst78000089, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000F8: {inst: inst7800008A, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000F9: {inst: inst7800012D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000FA: {inst: inst7800012E, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000FB: {inst: inst7800012F, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000FC: {inst: inst78000137, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000FD: {inst: inst7800013E, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000FE: {inst: inst78000140, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000FF: {inst: inst78000142, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000100: {inst: inst78000138, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant78000101: {inst: inst7800013D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000102: {inst: inst78000158, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000103: {inst: inst78000159, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000104: {inst: inst7800015A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000105: {inst: inst7800015B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000106: {inst: inst7800015C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000107: {inst: inst7800015D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000108: {inst: inst7800015D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def12}}
// CHECK:STDOUT:       symbolic_constant78000109: {inst: inst7800015E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800010A: {inst: inst7800015F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800010B: {inst: inst78000161, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800010C: {inst: inst78000162, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800010D: {inst: inst78000163, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800010E: {inst: inst78000165, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800010F: {inst: inst7800015C, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def11}}
// CHECK:STDOUT:       symbolic_constant78000110: {inst: inst78000158, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def10}}
// CHECK:STDOUT:       symbolic_constant78000111: {inst: inst7800015B, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def9}}
// CHECK:STDOUT:       symbolic_constant78000112: {inst: inst78000159, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def8}}
// CHECK:STDOUT:       symbolic_constant78000113: {inst: inst78000166, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000114: {inst: inst78000166, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant78000115: {inst: inst780000B5, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant78000116: {inst: inst780000B4, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant78000117: {inst: inst780000B0, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant78000118: {inst: inst780000B3, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant78000119: {inst: inst780000B1, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant7800011A: {inst: inst780000BE, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800011B: {inst: inst78000167, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800011C: {inst: inst78000167, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800011D: {inst: inst78000167, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800011E: {inst: inst780000BE, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800011F: {inst: inst780000B1, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000120: {inst: inst780000B3, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant78000121: {inst: inst780000B0, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant78000122: {inst: inst780000B4, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant78000123: {inst: inst780000B5, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant78000124: {inst: inst78000166, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant78000125: {inst: inst78000159, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def8}}
// CHECK:STDOUT:       symbolic_constant78000126: {inst: inst7800015B, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def9}}
// CHECK:STDOUT:       symbolic_constant78000127: {inst: inst78000158, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def10}}
// CHECK:STDOUT:       symbolic_constant78000128: {inst: inst7800015C, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def11}}
// CHECK:STDOUT:       symbolic_constant78000129: {inst: inst7800015D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def12}}
// CHECK:STDOUT:       symbolic_constant7800012A: {inst: inst78000130, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800012B: {inst: inst7800012F, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800012C: {inst: inst7800012E, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800012D: {inst: inst7800008A, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800012E: {inst: inst7800013A, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800012F: {inst: inst7800012D, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000130: {inst: inst7800013B, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000131: {inst: inst78000089, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000132: {inst: inst78000096, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000133: {inst: inst7800013B, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000134: {inst: inst78000096, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000135: {inst: inst78000096, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000136: {inst: inst78000089, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000137: {inst: inst7800013B, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000138: {inst: inst7800012D, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000139: {inst: inst7800013A, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800013A: {inst: inst7800008A, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800013B: {inst: inst7800012E, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800013C: {inst: inst7800012F, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800013D: {inst: inst78000130, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800013E: {inst: inst78000133, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant7800013F: {inst: inst78000136, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000140: {inst: inst78000135, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000141: {inst: inst78000136, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000142: {inst: inst7800018C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000143: {inst: inst7800018D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000144: {inst: inst7800018E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000145: {inst: inst7800018F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000146: {inst: inst7800018F, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant78000147: {inst: inst7800018E, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000148: {inst: inst78000192, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000149: {inst: inst78000192, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant7800014A: {inst: inst78000194, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800014B: {inst: inst78000195, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800014C: {inst: inst78000195, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800014D: {inst: inst78000196, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800014E: {inst: inst78000197, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800014F: {inst: inst78000194, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000150: {inst: inst78000199, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000151: {inst: inst7800019A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000152: {inst: inst7800019C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000153: {inst: inst7800019C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant78000154: {inst: inst78000196, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000155: {inst: inst7800019D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000156: {inst: inst7800019F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000157: {inst: inst7800019F, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant78000158: {inst: inst7800019C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant78000159: {inst: inst78000197, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant7800015A: {inst: inst780001A1, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800015B: {inst: inst780001A1, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant7800015C: {inst: inst7800019F, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant7800015D: {inst: inst7800019D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800015E: {inst: inst7800018E, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800015F: {inst: inst7800018D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000160: {inst: inst7800018C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000161: {inst: inst7800012E, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000162: {inst: inst7800012D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000163: {inst: inst7800008A, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000164: {inst: inst78000089, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000165: {inst: inst7800018C, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000166: {inst: inst7800012D, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000167: {inst: inst78000089, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000168: {inst: inst780001A1, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000169: {inst: inst7800018E, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800016A: {inst: inst78000197, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant7800016B: {inst: inst7800019D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800016C: {inst: inst78000089, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800016D: {inst: inst7800008A, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800016E: {inst: inst7800012D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800016F: {inst: inst7800012E, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000170: {inst: inst7800018C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000171: {inst: inst7800018D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000172: {inst: inst7800018E, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000173: {inst: inst78000196, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant78000174: {inst: inst7800019D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant78000175: {inst: inst7800019F, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant78000176: {inst: inst780001A1, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant78000177: {inst: inst78000197, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant78000178: {inst: inst7800019C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant78000179: {inst: inst780001BA, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800017A: {inst: inst780001BB, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800017B: {inst: inst780001BC, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800017C: {inst: inst780001BD, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800017D: {inst: inst780001BE, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800017E: {inst: inst780001BF, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800017F: {inst: inst780001BF, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def18}}
// CHECK:STDOUT:       symbolic_constant78000180: {inst: inst780001C0, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000181: {inst: inst780001C1, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000182: {inst: inst780001C3, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000183: {inst: inst780001C4, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000184: {inst: inst780001C5, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000185: {inst: inst780001C7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000186: {inst: inst780001BE, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def17}}
// CHECK:STDOUT:       symbolic_constant78000187: {inst: inst780001BA, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def16}}
// CHECK:STDOUT:       symbolic_constant78000188: {inst: inst780001BD, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def15}}
// CHECK:STDOUT:       symbolic_constant78000189: {inst: inst780001BB, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def14}}
// CHECK:STDOUT:       symbolic_constant7800018A: {inst: inst780001C8, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800018B: {inst: inst780001C8, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def13}}
// CHECK:STDOUT:       symbolic_constant7800018C: {inst: inst7800015D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def12}}
// CHECK:STDOUT:       symbolic_constant7800018D: {inst: inst7800015C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def11}}
// CHECK:STDOUT:       symbolic_constant7800018E: {inst: inst78000158, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def10}}
// CHECK:STDOUT:       symbolic_constant7800018F: {inst: inst7800015B, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def9}}
// CHECK:STDOUT:       symbolic_constant78000190: {inst: inst78000159, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def8}}
// CHECK:STDOUT:       symbolic_constant78000191: {inst: inst78000166, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant78000192: {inst: inst780000B5, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant78000193: {inst: inst780000B4, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant78000194: {inst: inst780000B0, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant78000195: {inst: inst780000B3, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant78000196: {inst: inst780000B1, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000197: {inst: inst780000BE, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000198: {inst: inst780001C9, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000199: {inst: inst780001C9, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800019A: {inst: inst780001C9, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800019B: {inst: inst780000BE, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800019C: {inst: inst780000B1, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant7800019D: {inst: inst780000B3, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant7800019E: {inst: inst780000B0, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant7800019F: {inst: inst780000B4, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant780001A0: {inst: inst780000B5, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant780001A1: {inst: inst78000166, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant780001A2: {inst: inst78000159, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def8}}
// CHECK:STDOUT:       symbolic_constant780001A3: {inst: inst7800015B, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def9}}
// CHECK:STDOUT:       symbolic_constant780001A4: {inst: inst78000158, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def10}}
// CHECK:STDOUT:       symbolic_constant780001A5: {inst: inst7800015C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def11}}
// CHECK:STDOUT:       symbolic_constant780001A6: {inst: inst7800015D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def12}}
// CHECK:STDOUT:       symbolic_constant780001A7: {inst: inst780001C8, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def13}}
// CHECK:STDOUT:       symbolic_constant780001A8: {inst: inst780001BB, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def14}}
// CHECK:STDOUT:       symbolic_constant780001A9: {inst: inst780001BD, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def15}}
// CHECK:STDOUT:       symbolic_constant780001AA: {inst: inst780001BA, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def16}}
// CHECK:STDOUT:       symbolic_constant780001AB: {inst: inst780001BE, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def17}}
// CHECK:STDOUT:       symbolic_constant780001AC: {inst: inst780001BF, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def18}}
// CHECK:STDOUT:       symbolic_constant780001AD: {inst: inst7800018F, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant780001AE: {inst: inst7800018E, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant780001AF: {inst: inst7800018D, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780001B0: {inst: inst7800012E, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780001B1: {inst: inst7800008A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780001B2: {inst: inst78000199, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780001B3: {inst: inst7800018C, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780001B4: {inst: inst7800019A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780001B5: {inst: inst7800012D, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780001B6: {inst: inst7800013B, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780001B7: {inst: inst78000089, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780001B8: {inst: inst78000096, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780001B9: {inst: inst7800019A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780001BA: {inst: inst7800013B, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780001BB: {inst: inst78000096, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780001BC: {inst: inst78000096, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780001BD: {inst: inst78000089, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780001BE: {inst: inst7800013B, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780001BF: {inst: inst7800012D, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780001C0: {inst: inst7800019A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780001C1: {inst: inst7800018C, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780001C2: {inst: inst78000199, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780001C3: {inst: inst7800008A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780001C4: {inst: inst7800012E, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780001C5: {inst: inst7800018D, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl9}}
// CHECK:STDOUT:       symbolic_constant780001C6: {inst: inst7800018E, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl10}}
// CHECK:STDOUT:       symbolic_constant780001C7: {inst: inst7800018F, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl11}}
// CHECK:STDOUT:       symbolic_constant780001C8: {inst: inst78000192, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl12}}
// CHECK:STDOUT:       symbolic_constant780001C9: {inst: inst78000195, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780001CA: {inst: inst78000194, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780001CB: {inst: inst78000195, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780001CC: {inst: inst780001F7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780001CD: {inst: inst780000F3, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780001CE: {inst: inst780001F9, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780001CF: {inst: inst780001F9, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant780001D0: {inst: inst780001FB, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780001D1: {inst: inst780001FC, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780001D2: {inst: inst780001FD, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780001D3: {inst: inst780001FE, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780001D4: {inst: inst78000200, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780001D5: {inst: inst780001F7, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant780001D6: {inst: inst780001FB, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant780001D7: {inst: inst780001FC, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant780001D8: {inst: inst780001FE, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant780001D9: {inst: inst78000200, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant780001DA: {inst: inst78000210, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780001DB: {inst: inst78000210, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def8}}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst78000047
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst78000011
// CHECK:STDOUT:       1:               inst78000057
// CHECK:STDOUT:       2:               inst78000058
// CHECK:STDOUT:       3:               inst7800005A
// CHECK:STDOUT:       4:               inst7800005C
// CHECK:STDOUT:       5:               inst7800005D
// CHECK:STDOUT:       6:               inst7800005E
// CHECK:STDOUT:       7:               inst7800005F
// CHECK:STDOUT:       8:               inst78000065
// CHECK:STDOUT:       9:               inst78000068
// CHECK:STDOUT:       10:              inst7800006A
// CHECK:STDOUT:       11:              inst7800006C
// CHECK:STDOUT:       12:              inst7800006D
// CHECK:STDOUT:       13:              inst7800006E
// CHECK:STDOUT:       14:              inst7800006F
// CHECK:STDOUT:       15:              inst78000070
// CHECK:STDOUT:       16:              inst78000071
// CHECK:STDOUT:       17:              inst78000072
// CHECK:STDOUT:       18:              inst78000073
// CHECK:STDOUT:       19:              inst78000074
// CHECK:STDOUT:       20:              inst7800007D
// CHECK:STDOUT:       21:              inst7800007E
// CHECK:STDOUT:       22:              inst78000082
// CHECK:STDOUT:       23:              inst78000084
// CHECK:STDOUT:       24:              inst78000087
// CHECK:STDOUT:       25:              inst78000088
// CHECK:STDOUT:       26:              inst7800008D
// CHECK:STDOUT:       27:              inst7800008E
// CHECK:STDOUT:       28:              inst78000090
// CHECK:STDOUT:       29:              inst78000097
// CHECK:STDOUT:       30:              inst7800009A
// CHECK:STDOUT:       31:              inst7800009C
// CHECK:STDOUT:       32:              inst7800009E
// CHECK:STDOUT:       33:              inst7800009F
// CHECK:STDOUT:       34:              inst780000A0
// CHECK:STDOUT:       35:              inst780000A1
// CHECK:STDOUT:       36:              inst780000A2
// CHECK:STDOUT:       37:              inst780000A3
// CHECK:STDOUT:       38:              inst780000A4
// CHECK:STDOUT:       39:              inst780000A5
// CHECK:STDOUT:       40:              inst780000A6
// CHECK:STDOUT:       41:              inst780000B8
// CHECK:STDOUT:       42:              inst780000BC
// CHECK:STDOUT:       43:              inst780000C7
// CHECK:STDOUT:       44:              inst780000C8
// CHECK:STDOUT:       45:              inst780000C9
// CHECK:STDOUT:       46:              inst780000CA
// CHECK:STDOUT:       47:              inst780000CB
// CHECK:STDOUT:       48:              inst780000CC
// CHECK:STDOUT:       49:              inst780000CD
// CHECK:STDOUT:       50:              inst780000D6
// CHECK:STDOUT:       51:              inst780000D7
// CHECK:STDOUT:       52:              inst780000D9
// CHECK:STDOUT:       53:              inst780000DA
// CHECK:STDOUT:       54:              inst780000DB
// CHECK:STDOUT:       55:              inst780000DC
// CHECK:STDOUT:       56:              inst780000DD
// CHECK:STDOUT:       57:              inst780000DF
// CHECK:STDOUT:       58:              inst780000E0
// CHECK:STDOUT:       59:              inst780000E1
// CHECK:STDOUT:       60:              inst780000E2
// CHECK:STDOUT:       61:              inst780000E3
// CHECK:STDOUT:       62:              inst780000E5
// CHECK:STDOUT:       63:              inst780000E6
// CHECK:STDOUT:       64:              inst780000E7
// CHECK:STDOUT:       65:              inst780000E8
// CHECK:STDOUT:       66:              inst780000E9
// CHECK:STDOUT:       67:              inst780000EB
// CHECK:STDOUT:       68:              inst780000EC
// CHECK:STDOUT:       69:              inst780000ED
// CHECK:STDOUT:       70:              inst780000EE
// CHECK:STDOUT:       71:              inst780000EF
// CHECK:STDOUT:       72:              inst780000F1
// CHECK:STDOUT:       73:              inst780000F2
// CHECK:STDOUT:       74:              inst780000F4
// CHECK:STDOUT:       75:              inst780000F5
// CHECK:STDOUT:       76:              inst780000F6
// CHECK:STDOUT:       77:              inst780000F7
// CHECK:STDOUT:       78:              inst780000F8
// CHECK:STDOUT:       79:              inst780000FE
// CHECK:STDOUT:       80:              inst78000102
// CHECK:STDOUT:       81:              inst78000105
// CHECK:STDOUT:       82:              inst78000107
// CHECK:STDOUT:       83:              inst78000109
// CHECK:STDOUT:       84:              inst7800010A
// CHECK:STDOUT:       85:              inst7800010B
// CHECK:STDOUT:       86:              inst7800010C
// CHECK:STDOUT:       87:              inst7800010D
// CHECK:STDOUT:       88:              inst7800010E
// CHECK:STDOUT:       89:              inst7800010F
// CHECK:STDOUT:       90:              inst78000110
// CHECK:STDOUT:       91:              inst78000111
// CHECK:STDOUT:       92:              inst7800011A
// CHECK:STDOUT:       93:              inst7800011B
// CHECK:STDOUT:       94:              inst7800011F
// CHECK:STDOUT:       95:              inst78000120
// CHECK:STDOUT:       96:              inst78000122
// CHECK:STDOUT:       97:              inst78000123
// CHECK:STDOUT:       98:              inst78000124
// CHECK:STDOUT:       99:              inst78000125
// CHECK:STDOUT:       100:             inst78000126
// CHECK:STDOUT:       101:             inst78000128
// CHECK:STDOUT:       102:             inst78000129
// CHECK:STDOUT:       103:             inst7800012A
// CHECK:STDOUT:       104:             inst7800012B
// CHECK:STDOUT:       105:             inst7800012C
// CHECK:STDOUT:       106:             inst78000131
// CHECK:STDOUT:       107:             inst78000132
// CHECK:STDOUT:       108:             inst78000134
// CHECK:STDOUT:       109:             inst7800013C
// CHECK:STDOUT:       110:             inst7800013F
// CHECK:STDOUT:       111:             inst78000141
// CHECK:STDOUT:       112:             inst78000143
// CHECK:STDOUT:       113:             inst78000144
// CHECK:STDOUT:       114:             inst78000145
// CHECK:STDOUT:       115:             inst78000146
// CHECK:STDOUT:       116:             inst78000147
// CHECK:STDOUT:       117:             inst78000148
// CHECK:STDOUT:       118:             inst78000149
// CHECK:STDOUT:       119:             inst7800014A
// CHECK:STDOUT:       120:             inst7800014B
// CHECK:STDOUT:       121:             inst7800014C
// CHECK:STDOUT:       122:             inst78000160
// CHECK:STDOUT:       123:             inst78000164
// CHECK:STDOUT:       124:             inst78000175
// CHECK:STDOUT:       125:             inst78000176
// CHECK:STDOUT:       126:             inst78000177
// CHECK:STDOUT:       127:             inst78000178
// CHECK:STDOUT:       128:             inst78000179
// CHECK:STDOUT:       129:             inst7800017A
// CHECK:STDOUT:       130:             inst7800017B
// CHECK:STDOUT:       131:             inst7800017C
// CHECK:STDOUT:       132:             inst7800017D
// CHECK:STDOUT:       133:             inst7800018A
// CHECK:STDOUT:       134:             inst7800018B
// CHECK:STDOUT:       135:             inst78000190
// CHECK:STDOUT:       136:             inst78000191
// CHECK:STDOUT:       137:             inst78000193
// CHECK:STDOUT:       138:             inst7800019B
// CHECK:STDOUT:       139:             inst7800019E
// CHECK:STDOUT:       140:             inst780001A0
// CHECK:STDOUT:       141:             inst780001A2
// CHECK:STDOUT:       142:             inst780001A3
// CHECK:STDOUT:       143:             inst780001A4
// CHECK:STDOUT:       144:             inst780001A5
// CHECK:STDOUT:       145:             inst780001A6
// CHECK:STDOUT:       146:             inst780001A7
// CHECK:STDOUT:       147:             inst780001A8
// CHECK:STDOUT:       148:             inst780001A9
// CHECK:STDOUT:       149:             inst780001AA
// CHECK:STDOUT:       150:             inst780001AB
// CHECK:STDOUT:       151:             inst780001AC
// CHECK:STDOUT:       152:             inst780001C2
// CHECK:STDOUT:       153:             inst780001C6
// CHECK:STDOUT:       154:             inst780001DD
// CHECK:STDOUT:       155:             inst780001DE
// CHECK:STDOUT:       156:             inst780001DF
// CHECK:STDOUT:       157:             inst780001E0
// CHECK:STDOUT:       158:             inst780001E1
// CHECK:STDOUT:       159:             inst780001E2
// CHECK:STDOUT:       160:             inst780001E3
// CHECK:STDOUT:       161:             inst780001E4
// CHECK:STDOUT:       162:             inst780001E5
// CHECK:STDOUT:       163:             inst780001E6
// CHECK:STDOUT:       164:             inst780001E7
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block78000005:
// CHECK:STDOUT:       0:               inst78000012
// CHECK:STDOUT:       1:               inst78000014
// CHECK:STDOUT:     inst_block78000006:
// CHECK:STDOUT:       0:               inst78000016
// CHECK:STDOUT:     inst_block78000007:
// CHECK:STDOUT:       0:               inst7800001C
// CHECK:STDOUT:       1:               inst7800001D
// CHECK:STDOUT:     inst_block78000008:
// CHECK:STDOUT:       0:               inst78000025
// CHECK:STDOUT:     inst_block78000009:
// CHECK:STDOUT:       0:               inst7800002A
// CHECK:STDOUT:       1:               inst7800002C
// CHECK:STDOUT:     inst_block7800000A:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst7800002B
// CHECK:STDOUT:     inst_block7800000B:
// CHECK:STDOUT:       0:               inst7800001E
// CHECK:STDOUT:       1:               inst7800002D
// CHECK:STDOUT:     inst_block7800000C:
// CHECK:STDOUT:       0:               inst7800001F
// CHECK:STDOUT:       1:               inst7800002D
// CHECK:STDOUT:     inst_block7800000D:
// CHECK:STDOUT:       0:               inst7800001E
// CHECK:STDOUT:       1:               inst78000033
// CHECK:STDOUT:     inst_block7800000E:
// CHECK:STDOUT:       0:               inst7800001E
// CHECK:STDOUT:       1:               inst7800002B
// CHECK:STDOUT:     inst_block7800000F:
// CHECK:STDOUT:       0:               inst7800001F
// CHECK:STDOUT:       1:               inst7800002B
// CHECK:STDOUT:     inst_block78000010:
// CHECK:STDOUT:       0:               inst78000021
// CHECK:STDOUT:       1:               inst7800003B
// CHECK:STDOUT:     inst_block78000011:
// CHECK:STDOUT:       0:               inst78000043
// CHECK:STDOUT:       1:               inst78000045
// CHECK:STDOUT:     inst_block78000012:
// CHECK:STDOUT:       0:               inst78000016
// CHECK:STDOUT:       1:               inst78000021
// CHECK:STDOUT:       2:               inst78000025
// CHECK:STDOUT:       3:               inst7800003B
// CHECK:STDOUT:       4:               inst7800003F
// CHECK:STDOUT:     inst_block78000013:
// CHECK:STDOUT:       0:               inst78000029
// CHECK:STDOUT:       1:               inst7800002A
// CHECK:STDOUT:       2:               inst7800002C
// CHECK:STDOUT:       3:               inst7800002F
// CHECK:STDOUT:       4:               inst78000033
// CHECK:STDOUT:       5:               inst78000035
// CHECK:STDOUT:       6:               inst78000037
// CHECK:STDOUT:       7:               inst78000042
// CHECK:STDOUT:       8:               inst78000019
// CHECK:STDOUT:       9:               inst78000043
// CHECK:STDOUT:       10:              inst78000044
// CHECK:STDOUT:       11:              inst78000028
// CHECK:STDOUT:       12:              inst78000045
// CHECK:STDOUT:       13:              inst78000046
// CHECK:STDOUT:     inst_block78000014:
// CHECK:STDOUT:       0:               inst78000019
// CHECK:STDOUT:     inst_block78000015:
// CHECK:STDOUT:       0:               inst7800001A
// CHECK:STDOUT:     inst_block78000016:
// CHECK:STDOUT:       0:               inst78000018
// CHECK:STDOUT:       1:               inst7800001B
// CHECK:STDOUT:       2:               inst7800001F
// CHECK:STDOUT:       3:               inst78000023
// CHECK:STDOUT:       4:               inst78000024
// CHECK:STDOUT:       5:               inst78000027
// CHECK:STDOUT:       6:               inst78000031
// CHECK:STDOUT:       7:               inst78000036
// CHECK:STDOUT:       8:               inst78000039
// CHECK:STDOUT:       9:               inst7800003D
// CHECK:STDOUT:       10:              inst7800003E
// CHECK:STDOUT:       11:              inst78000041
// CHECK:STDOUT:     inst_block78000017:
// CHECK:STDOUT:       0:               inst78000017
// CHECK:STDOUT:       1:               inst7800001A
// CHECK:STDOUT:       2:               inst7800001E
// CHECK:STDOUT:       3:               inst78000020
// CHECK:STDOUT:       4:               inst78000022
// CHECK:STDOUT:       5:               inst78000026
// CHECK:STDOUT:       6:               inst78000030
// CHECK:STDOUT:       7:               inst78000034
// CHECK:STDOUT:       8:               inst78000038
// CHECK:STDOUT:       9:               inst7800003A
// CHECK:STDOUT:       10:              inst7800003C
// CHECK:STDOUT:       11:              inst78000040
// CHECK:STDOUT:     inst_block78000018:
// CHECK:STDOUT:       0:               inst78000051
// CHECK:STDOUT:       1:               inst78000052
// CHECK:STDOUT:       2:               inst78000053
// CHECK:STDOUT:       3:               inst780001FF
// CHECK:STDOUT:       4:               inst78000206
// CHECK:STDOUT:       5:               inst78000209
// CHECK:STDOUT:       6:               inst7800020A
// CHECK:STDOUT:       7:               inst7800020D
// CHECK:STDOUT:       8:               inst7800020E
// CHECK:STDOUT:       9:               inst7800020F
// CHECK:STDOUT:       10:              inst78000212
// CHECK:STDOUT:       11:              inst78000213
// CHECK:STDOUT:       12:              inst78000055
// CHECK:STDOUT:       13:              inst78000214
// CHECK:STDOUT:       14:              inst78000215
// CHECK:STDOUT:       15:              inst78000216
// CHECK:STDOUT:       16:              inst78000217
// CHECK:STDOUT:       17:              inst78000218
// CHECK:STDOUT:       18:              inst78000219
// CHECK:STDOUT:       19:              inst7800021A
// CHECK:STDOUT:       20:              inst7800021B
// CHECK:STDOUT:     inst_block78000019:
// CHECK:STDOUT:       0:               inst78000051
// CHECK:STDOUT:       1:               inst78000052
// CHECK:STDOUT:     inst_block7800001A:
// CHECK:STDOUT:       0:               inst7800005D
// CHECK:STDOUT:     inst_block7800001B:
// CHECK:STDOUT:       0:               inst7800005E
// CHECK:STDOUT:     inst_block7800001C:
// CHECK:STDOUT:       0:               inst7800005B
// CHECK:STDOUT:     inst_block7800001D:
// CHECK:STDOUT:       0:               inst7800005B
// CHECK:STDOUT:       1:               inst78000060
// CHECK:STDOUT:       2:               inst78000061
// CHECK:STDOUT:     inst_block7800001E:
// CHECK:STDOUT:       0:               inst7800006D
// CHECK:STDOUT:       1:               inst7800006E
// CHECK:STDOUT:     inst_block7800001F:
// CHECK:STDOUT:       0:               inst78000070
// CHECK:STDOUT:     inst_block78000020:
// CHECK:STDOUT:       0:               inst78000074
// CHECK:STDOUT:     inst_block78000021:
// CHECK:STDOUT:       0:               inst78000075
// CHECK:STDOUT:       1:               inst78000076
// CHECK:STDOUT:       2:               inst78000077
// CHECK:STDOUT:       3:               inst78000078
// CHECK:STDOUT:       4:               inst78000079
// CHECK:STDOUT:       5:               inst7800007A
// CHECK:STDOUT:       6:               inst7800007B
// CHECK:STDOUT:       7:               inst7800007C
// CHECK:STDOUT:     inst_block78000022:
// CHECK:STDOUT:       0:               inst7800005B
// CHECK:STDOUT:       1:               inst78000062
// CHECK:STDOUT:       2:               inst78000063
// CHECK:STDOUT:       3:               inst78000067
// CHECK:STDOUT:       4:               inst78000069
// CHECK:STDOUT:       5:               inst7800006B
// CHECK:STDOUT:       6:               inst78000064
// CHECK:STDOUT:       7:               inst78000066
// CHECK:STDOUT:     inst_block78000023:
// CHECK:STDOUT:       0:               inst7800007F
// CHECK:STDOUT:     inst_block78000024:
// CHECK:STDOUT:       0:               inst7800007F
// CHECK:STDOUT:       1:               inst78000080
// CHECK:STDOUT:       2:               inst78000081
// CHECK:STDOUT:     inst_block78000025:
// CHECK:STDOUT:       0:               inst7800008D
// CHECK:STDOUT:     inst_block78000026:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:     inst_block78000027:
// CHECK:STDOUT:       0:               inst78000096
// CHECK:STDOUT:       1:               inst78000089
// CHECK:STDOUT:       2:               inst7800008A
// CHECK:STDOUT:       3:               inst7800008B
// CHECK:STDOUT:       4:               inst7800008C
// CHECK:STDOUT:       5:               inst7800008F
// CHECK:STDOUT:     inst_block78000028:
// CHECK:STDOUT:       0:               inst78000091
// CHECK:STDOUT:       1:               inst78000092
// CHECK:STDOUT:     inst_block78000029:
// CHECK:STDOUT:       0:               inst7800009F
// CHECK:STDOUT:       1:               inst780000A0
// CHECK:STDOUT:     inst_block7800002A:
// CHECK:STDOUT:       0:               inst780000A2
// CHECK:STDOUT:     inst_block7800002B:
// CHECK:STDOUT:       0:               inst780000A6
// CHECK:STDOUT:     inst_block7800002C:
// CHECK:STDOUT:       0:               inst780000A7
// CHECK:STDOUT:       1:               inst780000A8
// CHECK:STDOUT:       2:               inst780000A9
// CHECK:STDOUT:       3:               inst780000AA
// CHECK:STDOUT:       4:               inst780000AB
// CHECK:STDOUT:       5:               inst780000AC
// CHECK:STDOUT:       6:               inst780000AD
// CHECK:STDOUT:       7:               inst780000AE
// CHECK:STDOUT:       8:               inst780000AF
// CHECK:STDOUT:     inst_block7800002D:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:       1:               inst780000B1
// CHECK:STDOUT:       2:               inst780000B2
// CHECK:STDOUT:     inst_block7800002E:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:       1:               inst7800008A
// CHECK:STDOUT:       2:               inst780000B6
// CHECK:STDOUT:       3:               inst780000BB
// CHECK:STDOUT:       4:               inst780000BD
// CHECK:STDOUT:       5:               inst780000BA
// CHECK:STDOUT:       6:               inst780000B7
// CHECK:STDOUT:       7:               inst780000B9
// CHECK:STDOUT:     inst_block7800002F:
// CHECK:STDOUT:       0:               inst780000A7
// CHECK:STDOUT:     inst_block78000030:
// CHECK:STDOUT:       0:               inst780000A7
// CHECK:STDOUT:     inst_block78000031:
// CHECK:STDOUT:       0:               inst780000C0
// CHECK:STDOUT:       1:               inst780000C1
// CHECK:STDOUT:       2:               inst780000C2
// CHECK:STDOUT:       3:               inst780000C3
// CHECK:STDOUT:       4:               inst780000C4
// CHECK:STDOUT:       5:               inst780000C5
// CHECK:STDOUT:       6:               inst780000C6
// CHECK:STDOUT:     inst_block78000032:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:       1:               inst7800008A
// CHECK:STDOUT:       2:               inst7800008B
// CHECK:STDOUT:       3:               inst78000093
// CHECK:STDOUT:       4:               inst78000099
// CHECK:STDOUT:       5:               inst7800009B
// CHECK:STDOUT:       6:               inst7800009D
// CHECK:STDOUT:       7:               inst78000094
// CHECK:STDOUT:       8:               inst78000098
// CHECK:STDOUT:     inst_block78000033:
// CHECK:STDOUT:       0:               inst780000C9
// CHECK:STDOUT:     inst_block78000034:
// CHECK:STDOUT:       0:               inst780000CD
// CHECK:STDOUT:     inst_block78000035:
// CHECK:STDOUT:       0:               inst780000CF
// CHECK:STDOUT:     inst_block78000036:
// CHECK:STDOUT:       0:               inst780000CE
// CHECK:STDOUT:       1:               inst780000CF
// CHECK:STDOUT:       2:               inst780000D0
// CHECK:STDOUT:       3:               inst780000D1
// CHECK:STDOUT:       4:               inst780000D2
// CHECK:STDOUT:       5:               inst780000D3
// CHECK:STDOUT:     inst_block78000037:
// CHECK:STDOUT:       0:               inst780000CF
// CHECK:STDOUT:     inst_block78000038:
// CHECK:STDOUT:       0:               inst780000D4
// CHECK:STDOUT:       1:               inst780000D5
// CHECK:STDOUT:     inst_block78000039:
// CHECK:STDOUT:       0:               inst780000F1
// CHECK:STDOUT:     inst_block7800003A:
// CHECK:STDOUT:       0:               inst78000017
// CHECK:STDOUT:       1:               inst7800001A
// CHECK:STDOUT:       2:               inst7800001E
// CHECK:STDOUT:       3:               inst780000F0
// CHECK:STDOUT:       4:               inst780000F3
// CHECK:STDOUT:     inst_block7800003B:
// CHECK:STDOUT:       0:               inst780000F4
// CHECK:STDOUT:     inst_block7800003C:
// CHECK:STDOUT:       0:               inst780000F8
// CHECK:STDOUT:     inst_block7800003D:
// CHECK:STDOUT:       0:               inst780000FA
// CHECK:STDOUT:     inst_block7800003E:
// CHECK:STDOUT:       0:               inst780000F9
// CHECK:STDOUT:       1:               inst780000FA
// CHECK:STDOUT:       2:               inst780000FB
// CHECK:STDOUT:       3:               inst780000FC
// CHECK:STDOUT:       4:               inst780000FD
// CHECK:STDOUT:     inst_block7800003F:
// CHECK:STDOUT:       0:               inst7800010A
// CHECK:STDOUT:       1:               inst7800010B
// CHECK:STDOUT:     inst_block78000040:
// CHECK:STDOUT:       0:               inst7800010D
// CHECK:STDOUT:     inst_block78000041:
// CHECK:STDOUT:       0:               inst78000111
// CHECK:STDOUT:     inst_block78000042:
// CHECK:STDOUT:       0:               inst78000112
// CHECK:STDOUT:       1:               inst78000113
// CHECK:STDOUT:       2:               inst78000114
// CHECK:STDOUT:       3:               inst78000115
// CHECK:STDOUT:       4:               inst78000116
// CHECK:STDOUT:       5:               inst78000117
// CHECK:STDOUT:       6:               inst78000118
// CHECK:STDOUT:       7:               inst78000119
// CHECK:STDOUT:     inst_block78000043:
// CHECK:STDOUT:       0:               inst7800001A
// CHECK:STDOUT:       1:               inst7800001E
// CHECK:STDOUT:       2:               inst78000020
// CHECK:STDOUT:       3:               inst78000104
// CHECK:STDOUT:       4:               inst78000106
// CHECK:STDOUT:       5:               inst78000108
// CHECK:STDOUT:       6:               inst78000101
// CHECK:STDOUT:       7:               inst78000103
// CHECK:STDOUT:     inst_block78000044:
// CHECK:STDOUT:       0:               inst780000FA
// CHECK:STDOUT:     inst_block78000045:
// CHECK:STDOUT:       0:               inst7800011C
// CHECK:STDOUT:       1:               inst7800011D
// CHECK:STDOUT:       2:               inst7800011E
// CHECK:STDOUT:     inst_block78000046:
// CHECK:STDOUT:       0:               inst7800008A
// CHECK:STDOUT:       1:               inst7800012E
// CHECK:STDOUT:     inst_block78000047:
// CHECK:STDOUT:       0:               inst78000131
// CHECK:STDOUT:     inst_block78000048:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:       1:               inst7800012D
// CHECK:STDOUT:     inst_block78000049:
// CHECK:STDOUT:       0:               inst78000059
// CHECK:STDOUT:       1:               inst78000059
// CHECK:STDOUT:     inst_block7800004A:
// CHECK:STDOUT:       0:               inst78000096
// CHECK:STDOUT:       1:               inst78000089
// CHECK:STDOUT:       2:               inst7800013B
// CHECK:STDOUT:       3:               inst7800012D
// CHECK:STDOUT:       4:               inst7800013A
// CHECK:STDOUT:       5:               inst7800008A
// CHECK:STDOUT:       6:               inst7800012E
// CHECK:STDOUT:       7:               inst7800012F
// CHECK:STDOUT:       8:               inst78000130
// CHECK:STDOUT:       9:               inst78000133
// CHECK:STDOUT:     inst_block7800004B:
// CHECK:STDOUT:       0:               inst78000135
// CHECK:STDOUT:       1:               inst78000136
// CHECK:STDOUT:     inst_block7800004C:
// CHECK:STDOUT:       0:               inst78000144
// CHECK:STDOUT:       1:               inst78000145
// CHECK:STDOUT:     inst_block7800004D:
// CHECK:STDOUT:       0:               inst78000147
// CHECK:STDOUT:     inst_block7800004E:
// CHECK:STDOUT:       0:               inst7800014B
// CHECK:STDOUT:       1:               inst7800014C
// CHECK:STDOUT:     inst_block7800004F:
// CHECK:STDOUT:       0:               inst7800014E
// CHECK:STDOUT:       1:               inst78000150
// CHECK:STDOUT:     inst_block78000050:
// CHECK:STDOUT:       0:               inst7800014D
// CHECK:STDOUT:       1:               inst7800014E
// CHECK:STDOUT:       2:               inst7800014F
// CHECK:STDOUT:       3:               inst78000150
// CHECK:STDOUT:       4:               inst78000151
// CHECK:STDOUT:       5:               inst78000152
// CHECK:STDOUT:       6:               inst78000153
// CHECK:STDOUT:       7:               inst78000154
// CHECK:STDOUT:       8:               inst78000155
// CHECK:STDOUT:       9:               inst78000156
// CHECK:STDOUT:       10:              inst78000157
// CHECK:STDOUT:     inst_block78000051:
// CHECK:STDOUT:       0:               inst7800012D
// CHECK:STDOUT:     inst_block78000052:
// CHECK:STDOUT:       0:               inst7800012D
// CHECK:STDOUT:       1:               inst78000159
// CHECK:STDOUT:       2:               inst7800015A
// CHECK:STDOUT:     inst_block78000053:
// CHECK:STDOUT:       0:               inst7800012D
// CHECK:STDOUT:       1:               inst7800012E
// CHECK:STDOUT:       2:               inst7800015E
// CHECK:STDOUT:       3:               inst78000163
// CHECK:STDOUT:       4:               inst78000165
// CHECK:STDOUT:       5:               inst78000162
// CHECK:STDOUT:       6:               inst7800015F
// CHECK:STDOUT:       7:               inst78000161
// CHECK:STDOUT:     inst_block78000054:
// CHECK:STDOUT:       0:               inst7800014D
// CHECK:STDOUT:     inst_block78000055:
// CHECK:STDOUT:       0:               inst7800014D
// CHECK:STDOUT:     inst_block78000056:
// CHECK:STDOUT:       0:               inst7800014F
// CHECK:STDOUT:     inst_block78000057:
// CHECK:STDOUT:       0:               inst7800014F
// CHECK:STDOUT:     inst_block78000058:
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
// CHECK:STDOUT:     inst_block78000059:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:       1:               inst7800008A
// CHECK:STDOUT:       2:               inst7800012D
// CHECK:STDOUT:       3:               inst7800012E
// CHECK:STDOUT:       4:               inst7800012F
// CHECK:STDOUT:       5:               inst78000137
// CHECK:STDOUT:       6:               inst7800013E
// CHECK:STDOUT:       7:               inst78000140
// CHECK:STDOUT:       8:               inst78000142
// CHECK:STDOUT:       9:               inst78000138
// CHECK:STDOUT:       10:              inst7800013D
// CHECK:STDOUT:     inst_block7800005A:
// CHECK:STDOUT:       0:               inst78000177
// CHECK:STDOUT:       1:               inst78000178
// CHECK:STDOUT:     inst_block7800005B:
// CHECK:STDOUT:       0:               inst7800017C
// CHECK:STDOUT:       1:               inst7800017D
// CHECK:STDOUT:     inst_block7800005C:
// CHECK:STDOUT:       0:               inst7800017F
// CHECK:STDOUT:       1:               inst78000181
// CHECK:STDOUT:     inst_block7800005D:
// CHECK:STDOUT:       0:               inst78000183
// CHECK:STDOUT:       1:               inst78000184
// CHECK:STDOUT:     inst_block7800005E:
// CHECK:STDOUT:       0:               inst7800017F
// CHECK:STDOUT:       1:               inst78000181
// CHECK:STDOUT:     inst_block7800005F:
// CHECK:STDOUT:       0:               inst7800017E
// CHECK:STDOUT:       1:               inst7800017F
// CHECK:STDOUT:       2:               inst78000180
// CHECK:STDOUT:       3:               inst78000181
// CHECK:STDOUT:       4:               inst78000182
// CHECK:STDOUT:       5:               inst78000183
// CHECK:STDOUT:       6:               inst78000184
// CHECK:STDOUT:       7:               inst78000185
// CHECK:STDOUT:       8:               inst78000186
// CHECK:STDOUT:       9:               inst78000187
// CHECK:STDOUT:     inst_block78000060:
// CHECK:STDOUT:       0:               inst7800017F
// CHECK:STDOUT:       1:               inst78000181
// CHECK:STDOUT:     inst_block78000061:
// CHECK:STDOUT:       0:               inst78000188
// CHECK:STDOUT:       1:               inst78000189
// CHECK:STDOUT:     inst_block78000062:
// CHECK:STDOUT:       0:               inst7800008A
// CHECK:STDOUT:       1:               inst7800012E
// CHECK:STDOUT:       2:               inst7800018D
// CHECK:STDOUT:     inst_block78000063:
// CHECK:STDOUT:       0:               inst78000190
// CHECK:STDOUT:     inst_block78000064:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:       1:               inst7800012D
// CHECK:STDOUT:       2:               inst7800018C
// CHECK:STDOUT:     inst_block78000065:
// CHECK:STDOUT:       0:               inst78000059
// CHECK:STDOUT:       1:               inst78000059
// CHECK:STDOUT:       2:               inst78000059
// CHECK:STDOUT:     inst_block78000066:
// CHECK:STDOUT:       0:               inst78000096
// CHECK:STDOUT:       1:               inst78000089
// CHECK:STDOUT:       2:               inst7800013B
// CHECK:STDOUT:       3:               inst7800012D
// CHECK:STDOUT:       4:               inst7800019A
// CHECK:STDOUT:       5:               inst7800018C
// CHECK:STDOUT:       6:               inst78000199
// CHECK:STDOUT:       7:               inst7800008A
// CHECK:STDOUT:       8:               inst7800012E
// CHECK:STDOUT:       9:               inst7800018D
// CHECK:STDOUT:       10:              inst7800018E
// CHECK:STDOUT:       11:              inst7800018F
// CHECK:STDOUT:       12:              inst78000192
// CHECK:STDOUT:     inst_block78000067:
// CHECK:STDOUT:       0:               inst78000194
// CHECK:STDOUT:       1:               inst78000195
// CHECK:STDOUT:     inst_block78000068:
// CHECK:STDOUT:       0:               inst780001A3
// CHECK:STDOUT:       1:               inst780001A4
// CHECK:STDOUT:     inst_block78000069:
// CHECK:STDOUT:       0:               inst780001A6
// CHECK:STDOUT:     inst_block7800006A:
// CHECK:STDOUT:       0:               inst780001AA
// CHECK:STDOUT:       1:               inst780001AB
// CHECK:STDOUT:       2:               inst780001AC
// CHECK:STDOUT:     inst_block7800006B:
// CHECK:STDOUT:       0:               inst780001AE
// CHECK:STDOUT:       1:               inst780001B0
// CHECK:STDOUT:       2:               inst780001B2
// CHECK:STDOUT:     inst_block7800006C:
// CHECK:STDOUT:       0:               inst780001AD
// CHECK:STDOUT:       1:               inst780001AE
// CHECK:STDOUT:       2:               inst780001AF
// CHECK:STDOUT:       3:               inst780001B0
// CHECK:STDOUT:       4:               inst780001B1
// CHECK:STDOUT:       5:               inst780001B2
// CHECK:STDOUT:       6:               inst780001B3
// CHECK:STDOUT:       7:               inst780001B4
// CHECK:STDOUT:       8:               inst780001B5
// CHECK:STDOUT:       9:               inst780001B6
// CHECK:STDOUT:       10:              inst780001B7
// CHECK:STDOUT:       11:              inst780001B8
// CHECK:STDOUT:       12:              inst780001B9
// CHECK:STDOUT:     inst_block7800006D:
// CHECK:STDOUT:       0:               inst7800018C
// CHECK:STDOUT:     inst_block7800006E:
// CHECK:STDOUT:       0:               inst7800018C
// CHECK:STDOUT:       1:               inst780001BB
// CHECK:STDOUT:       2:               inst780001BC
// CHECK:STDOUT:     inst_block7800006F:
// CHECK:STDOUT:       0:               inst7800018C
// CHECK:STDOUT:       1:               inst7800018D
// CHECK:STDOUT:       2:               inst780001C0
// CHECK:STDOUT:       3:               inst780001C5
// CHECK:STDOUT:       4:               inst780001C7
// CHECK:STDOUT:       5:               inst780001C4
// CHECK:STDOUT:       6:               inst780001C1
// CHECK:STDOUT:       7:               inst780001C3
// CHECK:STDOUT:     inst_block78000070:
// CHECK:STDOUT:       0:               inst780001AD
// CHECK:STDOUT:     inst_block78000071:
// CHECK:STDOUT:       0:               inst780001AD
// CHECK:STDOUT:     inst_block78000072:
// CHECK:STDOUT:       0:               inst780001AF
// CHECK:STDOUT:     inst_block78000073:
// CHECK:STDOUT:       0:               inst780001AF
// CHECK:STDOUT:     inst_block78000074:
// CHECK:STDOUT:       0:               inst780001B1
// CHECK:STDOUT:     inst_block78000075:
// CHECK:STDOUT:       0:               inst780001B1
// CHECK:STDOUT:     inst_block78000076:
// CHECK:STDOUT:       0:               inst780001CA
// CHECK:STDOUT:       1:               inst780001CB
// CHECK:STDOUT:       2:               inst780001CC
// CHECK:STDOUT:       3:               inst780001CD
// CHECK:STDOUT:       4:               inst780001CE
// CHECK:STDOUT:       5:               inst780001CF
// CHECK:STDOUT:       6:               inst780001D0
// CHECK:STDOUT:       7:               inst780001D1
// CHECK:STDOUT:       8:               inst780001D2
// CHECK:STDOUT:       9:               inst780001D3
// CHECK:STDOUT:       10:              inst780001D4
// CHECK:STDOUT:       11:              inst780001D5
// CHECK:STDOUT:       12:              inst780001D6
// CHECK:STDOUT:       13:              inst780001D7
// CHECK:STDOUT:       14:              inst780001D8
// CHECK:STDOUT:       15:              inst780001D9
// CHECK:STDOUT:       16:              inst780001DA
// CHECK:STDOUT:       17:              inst780001DB
// CHECK:STDOUT:       18:              inst780001DC
// CHECK:STDOUT:     inst_block78000077:
// CHECK:STDOUT:       0:               inst78000089
// CHECK:STDOUT:       1:               inst7800008A
// CHECK:STDOUT:       2:               inst7800012D
// CHECK:STDOUT:       3:               inst7800012E
// CHECK:STDOUT:       4:               inst7800018C
// CHECK:STDOUT:       5:               inst7800018D
// CHECK:STDOUT:       6:               inst7800018E
// CHECK:STDOUT:       7:               inst78000196
// CHECK:STDOUT:       8:               inst7800019D
// CHECK:STDOUT:       9:               inst7800019F
// CHECK:STDOUT:       10:              inst780001A1
// CHECK:STDOUT:       11:              inst78000197
// CHECK:STDOUT:       12:              inst7800019C
// CHECK:STDOUT:     inst_block78000078:
// CHECK:STDOUT:       0:               inst780001DF
// CHECK:STDOUT:       1:               inst780001E0
// CHECK:STDOUT:       2:               inst780001E1
// CHECK:STDOUT:     inst_block78000079:
// CHECK:STDOUT:       0:               inst780001E5
// CHECK:STDOUT:       1:               inst780001E6
// CHECK:STDOUT:       2:               inst780001E7
// CHECK:STDOUT:     inst_block7800007A:
// CHECK:STDOUT:       0:               inst780001E9
// CHECK:STDOUT:       1:               inst780001EB
// CHECK:STDOUT:       2:               inst780001ED
// CHECK:STDOUT:     inst_block7800007B:
// CHECK:STDOUT:       0:               inst780001EF
// CHECK:STDOUT:       1:               inst780001F0
// CHECK:STDOUT:       2:               inst780001F1
// CHECK:STDOUT:     inst_block7800007C:
// CHECK:STDOUT:       0:               inst780001E9
// CHECK:STDOUT:       1:               inst780001EB
// CHECK:STDOUT:       2:               inst780001ED
// CHECK:STDOUT:     inst_block7800007D:
// CHECK:STDOUT:       0:               inst780001E8
// CHECK:STDOUT:       1:               inst780001E9
// CHECK:STDOUT:       2:               inst780001EA
// CHECK:STDOUT:       3:               inst780001EB
// CHECK:STDOUT:       4:               inst780001EC
// CHECK:STDOUT:       5:               inst780001ED
// CHECK:STDOUT:       6:               inst780001EE
// CHECK:STDOUT:       7:               inst780001EF
// CHECK:STDOUT:       8:               inst780001F0
// CHECK:STDOUT:       9:               inst780001F1
// CHECK:STDOUT:       10:              inst780001F2
// CHECK:STDOUT:       11:              inst780001F3
// CHECK:STDOUT:       12:              inst780001F4
// CHECK:STDOUT:     inst_block7800007E:
// CHECK:STDOUT:       0:               inst780001E9
// CHECK:STDOUT:       1:               inst780001EB
// CHECK:STDOUT:       2:               inst780001ED
// CHECK:STDOUT:     inst_block7800007F:
// CHECK:STDOUT:       0:               inst780001F5
// CHECK:STDOUT:       1:               inst780001F6
// CHECK:STDOUT:     inst_block78000080:
// CHECK:STDOUT:       0:               inst780000FF
// CHECK:STDOUT:       1:               inst78000100
// CHECK:STDOUT:       2:               inst7800004B
// CHECK:STDOUT:     inst_block78000081:
// CHECK:STDOUT:       0:               inst7800001B
// CHECK:STDOUT:     inst_block78000082:
// CHECK:STDOUT:       0:               inst780001F7
// CHECK:STDOUT:     inst_block78000083:
// CHECK:STDOUT:       0:               inst780001FB
// CHECK:STDOUT:     inst_block78000084:
// CHECK:STDOUT:       0:               inst780001FB
// CHECK:STDOUT:       1:               inst780001FC
// CHECK:STDOUT:       2:               inst780001FD
// CHECK:STDOUT:     inst_block78000085:
// CHECK:STDOUT:       0:               inst78000201
// CHECK:STDOUT:     inst_block78000086:
// CHECK:STDOUT:       0:               inst78000202
// CHECK:STDOUT:     inst_block78000087:
// CHECK:STDOUT:       0:               inst780001FB
// CHECK:STDOUT:       1:               inst7800001E
// CHECK:STDOUT:       2:               inst78000020
// CHECK:STDOUT:       3:               inst78000104
// CHECK:STDOUT:       4:               inst78000106
// CHECK:STDOUT:       5:               inst78000108
// CHECK:STDOUT:       6:               inst78000101
// CHECK:STDOUT:       7:               inst78000103
// CHECK:STDOUT:     inst_block78000088:
// CHECK:STDOUT:       0:               inst78000202
// CHECK:STDOUT:     inst_block78000089:
// CHECK:STDOUT:       0:               inst78000051
// CHECK:STDOUT:     inst_block7800008A:
// CHECK:STDOUT:       0:               inst78000214
// CHECK:STDOUT:       1:               inst78000218
// CHECK:STDOUT:     inst_block7800008B:
// CHECK:STDOUT:       0:               inst7800004C
// CHECK:STDOUT:       1:               inst78000050
// CHECK:STDOUT:       2:               inst780001FA
// CHECK:STDOUT:       3:               inst78000201
// CHECK:STDOUT:       4:               inst78000202
// CHECK:STDOUT:       5:               inst78000203
// CHECK:STDOUT:       6:               inst78000204
// CHECK:STDOUT:       7:               inst78000205
// CHECK:STDOUT:       8:               inst78000211
// CHECK:STDOUT:     inst_block7800008C:
// CHECK:STDOUT:       0:               inst(Package)
// CHECK:STDOUT:       1:               inst78000010
// CHECK:STDOUT:       2:               inst78000047
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
