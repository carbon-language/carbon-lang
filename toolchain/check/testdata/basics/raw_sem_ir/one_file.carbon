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
// CHECK:STDOUT:     import_ir_inst3: {ir_id: import_ir78000004, inst_id: inst70000035}
// CHECK:STDOUT:     import_ir_inst4: {ir_id: import_ir78000004, inst_id: inst7000002D}
// CHECK:STDOUT:     import_ir_inst5: {ir_id: import_ir78000004, inst_id: inst70000012}
// CHECK:STDOUT:     import_ir_inst6: {ir_id: import_ir78000004, inst_id: inst7000002D}
// CHECK:STDOUT:     import_ir_inst7: {ir_id: import_ir78000004, inst_id: inst70000027}
// CHECK:STDOUT:     import_ir_inst8: {ir_id: import_ir78000004, inst_id: inst70000028}
// CHECK:STDOUT:     import_ir_inst9: {ir_id: import_ir78000004, inst_id: inst7000001D}
// CHECK:STDOUT:     import_ir_instA: {ir_id: import_ir78000004, inst_id: inst70000020}
// CHECK:STDOUT:     import_ir_instB: {ir_id: import_ir78000004, inst_id: inst70000023}
// CHECK:STDOUT:     import_ir_instC: {ir_id: import_ir78000004, inst_id: inst70000024}
// CHECK:STDOUT:     import_ir_instD: {ir_id: import_ir78000004, inst_id: inst70000012}
// CHECK:STDOUT:     import_ir_instE: {ir_id: import_ir78000004, inst_id: inst70000017}
// CHECK:STDOUT:     import_ir_instF: {ir_id: import_ir78000004, inst_id: inst7000001A}
// CHECK:STDOUT:     import_ir_inst10: {ir_id: import_ir78000004, inst_id: inst7000001E}
// CHECK:STDOUT:     import_ir_inst11: {ir_id: import_ir78000004, inst_id: inst70000026}
// CHECK:STDOUT:     import_ir_inst12: {ir_id: import_ir78000004, inst_id: inst70000031}
// CHECK:STDOUT:     import_ir_inst13: {ir_id: import_ir78000004, inst_id: inst70000032}
// CHECK:STDOUT:     import_ir_inst14: {ir_id: import_ir78000004, inst_id: inst70000033}
// CHECK:STDOUT:     import_ir_inst15: {ir_id: import_ir78000004, inst_id: inst70000012}
// CHECK:STDOUT:     import_ir_inst16: {ir_id: import_ir78000004, inst_id: inst7000002D}
// CHECK:STDOUT:     import_ir_inst17: {ir_id: import_ir78000004, inst_id: inst70000072}
// CHECK:STDOUT:     import_ir_inst18: {ir_id: import_ir78000004, inst_id: inst70000070}
// CHECK:STDOUT:     import_ir_inst19: {ir_id: import_ir78000004, inst_id: inst70000089}
// CHECK:STDOUT:     import_ir_inst1A: {ir_id: import_ir78000004, inst_id: inst70000071}
// CHECK:STDOUT:     import_ir_inst1B: {ir_id: import_ir78000004, inst_id: inst70000089}
// CHECK:STDOUT:     import_ir_inst1C: {ir_id: import_ir78000004, inst_id: inst70000084}
// CHECK:STDOUT:     import_ir_inst1D: {ir_id: import_ir78000004, inst_id: inst70000085}
// CHECK:STDOUT:     import_ir_inst1E: {ir_id: import_ir78000004, inst_id: inst7000007C}
// CHECK:STDOUT:     import_ir_inst1F: {ir_id: import_ir78000004, inst_id: inst7000007F}
// CHECK:STDOUT:     import_ir_inst20: {ir_id: import_ir78000004, inst_id: inst70000080}
// CHECK:STDOUT:     import_ir_inst21: {ir_id: import_ir78000004, inst_id: inst70000081}
// CHECK:STDOUT:     import_ir_inst22: {ir_id: import_ir78000004, inst_id: inst70000062}
// CHECK:STDOUT:     import_ir_inst23: {ir_id: import_ir78000004, inst_id: inst70000078}
// CHECK:STDOUT:     import_ir_inst24: {ir_id: import_ir78000004, inst_id: inst70000079}
// CHECK:STDOUT:     import_ir_inst25: {ir_id: import_ir78000004, inst_id: inst7000007A}
// CHECK:STDOUT:     import_ir_inst26: {ir_id: import_ir78000004, inst_id: inst7000007D}
// CHECK:STDOUT:     import_ir_inst27: {ir_id: import_ir78000004, inst_id: inst70000083}
// CHECK:STDOUT:     import_ir_inst28: {ir_id: import_ir78000004, inst_id: inst70000093}
// CHECK:STDOUT:     import_ir_inst29: {ir_id: import_ir78000004, inst_id: inst7000009A}
// CHECK:STDOUT:     import_ir_inst2A: {ir_id: import_ir78000004, inst_id: inst700000A1}
// CHECK:STDOUT:     import_ir_inst2B: {ir_id: import_ir78000004, inst_id: inst700000A7}
// CHECK:STDOUT:     import_ir_inst2C: {ir_id: import_ir78000004, inst_id: inst700000A8}
// CHECK:STDOUT:     import_ir_inst2D: {ir_id: import_ir78000004, inst_id: inst700000A9}
// CHECK:STDOUT:     import_ir_inst2E: {ir_id: import_ir78000004, inst_id: inst700000AF}
// CHECK:STDOUT:     import_ir_inst2F: {ir_id: import_ir78000004, inst_id: inst70000065}
// CHECK:STDOUT:     import_ir_inst30: {ir_id: import_ir78000004, inst_id: inst7000006B}
// CHECK:STDOUT:     import_ir_inst31: {ir_id: import_ir78000004, inst_id: inst7000006E}
// CHECK:STDOUT:     import_ir_inst32: {ir_id: import_ir78000004, inst_id: inst70000062}
// CHECK:STDOUT:     import_ir_inst33: {ir_id: import_ir78000004, inst_id: inst70000064}
// CHECK:STDOUT:     import_ir_inst34: {ir_id: import_ir78000004, inst_id: inst70000069}
// CHECK:STDOUT:     import_ir_inst35: {ir_id: import_ir78000004, inst_id: inst7000006D}
// CHECK:STDOUT:     import_ir_inst36: {ir_id: import_ir78000004, inst_id: inst70000074}
// CHECK:STDOUT:     import_ir_inst37: {ir_id: import_ir78000004, inst_id: inst7000008C}
// CHECK:STDOUT:     import_ir_inst38: {ir_id: import_ir78000004, inst_id: inst7000008D}
// CHECK:STDOUT:     import_ir_inst39: {ir_id: import_ir78000004, inst_id: inst700000C2}
// CHECK:STDOUT:     import_ir_inst3A: {ir_id: import_ir78000004, inst_id: inst700000C0}
// CHECK:STDOUT:     import_ir_inst3B: {ir_id: import_ir78000004, inst_id: inst700000BE}
// CHECK:STDOUT:     import_ir_inst3C: {ir_id: import_ir78000004, inst_id: inst700000BF}
// CHECK:STDOUT:     import_ir_inst3D: {ir_id: import_ir78000004, inst_id: inst700000DE}
// CHECK:STDOUT:     import_ir_inst3E: {ir_id: import_ir78000004, inst_id: inst700000DC}
// CHECK:STDOUT:     import_ir_inst3F: {ir_id: import_ir78000004, inst_id: inst700000DA}
// CHECK:STDOUT:     import_ir_inst40: {ir_id: import_ir78000004, inst_id: inst700000DB}
// CHECK:STDOUT:     import_ir_inst41: {ir_id: import_ir78000004, inst_id: inst700000FA}
// CHECK:STDOUT:     import_ir_inst42: {ir_id: import_ir78000004, inst_id: inst700000F8}
// CHECK:STDOUT:     import_ir_inst43: {ir_id: import_ir78000004, inst_id: inst700000F6}
// CHECK:STDOUT:     import_ir_inst44: {ir_id: import_ir78000004, inst_id: inst700000F7}
// CHECK:STDOUT:     import_ir_inst45: {ir_id: import_ir78000004, inst_id: inst70000116}
// CHECK:STDOUT:     import_ir_inst46: {ir_id: import_ir78000004, inst_id: inst70000114}
// CHECK:STDOUT:     import_ir_inst47: {ir_id: import_ir78000004, inst_id: inst70000112}
// CHECK:STDOUT:     import_ir_inst48: {ir_id: import_ir78000004, inst_id: inst70000113}
// CHECK:STDOUT:     import_ir_inst49: {ir_id: import_ir78000004, inst_id: inst70000139}
// CHECK:STDOUT:     import_ir_inst4A: {ir_id: import_ir78000004, inst_id: inst70000137}
// CHECK:STDOUT:     import_ir_inst4B: {ir_id: import_ir78000004, inst_id: inst7000014D}
// CHECK:STDOUT:     import_ir_inst4C: {ir_id: import_ir78000004, inst_id: inst70000138}
// CHECK:STDOUT:     import_ir_inst4D: {ir_id: import_ir78000004, inst_id: inst70000130}
// CHECK:STDOUT:     import_ir_inst4E: {ir_id: import_ir78000004, inst_id: inst70000132}
// CHECK:STDOUT:     import_ir_inst4F: {ir_id: import_ir78000004, inst_id: inst70000135}
// CHECK:STDOUT:     import_ir_inst50: {ir_id: import_ir78000004, inst_id: inst7000012D}
// CHECK:STDOUT:     import_ir_inst51: {ir_id: import_ir78000004, inst_id: inst7000012F}
// CHECK:STDOUT:     import_ir_inst52: {ir_id: import_ir78000004, inst_id: inst70000134}
// CHECK:STDOUT:     import_ir_inst53: {ir_id: import_ir78000004, inst_id: inst7000013B}
// CHECK:STDOUT:     import_ir_inst54: {ir_id: import_ir78000004, inst_id: inst7000014D}
// CHECK:STDOUT:     import_ir_inst55: {ir_id: import_ir78000004, inst_id: inst70000148}
// CHECK:STDOUT:     import_ir_inst56: {ir_id: import_ir78000004, inst_id: inst70000149}
// CHECK:STDOUT:     import_ir_inst57: {ir_id: import_ir78000004, inst_id: inst70000140}
// CHECK:STDOUT:     import_ir_inst58: {ir_id: import_ir78000004, inst_id: inst70000143}
// CHECK:STDOUT:     import_ir_inst59: {ir_id: import_ir78000004, inst_id: inst70000144}
// CHECK:STDOUT:     import_ir_inst5A: {ir_id: import_ir78000004, inst_id: inst70000145}
// CHECK:STDOUT:     import_ir_inst5B: {ir_id: import_ir78000004, inst_id: inst7000012D}
// CHECK:STDOUT:     import_ir_inst5C: {ir_id: import_ir78000004, inst_id: inst7000013D}
// CHECK:STDOUT:     import_ir_inst5D: {ir_id: import_ir78000004, inst_id: inst7000013E}
// CHECK:STDOUT:     import_ir_inst5E: {ir_id: import_ir78000004, inst_id: inst70000141}
// CHECK:STDOUT:     import_ir_inst5F: {ir_id: import_ir78000004, inst_id: inst70000147}
// CHECK:STDOUT:     import_ir_inst60: {ir_id: import_ir78000004, inst_id: inst70000150}
// CHECK:STDOUT:     import_ir_inst61: {ir_id: import_ir78000004, inst_id: inst70000151}
// CHECK:STDOUT:     import_ir_inst62: {ir_id: import_ir78000004, inst_id: inst70000154}
// CHECK:STDOUT:     import_ir_inst63: {ir_id: import_ir78000004, inst_id: inst7000015C}
// CHECK:STDOUT:     import_ir_inst64: {ir_id: import_ir78000004, inst_id: inst7000015A}
// CHECK:STDOUT:     import_ir_inst65: {ir_id: import_ir78000004, inst_id: inst70000158}
// CHECK:STDOUT:     import_ir_inst66: {ir_id: import_ir78000004, inst_id: inst70000159}
// CHECK:STDOUT:     import_ir_inst67: {ir_id: import_ir78000004, inst_id: inst70000175}
// CHECK:STDOUT:     import_ir_inst68: {ir_id: import_ir78000004, inst_id: inst70000173}
// CHECK:STDOUT:     import_ir_inst69: {ir_id: import_ir78000004, inst_id: inst70000171}
// CHECK:STDOUT:     import_ir_inst6A: {ir_id: import_ir78000004, inst_id: inst70000172}
// CHECK:STDOUT:     import_ir_inst6B: {ir_id: import_ir78000004, inst_id: inst700001AB}
// CHECK:STDOUT:     import_ir_inst6C: {ir_id: import_ir78000004, inst_id: inst700001A9}
// CHECK:STDOUT:     import_ir_inst6D: {ir_id: import_ir78000004, inst_id: inst700001C6}
// CHECK:STDOUT:     import_ir_inst6E: {ir_id: import_ir78000004, inst_id: inst700001AA}
// CHECK:STDOUT:     import_ir_inst6F: {ir_id: import_ir78000004, inst_id: inst700001C6}
// CHECK:STDOUT:     import_ir_inst70: {ir_id: import_ir78000004, inst_id: inst700001C1}
// CHECK:STDOUT:     import_ir_inst71: {ir_id: import_ir78000004, inst_id: inst700001C2}
// CHECK:STDOUT:     import_ir_inst72: {ir_id: import_ir78000004, inst_id: inst700001B9}
// CHECK:STDOUT:     import_ir_inst73: {ir_id: import_ir78000004, inst_id: inst700001BC}
// CHECK:STDOUT:     import_ir_inst74: {ir_id: import_ir78000004, inst_id: inst700001BD}
// CHECK:STDOUT:     import_ir_inst75: {ir_id: import_ir78000004, inst_id: inst700001BE}
// CHECK:STDOUT:     import_ir_inst76: {ir_id: import_ir78000004, inst_id: inst7000018C}
// CHECK:STDOUT:     import_ir_inst77: {ir_id: import_ir78000004, inst_id: inst70000191}
// CHECK:STDOUT:     import_ir_inst78: {ir_id: import_ir78000004, inst_id: inst700001B3}
// CHECK:STDOUT:     import_ir_inst79: {ir_id: import_ir78000004, inst_id: inst700001B4}
// CHECK:STDOUT:     import_ir_inst7A: {ir_id: import_ir78000004, inst_id: inst700001B5}
// CHECK:STDOUT:     import_ir_inst7B: {ir_id: import_ir78000004, inst_id: inst700001B6}
// CHECK:STDOUT:     import_ir_inst7C: {ir_id: import_ir78000004, inst_id: inst700001B7}
// CHECK:STDOUT:     import_ir_inst7D: {ir_id: import_ir78000004, inst_id: inst700001BA}
// CHECK:STDOUT:     import_ir_inst7E: {ir_id: import_ir78000004, inst_id: inst700001C0}
// CHECK:STDOUT:     import_ir_inst7F: {ir_id: import_ir78000004, inst_id: inst700001D1}
// CHECK:STDOUT:     import_ir_inst80: {ir_id: import_ir78000004, inst_id: inst700001D7}
// CHECK:STDOUT:     import_ir_inst81: {ir_id: import_ir78000004, inst_id: inst700001DA}
// CHECK:STDOUT:     import_ir_inst82: {ir_id: import_ir78000004, inst_id: inst700001DC}
// CHECK:STDOUT:     import_ir_inst83: {ir_id: import_ir78000004, inst_id: inst700001DD}
// CHECK:STDOUT:     import_ir_inst84: {ir_id: import_ir78000004, inst_id: inst700001DE}
// CHECK:STDOUT:     import_ir_inst85: {ir_id: import_ir78000004, inst_id: inst700001E1}
// CHECK:STDOUT:     import_ir_inst86: {ir_id: import_ir78000004, inst_id: inst700001EB}
// CHECK:STDOUT:     import_ir_inst87: {ir_id: import_ir78000004, inst_id: inst700001F1}
// CHECK:STDOUT:     import_ir_inst88: {ir_id: import_ir78000004, inst_id: inst700001F5}
// CHECK:STDOUT:     import_ir_inst89: {ir_id: import_ir78000004, inst_id: inst700001F6}
// CHECK:STDOUT:     import_ir_inst8A: {ir_id: import_ir78000004, inst_id: inst700001F7}
// CHECK:STDOUT:     import_ir_inst8B: {ir_id: import_ir78000004, inst_id: inst700001FD}
// CHECK:STDOUT:     import_ir_inst8C: {ir_id: import_ir78000004, inst_id: inst70000194}
// CHECK:STDOUT:     import_ir_inst8D: {ir_id: import_ir78000004, inst_id: inst7000018E}
// CHECK:STDOUT:     import_ir_inst8E: {ir_id: import_ir78000004, inst_id: inst700001A4}
// CHECK:STDOUT:     import_ir_inst8F: {ir_id: import_ir78000004, inst_id: inst700001A6}
// CHECK:STDOUT:     import_ir_inst90: {ir_id: import_ir78000004, inst_id: inst7000018C}
// CHECK:STDOUT:     import_ir_inst91: {ir_id: import_ir78000004, inst_id: inst70000191}
// CHECK:STDOUT:     import_ir_inst92: {ir_id: import_ir78000004, inst_id: inst7000018D}
// CHECK:STDOUT:     import_ir_inst93: {ir_id: import_ir78000004, inst_id: inst70000193}
// CHECK:STDOUT:     import_ir_inst94: {ir_id: import_ir78000004, inst_id: inst7000019A}
// CHECK:STDOUT:     import_ir_inst95: {ir_id: import_ir78000004, inst_id: inst7000019D}
// CHECK:STDOUT:     import_ir_inst96: {ir_id: import_ir78000004, inst_id: inst700001A1}
// CHECK:STDOUT:     import_ir_inst97: {ir_id: import_ir78000004, inst_id: inst700001A5}
// CHECK:STDOUT:     import_ir_inst98: {ir_id: import_ir78000004, inst_id: inst700001AD}
// CHECK:STDOUT:     import_ir_inst99: {ir_id: import_ir78000004, inst_id: inst700001C9}
// CHECK:STDOUT:     import_ir_inst9A: {ir_id: import_ir78000004, inst_id: inst700001CA}
// CHECK:STDOUT:     import_ir_inst9B: {ir_id: import_ir78000004, inst_id: inst70000237}
// CHECK:STDOUT:     import_ir_inst9C: {ir_id: import_ir78000004, inst_id: inst70000235}
// CHECK:STDOUT:     import_ir_inst9D: {ir_id: import_ir78000004, inst_id: inst70000256}
// CHECK:STDOUT:     import_ir_inst9E: {ir_id: import_ir78000004, inst_id: inst70000236}
// CHECK:STDOUT:     import_ir_inst9F: {ir_id: import_ir78000004, inst_id: inst70000256}
// CHECK:STDOUT:     import_ir_instA0: {ir_id: import_ir78000004, inst_id: inst70000251}
// CHECK:STDOUT:     import_ir_instA1: {ir_id: import_ir78000004, inst_id: inst70000252}
// CHECK:STDOUT:     import_ir_instA2: {ir_id: import_ir78000004, inst_id: inst70000249}
// CHECK:STDOUT:     import_ir_instA3: {ir_id: import_ir78000004, inst_id: inst7000024C}
// CHECK:STDOUT:     import_ir_instA4: {ir_id: import_ir78000004, inst_id: inst7000024D}
// CHECK:STDOUT:     import_ir_instA5: {ir_id: import_ir78000004, inst_id: inst7000024E}
// CHECK:STDOUT:     import_ir_instA6: {ir_id: import_ir78000004, inst_id: inst7000020E}
// CHECK:STDOUT:     import_ir_instA7: {ir_id: import_ir78000004, inst_id: inst70000213}
// CHECK:STDOUT:     import_ir_instA8: {ir_id: import_ir78000004, inst_id: inst70000218}
// CHECK:STDOUT:     import_ir_instA9: {ir_id: import_ir78000004, inst_id: inst70000241}
// CHECK:STDOUT:     import_ir_instAA: {ir_id: import_ir78000004, inst_id: inst70000242}
// CHECK:STDOUT:     import_ir_instAB: {ir_id: import_ir78000004, inst_id: inst70000243}
// CHECK:STDOUT:     import_ir_instAC: {ir_id: import_ir78000004, inst_id: inst70000244}
// CHECK:STDOUT:     import_ir_instAD: {ir_id: import_ir78000004, inst_id: inst70000245}
// CHECK:STDOUT:     import_ir_instAE: {ir_id: import_ir78000004, inst_id: inst70000246}
// CHECK:STDOUT:     import_ir_instAF: {ir_id: import_ir78000004, inst_id: inst70000247}
// CHECK:STDOUT:     import_ir_instB0: {ir_id: import_ir78000004, inst_id: inst7000024A}
// CHECK:STDOUT:     import_ir_instB1: {ir_id: import_ir78000004, inst_id: inst70000250}
// CHECK:STDOUT:     import_ir_instB2: {ir_id: import_ir78000004, inst_id: inst70000261}
// CHECK:STDOUT:     import_ir_instB3: {ir_id: import_ir78000004, inst_id: inst70000266}
// CHECK:STDOUT:     import_ir_instB4: {ir_id: import_ir78000004, inst_id: inst70000269}
// CHECK:STDOUT:     import_ir_instB5: {ir_id: import_ir78000004, inst_id: inst7000026B}
// CHECK:STDOUT:     import_ir_instB6: {ir_id: import_ir78000004, inst_id: inst7000026C}
// CHECK:STDOUT:     import_ir_instB7: {ir_id: import_ir78000004, inst_id: inst7000026D}
// CHECK:STDOUT:     import_ir_instB8: {ir_id: import_ir78000004, inst_id: inst70000270}
// CHECK:STDOUT:     import_ir_instB9: {ir_id: import_ir78000004, inst_id: inst70000278}
// CHECK:STDOUT:     import_ir_instBA: {ir_id: import_ir78000004, inst_id: inst7000027B}
// CHECK:STDOUT:     import_ir_instBB: {ir_id: import_ir78000004, inst_id: inst7000027D}
// CHECK:STDOUT:     import_ir_instBC: {ir_id: import_ir78000004, inst_id: inst7000027E}
// CHECK:STDOUT:     import_ir_instBD: {ir_id: import_ir78000004, inst_id: inst7000027F}
// CHECK:STDOUT:     import_ir_instBE: {ir_id: import_ir78000004, inst_id: inst70000282}
// CHECK:STDOUT:     import_ir_instBF: {ir_id: import_ir78000004, inst_id: inst7000028C}
// CHECK:STDOUT:     import_ir_instC0: {ir_id: import_ir78000004, inst_id: inst70000292}
// CHECK:STDOUT:     import_ir_instC1: {ir_id: import_ir78000004, inst_id: inst70000296}
// CHECK:STDOUT:     import_ir_instC2: {ir_id: import_ir78000004, inst_id: inst70000297}
// CHECK:STDOUT:     import_ir_instC3: {ir_id: import_ir78000004, inst_id: inst70000298}
// CHECK:STDOUT:     import_ir_instC4: {ir_id: import_ir78000004, inst_id: inst7000029E}
// CHECK:STDOUT:     import_ir_instC5: {ir_id: import_ir78000004, inst_id: inst7000021B}
// CHECK:STDOUT:     import_ir_instC6: {ir_id: import_ir78000004, inst_id: inst70000215}
// CHECK:STDOUT:     import_ir_instC7: {ir_id: import_ir78000004, inst_id: inst70000210}
// CHECK:STDOUT:     import_ir_instC8: {ir_id: import_ir78000004, inst_id: inst7000022F}
// CHECK:STDOUT:     import_ir_instC9: {ir_id: import_ir78000004, inst_id: inst70000231}
// CHECK:STDOUT:     import_ir_instCA: {ir_id: import_ir78000004, inst_id: inst7000020E}
// CHECK:STDOUT:     import_ir_instCB: {ir_id: import_ir78000004, inst_id: inst70000213}
// CHECK:STDOUT:     import_ir_instCC: {ir_id: import_ir78000004, inst_id: inst70000218}
// CHECK:STDOUT:     import_ir_instCD: {ir_id: import_ir78000004, inst_id: inst7000020F}
// CHECK:STDOUT:     import_ir_instCE: {ir_id: import_ir78000004, inst_id: inst70000214}
// CHECK:STDOUT:     import_ir_instCF: {ir_id: import_ir78000004, inst_id: inst7000021A}
// CHECK:STDOUT:     import_ir_instD0: {ir_id: import_ir78000004, inst_id: inst70000222}
// CHECK:STDOUT:     import_ir_instD1: {ir_id: import_ir78000004, inst_id: inst70000225}
// CHECK:STDOUT:     import_ir_instD2: {ir_id: import_ir78000004, inst_id: inst70000228}
// CHECK:STDOUT:     import_ir_instD3: {ir_id: import_ir78000004, inst_id: inst7000022C}
// CHECK:STDOUT:     import_ir_instD4: {ir_id: import_ir78000004, inst_id: inst70000230}
// CHECK:STDOUT:     import_ir_instD5: {ir_id: import_ir78000004, inst_id: inst70000239}
// CHECK:STDOUT:     import_ir_instD6: {ir_id: import_ir78000004, inst_id: inst70000259}
// CHECK:STDOUT:     import_ir_instD7: {ir_id: import_ir78000004, inst_id: inst7000025A}
// CHECK:STDOUT:   clang_decls:     {}
// CHECK:STDOUT:   name_scopes:
// CHECK:STDOUT:     name_scope0:     {inst: instF, parent_scope: name_scope<none>, has_error: false, extended_scopes: [], names: {name(Core): inst78000011, name0: inst7800003E}}
// CHECK:STDOUT:     name_scope78000001: {inst: inst78000011, parent_scope: name_scope0, has_error: false, extended_scopes: [], names: {name3: inst7800004E}}
// CHECK:STDOUT:     name_scope78000002: {inst: inst7800004F, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {name(SelfType): inst7800006A}}
// CHECK:STDOUT:     name_scope78000003: {inst: inst78000051, parent_scope: name_scope78000002, has_error: false, extended_scopes: [], names: {name4: inst78000053}}
// CHECK:STDOUT:     name_scope78000004: {inst: inst78000070, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000005: {inst: inst780000A5, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000006: {inst: inst780000A9, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000007: {inst: inst780000AD, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000008: {inst: inst780000B1, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope78000009: {inst: inst780000B5, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000A: {inst: inst780000D3, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000B: {inst: inst780000D7, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000C: {inst: inst780000DB, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:     name_scope7800000D: {inst: inst7800011F, parent_scope: name_scope78000001, has_error: false, extended_scopes: [], names: {}}
// CHECK:STDOUT:   entity_names:
// CHECK:STDOUT:     entity_name78000000: {name: name(PeriodSelf), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000001: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000002: {name: name2, parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000003: {name: name3, parent_scope: name_scope78000001, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000004: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000005: {name: name4, parent_scope: name_scope78000003, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000006: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000007: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000008: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000009: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800000A: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800000B: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800000C: {name: name(SelfType), parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800000D: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800000E: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800000F: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000010: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000011: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000012: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000013: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000014: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000015: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000016: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000017: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000018: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000019: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800001A: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800001B: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800001C: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800001D: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800001E: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800001F: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000020: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000021: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000022: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000023: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000024: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000025: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000026: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000027: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000028: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000029: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800002A: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800002B: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800002C: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800002D: {name: name(SelfValue), parent_scope: name_scope<none>, index: -1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800002E: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800002F: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000030: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000031: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000032: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000033: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000034: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000035: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000036: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000037: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000038: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name78000039: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800003A: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800003B: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800003C: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800003D: {name: name6, parent_scope: name_scope<none>, index: 2, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800003E: {name: name5, parent_scope: name_scope<none>, index: 1, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:     entity_name7800003F: {name: name1, parent_scope: name_scope<none>, index: 0, is_template: 0, is_unused: 0, form: constant<none>}
// CHECK:STDOUT:   cpp_global_vars: {}
// CHECK:STDOUT:   functions:
// CHECK:STDOUT:     function78000000: {name: name0, parent_scope: name_scope0, call_param_patterns_id: inst_block78000011, call_params_id: inst_block78000012, return_type_inst_id: inst78000030, return_form_inst_id: inst78000032, return_patterns_id: inst_block78000010, body: [inst_block78000019]}
// CHECK:STDOUT:     function78000001: {name: name4, parent_scope: name_scope78000003, call_param_patterns_id: inst_block7800001F, return_type_inst_id: inst78000060, return_form_inst_id: inst78000061, return_patterns_id: inst_block78000021}
// CHECK:STDOUT:     function78000002: {name: name4, parent_scope: name_scope78000004, call_param_patterns_id: inst_block7800002B, return_type_inst_id: inst78000080, return_form_inst_id: inst78000081, return_patterns_id: inst_block7800002D}
// CHECK:STDOUT:     function78000003: {name: name4, parent_scope: name_scope78000009, call_param_patterns_id: inst_block78000042, return_type_inst_id: inst780000C8, return_form_inst_id: inst780000C9, return_patterns_id: inst_block78000044}
// CHECK:STDOUT:     function78000004: {name: name4, parent_scope: name_scope7800000C, call_param_patterns_id: inst_block78000050, return_type_inst_id: inst780000ED, return_form_inst_id: inst780000EE, return_patterns_id: inst_block78000052}
// CHECK:STDOUT:     function78000005: {name: name4, parent_scope: name_scope7800000D, call_param_patterns_id: inst_block7800006D, return_type_inst_id: inst78000131, return_form_inst_id: inst78000132, return_patterns_id: inst_block7800006F}
// CHECK:STDOUT:   classes:         {}
// CHECK:STDOUT:   interfaces:
// CHECK:STDOUT:     interface78000000: {name: name3, parent_scope: name_scope78000001, require_impls_block_id: require_block_empty}
// CHECK:STDOUT:   associated_constants: {}
// CHECK:STDOUT:   impls:
// CHECK:STDOUT:     impl78000000:    {self: inst7800009B, constraint: inst7800009C, witness: inst7800006F}
// CHECK:STDOUT:     impl78000001:    {self: inst780000A6, constraint: inst780000A7, witness: inst780000A4}
// CHECK:STDOUT:     impl78000002:    {self: inst780000AA, constraint: inst780000AB, witness: inst780000A8}
// CHECK:STDOUT:     impl78000003:    {self: inst780000AE, constraint: inst780000AF, witness: inst780000AC}
// CHECK:STDOUT:     impl78000004:    {self: inst780000B2, constraint: inst780000B3, witness: inst780000B0}
// CHECK:STDOUT:     impl78000005:    {self: inst780000BA, constraint: inst780000BB, witness: inst780000B4}
// CHECK:STDOUT:     impl78000006:    {self: inst780000D4, constraint: inst780000D5, witness: inst780000D2}
// CHECK:STDOUT:     impl78000007:    {self: inst780000D8, constraint: inst780000D9, witness: inst780000D6}
// CHECK:STDOUT:     impl78000008:    {self: inst78000111, constraint: inst78000112, witness: inst780000DA}
// CHECK:STDOUT:     impl78000009:    {self: inst7800015F, constraint: inst78000160, witness: inst7800011E}
// CHECK:STDOUT:   generics:
// CHECK:STDOUT:     generic78000000: {decl: inst7800003E, bindings: inst_block78000015, self_specific_id: specific78000000, decl_block_id: inst_block78000017, definition_block_id: inst_block78000091}
// CHECK:STDOUT:     generic78000001: {decl: inst78000051, bindings: inst_block7800001C, self_specific_id: specific78000001, decl_block_id: inst_block_empty, definition_block_id: inst_block78000026}
// CHECK:STDOUT:     generic78000002: {decl: inst78000056, bindings: inst_block78000022, self_specific_id: specific78000002, decl_block_id: inst_block78000023, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     generic78000003: {decl: inst78000070, bindings: inst_block78000037, self_specific_id: specific78000004, decl_block_id: inst_block78000039, definition_block_id: inst_block7800003B}
// CHECK:STDOUT:     generic78000004: {decl: inst78000077, bindings: inst_block7800002E, self_specific_id: specific78000007, decl_block_id: inst_block7800002F, definition_block_id: inst_block78000034}
// CHECK:STDOUT:     generic78000005: {decl: inst780000B5, bindings: inst_block7800003F, self_specific_id: specific7800000B, decl_block_id: inst_block78000041, definition_block_id: inst_block78000049}
// CHECK:STDOUT:     generic78000006: {decl: inst780000C0, bindings: inst_block78000045, self_specific_id: specific7800000D, decl_block_id: inst_block78000046, definition_block_id: inst_block_empty}
// CHECK:STDOUT:     generic78000007: {decl: inst780000DB, bindings: inst_block78000060, self_specific_id: specific7800000E, decl_block_id: inst_block78000064, definition_block_id: inst_block78000066}
// CHECK:STDOUT:     generic78000008: {decl: inst780000E2, bindings: inst_block78000053, self_specific_id: specific78000011, decl_block_id: inst_block78000055, definition_block_id: inst_block7800005D}
// CHECK:STDOUT:     generic78000009: {decl: inst7800011F, bindings: inst_block7800007F, self_specific_id: specific78000017, decl_block_id: inst_block78000083, definition_block_id: inst_block78000085}
// CHECK:STDOUT:     generic7800000A: {decl: inst78000126, bindings: inst_block78000070, self_specific_id: specific7800001A, decl_block_id: inst_block78000072, definition_block_id: inst_block7800007C}
// CHECK:STDOUT:   specifics:
// CHECK:STDOUT:     specific78000000: {generic: generic78000000, args: inst_block78000016, decl_block_id: inst_block78000018, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000001: {generic: generic78000001, args: inst_block7800001D, decl_block_id: inst_block_empty, definition_block_id: inst_block7800001E}
// CHECK:STDOUT:     specific78000002: {generic: generic78000002, args: inst_block7800001D, decl_block_id: inst_block78000024, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000003: {generic: generic78000001, args: inst_block78000025, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000004: {generic: generic78000003, args: inst_block78000028, decl_block_id: inst_block78000029, definition_block_id: inst_block7800002A}
// CHECK:STDOUT:     specific78000005: {generic: generic78000001, args: inst_block78000028, decl_block_id: inst_block_empty, definition_block_id: inst_block78000030}
// CHECK:STDOUT:     specific78000006: {generic: generic78000002, args: inst_block78000028, decl_block_id: inst_block78000031, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000007: {generic: generic78000004, args: inst_block78000028, decl_block_id: inst_block78000035, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000008: {generic: generic78000001, args: inst_block78000032, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000009: {generic: generic78000002, args: inst_block78000032, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800000A: {generic: generic78000003, args: inst_block78000038, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800000B: {generic: generic78000005, args: inst_block78000016, decl_block_id: inst_block7800003D, definition_block_id: inst_block78000086}
// CHECK:STDOUT:     specific7800000C: {generic: generic78000005, args: inst_block78000040, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800000D: {generic: generic78000006, args: inst_block78000016, decl_block_id: inst_block78000047, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800000E: {generic: generic78000007, args: inst_block7800004C, decl_block_id: inst_block7800004E, definition_block_id: inst_block7800004F}
// CHECK:STDOUT:     specific7800000F: {generic: generic78000001, args: inst_block78000056, decl_block_id: inst_block_empty, definition_block_id: inst_block78000057}
// CHECK:STDOUT:     specific78000010: {generic: generic78000002, args: inst_block78000056, decl_block_id: inst_block78000058, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000011: {generic: generic78000008, args: inst_block7800004C, decl_block_id: inst_block7800005E, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000012: {generic: generic78000001, args: inst_block78000059, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000013: {generic: generic78000002, args: inst_block78000059, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000014: {generic: generic78000001, args: inst_block7800005B, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000015: {generic: generic78000002, args: inst_block7800005B, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000016: {generic: generic78000007, args: inst_block78000061, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000017: {generic: generic78000009, args: inst_block78000069, decl_block_id: inst_block7800006B, definition_block_id: inst_block7800006C}
// CHECK:STDOUT:     specific78000018: {generic: generic78000001, args: inst_block78000073, decl_block_id: inst_block_empty, definition_block_id: inst_block78000074}
// CHECK:STDOUT:     specific78000019: {generic: generic78000002, args: inst_block78000073, decl_block_id: inst_block78000075, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800001A: {generic: generic7800000A, args: inst_block78000069, decl_block_id: inst_block7800007D, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800001B: {generic: generic78000001, args: inst_block78000076, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800001C: {generic: generic78000002, args: inst_block78000076, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800001D: {generic: generic78000001, args: inst_block78000078, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800001E: {generic: generic78000002, args: inst_block78000078, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific7800001F: {generic: generic78000001, args: inst_block7800007A, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000020: {generic: generic78000002, args: inst_block7800007A, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000021: {generic: generic78000009, args: inst_block78000080, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000022: {generic: generic78000005, args: inst_block78000087, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000023: {generic: generic78000001, args: inst_block78000089, decl_block_id: inst_block_empty, definition_block_id: inst_block7800008A}
// CHECK:STDOUT:     specific78000024: {generic: generic78000001, args: inst_block7800008C, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000025: {generic: generic78000002, args: inst_block78000089, decl_block_id: inst_block7800008D, definition_block_id: inst_block<none>}
// CHECK:STDOUT:     specific78000026: {generic: generic78000002, args: inst_block7800008C, decl_block_id: inst_block<none>, definition_block_id: inst_block<none>}
// CHECK:STDOUT:   specific_interfaces:
// CHECK:STDOUT:     specific_interface78000000: {interface_id: interface78000000, specific_id: specific<none>}
// CHECK:STDOUT:   struct_type_fields:
// CHECK:STDOUT:     struct_type_fields_empty: {}
// CHECK:STDOUT:   types:
// CHECK:STDOUT:     'type(TypeType)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(TypeType)}
// CHECK:STDOUT:     'type(inst(FormType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(FormType))}
// CHECK:STDOUT:     'type(Error)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(Error)}
// CHECK:STDOUT:     'type(inst(NamespaceType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     'type(inst78000026)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst78000026)}
// CHECK:STDOUT:     'type(inst7800002D)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst7800002D)}
// CHECK:STDOUT:     'type(inst78000029)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(inst7800002D)}
// CHECK:STDOUT:     'type(inst7800003F)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst78000026)}
// CHECK:STDOUT:     'type(symbolic_constant78000003)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant78000003)}
// CHECK:STDOUT:     'type(inst(WitnessType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(WitnessType))}
// CHECK:STDOUT:     'type(symbolic_constant78000011)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant78000011)}
// CHECK:STDOUT:     'type(symbolic_constant78000009)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant78000011)}
// CHECK:STDOUT:     'type(symbolic_constant78000004)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(symbolic_constant78000004)}
// CHECK:STDOUT:     'type(symbolic_constant7800000A)':
// CHECK:STDOUT:       value_repr:      {kind: pointer, type: type(symbolic_constant78000011)}
// CHECK:STDOUT:     'type(inst(InstType))':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst78000026)}
// CHECK:STDOUT:     'type(inst78000050)':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst78000050)}
// CHECK:STDOUT:     'type(inst(SpecificFunctionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     'type(inst(RequireSpecificDefinitionType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     'type(symbolic_constant78000146)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst78000026)}
// CHECK:STDOUT:     'type(symbolic_constant7800014A)':
// CHECK:STDOUT:       value_repr:      {kind: none, type: type(inst78000026)}
// CHECK:STDOUT:     'type(inst(BoundMethodType))':
// CHECK:STDOUT:       value_repr:      {kind: copy, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:   facet_types:
// CHECK:STDOUT:     facet_type78000000: {}
// CHECK:STDOUT:     facet_type78000001: {extends interface: interface78000000}
// CHECK:STDOUT:   insts:
// CHECK:STDOUT:     instF:           {kind: Namespace, arg0: name_scope0, arg1: inst<none>, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst78000010:    {kind: ImportDecl, arg0: name(Core)}
// CHECK:STDOUT:     inst78000011:    {kind: Namespace, arg0: name_scope78000001, arg1: inst78000010, type: type(inst(NamespaceType))}
// CHECK:STDOUT:     inst78000012:    {kind: FacetType, arg0: facet_type78000000, type: type(TypeType)}
// CHECK:STDOUT:     inst78000013:    {kind: SymbolicBinding, arg0: entity_name78000000, arg1: inst<none>, type: type(inst78000012)}
// CHECK:STDOUT:     inst78000014:    {kind: SymbolicBinding, arg0: entity_name78000000, arg1: inst<none>, type: type(inst78000012)}
// CHECK:STDOUT:     inst78000015:    {kind: TypeLiteral, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst78000016:    {kind: PatternType, arg0: inst(TypeType), type: type(TypeType)}
// CHECK:STDOUT:     inst78000017:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000018:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000019:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001A:    {kind: SymbolicBindingPattern, arg0: entity_name78000001, type: type(inst78000016)}
// CHECK:STDOUT:     inst7800001B:    {kind: NameRef, arg0: name1, arg1: inst78000017, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001C:    {kind: PointerType, arg0: inst7800001B, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001D:    {kind: PointerType, arg0: inst78000018, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001E:    {kind: PointerType, arg0: inst78000019, type: type(TypeType)}
// CHECK:STDOUT:     inst7800001F:    {kind: PatternType, arg0: inst7800001D, type: type(TypeType)}
// CHECK:STDOUT:     inst78000020:    {kind: ValueParamPattern, arg0: name2, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000021:    {kind: PatternType, arg0: inst7800001E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000022:    {kind: ValueBinding, arg0: entity_name78000002, arg1: inst7800003A, type: type(symbolic_constant78000004)}
// CHECK:STDOUT:     inst78000023:    {kind: WrapperBindingPattern, arg0: entity_name78000002, arg1: inst78000020, type: type(symbolic_constant78000006)}
// CHECK:STDOUT:     inst78000024:    {kind: NameRef, arg0: name1, arg1: inst78000017, type: type(TypeType)}
// CHECK:STDOUT:     inst78000025:    {kind: PointerType, arg0: inst78000024, type: type(TypeType)}
// CHECK:STDOUT:     inst78000026:    {kind: TupleType, arg0: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst78000027:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst78000026)}
// CHECK:STDOUT:     inst78000028:    {kind: TupleValue, arg0: inst_block_empty, type: type(inst78000026)}
// CHECK:STDOUT:     inst78000029:    {kind: TupleType, arg0: inst_block7800000A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002A:    {kind: TupleLiteral, arg0: inst_block78000009, type: type(inst78000029)}
// CHECK:STDOUT:     inst7800002B:    {kind: TupleValue, arg0: inst_block7800000B, type: type(inst78000029)}
// CHECK:STDOUT:     inst7800002C:    {kind: TupleValue, arg0: inst_block7800000C, type: type(inst78000029)}
// CHECK:STDOUT:     inst7800002D:    {kind: PointerType, arg0: inst78000029, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002E:    {kind: Converted, arg0: inst78000028, arg1: inst78000026, type: type(TypeType)}
// CHECK:STDOUT:     inst7800002F:    {kind: TupleType, arg0: inst_block7800000E, type: type(TypeType)}
// CHECK:STDOUT:     inst78000030:    {kind: Converted, arg0: inst7800002A, arg1: inst7800002F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000031:    {kind: TupleType, arg0: inst_block7800000F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000032:    {kind: InitForm, arg0: inst78000030, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000033:    {kind: InitForm, arg0: inst7800002F, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000034:    {kind: InitForm, arg0: inst78000031, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000035:    {kind: PatternType, arg0: inst7800002F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000036:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant7800000E)}
// CHECK:STDOUT:     inst78000037:    {kind: PatternType, arg0: inst78000031, type: type(TypeType)}
// CHECK:STDOUT:     inst78000038:    {kind: ReturnSlotPattern, arg0: inst78000036, arg1: inst78000030, type: type(symbolic_constant7800000E)}
// CHECK:STDOUT:     inst78000039:    {kind: SpliceBlock, arg0: inst_block78000005, arg1: inst78000015, type: type(TypeType)}
// CHECK:STDOUT:     inst7800003A:    {kind: ValueParam, arg0: call_param0, arg1: name2, type: type(symbolic_constant78000004)}
// CHECK:STDOUT:     inst7800003B:    {kind: SpliceBlock, arg0: inst_block78000007, arg1: inst7800001C, type: type(TypeType)}
// CHECK:STDOUT:     inst7800003C:    {kind: OutParam, arg0: call_param1, arg1: name(ReturnSlot), type: type(symbolic_constant7800000A)}
// CHECK:STDOUT:     inst7800003D:    {kind: ReturnSlot, arg0: inst7800002F, arg1: inst7800003C, type: type(symbolic_constant7800000A)}
// CHECK:STDOUT:     inst7800003E:    {kind: FunctionDecl, arg0: function78000000, arg1: inst_block78000014, type: type(inst7800003F)}
// CHECK:STDOUT:     inst7800003F:    {kind: FunctionType, arg0: function78000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000040:    {kind: StructValue, arg0: inst_block_empty, type: type(inst7800003F)}
// CHECK:STDOUT:     inst78000041:    {kind: RequireCompleteType, arg0: inst7800001D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000042:    {kind: RequireCompleteType, arg0: inst7800001D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000043:    {kind: RequireCompleteType, arg0: inst7800001E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000044:    {kind: PointerType, arg0: inst7800002F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000045:    {kind: RequireCompleteType, arg0: inst7800002F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000046:    {kind: RequireCompleteType, arg0: inst7800002F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000047:    {kind: RequireCompleteType, arg0: inst78000031, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000048:    {kind: NameRef, arg0: name2, arg1: inst78000022, type: type(symbolic_constant78000004)}
// CHECK:STDOUT:     inst78000049:    {kind: TupleLiteral, arg0: inst_block_empty, type: type(inst78000026)}
// CHECK:STDOUT:     inst7800004A:    {kind: TupleLiteral, arg0: inst_block7800001A, type: type(symbolic_constant7800000A)}
// CHECK:STDOUT:     inst7800004B:    {kind: RequireCompleteType, arg0: inst7800002F, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004C:    {kind: TupleAccess, arg0: inst7800003C, arg1: element0, type: type(symbolic_constant78000004)}
// CHECK:STDOUT:     inst7800004D:    {kind: RequireCompleteType, arg0: inst7800001D, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800004E:    {kind: ImportRefLoaded, arg0: import_ir_inst0, arg1: entity_name78000003, type: type(TypeType)}
// CHECK:STDOUT:     inst7800004F:    {kind: InterfaceDecl, arg0: interface78000000, arg1: inst_block_empty, type: type(TypeType)}
// CHECK:STDOUT:     inst78000050:    {kind: FacetType, arg0: facet_type78000001, type: type(TypeType)}
// CHECK:STDOUT:     inst78000051:    {kind: InterfaceWithSelfDecl, arg0: interface78000000}
// CHECK:STDOUT:     inst78000052:    {kind: SymbolicBinding, arg0: entity_name78000004, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000053:    {kind: ImportRefLoaded, arg0: import_ir_inst3, arg1: entity_name<none>, type: type(inst7800006B)}
// CHECK:STDOUT:     inst78000054:    {kind: ImportRefUnloaded, arg0: import_ir_inst4, arg1: entity_name78000005}
// CHECK:STDOUT:     inst78000055:    {kind: ImportRefLoaded, arg0: import_ir_inst5, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000056:    {kind: FunctionDecl, arg0: function78000001, arg1: inst_block_empty, type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst78000057:    {kind: FunctionType, arg0: function78000001, arg1: specific78000001, type: type(TypeType)}
// CHECK:STDOUT:     inst78000058:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000015)}
// CHECK:STDOUT:     inst78000059:    {kind: SymbolicBindingType, arg0: entity_name78000004, arg1: inst78000052, type: type(TypeType)}
// CHECK:STDOUT:     inst7800005A:    {kind: PatternType, arg0: inst78000059, type: type(TypeType)}
// CHECK:STDOUT:     inst7800005B:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant7800001B)}
// CHECK:STDOUT:     inst7800005C:    {kind: ReturnSlotPattern, arg0: inst7800005B, arg1: inst<none>, type: type(symbolic_constant7800001B)}
// CHECK:STDOUT:     inst7800005D:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant7800001B)}
// CHECK:STDOUT:     inst7800005E:    {kind: WrapperBindingPattern, arg0: entity_name78000008, arg1: inst7800005D, type: type(symbolic_constant7800001B)}
// CHECK:STDOUT:     inst7800005F:    {kind: InitForm, arg0: inst78000059, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000060:    {kind: ImportRefLoaded, arg0: import_ir_instB, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000061:    {kind: ImportRefLoaded, arg0: import_ir_instC, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000062:    {kind: ImportRefLoaded, arg0: import_ir_instD, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000063:    {kind: SymbolicBinding, arg0: entity_name78000004, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000064:    {kind: SymbolicBindingType, arg0: entity_name78000004, arg1: inst78000063, type: type(TypeType)}
// CHECK:STDOUT:     inst78000065:    {kind: PatternType, arg0: inst78000064, type: type(TypeType)}
// CHECK:STDOUT:     inst78000066:    {kind: InitForm, arg0: inst78000064, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000067:    {kind: SymbolicBinding, arg0: entity_name78000004, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000068:    {kind: FunctionType, arg0: function78000001, arg1: specific78000003, type: type(TypeType)}
// CHECK:STDOUT:     inst78000069:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000029)}
// CHECK:STDOUT:     inst7800006A:    {kind: ImportRefUnloaded, arg0: import_ir_inst15, arg1: entity_name<none>}
// CHECK:STDOUT:     inst7800006B:    {kind: AssociatedEntityType, arg0: interface78000000, arg1: specific<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800006C:    {kind: ImportRefLoaded, arg0: import_ir_inst16, arg1: entity_name<none>, type: type(symbolic_constant78000019)}
// CHECK:STDOUT:     inst7800006D:    {kind: AssociatedEntity, arg0: element0, arg1: inst7800006C, type: type(inst7800006B)}
// CHECK:STDOUT:     inst7800006E:    {kind: LookupImplWitness, arg0: inst7800001D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800006F:    {kind: ImportRefUnloaded, arg0: import_ir_inst17, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000070:    {kind: ImplDecl, arg0: impl78000000, arg1: inst_block_empty}
// CHECK:STDOUT:     inst78000071:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000072:    {kind: SymbolicBindingType, arg0: entity_name78000001, arg1: inst78000071, type: type(TypeType)}
// CHECK:STDOUT:     inst78000073:    {kind: ConstType, arg0: inst78000072, type: type(TypeType)}
// CHECK:STDOUT:     inst78000074:    {kind: ImportRefUnloaded, arg0: import_ir_inst19, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000075:    {kind: ImplWitnessTable, arg0: inst_block78000027, arg1: impl78000000}
// CHECK:STDOUT:     inst78000076:    {kind: ImplWitness, arg0: inst78000075, arg1: specific78000004, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000077:    {kind: FunctionDecl, arg0: function78000002, arg1: inst_block_empty, type: type(symbolic_constant78000031)}
// CHECK:STDOUT:     inst78000078:    {kind: FunctionType, arg0: function78000002, arg1: specific78000004, type: type(TypeType)}
// CHECK:STDOUT:     inst78000079:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000031)}
// CHECK:STDOUT:     inst7800007A:    {kind: PatternType, arg0: inst78000073, type: type(TypeType)}
// CHECK:STDOUT:     inst7800007B:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000036)}
// CHECK:STDOUT:     inst7800007C:    {kind: ReturnSlotPattern, arg0: inst7800007B, arg1: inst<none>, type: type(symbolic_constant78000036)}
// CHECK:STDOUT:     inst7800007D:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000036)}
// CHECK:STDOUT:     inst7800007E:    {kind: WrapperBindingPattern, arg0: entity_name7800000F, arg1: inst7800007D, type: type(symbolic_constant78000036)}
// CHECK:STDOUT:     inst7800007F:    {kind: InitForm, arg0: inst78000073, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000080:    {kind: ImportRefLoaded, arg0: import_ir_inst20, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000081:    {kind: ImportRefLoaded, arg0: import_ir_inst21, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000082:    {kind: ImportRefLoaded, arg0: import_ir_inst22, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000083:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000084:    {kind: SymbolicBindingType, arg0: entity_name78000001, arg1: inst78000083, type: type(TypeType)}
// CHECK:STDOUT:     inst78000085:    {kind: ConstType, arg0: inst78000084, type: type(TypeType)}
// CHECK:STDOUT:     inst78000086:    {kind: PatternType, arg0: inst78000085, type: type(TypeType)}
// CHECK:STDOUT:     inst78000087:    {kind: InitForm, arg0: inst78000085, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000088:    {kind: LookupImplWitness, arg0: inst78000071, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000089:    {kind: FunctionType, arg0: function78000001, arg1: specific78000005, type: type(TypeType)}
// CHECK:STDOUT:     inst7800008A:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000045)}
// CHECK:STDOUT:     inst7800008B:    {kind: FunctionTypeWithSelfType, arg0: inst78000089, arg1: inst78000071, type: type(TypeType)}
// CHECK:STDOUT:     inst7800008C:    {kind: ImplWitnessAccess, arg0: inst78000088, arg1: element0, type: type(symbolic_constant78000047)}
// CHECK:STDOUT:     inst7800008D:    {kind: SpecificImplFunction, arg0: inst7800008C, arg1: specific78000006, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst7800008E:    {kind: InitForm, arg0: inst78000072, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800008F:    {kind: PatternType, arg0: inst78000072, type: type(TypeType)}
// CHECK:STDOUT:     inst78000090:    {kind: RequireCompleteType, arg0: inst78000072, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000091:    {kind: RequireCompleteType, arg0: inst78000073, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000092:    {kind: RequireCompleteType, arg0: inst78000085, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000093:    {kind: RequireCompleteType, arg0: inst78000084, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000094:    {kind: LookupImplWitness, arg0: inst78000083, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000095:    {kind: FunctionType, arg0: function78000001, arg1: specific78000008, type: type(TypeType)}
// CHECK:STDOUT:     inst78000096:    {kind: FunctionTypeWithSelfType, arg0: inst78000095, arg1: inst78000083, type: type(TypeType)}
// CHECK:STDOUT:     inst78000097:    {kind: ImplWitnessAccess, arg0: inst78000094, arg1: element0, type: type(symbolic_constant78000059)}
// CHECK:STDOUT:     inst78000098:    {kind: SpecificImplFunction, arg0: inst78000097, arg1: specific78000009, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000099:    {kind: PatternType, arg0: inst78000050, type: type(TypeType)}
// CHECK:STDOUT:     inst7800009A:    {kind: SymbolicBindingPattern, arg0: entity_name78000015, type: type(inst78000099)}
// CHECK:STDOUT:     inst7800009B:    {kind: ImportRefLoaded, arg0: import_ir_inst30, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800009C:    {kind: ImportRefLoaded, arg0: import_ir_inst31, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst7800009D:    {kind: ImportRefLoaded, arg0: import_ir_inst32, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst7800009E:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst7800009F:    {kind: SymbolicBindingType, arg0: entity_name78000001, arg1: inst7800009E, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A0:    {kind: ConstType, arg0: inst7800009F, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A1:    {kind: ImplWitness, arg0: inst78000075, arg1: specific7800000A, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000A2:    {kind: FunctionType, arg0: function78000002, arg1: specific7800000A, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A3:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000064)}
// CHECK:STDOUT:     inst780000A4:    {kind: ImportRefUnloaded, arg0: import_ir_inst39, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000A5:    {kind: ImplDecl, arg0: impl78000001, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000A6:    {kind: ImportRefLoaded, arg0: import_ir_inst3B, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A7:    {kind: ImportRefLoaded, arg0: import_ir_inst3C, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000A8:    {kind: ImportRefUnloaded, arg0: import_ir_inst3D, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000A9:    {kind: ImplDecl, arg0: impl78000002, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000AA:    {kind: ImportRefLoaded, arg0: import_ir_inst3F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000AB:    {kind: ImportRefLoaded, arg0: import_ir_inst40, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000AC:    {kind: ImportRefUnloaded, arg0: import_ir_inst41, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000AD:    {kind: ImplDecl, arg0: impl78000003, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000AE:    {kind: ImportRefLoaded, arg0: import_ir_inst43, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000AF:    {kind: ImportRefLoaded, arg0: import_ir_inst44, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B0:    {kind: ImportRefUnloaded, arg0: import_ir_inst45, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000B1:    {kind: ImplDecl, arg0: impl78000004, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000B2:    {kind: ImportRefLoaded, arg0: import_ir_inst47, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B3:    {kind: ImportRefLoaded, arg0: import_ir_inst48, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000B4:    {kind: ImportRefLoaded, arg0: import_ir_inst49, arg1: entity_name<none>, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000B5:    {kind: ImplDecl, arg0: impl78000005, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000B6:    {kind: ImportRefUnloaded, arg0: import_ir_inst4B, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000B7:    {kind: ImplWitnessTable, arg0: inst_block7800003C, arg1: impl78000005}
// CHECK:STDOUT:     inst780000B8:    {kind: ImplWitness, arg0: inst780000B7, arg1: specific7800000B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000B9:    {kind: SymbolicBindingPattern, arg0: entity_name78000019, type: type(inst78000016)}
// CHECK:STDOUT:     inst780000BA:    {kind: ImportRefLoaded, arg0: import_ir_inst4E, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000BB:    {kind: ImportRefLoaded, arg0: import_ir_inst4F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000BC:    {kind: ImportRefLoaded, arg0: import_ir_inst50, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000BD:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000BE:    {kind: PointerType, arg0: inst780000BD, type: type(TypeType)}
// CHECK:STDOUT:     inst780000BF:    {kind: ImplWitness, arg0: inst780000B7, arg1: specific7800000C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000C0:    {kind: FunctionDecl, arg0: function78000003, arg1: inst_block_empty, type: type(symbolic_constant78000070)}
// CHECK:STDOUT:     inst780000C1:    {kind: FunctionType, arg0: function78000003, arg1: specific7800000B, type: type(TypeType)}
// CHECK:STDOUT:     inst780000C2:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000070)}
// CHECK:STDOUT:     inst780000C3:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant78000074)}
// CHECK:STDOUT:     inst780000C4:    {kind: ReturnSlotPattern, arg0: inst780000C3, arg1: inst<none>, type: type(symbolic_constant78000074)}
// CHECK:STDOUT:     inst780000C5:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant78000074)}
// CHECK:STDOUT:     inst780000C6:    {kind: WrapperBindingPattern, arg0: entity_name7800001A, arg1: inst780000C5, type: type(symbolic_constant78000074)}
// CHECK:STDOUT:     inst780000C7:    {kind: InitForm, arg0: inst7800001D, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000C8:    {kind: ImportRefLoaded, arg0: import_ir_inst59, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000C9:    {kind: ImportRefLoaded, arg0: import_ir_inst5A, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000CA:    {kind: ImportRefLoaded, arg0: import_ir_inst5B, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000CB:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000CC:    {kind: PointerType, arg0: inst780000CB, type: type(TypeType)}
// CHECK:STDOUT:     inst780000CD:    {kind: PatternType, arg0: inst780000CC, type: type(TypeType)}
// CHECK:STDOUT:     inst780000CE:    {kind: InitForm, arg0: inst780000CC, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000CF:    {kind: FunctionType, arg0: function78000003, arg1: specific7800000C, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D0:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000080)}
// CHECK:STDOUT:     inst780000D1:    {kind: RequireCompleteType, arg0: inst780000BE, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000D2:    {kind: ImportRefUnloaded, arg0: import_ir_inst63, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000D3:    {kind: ImplDecl, arg0: impl78000006, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000D4:    {kind: ImportRefLoaded, arg0: import_ir_inst65, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D5:    {kind: ImportRefLoaded, arg0: import_ir_inst66, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D6:    {kind: ImportRefUnloaded, arg0: import_ir_inst67, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000D7:    {kind: ImplDecl, arg0: impl78000007, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000D8:    {kind: ImportRefLoaded, arg0: import_ir_inst69, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000D9:    {kind: ImportRefLoaded, arg0: import_ir_inst6A, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DA:    {kind: ImportRefUnloaded, arg0: import_ir_inst6B, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000DB:    {kind: ImplDecl, arg0: impl78000008, arg1: inst_block_empty}
// CHECK:STDOUT:     inst780000DC:    {kind: SymbolicBinding, arg0: entity_name7800001C, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst780000DD:    {kind: SymbolicBindingType, arg0: entity_name7800001C, arg1: inst780000DC, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DE:    {kind: TupleType, arg0: inst_block7800004A, type: type(TypeType)}
// CHECK:STDOUT:     inst780000DF:    {kind: ImportRefUnloaded, arg0: import_ir_inst6D, arg1: entity_name<none>}
// CHECK:STDOUT:     inst780000E0:    {kind: ImplWitnessTable, arg0: inst_block7800004B, arg1: impl78000008}
// CHECK:STDOUT:     inst780000E1:    {kind: ImplWitness, arg0: inst780000E0, arg1: specific7800000E, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000E2:    {kind: FunctionDecl, arg0: function78000004, arg1: inst_block_empty, type: type(symbolic_constant78000089)}
// CHECK:STDOUT:     inst780000E3:    {kind: FunctionType, arg0: function78000004, arg1: specific7800000E, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E4:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000089)}
// CHECK:STDOUT:     inst780000E5:    {kind: TupleType, arg0: inst_block7800004D, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E6:    {kind: TupleValue, arg0: inst_block7800004C, type: type(inst780000E5)}
// CHECK:STDOUT:     inst780000E7:    {kind: PatternType, arg0: inst780000DE, type: type(TypeType)}
// CHECK:STDOUT:     inst780000E8:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant7800008F)}
// CHECK:STDOUT:     inst780000E9:    {kind: ReturnSlotPattern, arg0: inst780000E8, arg1: inst<none>, type: type(symbolic_constant7800008F)}
// CHECK:STDOUT:     inst780000EA:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant7800008F)}
// CHECK:STDOUT:     inst780000EB:    {kind: WrapperBindingPattern, arg0: entity_name7800001E, arg1: inst780000EA, type: type(symbolic_constant7800008F)}
// CHECK:STDOUT:     inst780000EC:    {kind: InitForm, arg0: inst780000DE, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000ED:    {kind: ImportRefLoaded, arg0: import_ir_inst74, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst780000EE:    {kind: ImportRefLoaded, arg0: import_ir_inst75, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000EF:    {kind: ImportRefLoaded, arg0: import_ir_inst76, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst780000F0:    {kind: ImportRefLoaded, arg0: import_ir_inst77, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst780000F1:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst780000F2:    {kind: SymbolicBindingType, arg0: entity_name78000001, arg1: inst780000F1, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F3:    {kind: SymbolicBinding, arg0: entity_name7800001C, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst780000F4:    {kind: SymbolicBindingType, arg0: entity_name7800001C, arg1: inst780000F3, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F5:    {kind: TupleType, arg0: inst_block78000054, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F6:    {kind: PatternType, arg0: inst780000F5, type: type(TypeType)}
// CHECK:STDOUT:     inst780000F7:    {kind: InitForm, arg0: inst780000F5, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000F8:    {kind: LookupImplWitness, arg0: inst780000DC, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst780000F9:    {kind: FunctionType, arg0: function78000001, arg1: specific7800000F, type: type(TypeType)}
// CHECK:STDOUT:     inst780000FA:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780000A3)}
// CHECK:STDOUT:     inst780000FB:    {kind: FunctionTypeWithSelfType, arg0: inst780000F9, arg1: inst780000DC, type: type(TypeType)}
// CHECK:STDOUT:     inst780000FC:    {kind: ImplWitnessAccess, arg0: inst780000F8, arg1: element0, type: type(symbolic_constant780000A5)}
// CHECK:STDOUT:     inst780000FD:    {kind: SpecificImplFunction, arg0: inst780000FC, arg1: specific78000010, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst780000FE:    {kind: InitForm, arg0: inst780000DD, type: type(inst(FormType))}
// CHECK:STDOUT:     inst780000FF:    {kind: PatternType, arg0: inst780000DD, type: type(TypeType)}
// CHECK:STDOUT:     inst78000100:    {kind: RequireCompleteType, arg0: inst780000DD, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000101:    {kind: RequireCompleteType, arg0: inst780000DE, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000102:    {kind: RequireCompleteType, arg0: inst780000F5, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000103:    {kind: RequireCompleteType, arg0: inst780000F2, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000104:    {kind: LookupImplWitness, arg0: inst780000F1, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000105:    {kind: FunctionType, arg0: function78000001, arg1: specific78000012, type: type(TypeType)}
// CHECK:STDOUT:     inst78000106:    {kind: FunctionTypeWithSelfType, arg0: inst78000105, arg1: inst780000F1, type: type(TypeType)}
// CHECK:STDOUT:     inst78000107:    {kind: ImplWitnessAccess, arg0: inst78000104, arg1: element0, type: type(symbolic_constant780000BD)}
// CHECK:STDOUT:     inst78000108:    {kind: SpecificImplFunction, arg0: inst78000107, arg1: specific78000013, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000109:    {kind: RequireCompleteType, arg0: inst780000F4, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800010A:    {kind: LookupImplWitness, arg0: inst780000F3, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800010B:    {kind: FunctionType, arg0: function78000001, arg1: specific78000014, type: type(TypeType)}
// CHECK:STDOUT:     inst7800010C:    {kind: FunctionTypeWithSelfType, arg0: inst7800010B, arg1: inst780000F3, type: type(TypeType)}
// CHECK:STDOUT:     inst7800010D:    {kind: ImplWitnessAccess, arg0: inst7800010A, arg1: element0, type: type(symbolic_constant780000C3)}
// CHECK:STDOUT:     inst7800010E:    {kind: SpecificImplFunction, arg0: inst7800010D, arg1: specific78000015, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst7800010F:    {kind: SymbolicBindingPattern, arg0: entity_name78000029, type: type(inst78000099)}
// CHECK:STDOUT:     inst78000110:    {kind: SymbolicBindingPattern, arg0: entity_name7800002A, type: type(inst78000099)}
// CHECK:STDOUT:     inst78000111:    {kind: ImportRefLoaded, arg0: import_ir_inst8E, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000112:    {kind: ImportRefLoaded, arg0: import_ir_inst8F, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000113:    {kind: ImportRefLoaded, arg0: import_ir_inst90, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000114:    {kind: ImportRefLoaded, arg0: import_ir_inst91, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000115:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000116:    {kind: SymbolicBinding, arg0: entity_name7800001C, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000117:    {kind: TupleValue, arg0: inst_block78000061, type: type(inst780000E5)}
// CHECK:STDOUT:     inst78000118:    {kind: SymbolicBindingType, arg0: entity_name78000001, arg1: inst78000115, type: type(TypeType)}
// CHECK:STDOUT:     inst78000119:    {kind: SymbolicBindingType, arg0: entity_name7800001C, arg1: inst78000116, type: type(TypeType)}
// CHECK:STDOUT:     inst7800011A:    {kind: TupleType, arg0: inst_block78000062, type: type(TypeType)}
// CHECK:STDOUT:     inst7800011B:    {kind: ImplWitness, arg0: inst780000E0, arg1: specific78000016, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800011C:    {kind: FunctionType, arg0: function78000004, arg1: specific78000016, type: type(TypeType)}
// CHECK:STDOUT:     inst7800011D:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780000D4)}
// CHECK:STDOUT:     inst7800011E:    {kind: ImportRefUnloaded, arg0: import_ir_inst9B, arg1: entity_name<none>}
// CHECK:STDOUT:     inst7800011F:    {kind: ImplDecl, arg0: impl78000009, arg1: inst_block_empty}
// CHECK:STDOUT:     inst78000120:    {kind: SymbolicBinding, arg0: entity_name7800002B, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000121:    {kind: SymbolicBindingType, arg0: entity_name7800002B, arg1: inst78000120, type: type(TypeType)}
// CHECK:STDOUT:     inst78000122:    {kind: TupleType, arg0: inst_block78000067, type: type(TypeType)}
// CHECK:STDOUT:     inst78000123:    {kind: ImportRefUnloaded, arg0: import_ir_inst9D, arg1: entity_name<none>}
// CHECK:STDOUT:     inst78000124:    {kind: ImplWitnessTable, arg0: inst_block78000068, arg1: impl78000009}
// CHECK:STDOUT:     inst78000125:    {kind: ImplWitness, arg0: inst78000124, arg1: specific78000017, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000126:    {kind: FunctionDecl, arg0: function78000005, arg1: inst_block_empty, type: type(symbolic_constant780000DC)}
// CHECK:STDOUT:     inst78000127:    {kind: FunctionType, arg0: function78000005, arg1: specific78000017, type: type(TypeType)}
// CHECK:STDOUT:     inst78000128:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780000DC)}
// CHECK:STDOUT:     inst78000129:    {kind: TupleType, arg0: inst_block7800006A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800012A:    {kind: TupleValue, arg0: inst_block78000069, type: type(inst78000129)}
// CHECK:STDOUT:     inst7800012B:    {kind: PatternType, arg0: inst78000122, type: type(TypeType)}
// CHECK:STDOUT:     inst7800012C:    {kind: OutParamPattern, arg0: name(ReturnSlot), type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst7800012D:    {kind: ReturnSlotPattern, arg0: inst7800012C, arg1: inst<none>, type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst7800012E:    {kind: ValueParamPattern, arg0: name(SelfValue), type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst7800012F:    {kind: WrapperBindingPattern, arg0: entity_name7800002D, arg1: inst7800012E, type: type(symbolic_constant780000E2)}
// CHECK:STDOUT:     inst78000130:    {kind: InitForm, arg0: inst78000122, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000131:    {kind: ImportRefLoaded, arg0: import_ir_instA4, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000132:    {kind: ImportRefLoaded, arg0: import_ir_instA5, arg1: entity_name<none>, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000133:    {kind: ImportRefLoaded, arg0: import_ir_instA6, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000134:    {kind: ImportRefLoaded, arg0: import_ir_instA7, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000135:    {kind: ImportRefLoaded, arg0: import_ir_instA8, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000136:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000137:    {kind: SymbolicBindingType, arg0: entity_name78000001, arg1: inst78000136, type: type(TypeType)}
// CHECK:STDOUT:     inst78000138:    {kind: SymbolicBinding, arg0: entity_name7800001C, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000139:    {kind: SymbolicBindingType, arg0: entity_name7800001C, arg1: inst78000138, type: type(TypeType)}
// CHECK:STDOUT:     inst7800013A:    {kind: SymbolicBinding, arg0: entity_name7800002B, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst7800013B:    {kind: SymbolicBindingType, arg0: entity_name7800002B, arg1: inst7800013A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800013C:    {kind: TupleType, arg0: inst_block78000071, type: type(TypeType)}
// CHECK:STDOUT:     inst7800013D:    {kind: PatternType, arg0: inst7800013C, type: type(TypeType)}
// CHECK:STDOUT:     inst7800013E:    {kind: InitForm, arg0: inst7800013C, type: type(inst(FormType))}
// CHECK:STDOUT:     inst7800013F:    {kind: LookupImplWitness, arg0: inst78000120, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000140:    {kind: FunctionType, arg0: function78000001, arg1: specific78000018, type: type(TypeType)}
// CHECK:STDOUT:     inst78000141:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant780000FB)}
// CHECK:STDOUT:     inst78000142:    {kind: FunctionTypeWithSelfType, arg0: inst78000140, arg1: inst78000120, type: type(TypeType)}
// CHECK:STDOUT:     inst78000143:    {kind: ImplWitnessAccess, arg0: inst7800013F, arg1: element0, type: type(symbolic_constant780000FD)}
// CHECK:STDOUT:     inst78000144:    {kind: SpecificImplFunction, arg0: inst78000143, arg1: specific78000019, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000145:    {kind: InitForm, arg0: inst78000121, type: type(inst(FormType))}
// CHECK:STDOUT:     inst78000146:    {kind: PatternType, arg0: inst78000121, type: type(TypeType)}
// CHECK:STDOUT:     inst78000147:    {kind: RequireCompleteType, arg0: inst78000121, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000148:    {kind: RequireCompleteType, arg0: inst78000122, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000149:    {kind: RequireCompleteType, arg0: inst7800013C, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800014A:    {kind: RequireCompleteType, arg0: inst78000137, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800014B:    {kind: LookupImplWitness, arg0: inst78000136, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800014C:    {kind: FunctionType, arg0: function78000001, arg1: specific7800001B, type: type(TypeType)}
// CHECK:STDOUT:     inst7800014D:    {kind: FunctionTypeWithSelfType, arg0: inst7800014C, arg1: inst78000136, type: type(TypeType)}
// CHECK:STDOUT:     inst7800014E:    {kind: ImplWitnessAccess, arg0: inst7800014B, arg1: element0, type: type(symbolic_constant7800011B)}
// CHECK:STDOUT:     inst7800014F:    {kind: SpecificImplFunction, arg0: inst7800014E, arg1: specific7800001C, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000150:    {kind: RequireCompleteType, arg0: inst78000139, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000151:    {kind: LookupImplWitness, arg0: inst78000138, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000152:    {kind: FunctionType, arg0: function78000001, arg1: specific7800001D, type: type(TypeType)}
// CHECK:STDOUT:     inst78000153:    {kind: FunctionTypeWithSelfType, arg0: inst78000152, arg1: inst78000138, type: type(TypeType)}
// CHECK:STDOUT:     inst78000154:    {kind: ImplWitnessAccess, arg0: inst78000151, arg1: element0, type: type(symbolic_constant78000121)}
// CHECK:STDOUT:     inst78000155:    {kind: SpecificImplFunction, arg0: inst78000154, arg1: specific7800001E, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000156:    {kind: RequireCompleteType, arg0: inst7800013B, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000157:    {kind: LookupImplWitness, arg0: inst7800013A, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000158:    {kind: FunctionType, arg0: function78000001, arg1: specific7800001F, type: type(TypeType)}
// CHECK:STDOUT:     inst78000159:    {kind: FunctionTypeWithSelfType, arg0: inst78000158, arg1: inst7800013A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800015A:    {kind: ImplWitnessAccess, arg0: inst78000157, arg1: element0, type: type(symbolic_constant78000127)}
// CHECK:STDOUT:     inst7800015B:    {kind: SpecificImplFunction, arg0: inst7800015A, arg1: specific78000020, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst7800015C:    {kind: SymbolicBindingPattern, arg0: entity_name7800003D, type: type(inst78000099)}
// CHECK:STDOUT:     inst7800015D:    {kind: SymbolicBindingPattern, arg0: entity_name7800003E, type: type(inst78000099)}
// CHECK:STDOUT:     inst7800015E:    {kind: SymbolicBindingPattern, arg0: entity_name7800003F, type: type(inst78000099)}
// CHECK:STDOUT:     inst7800015F:    {kind: ImportRefLoaded, arg0: import_ir_instC8, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000160:    {kind: ImportRefLoaded, arg0: import_ir_instC9, arg1: entity_name<none>, type: type(TypeType)}
// CHECK:STDOUT:     inst78000161:    {kind: ImportRefLoaded, arg0: import_ir_instCA, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000162:    {kind: ImportRefLoaded, arg0: import_ir_instCB, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000163:    {kind: ImportRefLoaded, arg0: import_ir_instCC, arg1: entity_name<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000164:    {kind: SymbolicBinding, arg0: entity_name78000001, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000165:    {kind: SymbolicBinding, arg0: entity_name7800001C, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000166:    {kind: SymbolicBinding, arg0: entity_name7800002B, arg1: inst<none>, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000167:    {kind: TupleValue, arg0: inst_block78000080, type: type(inst78000129)}
// CHECK:STDOUT:     inst78000168:    {kind: SymbolicBindingType, arg0: entity_name78000001, arg1: inst78000164, type: type(TypeType)}
// CHECK:STDOUT:     inst78000169:    {kind: SymbolicBindingType, arg0: entity_name7800001C, arg1: inst78000165, type: type(TypeType)}
// CHECK:STDOUT:     inst7800016A:    {kind: SymbolicBindingType, arg0: entity_name7800002B, arg1: inst78000166, type: type(TypeType)}
// CHECK:STDOUT:     inst7800016B:    {kind: TupleType, arg0: inst_block78000081, type: type(TypeType)}
// CHECK:STDOUT:     inst7800016C:    {kind: ImplWitness, arg0: inst78000124, arg1: specific78000021, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst7800016D:    {kind: FunctionType, arg0: function78000005, arg1: specific78000021, type: type(TypeType)}
// CHECK:STDOUT:     inst7800016E:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant7800013C)}
// CHECK:STDOUT:     inst7800016F:    {kind: RequireSpecificDefinition, arg0: specific7800000B, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     inst78000170:    {kind: RequireSpecificDefinition, arg0: specific7800000B, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     inst78000171:    {kind: RequireSpecificDefinition, arg0: specific78000022, type: type(inst(RequireSpecificDefinitionType))}
// CHECK:STDOUT:     inst78000172:    {kind: LookupImplWitness, arg0: inst7800001D, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000173:    {kind: LookupImplWitness, arg0: inst7800001E, arg1: specific_interface78000000, type: type(inst(WitnessType))}
// CHECK:STDOUT:     inst78000174:    {kind: FacetValue, arg0: inst7800001D, arg1: inst_block78000088, type: type(inst78000050)}
// CHECK:STDOUT:     inst78000175:    {kind: FunctionType, arg0: function78000001, arg1: specific78000023, type: type(TypeType)}
// CHECK:STDOUT:     inst78000176:    {kind: StructValue, arg0: inst_block_empty, type: type(symbolic_constant78000144)}
// CHECK:STDOUT:     inst78000177:    {kind: FunctionTypeWithSelfType, arg0: inst78000175, arg1: inst78000174, type: type(TypeType)}
// CHECK:STDOUT:     inst78000178:    {kind: ImplWitnessAccess, arg0: inst78000172, arg1: element0, type: type(symbolic_constant7800014A)}
// CHECK:STDOUT:     inst78000179:    {kind: ImplWitnessAccess, arg0: inst78000172, arg1: element0, type: type(symbolic_constant78000146)}
// CHECK:STDOUT:     inst7800017A:    {kind: FacetValue, arg0: inst7800001E, arg1: inst_block7800008B, type: type(inst78000050)}
// CHECK:STDOUT:     inst7800017B:    {kind: FunctionType, arg0: function78000001, arg1: specific78000024, type: type(TypeType)}
// CHECK:STDOUT:     inst7800017C:    {kind: FunctionTypeWithSelfType, arg0: inst7800017B, arg1: inst7800017A, type: type(TypeType)}
// CHECK:STDOUT:     inst7800017D:    {kind: ImplWitnessAccess, arg0: inst78000173, arg1: element0, type: type(symbolic_constant7800014A)}
// CHECK:STDOUT:     inst7800017E:    {kind: BoundMethod, arg0: inst78000048, arg1: inst78000178, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst7800017F:    {kind: SpecificImplFunction, arg0: inst78000178, arg1: specific78000025, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000180:    {kind: SpecificImplFunction, arg0: inst78000179, arg1: specific78000025, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000181:    {kind: SpecificImplFunction, arg0: inst7800017D, arg1: specific78000026, type: type(inst(SpecificFunctionType))}
// CHECK:STDOUT:     inst78000182:    {kind: BoundMethod, arg0: inst78000048, arg1: inst7800017F, type: type(inst(BoundMethodType))}
// CHECK:STDOUT:     inst78000183:    {kind: Call, arg0: inst78000182, arg1: inst_block7800008F, type: type(symbolic_constant78000004)}
// CHECK:STDOUT:     inst78000184:    {kind: InPlaceInit, arg0: inst78000183, arg1: inst7800004C, type: type(symbolic_constant78000004)}
// CHECK:STDOUT:     inst78000185:    {kind: TupleAccess, arg0: inst7800003C, arg1: element1, type: type(inst78000026)}
// CHECK:STDOUT:     inst78000186:    {kind: TupleInit, arg0: inst_block_empty, arg1: inst<none>, type: type(inst78000026)}
// CHECK:STDOUT:     inst78000187:    {kind: Converted, arg0: inst78000049, arg1: inst78000186, type: type(inst78000026)}
// CHECK:STDOUT:     inst78000188:    {kind: TupleInit, arg0: inst_block78000090, arg1: inst7800003C, type: type(symbolic_constant7800000A)}
// CHECK:STDOUT:     inst78000189:    {kind: Converted, arg0: inst7800004A, arg1: inst78000188, type: type(symbolic_constant7800000A)}
// CHECK:STDOUT:     inst7800018A:    {kind: ReturnExpr, arg0: inst78000189, arg1: inst7800003C}
// CHECK:STDOUT:   constant_values:
// CHECK:STDOUT:     values:
// CHECK:STDOUT:       instF:           concrete_constant(instF)
// CHECK:STDOUT:       inst78000011:    concrete_constant(inst78000011)
// CHECK:STDOUT:       inst78000012:    concrete_constant(inst78000012)
// CHECK:STDOUT:       inst78000013:    symbolic_constant78000000
// CHECK:STDOUT:       inst78000014:    symbolic_constant78000000
// CHECK:STDOUT:       inst78000015:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst78000016:    concrete_constant(inst78000016)
// CHECK:STDOUT:       inst78000017:    symbolic_constant78000002
// CHECK:STDOUT:       inst78000018:    symbolic_constant78000001
// CHECK:STDOUT:       inst78000019:    symbolic_constant78000002
// CHECK:STDOUT:       inst7800001A:    concrete_constant(inst7800001A)
// CHECK:STDOUT:       inst7800001B:    symbolic_constant78000002
// CHECK:STDOUT:       inst7800001C:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800001D:    symbolic_constant78000003
// CHECK:STDOUT:       inst7800001E:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800001F:    symbolic_constant78000005
// CHECK:STDOUT:       inst78000020:    concrete_constant(inst78000020)
// CHECK:STDOUT:       inst78000021:    symbolic_constant78000006
// CHECK:STDOUT:       inst78000023:    concrete_constant(inst78000023)
// CHECK:STDOUT:       inst78000024:    symbolic_constant78000002
// CHECK:STDOUT:       inst78000025:    symbolic_constant78000004
// CHECK:STDOUT:       inst78000026:    concrete_constant(inst78000026)
// CHECK:STDOUT:       inst78000027:    concrete_constant(inst78000028)
// CHECK:STDOUT:       inst78000028:    concrete_constant(inst78000028)
// CHECK:STDOUT:       inst78000029:    concrete_constant(inst78000029)
// CHECK:STDOUT:       inst7800002A:    symbolic_constant78000008
// CHECK:STDOUT:       inst7800002B:    symbolic_constant78000007
// CHECK:STDOUT:       inst7800002C:    symbolic_constant78000008
// CHECK:STDOUT:       inst7800002D:    concrete_constant(inst7800002D)
// CHECK:STDOUT:       inst7800002E:    concrete_constant(inst78000026)
// CHECK:STDOUT:       inst7800002F:    symbolic_constant78000009
// CHECK:STDOUT:       inst78000030:    symbolic_constant7800000A
// CHECK:STDOUT:       inst78000031:    symbolic_constant7800000A
// CHECK:STDOUT:       inst78000032:    symbolic_constant7800000C
// CHECK:STDOUT:       inst78000033:    symbolic_constant7800000B
// CHECK:STDOUT:       inst78000034:    symbolic_constant7800000C
// CHECK:STDOUT:       inst78000035:    symbolic_constant7800000D
// CHECK:STDOUT:       inst78000036:    concrete_constant(inst78000036)
// CHECK:STDOUT:       inst78000037:    symbolic_constant7800000E
// CHECK:STDOUT:       inst78000038:    concrete_constant(inst78000038)
// CHECK:STDOUT:       inst78000039:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst7800003B:    symbolic_constant78000004
// CHECK:STDOUT:       inst7800003E:    concrete_constant(inst78000040)
// CHECK:STDOUT:       inst7800003F:    concrete_constant(inst7800003F)
// CHECK:STDOUT:       inst78000040:    concrete_constant(inst78000040)
// CHECK:STDOUT:       inst78000041:    symbolic_constant78000010
// CHECK:STDOUT:       inst78000042:    symbolic_constant7800000F
// CHECK:STDOUT:       inst78000043:    symbolic_constant78000010
// CHECK:STDOUT:       inst78000044:    symbolic_constant78000011
// CHECK:STDOUT:       inst78000045:    symbolic_constant78000013
// CHECK:STDOUT:       inst78000046:    symbolic_constant78000012
// CHECK:STDOUT:       inst78000047:    symbolic_constant78000013
// CHECK:STDOUT:       inst78000049:    concrete_constant(inst78000028)
// CHECK:STDOUT:       inst7800004B:    symbolic_constant78000013
// CHECK:STDOUT:       inst7800004D:    symbolic_constant78000010
// CHECK:STDOUT:       inst7800004E:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst7800004F:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst78000050:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst78000051:    concrete_constant(inst78000051)
// CHECK:STDOUT:       inst78000052:    symbolic_constant78000014
// CHECK:STDOUT:       inst78000053:    concrete_constant(inst7800006D)
// CHECK:STDOUT:       inst78000054:    constant<none>
// CHECK:STDOUT:       inst78000055:    symbolic_constant78000014
// CHECK:STDOUT:       inst78000056:    symbolic_constant78000017
// CHECK:STDOUT:       inst78000057:    symbolic_constant78000015
// CHECK:STDOUT:       inst78000058:    symbolic_constant78000016
// CHECK:STDOUT:       inst78000059:    symbolic_constant78000018
// CHECK:STDOUT:       inst7800005A:    symbolic_constant7800001A
// CHECK:STDOUT:       inst7800005B:    concrete_constant(inst7800005B)
// CHECK:STDOUT:       inst7800005C:    concrete_constant(inst7800005C)
// CHECK:STDOUT:       inst7800005D:    concrete_constant(inst7800005D)
// CHECK:STDOUT:       inst7800005E:    concrete_constant(inst7800005E)
// CHECK:STDOUT:       inst7800005F:    symbolic_constant7800001C
// CHECK:STDOUT:       inst78000060:    symbolic_constant78000021
// CHECK:STDOUT:       inst78000061:    symbolic_constant78000020
// CHECK:STDOUT:       inst78000062:    symbolic_constant78000014
// CHECK:STDOUT:       inst78000063:    symbolic_constant78000022
// CHECK:STDOUT:       inst78000064:    symbolic_constant78000023
// CHECK:STDOUT:       inst78000065:    symbolic_constant78000024
// CHECK:STDOUT:       inst78000066:    symbolic_constant78000025
// CHECK:STDOUT:       inst78000067:    symbolic_constant78000028
// CHECK:STDOUT:       inst78000068:    symbolic_constant78000029
// CHECK:STDOUT:       inst78000069:    symbolic_constant7800002A
// CHECK:STDOUT:       inst7800006A:    constant<none>
// CHECK:STDOUT:       inst7800006B:    concrete_constant(inst7800006B)
// CHECK:STDOUT:       inst7800006C:    symbolic_constant78000017
// CHECK:STDOUT:       inst7800006D:    concrete_constant(inst7800006D)
// CHECK:STDOUT:       inst7800006E:    symbolic_constant78000142
// CHECK:STDOUT:       inst7800006F:    constant<none>
// CHECK:STDOUT:       inst78000070:    concrete_constant(inst78000070)
// CHECK:STDOUT:       inst78000071:    symbolic_constant7800002B
// CHECK:STDOUT:       inst78000072:    symbolic_constant7800002C
// CHECK:STDOUT:       inst78000073:    symbolic_constant7800002D
// CHECK:STDOUT:       inst78000074:    constant<none>
// CHECK:STDOUT:       inst78000075:    concrete_constant(inst78000075)
// CHECK:STDOUT:       inst78000076:    symbolic_constant7800002F
// CHECK:STDOUT:       inst78000077:    symbolic_constant78000033
// CHECK:STDOUT:       inst78000078:    symbolic_constant78000031
// CHECK:STDOUT:       inst78000079:    symbolic_constant78000032
// CHECK:STDOUT:       inst7800007A:    symbolic_constant78000035
// CHECK:STDOUT:       inst7800007B:    concrete_constant(inst7800007B)
// CHECK:STDOUT:       inst7800007C:    concrete_constant(inst7800007C)
// CHECK:STDOUT:       inst7800007D:    concrete_constant(inst7800007D)
// CHECK:STDOUT:       inst7800007E:    concrete_constant(inst7800007E)
// CHECK:STDOUT:       inst7800007F:    symbolic_constant78000037
// CHECK:STDOUT:       inst78000080:    symbolic_constant7800003E
// CHECK:STDOUT:       inst78000081:    symbolic_constant7800003D
// CHECK:STDOUT:       inst78000082:    symbolic_constant7800003C
// CHECK:STDOUT:       inst78000083:    symbolic_constant7800003F
// CHECK:STDOUT:       inst78000084:    symbolic_constant78000040
// CHECK:STDOUT:       inst78000085:    symbolic_constant78000041
// CHECK:STDOUT:       inst78000086:    symbolic_constant78000042
// CHECK:STDOUT:       inst78000087:    symbolic_constant78000043
// CHECK:STDOUT:       inst78000088:    symbolic_constant78000044
// CHECK:STDOUT:       inst78000089:    symbolic_constant78000045
// CHECK:STDOUT:       inst7800008A:    symbolic_constant78000046
// CHECK:STDOUT:       inst7800008B:    symbolic_constant78000047
// CHECK:STDOUT:       inst7800008C:    symbolic_constant78000048
// CHECK:STDOUT:       inst7800008D:    symbolic_constant78000049
// CHECK:STDOUT:       inst7800008E:    symbolic_constant7800004B
// CHECK:STDOUT:       inst7800008F:    symbolic_constant7800004C
// CHECK:STDOUT:       inst78000090:    symbolic_constant78000051
// CHECK:STDOUT:       inst78000091:    symbolic_constant78000053
// CHECK:STDOUT:       inst78000092:    symbolic_constant78000055
// CHECK:STDOUT:       inst78000093:    symbolic_constant78000056
// CHECK:STDOUT:       inst78000094:    symbolic_constant78000057
// CHECK:STDOUT:       inst78000095:    symbolic_constant78000058
// CHECK:STDOUT:       inst78000096:    symbolic_constant78000059
// CHECK:STDOUT:       inst78000097:    symbolic_constant7800005A
// CHECK:STDOUT:       inst78000098:    symbolic_constant7800005B
// CHECK:STDOUT:       inst78000099:    concrete_constant(inst78000099)
// CHECK:STDOUT:       inst7800009A:    concrete_constant(inst7800009A)
// CHECK:STDOUT:       inst7800009B:    symbolic_constant7800002E
// CHECK:STDOUT:       inst7800009C:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst7800009D:    symbolic_constant7800003C
// CHECK:STDOUT:       inst7800009E:    symbolic_constant7800005F
// CHECK:STDOUT:       inst7800009F:    symbolic_constant78000060
// CHECK:STDOUT:       inst780000A0:    symbolic_constant78000061
// CHECK:STDOUT:       inst780000A1:    symbolic_constant78000062
// CHECK:STDOUT:       inst780000A2:    symbolic_constant78000064
// CHECK:STDOUT:       inst780000A3:    symbolic_constant78000065
// CHECK:STDOUT:       inst780000A4:    constant<none>
// CHECK:STDOUT:       inst780000A5:    concrete_constant(inst780000A5)
// CHECK:STDOUT:       inst780000A6:    concrete_constant(inst(BoolType))
// CHECK:STDOUT:       inst780000A7:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst780000A8:    constant<none>
// CHECK:STDOUT:       inst780000A9:    concrete_constant(inst780000A9)
// CHECK:STDOUT:       inst780000AA:    concrete_constant(inst(CharLiteralType))
// CHECK:STDOUT:       inst780000AB:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst780000AC:    constant<none>
// CHECK:STDOUT:       inst780000AD:    concrete_constant(inst780000AD)
// CHECK:STDOUT:       inst780000AE:    concrete_constant(inst(FloatLiteralType))
// CHECK:STDOUT:       inst780000AF:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst780000B0:    constant<none>
// CHECK:STDOUT:       inst780000B1:    concrete_constant(inst780000B1)
// CHECK:STDOUT:       inst780000B2:    concrete_constant(inst(IntLiteralType))
// CHECK:STDOUT:       inst780000B3:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst780000B4:    symbolic_constant7800013E
// CHECK:STDOUT:       inst780000B5:    concrete_constant(inst780000B5)
// CHECK:STDOUT:       inst780000B6:    constant<none>
// CHECK:STDOUT:       inst780000B7:    concrete_constant(inst780000B7)
// CHECK:STDOUT:       inst780000B8:    symbolic_constant78000067
// CHECK:STDOUT:       inst780000B9:    concrete_constant(inst780000B9)
// CHECK:STDOUT:       inst780000BA:    symbolic_constant78000066
// CHECK:STDOUT:       inst780000BB:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst780000BC:    symbolic_constant7800006B
// CHECK:STDOUT:       inst780000BD:    symbolic_constant7800006C
// CHECK:STDOUT:       inst780000BE:    symbolic_constant7800006D
// CHECK:STDOUT:       inst780000BF:    symbolic_constant7800006E
// CHECK:STDOUT:       inst780000C0:    symbolic_constant78000072
// CHECK:STDOUT:       inst780000C1:    symbolic_constant78000070
// CHECK:STDOUT:       inst780000C2:    symbolic_constant78000071
// CHECK:STDOUT:       inst780000C3:    concrete_constant(inst780000C3)
// CHECK:STDOUT:       inst780000C4:    concrete_constant(inst780000C4)
// CHECK:STDOUT:       inst780000C5:    concrete_constant(inst780000C5)
// CHECK:STDOUT:       inst780000C6:    concrete_constant(inst780000C6)
// CHECK:STDOUT:       inst780000C7:    symbolic_constant78000075
// CHECK:STDOUT:       inst780000C8:    symbolic_constant7800007A
// CHECK:STDOUT:       inst780000C9:    symbolic_constant78000079
// CHECK:STDOUT:       inst780000CA:    symbolic_constant7800006B
// CHECK:STDOUT:       inst780000CB:    symbolic_constant7800007B
// CHECK:STDOUT:       inst780000CC:    symbolic_constant7800007C
// CHECK:STDOUT:       inst780000CD:    symbolic_constant7800007D
// CHECK:STDOUT:       inst780000CE:    symbolic_constant7800007E
// CHECK:STDOUT:       inst780000CF:    symbolic_constant78000080
// CHECK:STDOUT:       inst780000D0:    symbolic_constant78000081
// CHECK:STDOUT:       inst780000D1:    symbolic_constant78000082
// CHECK:STDOUT:       inst780000D2:    constant<none>
// CHECK:STDOUT:       inst780000D3:    concrete_constant(inst780000D3)
// CHECK:STDOUT:       inst780000D4:    concrete_constant(inst(TypeType))
// CHECK:STDOUT:       inst780000D5:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst780000D6:    constant<none>
// CHECK:STDOUT:       inst780000D7:    concrete_constant(inst780000D7)
// CHECK:STDOUT:       inst780000D8:    concrete_constant(inst78000026)
// CHECK:STDOUT:       inst780000D9:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst780000DA:    constant<none>
// CHECK:STDOUT:       inst780000DB:    concrete_constant(inst780000DB)
// CHECK:STDOUT:       inst780000DC:    symbolic_constant78000083
// CHECK:STDOUT:       inst780000DD:    symbolic_constant78000084
// CHECK:STDOUT:       inst780000DE:    symbolic_constant78000085
// CHECK:STDOUT:       inst780000DF:    constant<none>
// CHECK:STDOUT:       inst780000E0:    concrete_constant(inst780000E0)
// CHECK:STDOUT:       inst780000E1:    symbolic_constant78000087
// CHECK:STDOUT:       inst780000E2:    symbolic_constant7800008B
// CHECK:STDOUT:       inst780000E3:    symbolic_constant78000089
// CHECK:STDOUT:       inst780000E4:    symbolic_constant7800008A
// CHECK:STDOUT:       inst780000E5:    concrete_constant(inst780000E5)
// CHECK:STDOUT:       inst780000E6:    symbolic_constant7800008D
// CHECK:STDOUT:       inst780000E7:    symbolic_constant7800008E
// CHECK:STDOUT:       inst780000E8:    concrete_constant(inst780000E8)
// CHECK:STDOUT:       inst780000E9:    concrete_constant(inst780000E9)
// CHECK:STDOUT:       inst780000EA:    concrete_constant(inst780000EA)
// CHECK:STDOUT:       inst780000EB:    concrete_constant(inst780000EB)
// CHECK:STDOUT:       inst780000EC:    symbolic_constant78000090
// CHECK:STDOUT:       inst780000ED:    symbolic_constant7800009A
// CHECK:STDOUT:       inst780000EE:    symbolic_constant78000099
// CHECK:STDOUT:       inst780000EF:    symbolic_constant78000098
// CHECK:STDOUT:       inst780000F0:    symbolic_constant78000097
// CHECK:STDOUT:       inst780000F1:    symbolic_constant7800009B
// CHECK:STDOUT:       inst780000F2:    symbolic_constant7800009C
// CHECK:STDOUT:       inst780000F3:    symbolic_constant7800009D
// CHECK:STDOUT:       inst780000F4:    symbolic_constant7800009E
// CHECK:STDOUT:       inst780000F5:    symbolic_constant7800009F
// CHECK:STDOUT:       inst780000F6:    symbolic_constant780000A0
// CHECK:STDOUT:       inst780000F7:    symbolic_constant780000A1
// CHECK:STDOUT:       inst780000F8:    symbolic_constant780000A2
// CHECK:STDOUT:       inst780000F9:    symbolic_constant780000A3
// CHECK:STDOUT:       inst780000FA:    symbolic_constant780000A4
// CHECK:STDOUT:       inst780000FB:    symbolic_constant780000A5
// CHECK:STDOUT:       inst780000FC:    symbolic_constant780000A6
// CHECK:STDOUT:       inst780000FD:    symbolic_constant780000A7
// CHECK:STDOUT:       inst780000FE:    symbolic_constant780000A9
// CHECK:STDOUT:       inst780000FF:    symbolic_constant780000AA
// CHECK:STDOUT:       inst78000100:    symbolic_constant780000AF
// CHECK:STDOUT:       inst78000101:    symbolic_constant780000B7
// CHECK:STDOUT:       inst78000102:    symbolic_constant780000B9
// CHECK:STDOUT:       inst78000103:    symbolic_constant780000BA
// CHECK:STDOUT:       inst78000104:    symbolic_constant780000BB
// CHECK:STDOUT:       inst78000105:    symbolic_constant780000BC
// CHECK:STDOUT:       inst78000106:    symbolic_constant780000BD
// CHECK:STDOUT:       inst78000107:    symbolic_constant780000BE
// CHECK:STDOUT:       inst78000108:    symbolic_constant780000BF
// CHECK:STDOUT:       inst78000109:    symbolic_constant780000C0
// CHECK:STDOUT:       inst7800010A:    symbolic_constant780000C1
// CHECK:STDOUT:       inst7800010B:    symbolic_constant780000C2
// CHECK:STDOUT:       inst7800010C:    symbolic_constant780000C3
// CHECK:STDOUT:       inst7800010D:    symbolic_constant780000C4
// CHECK:STDOUT:       inst7800010E:    symbolic_constant780000C5
// CHECK:STDOUT:       inst7800010F:    concrete_constant(inst7800010F)
// CHECK:STDOUT:       inst78000110:    concrete_constant(inst78000110)
// CHECK:STDOUT:       inst78000111:    symbolic_constant78000086
// CHECK:STDOUT:       inst78000112:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst78000113:    symbolic_constant78000098
// CHECK:STDOUT:       inst78000114:    symbolic_constant78000097
// CHECK:STDOUT:       inst78000115:    symbolic_constant780000CC
// CHECK:STDOUT:       inst78000116:    symbolic_constant780000CD
// CHECK:STDOUT:       inst78000117:    symbolic_constant780000CE
// CHECK:STDOUT:       inst78000118:    symbolic_constant780000CF
// CHECK:STDOUT:       inst78000119:    symbolic_constant780000D0
// CHECK:STDOUT:       inst7800011A:    symbolic_constant780000D1
// CHECK:STDOUT:       inst7800011B:    symbolic_constant780000D2
// CHECK:STDOUT:       inst7800011C:    symbolic_constant780000D4
// CHECK:STDOUT:       inst7800011D:    symbolic_constant780000D5
// CHECK:STDOUT:       inst7800011E:    constant<none>
// CHECK:STDOUT:       inst7800011F:    concrete_constant(inst7800011F)
// CHECK:STDOUT:       inst78000120:    symbolic_constant780000D6
// CHECK:STDOUT:       inst78000121:    symbolic_constant780000D7
// CHECK:STDOUT:       inst78000122:    symbolic_constant780000D8
// CHECK:STDOUT:       inst78000123:    constant<none>
// CHECK:STDOUT:       inst78000124:    concrete_constant(inst78000124)
// CHECK:STDOUT:       inst78000125:    symbolic_constant780000DA
// CHECK:STDOUT:       inst78000126:    symbolic_constant780000DE
// CHECK:STDOUT:       inst78000127:    symbolic_constant780000DC
// CHECK:STDOUT:       inst78000128:    symbolic_constant780000DD
// CHECK:STDOUT:       inst78000129:    concrete_constant(inst78000129)
// CHECK:STDOUT:       inst7800012A:    symbolic_constant780000E0
// CHECK:STDOUT:       inst7800012B:    symbolic_constant780000E1
// CHECK:STDOUT:       inst7800012C:    concrete_constant(inst7800012C)
// CHECK:STDOUT:       inst7800012D:    concrete_constant(inst7800012D)
// CHECK:STDOUT:       inst7800012E:    concrete_constant(inst7800012E)
// CHECK:STDOUT:       inst7800012F:    concrete_constant(inst7800012F)
// CHECK:STDOUT:       inst78000130:    symbolic_constant780000E3
// CHECK:STDOUT:       inst78000131:    symbolic_constant780000F0
// CHECK:STDOUT:       inst78000132:    symbolic_constant780000EF
// CHECK:STDOUT:       inst78000133:    symbolic_constant780000EE
// CHECK:STDOUT:       inst78000134:    symbolic_constant780000ED
// CHECK:STDOUT:       inst78000135:    symbolic_constant780000EC
// CHECK:STDOUT:       inst78000136:    symbolic_constant780000F1
// CHECK:STDOUT:       inst78000137:    symbolic_constant780000F2
// CHECK:STDOUT:       inst78000138:    symbolic_constant780000F3
// CHECK:STDOUT:       inst78000139:    symbolic_constant780000F4
// CHECK:STDOUT:       inst7800013A:    symbolic_constant780000F5
// CHECK:STDOUT:       inst7800013B:    symbolic_constant780000F6
// CHECK:STDOUT:       inst7800013C:    symbolic_constant780000F7
// CHECK:STDOUT:       inst7800013D:    symbolic_constant780000F8
// CHECK:STDOUT:       inst7800013E:    symbolic_constant780000F9
// CHECK:STDOUT:       inst7800013F:    symbolic_constant780000FA
// CHECK:STDOUT:       inst78000140:    symbolic_constant780000FB
// CHECK:STDOUT:       inst78000141:    symbolic_constant780000FC
// CHECK:STDOUT:       inst78000142:    symbolic_constant780000FD
// CHECK:STDOUT:       inst78000143:    symbolic_constant780000FE
// CHECK:STDOUT:       inst78000144:    symbolic_constant780000FF
// CHECK:STDOUT:       inst78000145:    symbolic_constant78000101
// CHECK:STDOUT:       inst78000146:    symbolic_constant78000102
// CHECK:STDOUT:       inst78000147:    symbolic_constant78000107
// CHECK:STDOUT:       inst78000148:    symbolic_constant78000115
// CHECK:STDOUT:       inst78000149:    symbolic_constant78000117
// CHECK:STDOUT:       inst7800014A:    symbolic_constant78000118
// CHECK:STDOUT:       inst7800014B:    symbolic_constant78000119
// CHECK:STDOUT:       inst7800014C:    symbolic_constant7800011A
// CHECK:STDOUT:       inst7800014D:    symbolic_constant7800011B
// CHECK:STDOUT:       inst7800014E:    symbolic_constant7800011C
// CHECK:STDOUT:       inst7800014F:    symbolic_constant7800011D
// CHECK:STDOUT:       inst78000150:    symbolic_constant7800011E
// CHECK:STDOUT:       inst78000151:    symbolic_constant7800011F
// CHECK:STDOUT:       inst78000152:    symbolic_constant78000120
// CHECK:STDOUT:       inst78000153:    symbolic_constant78000121
// CHECK:STDOUT:       inst78000154:    symbolic_constant78000122
// CHECK:STDOUT:       inst78000155:    symbolic_constant78000123
// CHECK:STDOUT:       inst78000156:    symbolic_constant78000124
// CHECK:STDOUT:       inst78000157:    symbolic_constant78000125
// CHECK:STDOUT:       inst78000158:    symbolic_constant78000126
// CHECK:STDOUT:       inst78000159:    symbolic_constant78000127
// CHECK:STDOUT:       inst7800015A:    symbolic_constant78000128
// CHECK:STDOUT:       inst7800015B:    symbolic_constant78000129
// CHECK:STDOUT:       inst7800015C:    concrete_constant(inst7800015C)
// CHECK:STDOUT:       inst7800015D:    concrete_constant(inst7800015D)
// CHECK:STDOUT:       inst7800015E:    concrete_constant(inst7800015E)
// CHECK:STDOUT:       inst7800015F:    symbolic_constant780000D9
// CHECK:STDOUT:       inst78000160:    concrete_constant(inst78000050)
// CHECK:STDOUT:       inst78000161:    symbolic_constant780000EE
// CHECK:STDOUT:       inst78000162:    symbolic_constant780000ED
// CHECK:STDOUT:       inst78000163:    symbolic_constant780000EC
// CHECK:STDOUT:       inst78000164:    symbolic_constant78000132
// CHECK:STDOUT:       inst78000165:    symbolic_constant78000133
// CHECK:STDOUT:       inst78000166:    symbolic_constant78000134
// CHECK:STDOUT:       inst78000167:    symbolic_constant78000135
// CHECK:STDOUT:       inst78000168:    symbolic_constant78000136
// CHECK:STDOUT:       inst78000169:    symbolic_constant78000137
// CHECK:STDOUT:       inst7800016A:    symbolic_constant78000138
// CHECK:STDOUT:       inst7800016B:    symbolic_constant78000139
// CHECK:STDOUT:       inst7800016C:    symbolic_constant7800013A
// CHECK:STDOUT:       inst7800016D:    symbolic_constant7800013C
// CHECK:STDOUT:       inst7800016E:    symbolic_constant7800013D
// CHECK:STDOUT:       inst7800016F:    symbolic_constant78000140
// CHECK:STDOUT:       inst78000170:    symbolic_constant7800013F
// CHECK:STDOUT:       inst78000171:    symbolic_constant78000140
// CHECK:STDOUT:       inst78000172:    symbolic_constant78000141
// CHECK:STDOUT:       inst78000173:    symbolic_constant78000142
// CHECK:STDOUT:       inst78000174:    symbolic_constant78000143
// CHECK:STDOUT:       inst78000175:    symbolic_constant78000144
// CHECK:STDOUT:       inst78000176:    symbolic_constant78000145
// CHECK:STDOUT:       inst78000177:    symbolic_constant78000146
// CHECK:STDOUT:       inst78000178:    symbolic_constant7800014B
// CHECK:STDOUT:       inst78000179:    symbolic_constant78000147
// CHECK:STDOUT:       inst7800017A:    symbolic_constant78000148
// CHECK:STDOUT:       inst7800017B:    symbolic_constant78000149
// CHECK:STDOUT:       inst7800017C:    symbolic_constant7800014A
// CHECK:STDOUT:       inst7800017D:    symbolic_constant7800014B
// CHECK:STDOUT:       inst7800017F:    symbolic_constant7800014D
// CHECK:STDOUT:       inst78000180:    symbolic_constant7800014C
// CHECK:STDOUT:       inst78000181:    symbolic_constant7800014D
// CHECK:STDOUT:       inst78000186:    concrete_constant(inst78000028)
// CHECK:STDOUT:       inst78000187:    concrete_constant(inst78000028)
// CHECK:STDOUT:     symbolic_constants:
// CHECK:STDOUT:       symbolic_constant78000000: {inst: inst78000014, kind: self, attached: null}
// CHECK:STDOUT:       symbolic_constant78000001: {inst: inst78000018, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000002: {inst: inst78000018, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000003: {inst: inst7800001D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000004: {inst: inst7800001D, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000005: {inst: inst7800001F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000006: {inst: inst7800001F, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000007: {inst: inst7800002B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000008: {inst: inst7800002B, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000009: {inst: inst7800002F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800000A: {inst: inst7800002F, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800000B: {inst: inst78000033, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800000C: {inst: inst78000033, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800000D: {inst: inst78000035, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800000E: {inst: inst78000035, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800000F: {inst: inst78000042, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000010: {inst: inst78000042, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000011: {inst: inst78000044, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000012: {inst: inst78000046, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000013: {inst: inst78000046, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000014: {inst: inst78000052, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000015: {inst: inst78000057, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000016: {inst: inst78000058, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000017: {inst: inst78000058, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000018: {inst: inst78000059, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000019: {inst: inst78000057, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800001A: {inst: inst7800005A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001B: {inst: inst7800005A, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800001C: {inst: inst7800005F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800001D: {inst: inst7800005F, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800001E: {inst: inst78000059, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800001F: {inst: inst78000052, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000020: {inst: inst7800005F, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000021: {inst: inst78000059, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000022: {inst: inst78000052, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000023: {inst: inst78000059, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000024: {inst: inst7800005A, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000025: {inst: inst7800005F, kind: checked, attached: {generic: generic78000002, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000026: {inst: inst78000058, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000027: {inst: inst78000052, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000028: {inst: inst78000052, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000029: {inst: inst78000057, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800002A: {inst: inst78000058, kind: checked, attached: {generic: generic78000001, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant7800002B: {inst: inst78000071, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002C: {inst: inst78000072, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002D: {inst: inst78000073, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800002E: {inst: inst78000073, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800002F: {inst: inst78000076, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000030: {inst: inst78000076, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000031: {inst: inst78000078, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000032: {inst: inst78000079, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000033: {inst: inst78000079, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000034: {inst: inst78000078, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000035: {inst: inst7800007A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000036: {inst: inst7800007A, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000037: {inst: inst7800007F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000038: {inst: inst7800007F, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000039: {inst: inst78000073, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800003A: {inst: inst78000072, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800003B: {inst: inst78000071, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800003C: {inst: inst78000071, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800003D: {inst: inst7800007F, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800003E: {inst: inst78000073, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800003F: {inst: inst78000071, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000040: {inst: inst78000072, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000041: {inst: inst78000073, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000042: {inst: inst7800007A, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000043: {inst: inst7800007F, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000044: {inst: inst78000088, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000045: {inst: inst78000089, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000046: {inst: inst7800008A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000047: {inst: inst7800008B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000048: {inst: inst7800008C, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000049: {inst: inst7800008D, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004A: {inst: inst7800008D, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant7800004B: {inst: inst7800008E, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004C: {inst: inst7800008F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800004D: {inst: inst7800008C, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant7800004E: {inst: inst7800008B, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant7800004F: {inst: inst78000089, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant78000050: {inst: inst78000088, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000051: {inst: inst78000090, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000052: {inst: inst78000090, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000053: {inst: inst78000091, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000054: {inst: inst78000091, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000055: {inst: inst78000091, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000056: {inst: inst78000090, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000057: {inst: inst78000088, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000058: {inst: inst78000089, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant78000059: {inst: inst7800008B, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant7800005A: {inst: inst7800008C, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant7800005B: {inst: inst7800008D, kind: checked, attached: {generic: generic78000004, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant7800005C: {inst: inst78000073, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800005D: {inst: inst78000072, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800005E: {inst: inst78000071, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800005F: {inst: inst78000071, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000060: {inst: inst78000072, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000061: {inst: inst78000073, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000062: {inst: inst78000076, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000063: {inst: inst78000079, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000064: {inst: inst78000078, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000065: {inst: inst78000079, kind: checked, attached: {generic: generic78000003, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000066: {inst: inst7800001D, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000067: {inst: inst780000B8, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000068: {inst: inst780000B8, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000069: {inst: inst7800001D, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800006A: {inst: inst78000018, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800006B: {inst: inst78000018, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800006C: {inst: inst78000018, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800006D: {inst: inst7800001D, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800006E: {inst: inst780000B8, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800006F: {inst: inst78000042, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000070: {inst: inst780000C1, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000071: {inst: inst780000C2, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000072: {inst: inst780000C2, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000073: {inst: inst780000C1, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000074: {inst: inst7800001F, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000075: {inst: inst780000C7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000076: {inst: inst780000C7, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000077: {inst: inst7800001D, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000078: {inst: inst78000018, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000079: {inst: inst780000C7, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800007A: {inst: inst7800001D, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800007B: {inst: inst78000018, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800007C: {inst: inst7800001D, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800007D: {inst: inst7800001F, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800007E: {inst: inst780000C7, kind: checked, attached: {generic: generic78000006, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800007F: {inst: inst780000C2, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000080: {inst: inst780000C1, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000081: {inst: inst780000C2, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000082: {inst: inst78000042, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000083: {inst: inst780000DC, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000084: {inst: inst780000DD, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000085: {inst: inst780000DE, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000086: {inst: inst780000DE, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000087: {inst: inst780000E1, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000088: {inst: inst780000E1, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000089: {inst: inst780000E3, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800008A: {inst: inst780000E4, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800008B: {inst: inst780000E4, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800008C: {inst: inst780000E3, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800008D: {inst: inst780000E6, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800008E: {inst: inst780000E7, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800008F: {inst: inst780000E7, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000090: {inst: inst780000EC, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000091: {inst: inst780000EC, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000092: {inst: inst780000DE, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000093: {inst: inst780000DD, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000094: {inst: inst780000DC, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000095: {inst: inst78000072, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000096: {inst: inst78000071, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000097: {inst: inst780000DC, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000098: {inst: inst78000071, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000099: {inst: inst780000EC, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800009A: {inst: inst780000DE, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800009B: {inst: inst78000071, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant7800009C: {inst: inst78000072, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant7800009D: {inst: inst780000DC, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800009E: {inst: inst780000DD, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800009F: {inst: inst780000DE, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000A0: {inst: inst780000E7, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000A1: {inst: inst780000EC, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000A2: {inst: inst780000F8, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000A3: {inst: inst780000F9, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000A4: {inst: inst780000FA, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000A5: {inst: inst780000FB, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000A6: {inst: inst780000FC, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000A7: {inst: inst780000FD, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000A8: {inst: inst780000FD, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def12}}
// CHECK:STDOUT:       symbolic_constant780000A9: {inst: inst780000FE, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000AA: {inst: inst780000FF, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000AB: {inst: inst780000FC, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def11}}
// CHECK:STDOUT:       symbolic_constant780000AC: {inst: inst780000FB, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def10}}
// CHECK:STDOUT:       symbolic_constant780000AD: {inst: inst780000F9, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def9}}
// CHECK:STDOUT:       symbolic_constant780000AE: {inst: inst780000F8, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def8}}
// CHECK:STDOUT:       symbolic_constant780000AF: {inst: inst78000100, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000B0: {inst: inst78000100, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant780000B1: {inst: inst7800008D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant780000B2: {inst: inst7800008C, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant780000B3: {inst: inst7800008B, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant780000B4: {inst: inst78000089, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant780000B5: {inst: inst78000088, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant780000B6: {inst: inst78000090, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000B7: {inst: inst78000101, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000B8: {inst: inst78000101, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780000B9: {inst: inst78000101, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780000BA: {inst: inst78000090, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000BB: {inst: inst78000088, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant780000BC: {inst: inst78000089, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant780000BD: {inst: inst7800008B, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant780000BE: {inst: inst7800008C, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant780000BF: {inst: inst7800008D, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant780000C0: {inst: inst78000100, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant780000C1: {inst: inst780000F8, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def8}}
// CHECK:STDOUT:       symbolic_constant780000C2: {inst: inst780000F9, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def9}}
// CHECK:STDOUT:       symbolic_constant780000C3: {inst: inst780000FB, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def10}}
// CHECK:STDOUT:       symbolic_constant780000C4: {inst: inst780000FC, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def11}}
// CHECK:STDOUT:       symbolic_constant780000C5: {inst: inst780000FD, kind: checked, attached: {generic: generic78000008, index: generic_inst_in_def12}}
// CHECK:STDOUT:       symbolic_constant780000C6: {inst: inst780000DE, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000C7: {inst: inst780000DD, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000C8: {inst: inst78000072, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000C9: {inst: inst780000E6, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000CA: {inst: inst780000DC, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000CB: {inst: inst78000071, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000CC: {inst: inst78000071, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000CD: {inst: inst780000DC, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000CE: {inst: inst780000E6, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000CF: {inst: inst78000072, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000D0: {inst: inst780000DD, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000D1: {inst: inst780000DE, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000D2: {inst: inst780000E1, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000D3: {inst: inst780000E4, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000D4: {inst: inst780000E3, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780000D5: {inst: inst780000E4, kind: checked, attached: {generic: generic78000007, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000D6: {inst: inst78000120, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D7: {inst: inst78000121, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D8: {inst: inst78000122, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000D9: {inst: inst78000122, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000DA: {inst: inst78000125, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DB: {inst: inst78000125, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000DC: {inst: inst78000127, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DD: {inst: inst78000128, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000DE: {inst: inst78000128, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant780000DF: {inst: inst78000127, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant780000E0: {inst: inst7800012A, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E1: {inst: inst7800012B, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E2: {inst: inst7800012B, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000E3: {inst: inst78000130, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000E4: {inst: inst78000130, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000E5: {inst: inst78000122, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000E6: {inst: inst78000121, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000E7: {inst: inst78000120, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000E8: {inst: inst780000DD, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000E9: {inst: inst780000DC, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000EA: {inst: inst78000072, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000EB: {inst: inst78000071, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000EC: {inst: inst78000120, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000ED: {inst: inst780000DC, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000EE: {inst: inst78000071, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000EF: {inst: inst78000130, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000F0: {inst: inst78000122, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000F1: {inst: inst78000071, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant780000F2: {inst: inst78000072, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant780000F3: {inst: inst780000DC, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant780000F4: {inst: inst780000DD, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant780000F5: {inst: inst78000120, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant780000F6: {inst: inst78000121, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant780000F7: {inst: inst78000122, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant780000F8: {inst: inst7800012B, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant780000F9: {inst: inst78000130, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant780000FA: {inst: inst7800013F, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000FB: {inst: inst78000140, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000FC: {inst: inst78000141, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000FD: {inst: inst78000142, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000FE: {inst: inst78000143, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant780000FF: {inst: inst78000144, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000100: {inst: inst78000144, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def18}}
// CHECK:STDOUT:       symbolic_constant78000101: {inst: inst78000145, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000102: {inst: inst78000146, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000103: {inst: inst78000143, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def17}}
// CHECK:STDOUT:       symbolic_constant78000104: {inst: inst78000142, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def16}}
// CHECK:STDOUT:       symbolic_constant78000105: {inst: inst78000140, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def15}}
// CHECK:STDOUT:       symbolic_constant78000106: {inst: inst7800013F, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def14}}
// CHECK:STDOUT:       symbolic_constant78000107: {inst: inst78000147, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000108: {inst: inst78000147, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def13}}
// CHECK:STDOUT:       symbolic_constant78000109: {inst: inst780000FD, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def12}}
// CHECK:STDOUT:       symbolic_constant7800010A: {inst: inst780000FC, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def11}}
// CHECK:STDOUT:       symbolic_constant7800010B: {inst: inst780000FB, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def10}}
// CHECK:STDOUT:       symbolic_constant7800010C: {inst: inst780000F9, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def9}}
// CHECK:STDOUT:       symbolic_constant7800010D: {inst: inst780000F8, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def8}}
// CHECK:STDOUT:       symbolic_constant7800010E: {inst: inst78000100, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant7800010F: {inst: inst7800008D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant78000110: {inst: inst7800008C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant78000111: {inst: inst7800008B, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant78000112: {inst: inst78000089, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant78000113: {inst: inst78000088, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000114: {inst: inst78000090, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000115: {inst: inst78000148, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000116: {inst: inst78000148, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000117: {inst: inst78000148, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant78000118: {inst: inst78000090, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant78000119: {inst: inst78000088, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant7800011A: {inst: inst78000089, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant7800011B: {inst: inst7800008B, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant7800011C: {inst: inst7800008C, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant7800011D: {inst: inst7800008D, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant7800011E: {inst: inst78000100, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant7800011F: {inst: inst780000F8, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def8}}
// CHECK:STDOUT:       symbolic_constant78000120: {inst: inst780000F9, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def9}}
// CHECK:STDOUT:       symbolic_constant78000121: {inst: inst780000FB, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def10}}
// CHECK:STDOUT:       symbolic_constant78000122: {inst: inst780000FC, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def11}}
// CHECK:STDOUT:       symbolic_constant78000123: {inst: inst780000FD, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def12}}
// CHECK:STDOUT:       symbolic_constant78000124: {inst: inst78000147, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def13}}
// CHECK:STDOUT:       symbolic_constant78000125: {inst: inst7800013F, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def14}}
// CHECK:STDOUT:       symbolic_constant78000126: {inst: inst78000140, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def15}}
// CHECK:STDOUT:       symbolic_constant78000127: {inst: inst78000142, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def16}}
// CHECK:STDOUT:       symbolic_constant78000128: {inst: inst78000143, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def17}}
// CHECK:STDOUT:       symbolic_constant78000129: {inst: inst78000144, kind: checked, attached: {generic: generic7800000A, index: generic_inst_in_def18}}
// CHECK:STDOUT:       symbolic_constant7800012A: {inst: inst78000122, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800012B: {inst: inst78000121, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant7800012C: {inst: inst780000DD, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant7800012D: {inst: inst78000072, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant7800012E: {inst: inst7800012A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant7800012F: {inst: inst78000120, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000130: {inst: inst780000DC, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000131: {inst: inst78000071, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000132: {inst: inst78000071, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl0}}
// CHECK:STDOUT:       symbolic_constant78000133: {inst: inst780000DC, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl1}}
// CHECK:STDOUT:       symbolic_constant78000134: {inst: inst78000120, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant78000135: {inst: inst7800012A, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl3}}
// CHECK:STDOUT:       symbolic_constant78000136: {inst: inst78000072, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl4}}
// CHECK:STDOUT:       symbolic_constant78000137: {inst: inst780000DD, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl5}}
// CHECK:STDOUT:       symbolic_constant78000138: {inst: inst78000121, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl6}}
// CHECK:STDOUT:       symbolic_constant78000139: {inst: inst78000122, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl7}}
// CHECK:STDOUT:       symbolic_constant7800013A: {inst: inst78000125, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_decl8}}
// CHECK:STDOUT:       symbolic_constant7800013B: {inst: inst78000128, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800013C: {inst: inst78000127, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def0}}
// CHECK:STDOUT:       symbolic_constant7800013D: {inst: inst78000128, kind: checked, attached: {generic: generic78000009, index: generic_inst_in_def1}}
// CHECK:STDOUT:       symbolic_constant7800013E: {inst: inst780000B8, kind: checked, attached: {generic: generic78000005, index: generic_inst_in_decl2}}
// CHECK:STDOUT:       symbolic_constant7800013F: {inst: inst78000170, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000140: {inst: inst78000170, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def2}}
// CHECK:STDOUT:       symbolic_constant78000141: {inst: inst78000172, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000142: {inst: inst78000172, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def3}}
// CHECK:STDOUT:       symbolic_constant78000143: {inst: inst78000174, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000144: {inst: inst78000175, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000145: {inst: inst78000176, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000146: {inst: inst78000177, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000147: {inst: inst78000179, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant78000148: {inst: inst78000174, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def4}}
// CHECK:STDOUT:       symbolic_constant78000149: {inst: inst78000175, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def5}}
// CHECK:STDOUT:       symbolic_constant7800014A: {inst: inst78000177, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def6}}
// CHECK:STDOUT:       symbolic_constant7800014B: {inst: inst78000179, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def7}}
// CHECK:STDOUT:       symbolic_constant7800014C: {inst: inst78000180, kind: checked, attached: null}
// CHECK:STDOUT:       symbolic_constant7800014D: {inst: inst78000180, kind: checked, attached: {generic: generic78000000, index: generic_inst_in_def8}}
// CHECK:STDOUT:   inst_blocks:
// CHECK:STDOUT:     inst_block_empty: {}
// CHECK:STDOUT:     exports:
// CHECK:STDOUT:       0:               inst7800003E
// CHECK:STDOUT:     generated:       {}
// CHECK:STDOUT:     imports:
// CHECK:STDOUT:       0:               inst78000011
// CHECK:STDOUT:       1:               inst7800004E
// CHECK:STDOUT:       2:               inst7800004F
// CHECK:STDOUT:       3:               inst78000051
// CHECK:STDOUT:       4:               inst78000053
// CHECK:STDOUT:       5:               inst78000054
// CHECK:STDOUT:       6:               inst78000055
// CHECK:STDOUT:       7:               inst78000056
// CHECK:STDOUT:       8:               inst7800005B
// CHECK:STDOUT:       9:               inst7800005C
// CHECK:STDOUT:       10:              inst7800005D
// CHECK:STDOUT:       11:              inst7800005E
// CHECK:STDOUT:       12:              inst78000060
// CHECK:STDOUT:       13:              inst78000061
// CHECK:STDOUT:       14:              inst78000062
// CHECK:STDOUT:       15:              inst7800006A
// CHECK:STDOUT:       16:              inst7800006C
// CHECK:STDOUT:       17:              inst7800006F
// CHECK:STDOUT:       18:              inst78000070
// CHECK:STDOUT:       19:              inst78000074
// CHECK:STDOUT:       20:              inst78000075
// CHECK:STDOUT:       21:              inst78000077
// CHECK:STDOUT:       22:              inst7800007B
// CHECK:STDOUT:       23:              inst7800007C
// CHECK:STDOUT:       24:              inst7800007D
// CHECK:STDOUT:       25:              inst7800007E
// CHECK:STDOUT:       26:              inst78000080
// CHECK:STDOUT:       27:              inst78000081
// CHECK:STDOUT:       28:              inst78000082
// CHECK:STDOUT:       29:              inst7800009A
// CHECK:STDOUT:       30:              inst7800009B
// CHECK:STDOUT:       31:              inst7800009C
// CHECK:STDOUT:       32:              inst7800009D
// CHECK:STDOUT:       33:              inst780000A4
// CHECK:STDOUT:       34:              inst780000A5
// CHECK:STDOUT:       35:              inst780000A6
// CHECK:STDOUT:       36:              inst780000A7
// CHECK:STDOUT:       37:              inst780000A8
// CHECK:STDOUT:       38:              inst780000A9
// CHECK:STDOUT:       39:              inst780000AA
// CHECK:STDOUT:       40:              inst780000AB
// CHECK:STDOUT:       41:              inst780000AC
// CHECK:STDOUT:       42:              inst780000AD
// CHECK:STDOUT:       43:              inst780000AE
// CHECK:STDOUT:       44:              inst780000AF
// CHECK:STDOUT:       45:              inst780000B0
// CHECK:STDOUT:       46:              inst780000B1
// CHECK:STDOUT:       47:              inst780000B2
// CHECK:STDOUT:       48:              inst780000B3
// CHECK:STDOUT:       49:              inst780000B4
// CHECK:STDOUT:       50:              inst780000B5
// CHECK:STDOUT:       51:              inst780000B6
// CHECK:STDOUT:       52:              inst780000B7
// CHECK:STDOUT:       53:              inst780000B9
// CHECK:STDOUT:       54:              inst780000BA
// CHECK:STDOUT:       55:              inst780000BB
// CHECK:STDOUT:       56:              inst780000BC
// CHECK:STDOUT:       57:              inst780000C0
// CHECK:STDOUT:       58:              inst780000C3
// CHECK:STDOUT:       59:              inst780000C4
// CHECK:STDOUT:       60:              inst780000C5
// CHECK:STDOUT:       61:              inst780000C6
// CHECK:STDOUT:       62:              inst780000C8
// CHECK:STDOUT:       63:              inst780000C9
// CHECK:STDOUT:       64:              inst780000CA
// CHECK:STDOUT:       65:              inst780000D2
// CHECK:STDOUT:       66:              inst780000D3
// CHECK:STDOUT:       67:              inst780000D4
// CHECK:STDOUT:       68:              inst780000D5
// CHECK:STDOUT:       69:              inst780000D6
// CHECK:STDOUT:       70:              inst780000D7
// CHECK:STDOUT:       71:              inst780000D8
// CHECK:STDOUT:       72:              inst780000D9
// CHECK:STDOUT:       73:              inst780000DA
// CHECK:STDOUT:       74:              inst780000DB
// CHECK:STDOUT:       75:              inst780000DF
// CHECK:STDOUT:       76:              inst780000E0
// CHECK:STDOUT:       77:              inst780000E2
// CHECK:STDOUT:       78:              inst780000E8
// CHECK:STDOUT:       79:              inst780000E9
// CHECK:STDOUT:       80:              inst780000EA
// CHECK:STDOUT:       81:              inst780000EB
// CHECK:STDOUT:       82:              inst780000ED
// CHECK:STDOUT:       83:              inst780000EE
// CHECK:STDOUT:       84:              inst780000EF
// CHECK:STDOUT:       85:              inst780000F0
// CHECK:STDOUT:       86:              inst7800010F
// CHECK:STDOUT:       87:              inst78000110
// CHECK:STDOUT:       88:              inst78000111
// CHECK:STDOUT:       89:              inst78000112
// CHECK:STDOUT:       90:              inst78000113
// CHECK:STDOUT:       91:              inst78000114
// CHECK:STDOUT:       92:              inst7800011E
// CHECK:STDOUT:       93:              inst7800011F
// CHECK:STDOUT:       94:              inst78000123
// CHECK:STDOUT:       95:              inst78000124
// CHECK:STDOUT:       96:              inst78000126
// CHECK:STDOUT:       97:              inst7800012C
// CHECK:STDOUT:       98:              inst7800012D
// CHECK:STDOUT:       99:              inst7800012E
// CHECK:STDOUT:       100:             inst7800012F
// CHECK:STDOUT:       101:             inst78000131
// CHECK:STDOUT:       102:             inst78000132
// CHECK:STDOUT:       103:             inst78000133
// CHECK:STDOUT:       104:             inst78000134
// CHECK:STDOUT:       105:             inst78000135
// CHECK:STDOUT:       106:             inst7800015C
// CHECK:STDOUT:       107:             inst7800015D
// CHECK:STDOUT:       108:             inst7800015E
// CHECK:STDOUT:       109:             inst7800015F
// CHECK:STDOUT:       110:             inst78000160
// CHECK:STDOUT:       111:             inst78000161
// CHECK:STDOUT:       112:             inst78000162
// CHECK:STDOUT:       113:             inst78000163
// CHECK:STDOUT:     global_init:     {}
// CHECK:STDOUT:     inst_block78000005:
// CHECK:STDOUT:       0:               inst78000013
// CHECK:STDOUT:       1:               inst78000015
// CHECK:STDOUT:     inst_block78000006:
// CHECK:STDOUT:       0:               inst7800001A
// CHECK:STDOUT:     inst_block78000007:
// CHECK:STDOUT:       0:               inst7800001B
// CHECK:STDOUT:       1:               inst7800001C
// CHECK:STDOUT:     inst_block78000008:
// CHECK:STDOUT:       0:               inst78000023
// CHECK:STDOUT:     inst_block78000009:
// CHECK:STDOUT:       0:               inst78000025
// CHECK:STDOUT:       1:               inst78000027
// CHECK:STDOUT:     inst_block7800000A:
// CHECK:STDOUT:       0:               inst(TypeType)
// CHECK:STDOUT:       1:               inst78000026
// CHECK:STDOUT:     inst_block7800000B:
// CHECK:STDOUT:       0:               inst7800001D
// CHECK:STDOUT:       1:               inst78000028
// CHECK:STDOUT:     inst_block7800000C:
// CHECK:STDOUT:       0:               inst7800001E
// CHECK:STDOUT:       1:               inst78000028
// CHECK:STDOUT:     inst_block7800000D:
// CHECK:STDOUT:       0:               inst7800001D
// CHECK:STDOUT:       1:               inst7800002E
// CHECK:STDOUT:     inst_block7800000E:
// CHECK:STDOUT:       0:               inst7800001D
// CHECK:STDOUT:       1:               inst78000026
// CHECK:STDOUT:     inst_block7800000F:
// CHECK:STDOUT:       0:               inst7800001E
// CHECK:STDOUT:       1:               inst78000026
// CHECK:STDOUT:     inst_block78000010:
// CHECK:STDOUT:       0:               inst78000038
// CHECK:STDOUT:     inst_block78000011:
// CHECK:STDOUT:       0:               inst78000020
// CHECK:STDOUT:       1:               inst78000036
// CHECK:STDOUT:     inst_block78000012:
// CHECK:STDOUT:       0:               inst7800003A
// CHECK:STDOUT:       1:               inst7800003C
// CHECK:STDOUT:     inst_block78000013:
// CHECK:STDOUT:       0:               inst7800001A
// CHECK:STDOUT:       1:               inst78000020
// CHECK:STDOUT:       2:               inst78000023
// CHECK:STDOUT:       3:               inst78000036
// CHECK:STDOUT:       4:               inst78000038
// CHECK:STDOUT:     inst_block78000014:
// CHECK:STDOUT:       0:               inst78000024
// CHECK:STDOUT:       1:               inst78000025
// CHECK:STDOUT:       2:               inst78000027
// CHECK:STDOUT:       3:               inst7800002A
// CHECK:STDOUT:       4:               inst7800002E
// CHECK:STDOUT:       5:               inst78000030
// CHECK:STDOUT:       6:               inst78000032
// CHECK:STDOUT:       7:               inst78000039
// CHECK:STDOUT:       8:               inst78000017
// CHECK:STDOUT:       9:               inst7800003A
// CHECK:STDOUT:       10:              inst7800003B
// CHECK:STDOUT:       11:              inst78000022
// CHECK:STDOUT:       12:              inst7800003C
// CHECK:STDOUT:       13:              inst7800003D
// CHECK:STDOUT:     inst_block78000015:
// CHECK:STDOUT:       0:               inst78000017
// CHECK:STDOUT:     inst_block78000016:
// CHECK:STDOUT:       0:               inst78000018
// CHECK:STDOUT:     inst_block78000017:
// CHECK:STDOUT:       0:               inst78000019
// CHECK:STDOUT:       1:               inst7800001E
// CHECK:STDOUT:       2:               inst78000021
// CHECK:STDOUT:       3:               inst7800002C
// CHECK:STDOUT:       4:               inst78000031
// CHECK:STDOUT:       5:               inst78000034
// CHECK:STDOUT:       6:               inst78000037
// CHECK:STDOUT:     inst_block78000018:
// CHECK:STDOUT:       0:               inst78000018
// CHECK:STDOUT:       1:               inst7800001D
// CHECK:STDOUT:       2:               inst7800001F
// CHECK:STDOUT:       3:               inst7800002B
// CHECK:STDOUT:       4:               inst7800002F
// CHECK:STDOUT:       5:               inst78000033
// CHECK:STDOUT:       6:               inst78000035
// CHECK:STDOUT:     inst_block78000019:
// CHECK:STDOUT:       0:               inst78000048
// CHECK:STDOUT:       1:               inst78000049
// CHECK:STDOUT:       2:               inst7800004A
// CHECK:STDOUT:       3:               inst78000178
// CHECK:STDOUT:       4:               inst7800017E
// CHECK:STDOUT:       5:               inst7800017F
// CHECK:STDOUT:       6:               inst78000182
// CHECK:STDOUT:       7:               inst78000183
// CHECK:STDOUT:       8:               inst7800004C
// CHECK:STDOUT:       9:               inst78000184
// CHECK:STDOUT:       10:              inst78000185
// CHECK:STDOUT:       11:              inst78000186
// CHECK:STDOUT:       12:              inst78000187
// CHECK:STDOUT:       13:              inst78000188
// CHECK:STDOUT:       14:              inst78000189
// CHECK:STDOUT:       15:              inst7800018A
// CHECK:STDOUT:     inst_block7800001A:
// CHECK:STDOUT:       0:               inst78000048
// CHECK:STDOUT:       1:               inst78000049
// CHECK:STDOUT:     inst_block7800001B:
// CHECK:STDOUT:       0:               inst78000054
// CHECK:STDOUT:     inst_block7800001C:
// CHECK:STDOUT:       0:               inst78000055
// CHECK:STDOUT:     inst_block7800001D:
// CHECK:STDOUT:       0:               inst78000052
// CHECK:STDOUT:     inst_block7800001E:
// CHECK:STDOUT:       0:               inst78000052
// CHECK:STDOUT:       1:               inst78000057
// CHECK:STDOUT:       2:               inst78000058
// CHECK:STDOUT:     inst_block7800001F:
// CHECK:STDOUT:       0:               inst7800005D
// CHECK:STDOUT:       1:               inst7800005B
// CHECK:STDOUT:     inst_block78000020:
// CHECK:STDOUT:       0:               inst7800005E
// CHECK:STDOUT:     inst_block78000021:
// CHECK:STDOUT:       0:               inst7800005C
// CHECK:STDOUT:     inst_block78000022:
// CHECK:STDOUT:       0:               inst78000062
// CHECK:STDOUT:     inst_block78000023:
// CHECK:STDOUT:       0:               inst78000063
// CHECK:STDOUT:       1:               inst78000064
// CHECK:STDOUT:       2:               inst78000065
// CHECK:STDOUT:       3:               inst78000066
// CHECK:STDOUT:     inst_block78000024:
// CHECK:STDOUT:       0:               inst78000052
// CHECK:STDOUT:       1:               inst78000059
// CHECK:STDOUT:       2:               inst7800005A
// CHECK:STDOUT:       3:               inst7800005F
// CHECK:STDOUT:     inst_block78000025:
// CHECK:STDOUT:       0:               inst78000067
// CHECK:STDOUT:     inst_block78000026:
// CHECK:STDOUT:       0:               inst78000067
// CHECK:STDOUT:       1:               inst78000068
// CHECK:STDOUT:       2:               inst78000069
// CHECK:STDOUT:     inst_block78000027:
// CHECK:STDOUT:       0:               inst78000074
// CHECK:STDOUT:     inst_block78000028:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:     inst_block78000029:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst78000072
// CHECK:STDOUT:       2:               inst78000073
// CHECK:STDOUT:       3:               inst78000076
// CHECK:STDOUT:     inst_block7800002A:
// CHECK:STDOUT:       0:               inst78000078
// CHECK:STDOUT:       1:               inst78000079
// CHECK:STDOUT:     inst_block7800002B:
// CHECK:STDOUT:       0:               inst7800007D
// CHECK:STDOUT:       1:               inst7800007B
// CHECK:STDOUT:     inst_block7800002C:
// CHECK:STDOUT:       0:               inst7800007E
// CHECK:STDOUT:     inst_block7800002D:
// CHECK:STDOUT:       0:               inst7800007C
// CHECK:STDOUT:     inst_block7800002E:
// CHECK:STDOUT:       0:               inst78000082
// CHECK:STDOUT:     inst_block7800002F:
// CHECK:STDOUT:       0:               inst78000083
// CHECK:STDOUT:       1:               inst78000084
// CHECK:STDOUT:       2:               inst78000085
// CHECK:STDOUT:       3:               inst78000086
// CHECK:STDOUT:       4:               inst78000087
// CHECK:STDOUT:     inst_block78000030:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst78000089
// CHECK:STDOUT:       2:               inst7800008A
// CHECK:STDOUT:     inst_block78000031:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst78000072
// CHECK:STDOUT:       2:               inst7800008F
// CHECK:STDOUT:       3:               inst7800008E
// CHECK:STDOUT:     inst_block78000032:
// CHECK:STDOUT:       0:               inst78000083
// CHECK:STDOUT:     inst_block78000033:
// CHECK:STDOUT:       0:               inst78000083
// CHECK:STDOUT:     inst_block78000034:
// CHECK:STDOUT:       0:               inst78000092
// CHECK:STDOUT:       1:               inst78000093
// CHECK:STDOUT:       2:               inst78000094
// CHECK:STDOUT:       3:               inst78000095
// CHECK:STDOUT:       4:               inst78000096
// CHECK:STDOUT:       5:               inst78000097
// CHECK:STDOUT:       6:               inst78000098
// CHECK:STDOUT:     inst_block78000035:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst78000072
// CHECK:STDOUT:       2:               inst78000073
// CHECK:STDOUT:       3:               inst7800007A
// CHECK:STDOUT:       4:               inst7800007F
// CHECK:STDOUT:     inst_block78000036:
// CHECK:STDOUT:       0:               inst7800009A
// CHECK:STDOUT:     inst_block78000037:
// CHECK:STDOUT:       0:               inst7800009D
// CHECK:STDOUT:     inst_block78000038:
// CHECK:STDOUT:       0:               inst7800009E
// CHECK:STDOUT:     inst_block78000039:
// CHECK:STDOUT:       0:               inst7800009E
// CHECK:STDOUT:       1:               inst7800009F
// CHECK:STDOUT:       2:               inst780000A0
// CHECK:STDOUT:       3:               inst780000A1
// CHECK:STDOUT:     inst_block7800003A:
// CHECK:STDOUT:       0:               inst7800009E
// CHECK:STDOUT:     inst_block7800003B:
// CHECK:STDOUT:       0:               inst780000A2
// CHECK:STDOUT:       1:               inst780000A3
// CHECK:STDOUT:     inst_block7800003C:
// CHECK:STDOUT:       0:               inst780000B6
// CHECK:STDOUT:     inst_block7800003D:
// CHECK:STDOUT:       0:               inst78000018
// CHECK:STDOUT:       1:               inst7800001D
// CHECK:STDOUT:       2:               inst780000B8
// CHECK:STDOUT:     inst_block7800003E:
// CHECK:STDOUT:       0:               inst780000B9
// CHECK:STDOUT:     inst_block7800003F:
// CHECK:STDOUT:       0:               inst780000BC
// CHECK:STDOUT:     inst_block78000040:
// CHECK:STDOUT:       0:               inst780000BD
// CHECK:STDOUT:     inst_block78000041:
// CHECK:STDOUT:       0:               inst780000BD
// CHECK:STDOUT:       1:               inst780000BE
// CHECK:STDOUT:       2:               inst780000BF
// CHECK:STDOUT:     inst_block78000042:
// CHECK:STDOUT:       0:               inst780000C5
// CHECK:STDOUT:       1:               inst780000C3
// CHECK:STDOUT:     inst_block78000043:
// CHECK:STDOUT:       0:               inst780000C6
// CHECK:STDOUT:     inst_block78000044:
// CHECK:STDOUT:       0:               inst780000C4
// CHECK:STDOUT:     inst_block78000045:
// CHECK:STDOUT:       0:               inst780000CA
// CHECK:STDOUT:     inst_block78000046:
// CHECK:STDOUT:       0:               inst780000CB
// CHECK:STDOUT:       1:               inst780000CC
// CHECK:STDOUT:       2:               inst780000CD
// CHECK:STDOUT:       3:               inst780000CE
// CHECK:STDOUT:     inst_block78000047:
// CHECK:STDOUT:       0:               inst78000018
// CHECK:STDOUT:       1:               inst7800001D
// CHECK:STDOUT:       2:               inst7800001F
// CHECK:STDOUT:       3:               inst780000C7
// CHECK:STDOUT:     inst_block78000048:
// CHECK:STDOUT:       0:               inst780000BD
// CHECK:STDOUT:     inst_block78000049:
// CHECK:STDOUT:       0:               inst780000CF
// CHECK:STDOUT:       1:               inst780000D0
// CHECK:STDOUT:       2:               inst780000D1
// CHECK:STDOUT:     inst_block7800004A:
// CHECK:STDOUT:       0:               inst78000072
// CHECK:STDOUT:       1:               inst780000DD
// CHECK:STDOUT:     inst_block7800004B:
// CHECK:STDOUT:       0:               inst780000DF
// CHECK:STDOUT:     inst_block7800004C:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst780000DC
// CHECK:STDOUT:     inst_block7800004D:
// CHECK:STDOUT:       0:               inst78000050
// CHECK:STDOUT:       1:               inst78000050
// CHECK:STDOUT:     inst_block7800004E:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst780000DC
// CHECK:STDOUT:       2:               inst780000E6
// CHECK:STDOUT:       3:               inst78000072
// CHECK:STDOUT:       4:               inst780000DD
// CHECK:STDOUT:       5:               inst780000DE
// CHECK:STDOUT:       6:               inst780000E1
// CHECK:STDOUT:     inst_block7800004F:
// CHECK:STDOUT:       0:               inst780000E3
// CHECK:STDOUT:       1:               inst780000E4
// CHECK:STDOUT:     inst_block78000050:
// CHECK:STDOUT:       0:               inst780000EA
// CHECK:STDOUT:       1:               inst780000E8
// CHECK:STDOUT:     inst_block78000051:
// CHECK:STDOUT:       0:               inst780000EB
// CHECK:STDOUT:     inst_block78000052:
// CHECK:STDOUT:       0:               inst780000E9
// CHECK:STDOUT:     inst_block78000053:
// CHECK:STDOUT:       0:               inst780000EF
// CHECK:STDOUT:       1:               inst780000F0
// CHECK:STDOUT:     inst_block78000054:
// CHECK:STDOUT:       0:               inst780000F2
// CHECK:STDOUT:       1:               inst780000F4
// CHECK:STDOUT:     inst_block78000055:
// CHECK:STDOUT:       0:               inst780000F1
// CHECK:STDOUT:       1:               inst780000F2
// CHECK:STDOUT:       2:               inst780000F3
// CHECK:STDOUT:       3:               inst780000F4
// CHECK:STDOUT:       4:               inst780000F5
// CHECK:STDOUT:       5:               inst780000F6
// CHECK:STDOUT:       6:               inst780000F7
// CHECK:STDOUT:     inst_block78000056:
// CHECK:STDOUT:       0:               inst780000DC
// CHECK:STDOUT:     inst_block78000057:
// CHECK:STDOUT:       0:               inst780000DC
// CHECK:STDOUT:       1:               inst780000F9
// CHECK:STDOUT:       2:               inst780000FA
// CHECK:STDOUT:     inst_block78000058:
// CHECK:STDOUT:       0:               inst780000DC
// CHECK:STDOUT:       1:               inst780000DD
// CHECK:STDOUT:       2:               inst780000FF
// CHECK:STDOUT:       3:               inst780000FE
// CHECK:STDOUT:     inst_block78000059:
// CHECK:STDOUT:       0:               inst780000F1
// CHECK:STDOUT:     inst_block7800005A:
// CHECK:STDOUT:       0:               inst780000F1
// CHECK:STDOUT:     inst_block7800005B:
// CHECK:STDOUT:       0:               inst780000F3
// CHECK:STDOUT:     inst_block7800005C:
// CHECK:STDOUT:       0:               inst780000F3
// CHECK:STDOUT:     inst_block7800005D:
// CHECK:STDOUT:       0:               inst78000102
// CHECK:STDOUT:       1:               inst78000103
// CHECK:STDOUT:       2:               inst78000104
// CHECK:STDOUT:       3:               inst78000105
// CHECK:STDOUT:       4:               inst78000106
// CHECK:STDOUT:       5:               inst78000107
// CHECK:STDOUT:       6:               inst78000108
// CHECK:STDOUT:       7:               inst78000109
// CHECK:STDOUT:       8:               inst7800010A
// CHECK:STDOUT:       9:               inst7800010B
// CHECK:STDOUT:       10:              inst7800010C
// CHECK:STDOUT:       11:              inst7800010D
// CHECK:STDOUT:       12:              inst7800010E
// CHECK:STDOUT:     inst_block7800005E:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst78000072
// CHECK:STDOUT:       2:               inst780000DC
// CHECK:STDOUT:       3:               inst780000DD
// CHECK:STDOUT:       4:               inst780000DE
// CHECK:STDOUT:       5:               inst780000E7
// CHECK:STDOUT:       6:               inst780000EC
// CHECK:STDOUT:     inst_block7800005F:
// CHECK:STDOUT:       0:               inst78000110
// CHECK:STDOUT:       1:               inst7800010F
// CHECK:STDOUT:     inst_block78000060:
// CHECK:STDOUT:       0:               inst78000113
// CHECK:STDOUT:       1:               inst78000114
// CHECK:STDOUT:     inst_block78000061:
// CHECK:STDOUT:       0:               inst78000115
// CHECK:STDOUT:       1:               inst78000116
// CHECK:STDOUT:     inst_block78000062:
// CHECK:STDOUT:       0:               inst78000118
// CHECK:STDOUT:       1:               inst78000119
// CHECK:STDOUT:     inst_block78000063:
// CHECK:STDOUT:       0:               inst78000115
// CHECK:STDOUT:       1:               inst78000116
// CHECK:STDOUT:     inst_block78000064:
// CHECK:STDOUT:       0:               inst78000115
// CHECK:STDOUT:       1:               inst78000116
// CHECK:STDOUT:       2:               inst78000117
// CHECK:STDOUT:       3:               inst78000118
// CHECK:STDOUT:       4:               inst78000119
// CHECK:STDOUT:       5:               inst7800011A
// CHECK:STDOUT:       6:               inst7800011B
// CHECK:STDOUT:     inst_block78000065:
// CHECK:STDOUT:       0:               inst78000115
// CHECK:STDOUT:       1:               inst78000116
// CHECK:STDOUT:     inst_block78000066:
// CHECK:STDOUT:       0:               inst7800011C
// CHECK:STDOUT:       1:               inst7800011D
// CHECK:STDOUT:     inst_block78000067:
// CHECK:STDOUT:       0:               inst78000072
// CHECK:STDOUT:       1:               inst780000DD
// CHECK:STDOUT:       2:               inst78000121
// CHECK:STDOUT:     inst_block78000068:
// CHECK:STDOUT:       0:               inst78000123
// CHECK:STDOUT:     inst_block78000069:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst780000DC
// CHECK:STDOUT:       2:               inst78000120
// CHECK:STDOUT:     inst_block7800006A:
// CHECK:STDOUT:       0:               inst78000050
// CHECK:STDOUT:       1:               inst78000050
// CHECK:STDOUT:       2:               inst78000050
// CHECK:STDOUT:     inst_block7800006B:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst780000DC
// CHECK:STDOUT:       2:               inst78000120
// CHECK:STDOUT:       3:               inst7800012A
// CHECK:STDOUT:       4:               inst78000072
// CHECK:STDOUT:       5:               inst780000DD
// CHECK:STDOUT:       6:               inst78000121
// CHECK:STDOUT:       7:               inst78000122
// CHECK:STDOUT:       8:               inst78000125
// CHECK:STDOUT:     inst_block7800006C:
// CHECK:STDOUT:       0:               inst78000127
// CHECK:STDOUT:       1:               inst78000128
// CHECK:STDOUT:     inst_block7800006D:
// CHECK:STDOUT:       0:               inst7800012E
// CHECK:STDOUT:       1:               inst7800012C
// CHECK:STDOUT:     inst_block7800006E:
// CHECK:STDOUT:       0:               inst7800012F
// CHECK:STDOUT:     inst_block7800006F:
// CHECK:STDOUT:       0:               inst7800012D
// CHECK:STDOUT:     inst_block78000070:
// CHECK:STDOUT:       0:               inst78000133
// CHECK:STDOUT:       1:               inst78000134
// CHECK:STDOUT:       2:               inst78000135
// CHECK:STDOUT:     inst_block78000071:
// CHECK:STDOUT:       0:               inst78000137
// CHECK:STDOUT:       1:               inst78000139
// CHECK:STDOUT:       2:               inst7800013B
// CHECK:STDOUT:     inst_block78000072:
// CHECK:STDOUT:       0:               inst78000136
// CHECK:STDOUT:       1:               inst78000137
// CHECK:STDOUT:       2:               inst78000138
// CHECK:STDOUT:       3:               inst78000139
// CHECK:STDOUT:       4:               inst7800013A
// CHECK:STDOUT:       5:               inst7800013B
// CHECK:STDOUT:       6:               inst7800013C
// CHECK:STDOUT:       7:               inst7800013D
// CHECK:STDOUT:       8:               inst7800013E
// CHECK:STDOUT:     inst_block78000073:
// CHECK:STDOUT:       0:               inst78000120
// CHECK:STDOUT:     inst_block78000074:
// CHECK:STDOUT:       0:               inst78000120
// CHECK:STDOUT:       1:               inst78000140
// CHECK:STDOUT:       2:               inst78000141
// CHECK:STDOUT:     inst_block78000075:
// CHECK:STDOUT:       0:               inst78000120
// CHECK:STDOUT:       1:               inst78000121
// CHECK:STDOUT:       2:               inst78000146
// CHECK:STDOUT:       3:               inst78000145
// CHECK:STDOUT:     inst_block78000076:
// CHECK:STDOUT:       0:               inst78000136
// CHECK:STDOUT:     inst_block78000077:
// CHECK:STDOUT:       0:               inst78000136
// CHECK:STDOUT:     inst_block78000078:
// CHECK:STDOUT:       0:               inst78000138
// CHECK:STDOUT:     inst_block78000079:
// CHECK:STDOUT:       0:               inst78000138
// CHECK:STDOUT:     inst_block7800007A:
// CHECK:STDOUT:       0:               inst7800013A
// CHECK:STDOUT:     inst_block7800007B:
// CHECK:STDOUT:       0:               inst7800013A
// CHECK:STDOUT:     inst_block7800007C:
// CHECK:STDOUT:       0:               inst78000149
// CHECK:STDOUT:       1:               inst7800014A
// CHECK:STDOUT:       2:               inst7800014B
// CHECK:STDOUT:       3:               inst7800014C
// CHECK:STDOUT:       4:               inst7800014D
// CHECK:STDOUT:       5:               inst7800014E
// CHECK:STDOUT:       6:               inst7800014F
// CHECK:STDOUT:       7:               inst78000150
// CHECK:STDOUT:       8:               inst78000151
// CHECK:STDOUT:       9:               inst78000152
// CHECK:STDOUT:       10:              inst78000153
// CHECK:STDOUT:       11:              inst78000154
// CHECK:STDOUT:       12:              inst78000155
// CHECK:STDOUT:       13:              inst78000156
// CHECK:STDOUT:       14:              inst78000157
// CHECK:STDOUT:       15:              inst78000158
// CHECK:STDOUT:       16:              inst78000159
// CHECK:STDOUT:       17:              inst7800015A
// CHECK:STDOUT:       18:              inst7800015B
// CHECK:STDOUT:     inst_block7800007D:
// CHECK:STDOUT:       0:               inst78000071
// CHECK:STDOUT:       1:               inst78000072
// CHECK:STDOUT:       2:               inst780000DC
// CHECK:STDOUT:       3:               inst780000DD
// CHECK:STDOUT:       4:               inst78000120
// CHECK:STDOUT:       5:               inst78000121
// CHECK:STDOUT:       6:               inst78000122
// CHECK:STDOUT:       7:               inst7800012B
// CHECK:STDOUT:       8:               inst78000130
// CHECK:STDOUT:     inst_block7800007E:
// CHECK:STDOUT:       0:               inst7800015E
// CHECK:STDOUT:       1:               inst7800015D
// CHECK:STDOUT:       2:               inst7800015C
// CHECK:STDOUT:     inst_block7800007F:
// CHECK:STDOUT:       0:               inst78000161
// CHECK:STDOUT:       1:               inst78000162
// CHECK:STDOUT:       2:               inst78000163
// CHECK:STDOUT:     inst_block78000080:
// CHECK:STDOUT:       0:               inst78000164
// CHECK:STDOUT:       1:               inst78000165
// CHECK:STDOUT:       2:               inst78000166
// CHECK:STDOUT:     inst_block78000081:
// CHECK:STDOUT:       0:               inst78000168
// CHECK:STDOUT:       1:               inst78000169
// CHECK:STDOUT:       2:               inst7800016A
// CHECK:STDOUT:     inst_block78000082:
// CHECK:STDOUT:       0:               inst78000164
// CHECK:STDOUT:       1:               inst78000165
// CHECK:STDOUT:       2:               inst78000166
// CHECK:STDOUT:     inst_block78000083:
// CHECK:STDOUT:       0:               inst78000164
// CHECK:STDOUT:       1:               inst78000165
// CHECK:STDOUT:       2:               inst78000166
// CHECK:STDOUT:       3:               inst78000167
// CHECK:STDOUT:       4:               inst78000168
// CHECK:STDOUT:       5:               inst78000169
// CHECK:STDOUT:       6:               inst7800016A
// CHECK:STDOUT:       7:               inst7800016B
// CHECK:STDOUT:       8:               inst7800016C
// CHECK:STDOUT:     inst_block78000084:
// CHECK:STDOUT:       0:               inst78000164
// CHECK:STDOUT:       1:               inst78000165
// CHECK:STDOUT:       2:               inst78000166
// CHECK:STDOUT:     inst_block78000085:
// CHECK:STDOUT:       0:               inst7800016D
// CHECK:STDOUT:       1:               inst7800016E
// CHECK:STDOUT:     inst_block78000086:
// CHECK:STDOUT:       0:               inst780000C1
// CHECK:STDOUT:       1:               inst780000C2
// CHECK:STDOUT:       2:               inst78000042
// CHECK:STDOUT:     inst_block78000087:
// CHECK:STDOUT:       0:               inst78000019
// CHECK:STDOUT:     inst_block78000088:
// CHECK:STDOUT:       0:               inst78000172
// CHECK:STDOUT:     inst_block78000089:
// CHECK:STDOUT:       0:               inst78000174
// CHECK:STDOUT:     inst_block7800008A:
// CHECK:STDOUT:       0:               inst78000174
// CHECK:STDOUT:       1:               inst78000175
// CHECK:STDOUT:       2:               inst78000176
// CHECK:STDOUT:     inst_block7800008B:
// CHECK:STDOUT:       0:               inst78000173
// CHECK:STDOUT:     inst_block7800008C:
// CHECK:STDOUT:       0:               inst7800017A
// CHECK:STDOUT:     inst_block7800008D:
// CHECK:STDOUT:       0:               inst78000174
// CHECK:STDOUT:       1:               inst7800001D
// CHECK:STDOUT:       2:               inst7800001F
// CHECK:STDOUT:       3:               inst780000C7
// CHECK:STDOUT:     inst_block7800008E:
// CHECK:STDOUT:       0:               inst7800017A
// CHECK:STDOUT:     inst_block7800008F:
// CHECK:STDOUT:       0:               inst78000048
// CHECK:STDOUT:     inst_block78000090:
// CHECK:STDOUT:       0:               inst78000184
// CHECK:STDOUT:       1:               inst78000187
// CHECK:STDOUT:     inst_block78000091:
// CHECK:STDOUT:       0:               inst78000043
// CHECK:STDOUT:       1:               inst78000047
// CHECK:STDOUT:       2:               inst78000171
// CHECK:STDOUT:       3:               inst78000173
// CHECK:STDOUT:       4:               inst7800017A
// CHECK:STDOUT:       5:               inst7800017B
// CHECK:STDOUT:       6:               inst7800017C
// CHECK:STDOUT:       7:               inst7800017D
// CHECK:STDOUT:       8:               inst78000181
// CHECK:STDOUT:     inst_block78000092:
// CHECK:STDOUT:       0:               instF
// CHECK:STDOUT:       1:               inst78000010
// CHECK:STDOUT:       2:               inst7800003E
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
