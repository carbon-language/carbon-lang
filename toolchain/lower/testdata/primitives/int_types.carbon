// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/int_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/int_types.carbon

fn F_i8(a: i8) -> i8 { return a; }
fn F_u16(a: u16) -> u16 { return a; }
fn F_i64(a: i64) -> i64 { return a; }
fn F_u65536(a: u65536) -> u65536 { return a; }

fn LoadStore_Bytes(p: Core.Int(24)*, n: Core.Int(24)) -> Core.Int(24) {
  let v: Core.Int(24) = *p;
  *p = n;
  return v;
}

fn LoadStore_NotBytes(p: Core.Int(13)*, n: Core.Int(13)) -> Core.Int(13) {
  let v: Core.Int(13) = *p;
  *p = n;
  return v;
}

// CHECK:STDOUT: ; ModuleID = 'int_types.carbon'
// CHECK:STDOUT: source_filename = "int_types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CF_i8.Main(i8 %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 %a, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CF_u16.Main(i16 %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i16 %a, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CF_i64.Main(i64 %a) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i64 %a, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i65536 @_CF_u65536.Main(i65536 %a) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i65536 %a, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i24 @_CLoadStore_Bytes.Main(ptr %p, i24 %n) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_25.2 = load i24, ptr %p, align 4, !dbg !40
// CHECK:STDOUT:   store i24 %n, ptr %p, align 4, !dbg !41
// CHECK:STDOUT:   ret i24 %.loc19_25.2, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i13 @_CLoadStore_NotBytes.Main(ptr %p, i13 %n) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc25_25.2 = load i16, ptr %p, align 2, !dbg !50
// CHECK:STDOUT:   %.loc25_25.21 = trunc i16 %.loc25_25.2 to i13, !dbg !50
// CHECK:STDOUT:   %0 = zext i13 %n to i16, !dbg !51
// CHECK:STDOUT:   store i16 %0, ptr %p, align 2, !dbg !51
// CHECK:STDOUT:   ret i13 %.loc25_25.21, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "int_types.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F_i8", linkageName: "_CF_i8.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 24, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F_u16", linkageName: "_CF_u16.Main", scope: null, file: !3, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !14}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 14, column: 27, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F_i64", linkageName: "_CF_i64.Main", scope: null, file: !3, line: 15, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{!21, !21}
// CHECK:STDOUT: !21 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !18, type: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 15, column: 27, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F_u65536", linkageName: "_CF_u65536.Main", scope: null, file: !3, line: 16, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{!28, !28}
// CHECK:STDOUT: !28 = !DIBasicType(name: "int", size: 65536, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !25, type: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 36, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "LoadStore_Bytes", linkageName: "_CLoadStore_Bytes.Main", scope: null, file: !3, line: 18, type: !33, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !33 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !34 = !{!35, !36, !35}
// CHECK:STDOUT: !35 = !DIBasicType(name: "int", size: 24, encoding: DW_ATE_signed)
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !37 = !{!38, !39}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !32, type: !36)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !32, type: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 25, scope: !32)
// CHECK:STDOUT: !41 = !DILocation(line: 20, column: 3, scope: !32)
// CHECK:STDOUT: !42 = !DILocation(line: 21, column: 3, scope: !32)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "LoadStore_NotBytes", linkageName: "_CLoadStore_NotBytes.Main", scope: null, file: !3, line: 24, type: !44, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !47)
// CHECK:STDOUT: !44 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !45 = !{!46, !36, !46}
// CHECK:STDOUT: !46 = !DIBasicType(name: "int", size: 13, encoding: DW_ATE_signed)
// CHECK:STDOUT: !47 = !{!48, !49}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !43, type: !36)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 2, scope: !43, type: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 25, column: 25, scope: !43)
// CHECK:STDOUT: !51 = !DILocation(line: 26, column: 3, scope: !43)
// CHECK:STDOUT: !52 = !DILocation(line: 27, column: 3, scope: !43)
