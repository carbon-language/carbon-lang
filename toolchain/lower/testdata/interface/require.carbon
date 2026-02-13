// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interface/require.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interface/require.carbon

interface A {
    fn F1[self: Self]() -> i32;
    fn F2[self: Self]() -> i32;
}

interface B {
    require impls A;
}

fn TakesA[T:! A](x: T) -> i32 {
    return x.F2();
}

fn TakesB[T:! B](x: T) -> i32 {
    return TakesA(x);
}

class C {}

impl C as A {
    fn F1[self: C]() -> i32 {
        return 86;
    }
    fn F2[self: C]() -> i32 {
        return 42;
    }
}

impl C as B {}

fn Run() -> i32 {
    var c: C;
    return TakesB(c);
}

// CHECK:STDOUT: ; ModuleID = 'require.carbon'
// CHECK:STDOUT: source_filename = "require.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF1.C.Main:A.Main"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 86, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF2.C.Main:A.Main"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 42, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !19
// CHECK:STDOUT:   %TakesB.call = call i32 @_CTakesB.Main.28c67e05fd10f829(ptr %c.var), !dbg !20
// CHECK:STDOUT:   ret i32 %TakesB.call, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CTakesB.Main.28c67e05fd10f829(ptr %x) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %TakesA.call = call i32 @_CTakesA.Main.b6d50ff22524203f(ptr %x), !dbg !25
// CHECK:STDOUT:   ret i32 %TakesA.call, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CTakesA.Main.b6d50ff22524203f(ptr %x) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.F2.call = call i32 @"_CF2.C.Main:A.Main"(ptr %x), !dbg !30
// CHECK:STDOUT:   ret i32 %A.F2.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "require.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F1", linkageName: "_CF1.C.Main:A.Main", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 34, column: 9, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "F2", linkageName: "_CF2.C.Main:A.Main", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !13)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !12, type: !8)
// CHECK:STDOUT: !15 = !DILocation(line: 37, column: 9, scope: !12)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 43, type: !17, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{!7}
// CHECK:STDOUT: !19 = !DILocation(line: 44, column: 5, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 45, column: 12, scope: !16)
// CHECK:STDOUT: !21 = !DILocation(line: 45, column: 5, scope: !16)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "TakesB", linkageName: "_CTakesB.Main.28c67e05fd10f829", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !8)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 12, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 5, scope: !22)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "TakesA", linkageName: "_CTakesA.Main.b6d50ff22524203f", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !8)
// CHECK:STDOUT: !30 = !DILocation(line: 23, column: 12, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 23, column: 5, scope: !27)
