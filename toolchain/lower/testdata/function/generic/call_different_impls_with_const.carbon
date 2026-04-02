// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_different_impls_with_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_different_impls_with_const.carbon

import Core library "io";

interface I {
  let T:! Core.Destroy;
  fn F() -> T;
}
class X {}
impl X as I where .T = bool {
  fn F() -> bool {
    Core.Print(1);
    return false;
  }
}
class Y {}
impl Y as I where .T = i32 {
  fn F() -> i32 {
    Core.Print(2);
    return 2;
  }
}

// Cannot coalesce the lowering for G specifics, as they call different functions.
// Check different functions are still emitted when trying to deduplicate emitted definitons.
fn G(U:! I) {
  let _: U.T = U.F();
}

fn Run() {
  G(X);
  G(Y);
}

// CHECK:STDOUT: ; ModuleID = 'call_different_impls_with_const.carbon'
// CHECK:STDOUT: source_filename = "call_different_impls_with_const.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @"_CF.X.Main:I.Main"() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !8
// CHECK:STDOUT:   ret i1 false, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.Y.Main:I.Main"() #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !20
// CHECK:STDOUT:   ret i32 2, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.27a852a6bab6ed2e(), !dbg !25
// CHECK:STDOUT:   call void @_CG.Main.2de7c972fe093485(), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.27a852a6bab6ed2e() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc37_20.1.temp = alloca i1, align 1, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc37_20.1.temp), !dbg !29
// CHECK:STDOUT:   %I.WithSelf.F.call = call i1 @"_CF.X.Main:I.Main"(), !dbg !29
// CHECK:STDOUT:   %.loc37_20.2 = zext i1 %I.WithSelf.F.call to i8, !dbg !29
// CHECK:STDOUT:   store i8 %.loc37_20.2, ptr %.loc37_20.1.temp, align 1, !dbg !29
// CHECK:STDOUT:   %.loc37_20.3 = load i8, ptr %.loc37_20.1.temp, align 1, !dbg !29
// CHECK:STDOUT:   %.loc37_20.31 = trunc i8 %.loc37_20.3 to i1, !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.2de7c972fe093485() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc37_20.1.temp = alloca i32, align 4, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc37_20.1.temp), !dbg !32
// CHECK:STDOUT:   %I.WithSelf.F.call = call i32 @"_CF.Y.Main:I.Main"(), !dbg !32
// CHECK:STDOUT:   store i32 %I.WithSelf.F.call, ptr %.loc37_20.1.temp, align 4, !dbg !32
// CHECK:STDOUT:   %.loc37_20.3 = load i32, ptr %.loc37_20.1.temp, align 4, !dbg !32
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %.loc37_20.1.temp), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @printf, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "call_different_impls_with_const.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !DILocation(line: 22, column: 5, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 23, column: 5, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 27, type: !11, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{null, !13}
// CHECK:STDOUT: !13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !14 = !{!15}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !10, type: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 27, column: 24, scope: !10)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "F", linkageName: "_CF.Y.Main:I.Main", scope: null, file: !3, line: 28, type: !18, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{!13}
// CHECK:STDOUT: !20 = !DILocation(line: 29, column: 5, scope: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 30, column: 5, scope: !17)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 40, type: !23, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DILocation(line: 41, column: 3, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 42, column: 3, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 40, column: 1, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.27a852a6bab6ed2e", scope: null, file: !3, line: 36, type: !23, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !29 = !DILocation(line: 37, column: 16, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 36, column: 1, scope: !28)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.2de7c972fe093485", scope: null, file: !3, line: 36, type: !23, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !32 = !DILocation(line: 37, column: 16, scope: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 36, column: 1, scope: !31)
