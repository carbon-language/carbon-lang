// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --include-carbon-core
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_different_impls_with_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_different_impls_with_const.carbon

import Core library "io";

interface I {
  let T: Core.Destroy;
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
fn G(generic U: I) {
  let _: U.T = U.F();
}

fn Run() {
  G(X);
  G(Y);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_different_impls_with_const.carbon'
// CHECK:STDOUT: source_filename = "call_different_impls_with_const.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @"_CF.X.Main:I.Main"() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !17
// CHECK:STDOUT:   ret i1 false, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CF.Y.Main:I.Main"() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !26
// CHECK:STDOUT:   ret i32 2, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.d0cc85b6c89517e3(), !dbg !29
// CHECK:STDOUT:   call void @_CG.Main.e8d4d639bae9ba40(), !dbg !32
// CHECK:STDOUT:   ret i32 0, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.d0cc85b6c89517e3() #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc38_20.1.temp = alloca i1, align 1, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc38_20.1.temp), !dbg !35
// CHECK:STDOUT:   %I.WithSelf.F.call = call i1 @"_CF.X.Main:I.Main"(), !dbg !35
// CHECK:STDOUT:   %.loc38_20.2 = zext i1 %I.WithSelf.F.call to i8, !dbg !35
// CHECK:STDOUT:   store i8 %.loc38_20.2, ptr %.loc38_20.1.temp, align 1, !dbg !35
// CHECK:STDOUT:   %.loc38_20.3 = load i8, ptr %.loc38_20.1.temp, align 1, !dbg !35
// CHECK:STDOUT:   %.loc38_20.31 = trunc i8 %.loc38_20.3 to i1, !dbg !35
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.e8d4d639bae9ba40() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc38_20.1.temp = alloca i32, align 4, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc38_20.1.temp), !dbg !40
// CHECK:STDOUT:   %I.WithSelf.F.call = call i32 @"_CF.Y.Main:I.Main"(), !dbg !40
// CHECK:STDOUT:   store i32 %I.WithSelf.F.call, ptr %.loc38_20.1.temp, align 4, !dbg !40
// CHECK:STDOUT:   %.loc38_20.3 = load i32, ptr %.loc38_20.1.temp, align 4, !dbg !40
// CHECK:STDOUT:   call void @"_COp.f58524831c37807e:core.Destroy.Core"(ptr %.loc38_20.1.temp), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !43
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_different_impls_with_const.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !4, line: 22, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DILocation(line: 23, column: 5, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 24, column: 5, scope: !14)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 28, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !6)
// CHECK:STDOUT: !21 = !DILocation(line: 28, column: 24, scope: !19)
// CHECK:STDOUT: !22 = !{!20}
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Y.Main:I.Main", scope: null, file: !4, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !26 = !DILocation(line: 30, column: 5, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 31, column: 5, scope: !23)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !4, line: 41, type: !13, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !29 = !DILocation(line: 42, column: 3, scope: !28)
// CHECK:STDOUT: !30 = !{null}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 43, column: 3, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 41, column: 1, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.d0cc85b6c89517e3", scope: null, file: !4, line: 37, type: !31, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !35 = !DILocation(line: 38, column: 16, scope: !34)
// CHECK:STDOUT: !38 = !DILocation(line: 37, column: 1, scope: !34)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e8d4d639bae9ba40", scope: null, file: !4, line: 37, type: !31, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !40 = !DILocation(line: 38, column: 16, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 37, column: 1, scope: !39)
// CHECK:STDOUT:
