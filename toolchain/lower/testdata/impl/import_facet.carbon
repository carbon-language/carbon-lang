// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is testing a facet type that combines two interfaces across multiple
// import boundaries.
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/impl/import_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/impl/import_facet.carbon

// --- a.carbon
library "[[@TEST_NAME]]";

class C(T:! Core.Copy & Core.UnformedInit & Core.Destroy) {
  fn GetC[unused self: Self]() -> T {
    var value: T;
    return value;
  }
}

// --- b.carbon
library "[[@TEST_NAME]]";
export import library "a";

interface I {
  let T:! Core.Copy & Core.UnformedInit & Core.Destroy;
  fn GetI[unused self: Self]() -> C(T);
}

// --- c.carbon
library "[[@TEST_NAME]]";
export import library "b";

class D(U:! type) {
  impl as I where .T = U* {
    fn GetI[unused self: Self]() -> C(U*) {
      return {};
    }
  }
}

// --- d.carbon
library "[[@TEST_NAME]]";
import library "c";

fn F(d: D(i32)) -> i32* {
  return d.(I.GetI)().GetC();
}

// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'c.carbon'
// CHECK:STDOUT: source_filename = "c.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "c.carbon", directory: "")
// CHECK:STDOUT: ; ModuleID = 'd.carbon'
// CHECK:STDOUT: source_filename = "d.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.096.anon = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %d) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_21.1.temp = alloca {}, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_21.1.temp), !dbg !10
// CHECK:STDOUT:   call void @"_CGetI.D.eb057aa32837c84e.Main:I.Main.b88d1103f417c6d4"(ptr %.loc5_21.1.temp, ptr %d), !dbg !10
// CHECK:STDOUT:   %C.GetC.call = call ptr @_CGetC.C.Main.7ea07d77fe061e6d(ptr %.loc5_21.1.temp), !dbg !10
// CHECK:STDOUT:   call void @"_COp.a0926fd4b0145051:core.Destroy.Core"(ptr %.loc5_21.1.temp), !dbg !10
// CHECK:STDOUT:   ret ptr %C.GetC.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a0926fd4b0145051:core.Destroy.Core"(ptr %self) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGetI.D.eb057aa32837c84e.Main:I.Main.b88d1103f417c6d4"(ptr sret({}) %return, ptr %self) #0 !dbg !18 {
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.096.anon, i64 0, i1 false), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGetC.C.Main.7ea07d77fe061e6d(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !27
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !27
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !28
// CHECK:STDOUT:   ret ptr %2, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
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
// CHECK:STDOUT: !3 = !DIFile(filename: "d.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 5, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 5, column: 3, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a0926fd4b0145051:core.Destroy.Core", scope: null, file: !3, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !7}
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 5, column: 10, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "GetI", linkageName: "_CGetI.D.eb057aa32837c84e.Main:I.Main.b88d1103f417c6d4", scope: null, file: !19, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !19 = !DIFile(filename: "c.carbon", directory: "")
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 7, column: 7, scope: !18)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "GetC", linkageName: "_CGetC.C.Main.7ea07d77fe061e6d", scope: null, file: !24, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !24 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !23, type: !7)
// CHECK:STDOUT: !27 = !DILocation(line: 5, column: 5, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 6, column: 12, scope: !23)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 5, scope: !23)
