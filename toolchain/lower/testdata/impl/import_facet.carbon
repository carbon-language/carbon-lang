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

class C(T: Core.Copy & Core.UnformedInit & Core.Destroy) {
  fn GetC(unused self) -> T {
    var value: T;
    return value;
  }
}

// --- b.carbon
library "[[@TEST_NAME]]";
export import library "a";

interface I {
  let T: Core.Copy & Core.UnformedInit & Core.Destroy;
  fn GetI(self) -> C(T);
}

// --- c.carbon
library "[[@TEST_NAME]]";
export import library "b";

class D(U: type) {
  impl as I where .T = U* {
    fn GetI(unused self) -> C(U*) {
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!7}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'c.carbon'
// CHECK:STDOUT: source_filename = "c.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!9}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !8 = !DIFile(filename: "c.carbon", directory: "")
// CHECK:STDOUT: !9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !8, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'd.carbon'
// CHECK:STDOUT: source_filename = "d.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.207.anon = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %d) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_21.1.temp = alloca {}, align 1, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_21.1.temp), !dbg !22
// CHECK:STDOUT:   call void @"_CGetI.D.eb057aa32837c84e.Main:I.Main.84588f41d61dafba"(ptr %.loc5_21.1.temp, ptr %d), !dbg !22
// CHECK:STDOUT:   %C.GetC.call = call ptr @_CGetC.C.Main.aac7339a58c49d52(ptr %.loc5_21.1.temp), !dbg !22
// CHECK:STDOUT:   call void @"_CSelfDestruct.14de9dd8f7d35649:core.Destroy.Core"(ptr %.loc5_21.1.temp), !dbg !22
// CHECK:STDOUT:   ret ptr %C.GetC.call, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.14de9dd8f7d35649:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.14de9dd8f7d35649:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.14de9dd8f7d35649:core.Destroy.Core"(ptr %self), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGetI.D.eb057aa32837c84e.Main:I.Main.84588f41d61dafba"(ptr sret({}) %return, ptr %self) #0 !dbg !41 {
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.207.anon, i64 0, i1 false), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGetC.C.Main.aac7339a58c49d52(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !53
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !53
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !53
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !55
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %1), !dbg !53
// CHECK:STDOUT:   ret ptr %2, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!11}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !8 = !DIFile(filename: "c.carbon", directory: "")
// CHECK:STDOUT: !10 = !DIFile(filename: "d.carbon", directory: "")
// CHECK:STDOUT: !11 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !10, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !13 = !{!12, !12}
// CHECK:STDOUT: !14 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !15 = !{null, !12}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !10, line: 4, type: !14, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !24)
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !17, type: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 5, column: 10, scope: !17)
// CHECK:STDOUT: !23 = !DILocation(line: 5, column: 3, scope: !17)
// CHECK:STDOUT: !24 = !{!18}
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !8, line: 5, type: !16, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !28)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !12)
// CHECK:STDOUT: !27 = !DILocation(line: 5, column: 24, scope: !25)
// CHECK:STDOUT: !28 = !{!26}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !10, line: 5, type: !16, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !32)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !12)
// CHECK:STDOUT: !31 = !DILocation(line: 5, column: 10, scope: !29)
// CHECK:STDOUT: !32 = !{!30}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.14de9dd8f7d35649:core.Destroy.Core", scope: null, file: !10, line: 5, type: !16, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !36)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !12)
// CHECK:STDOUT: !35 = !DILocation(line: 5, column: 10, scope: !33)
// CHECK:STDOUT: !36 = !{!34}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.14de9dd8f7d35649:core.Destroy.Core", scope: null, file: !10, line: 5, type: !16, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !12)
// CHECK:STDOUT: !39 = !DILocation(line: 5, column: 10, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "GetI", linkageName: "_CGetI.D.eb057aa32837c84e.Main:I.Main.84588f41d61dafba", scope: null, file: !8, line: 6, type: !14, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !47)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !12)
// CHECK:STDOUT: !46 = !DILocation(line: 7, column: 7, scope: !41)
// CHECK:STDOUT: !47 = !{!42}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "GetC", linkageName: "_CGetC.C.Main.aac7339a58c49d52", scope: null, file: !4, line: 4, type: !14, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !57)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !12)
// CHECK:STDOUT: !53 = !DILocation(line: 5, column: 5, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 6, column: 12, scope: !48)
// CHECK:STDOUT: !56 = !DILocation(line: 6, column: 5, scope: !48)
// CHECK:STDOUT: !57 = !{!49}
// CHECK:STDOUT:
