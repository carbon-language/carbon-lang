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
// CHECK:STDOUT: @C.val.1e1.anon = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CF.Main(ptr %d) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_21.1.temp = alloca {}, align 1, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_21.1.temp), !dbg !20
// CHECK:STDOUT:   call void @"_CGetI.D.eb057aa32837c84e.Main:I.Main.55456748a8a2eb70"(ptr %.loc5_21.1.temp, ptr %d), !dbg !20
// CHECK:STDOUT:   %C.GetC.call = call ptr @_CGetC.C.Main.ba9e1b1212a208fc(ptr %.loc5_21.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.15aa9db226b6d202(ptr %.loc5_21.1.temp), !dbg !20
// CHECK:STDOUT:   ret ptr %C.GetC.call, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CGetI.D.eb057aa32837c84e.Main:I.Main.55456748a8a2eb70"(ptr sret({}) %return, ptr %self) #0 !dbg !25 {
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.1e1.anon, i64 0, i1 false), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGetC.C.Main.ba9e1b1212a208fc(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !37
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !37
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5da679f5f4cb5255(ptr %1), !dbg !37
// CHECK:STDOUT:   ret ptr %2, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.15aa9db226b6d202(ptr %t) #0 !dbg !43 {
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5da679f5f4cb5255(ptr %t) #0 !dbg !50 {
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
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
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !10, line: 4, type: !14, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !24)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !15, type: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 5, column: 10, scope: !15)
// CHECK:STDOUT: !21 = !{null, !12}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 5, column: 3, scope: !15)
// CHECK:STDOUT: !24 = !{!16}
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "GetI", linkageName: "_CGetI.D.eb057aa32837c84e.Main:I.Main.55456748a8a2eb70", scope: null, file: !8, line: 6, type: !14, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !31)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !12)
// CHECK:STDOUT: !30 = !DILocation(line: 7, column: 7, scope: !25)
// CHECK:STDOUT: !31 = !{!26}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "GetC", linkageName: "_CGetC.C.Main.ba9e1b1212a208fc", scope: null, file: !4, line: 4, type: !14, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !41)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !12)
// CHECK:STDOUT: !37 = !DILocation(line: 5, column: 5, scope: !32)
// CHECK:STDOUT: !39 = !DILocation(line: 6, column: 12, scope: !32)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 5, scope: !32)
// CHECK:STDOUT: !41 = !{!33}
// CHECK:STDOUT: !42 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.15aa9db226b6d202", scope: null, file: !42, line: 35, type: !22, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !49)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !12)
// CHECK:STDOUT: !48 = !DILocation(line: 35, column: 1, scope: !43)
// CHECK:STDOUT: !49 = !{!44}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5da679f5f4cb5255", scope: null, file: !42, line: 35, type: !22, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !56)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !12)
// CHECK:STDOUT: !55 = !DILocation(line: 35, column: 1, scope: !50)
// CHECK:STDOUT: !56 = !{!51}
// CHECK:STDOUT:
