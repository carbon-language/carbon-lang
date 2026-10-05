// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/template/class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/template/class.carbon

// --- init_class.carbon
library "[[@TEST_NAME]]";

class B {
  impl as Core.Copy {
    fn Op(unused self) -> Self {
      return {};
    }
  }
}

class A(template T: type) {
  fn Make(v: T) -> A(T) {
    return {
      .field = v,
    };
  }

  var field: T;
}

fn Run() {
  A(B).Make({});
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'init_class.carbon'
// CHECK:STDOUT: source_filename = "init_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @B.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @B.val.loc6_16 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.B.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc6_16, i64 0, i1 false), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_15.1.temp = alloca { {} }, align 1, !dbg !32
// CHECK:STDOUT:   %.loc22_14.2.temp = alloca {}, align 1, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_15.1.temp), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_14.2.temp), !dbg !33
// CHECK:STDOUT:   call void @_CMake.A.Main.a6a60dfee2d7d273(ptr %.loc22_15.1.temp, ptr @B.val.loc6_16), !dbg !32
// CHECK:STDOUT:   call void @"_CSelfDestruct.d32110776493a13a:core.Destroy.Core"(ptr %.loc22_15.1.temp), !dbg !32
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr @B.val), !dbg !33
// CHECK:STDOUT:   ret i32 0, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.8fbe0c6a865381a4:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22.field = getelementptr inbounds nuw { {} }, ptr %self, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %.loc22.field), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8fbe0c6a865381a4:core.Destroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.8fbe0c6a865381a4:core.Destroy.Core"(ptr %self), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d32110776493a13a:core.Destroy.Core"(ptr %self) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22.field = getelementptr inbounds nuw { {} }, ptr %self, i32 0, i32 0, !dbg !53
// CHECK:STDOUT:   call void @"_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core"(ptr %.loc22.field), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d32110776493a13a:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d32110776493a13a:core.Destroy.Core"(ptr %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.A.Main.a6a60dfee2d7d273(ptr sret({ {} }) %return, ptr %v) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.b3e.field = getelementptr inbounds nuw { {} }, ptr %return, i32 0, i32 0, !dbg !65
// CHECK:STDOUT:   call void @"_COp.B.Main:Copy.Core"(ptr %.b3e.field, ptr %v), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "init_class.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!16, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !20 = !{!19}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = !{null, !16}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.B.Main:Copy.Core", scope: null, file: !14, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !30)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !16)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 7, scope: !24)
// CHECK:STDOUT: !30 = !{!25}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !14, line: 21, type: !21, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 3, scope: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 22, column: 13, scope: !31)
// CHECK:STDOUT: !34 = !DILocation(line: 21, column: 1, scope: !31)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !14, line: 22, type: !23, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !38)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !16)
// CHECK:STDOUT: !37 = !DILocation(line: 22, column: 3, scope: !35)
// CHECK:STDOUT: !38 = !{!36}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8c4b83ecd900f1ea:core.Destroy.Core", scope: null, file: !14, line: 22, type: !23, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !42)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !16)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 3, scope: !39)
// CHECK:STDOUT: !42 = !{!40}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.8fbe0c6a865381a4:core.Destroy.Core", scope: null, file: !14, line: 22, type: !23, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !16)
// CHECK:STDOUT: !45 = !DILocation(line: 22, column: 3, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8fbe0c6a865381a4:core.Destroy.Core", scope: null, file: !14, line: 22, type: !23, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !16)
// CHECK:STDOUT: !49 = !DILocation(line: 22, column: 3, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d32110776493a13a:core.Destroy.Core", scope: null, file: !14, line: 22, type: !23, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !16)
// CHECK:STDOUT: !53 = !DILocation(line: 22, column: 3, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d32110776493a13a:core.Destroy.Core", scope: null, file: !14, line: 22, type: !23, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !16)
// CHECK:STDOUT: !57 = !DILocation(line: 22, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.A.Main.a6a60dfee2d7d273", scope: null, file: !14, line: 12, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !67)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !16)
// CHECK:STDOUT: !64 = !DILocation(line: 14, column: 16, scope: !59)
// CHECK:STDOUT: !65 = !DILocation(line: 13, column: 12, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 13, column: 5, scope: !59)
// CHECK:STDOUT: !67 = !{!60}
// CHECK:STDOUT:
