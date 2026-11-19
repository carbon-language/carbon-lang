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
// CHECK:STDOUT: define void @"_COp.B.Main:Copy.Core"(ptr sret({}) %return, ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @B.val.loc6_16, i64 0, i1 false), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_15.1.temp = alloca { {} }, align 1, !dbg !30
// CHECK:STDOUT:   %.loc22_14.2.temp = alloca {}, align 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_15.1.temp), !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_14.2.temp), !dbg !31
// CHECK:STDOUT:   call void @_CMake.A.Main.a6a60dfee2d7d273(ptr %.loc22_15.1.temp, ptr @B.val.loc6_16), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a2f48eddae22d125(ptr %.loc22_15.1.temp), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr @B.val), !dbg !31
// CHECK:STDOUT:   ret i32 0, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.A.Main.a6a60dfee2d7d273(ptr sret({ {} }) %return, ptr %v) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.549.field = getelementptr inbounds nuw { {} }, ptr %return, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   call void @"_COp.B.Main:Copy.Core"(ptr %.549.field, ptr %v), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a2f48eddae22d125(ptr %t) #0 !dbg !44 {
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6579a5013ba3dcd2(ptr %t) #0 !dbg !51 {
// CHECK:STDOUT:   ret void, !dbg !56
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
// CHECK:STDOUT: !6 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "init_class.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{!16, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !20 = !{!19}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.B.Main:Copy.Core", scope: null, file: !14, line: 5, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !28)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !22, type: !16)
// CHECK:STDOUT: !27 = !DILocation(line: 6, column: 7, scope: !22)
// CHECK:STDOUT: !28 = !{!23}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !14, line: 21, type: !21, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !30 = !DILocation(line: 22, column: 3, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 13, scope: !29)
// CHECK:STDOUT: !32 = !{null, !16}
// CHECK:STDOUT: !33 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 21, column: 1, scope: !29)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.A.Main.a6a60dfee2d7d273", scope: null, file: !14, line: 12, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !43)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !16)
// CHECK:STDOUT: !40 = !DILocation(line: 14, column: 16, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 13, column: 12, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 13, column: 5, scope: !35)
// CHECK:STDOUT: !43 = !{!36}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a2f48eddae22d125", scope: null, file: !6, line: 32, type: !33, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !50)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !16)
// CHECK:STDOUT: !49 = !DILocation(line: 32, column: 1, scope: !44)
// CHECK:STDOUT: !50 = !{!45}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6579a5013ba3dcd2", scope: null, file: !6, line: 32, type: !33, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !16)
// CHECK:STDOUT: !56 = !DILocation(line: 32, column: 1, scope: !51)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT:
