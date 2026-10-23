// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/definition/deduced_return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/definition/deduced_return.carbon

class C {
  var a: i32;
  var b: i32;
}

fn Copy() -> auto {
  return 1 as i32;
}

fn InPlace() -> auto {
  return {.a = 1, .b = 2} as C;
}

fn ReturnedVar() -> auto {
  returned var c: C = {.a = 3, .b = 4};
  return var;
}

fn Use() -> i32 {
  let unused c: C = InPlace();
  let unused d: C = ReturnedVar();
  return Copy();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'deduced_return.carbon'
// CHECK:STDOUT: source_filename = "deduced_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.32e.loc23_31 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @C.val.7af.loc27_12 = internal constant { i32, i32 } { i32 3, i32 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCopy.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInPlace.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_25.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !48
// CHECK:STDOUT:   %.loc23_25.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !48
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @C.val.32e.loc23_31, i64 8, i1 false), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CReturnedVar.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_38.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !52
// CHECK:STDOUT:   %.loc27_38.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !52
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @C.val.7af.loc27_12, i64 8, i1 false), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_29.1.temp = alloca { i32, i32 }, align 4, !dbg !60
// CHECK:STDOUT:   %.loc33_33.1.temp = alloca { i32, i32 }, align 4, !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_29.1.temp), !dbg !60
// CHECK:STDOUT:   call void @_CInPlace.Main(ptr %.loc32_29.1.temp), !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_33.1.temp), !dbg !63
// CHECK:STDOUT:   call void @_CReturnedVar.Main(ptr %.loc33_33.1.temp), !dbg !63
// CHECK:STDOUT:   %Copy.call = call i32 @_CCopy.Main(), !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %.loc33_33.1.temp), !dbg !63
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %.loc32_29.1.temp), !dbg !60
// CHECK:STDOUT:   ret i32 %Copy.call, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2508ed710019eba8:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33_33.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !70
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc33_33.1.b), !dbg !70
// CHECK:STDOUT:   %.loc33_33.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !70
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc33_33.4.a), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.08d9e38b928ba575:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33_33.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !76
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc33_33.1.b), !dbg !76
// CHECK:STDOUT:   %.loc33_33.4.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !76
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc33_33.4.a), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.356af3b6aef18564(ptr %t) #0 !dbg !78 {
// CHECK:STDOUT:   call void @"_COp.08d9e38b928ba575:core.SubobjectDestroy.Core"(ptr %t), !dbg !82
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !85 {
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.356af3b6aef18564, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "deduced_return.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{null, !39}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !34, line: 18, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 3, scope: !44)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "InPlace", linkageName: "_CInPlace.Main", scope: null, file: !34, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !48 = !DILocation(line: 23, column: 10, scope: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 23, column: 3, scope: !47)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "ReturnedVar", linkageName: "_CReturnedVar.Main", scope: null, file: !34, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !52 = !DILocation(line: 27, column: 23, scope: !50)
// CHECK:STDOUT: !53 = !DILocation(line: 27, column: 12, scope: !50)
// CHECK:STDOUT: !56 = !DILocation(line: 28, column: 3, scope: !50)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !34, line: 31, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !60 = !DILocation(line: 32, column: 21, scope: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 33, column: 21, scope: !57)
// CHECK:STDOUT: !66 = !DILocation(line: 34, column: 10, scope: !57)
// CHECK:STDOUT: !67 = !DILocation(line: 34, column: 3, scope: !57)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2508ed710019eba8:core.SubobjectDestroy.Core", scope: null, file: !34, line: 33, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !73)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !39)
// CHECK:STDOUT: !70 = !DILocation(line: 33, column: 21, scope: !68)
// CHECK:STDOUT: !71 = !{null, !36}
// CHECK:STDOUT: !72 = !DISubroutineType(types: !71)
// CHECK:STDOUT: !73 = !{!69}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Op", linkageName: "_COp.08d9e38b928ba575:core.SubobjectDestroy.Core", scope: null, file: !34, line: 33, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !77)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !39)
// CHECK:STDOUT: !76 = !DILocation(line: 33, column: 21, scope: !74)
// CHECK:STDOUT: !77 = !{!75}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !26, line: 32, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !84)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !39)
// CHECK:STDOUT: !82 = !DILocation(line: 34, column: 5, scope: !78)
// CHECK:STDOUT: !83 = !DILocation(line: 32, column: 1, scope: !78)
// CHECK:STDOUT: !84 = !{!79}
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !26, line: 32, type: !72, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !91)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !36)
// CHECK:STDOUT: !90 = !DILocation(line: 32, column: 1, scope: !85)
// CHECK:STDOUT: !91 = !{!86}
// CHECK:STDOUT:
