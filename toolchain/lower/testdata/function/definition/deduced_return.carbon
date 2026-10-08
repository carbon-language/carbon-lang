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
// CHECK:STDOUT: @C.val.a10.loc23_31 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @C.val.8f0.loc27_12 = internal constant { i32, i32 } { i32 3, i32 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCopy.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInPlace.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_25.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !50
// CHECK:STDOUT:   %.loc23_25.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !50
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @C.val.a10.loc23_31, i64 8, i1 false), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CReturnedVar.Main(ptr sret({ i32, i32 }) %return) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_38.3.a = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 0, !dbg !54
// CHECK:STDOUT:   %.loc27_38.6.b = getelementptr inbounds nuw { i32, i32 }, ptr %return, i32 0, i32 1, !dbg !54
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @C.val.8f0.loc27_12, i64 8, i1 false), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CUse.Main() #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_29.1.temp = alloca { i32, i32 }, align 4, !dbg !62
// CHECK:STDOUT:   %.loc33_33.1.temp = alloca { i32, i32 }, align 4, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_29.1.temp), !dbg !62
// CHECK:STDOUT:   call void @_CInPlace.Main(ptr %.loc32_29.1.temp), !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc33_33.1.temp), !dbg !65
// CHECK:STDOUT:   call void @_CReturnedVar.Main(ptr %.loc33_33.1.temp), !dbg !65
// CHECK:STDOUT:   %Copy.call = call i32 @_CCopy.Main(), !dbg !68
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %.loc33_33.1.temp), !dbg !65
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %.loc32_29.1.temp), !dbg !62
// CHECK:STDOUT:   ret i32 %Copy.call, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core"(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33_33.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !76
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc33_33.1.b), !dbg !76
// CHECK:STDOUT:   %.loc33_33.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !76
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc33_33.2.a), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b4c762864cd9a02b:core.Destroy.Core"(ptr %self) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core"(ptr %self), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc33_33.1.b = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !84
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc33_33.1.b), !dbg !84
// CHECK:STDOUT:   %.loc33_33.2.a = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !84
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc33_33.2.a), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "deduced_return.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !{!36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{null, !36}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{null, !39}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !34, line: 18, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !48 = !DILocation(line: 19, column: 3, scope: !46)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "InPlace", linkageName: "_CInPlace.Main", scope: null, file: !34, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !50 = !DILocation(line: 23, column: 10, scope: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 23, column: 3, scope: !49)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "ReturnedVar", linkageName: "_CReturnedVar.Main", scope: null, file: !34, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !54 = !DILocation(line: 27, column: 23, scope: !52)
// CHECK:STDOUT: !55 = !DILocation(line: 27, column: 12, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 28, column: 3, scope: !52)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Use", linkageName: "_CUse.Main", scope: null, file: !34, line: 31, type: !38, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !62 = !DILocation(line: 32, column: 21, scope: !59)
// CHECK:STDOUT: !65 = !DILocation(line: 33, column: 21, scope: !59)
// CHECK:STDOUT: !68 = !DILocation(line: 34, column: 10, scope: !59)
// CHECK:STDOUT: !69 = !DILocation(line: 34, column: 3, scope: !59)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !34, line: 33, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !36)
// CHECK:STDOUT: !72 = !DILocation(line: 33, column: 21, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b4c762864cd9a02b:core.Destroy.Core", scope: null, file: !34, line: 33, type: !45, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !77)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !39)
// CHECK:STDOUT: !76 = !DILocation(line: 33, column: 21, scope: !74)
// CHECK:STDOUT: !77 = !{!75}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b4c762864cd9a02b:core.Destroy.Core", scope: null, file: !34, line: 33, type: !45, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !39)
// CHECK:STDOUT: !80 = !DILocation(line: 33, column: 21, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !34, line: 33, type: !45, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !39)
// CHECK:STDOUT: !84 = !DILocation(line: 33, column: 21, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !34, line: 33, type: !45, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !89)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !39)
// CHECK:STDOUT: !88 = !DILocation(line: 33, column: 21, scope: !86)
// CHECK:STDOUT: !89 = !{!87}
// CHECK:STDOUT:
