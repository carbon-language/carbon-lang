// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_dedup_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_dedup_ptr.carbon

fn F[T: Core.Copy](x: T) -> T {
  return x;
}

fn M() {
  var ptr_i32 : i32*;
  F(ptr_i32);
  var ptr_f64 : f64*;
  F(ptr_f64);
  var ptr_i8 : i8*;
  F(ptr_i8);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_dedup_ptr.carbon'
// CHECK:STDOUT: source_filename = "call_dedup_ptr.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !40
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !46
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !50
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !40
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !40
// CHECK:STDOUT:   %.loc19_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !42
// CHECK:STDOUT:   %F.call.loc19 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc19_5), !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !46
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !46
// CHECK:STDOUT:   %.loc21_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !48
// CHECK:STDOUT:   %F.call.loc21 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc21_5), !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !50
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !50
// CHECK:STDOUT:   %.loc23_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !52
// CHECK:STDOUT:   %F.call.loc23 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc23_5), !dbg !53
// CHECK:STDOUT:   call void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !50
// CHECK:STDOUT:   call void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !46
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core"(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.749b51755634ad17:core.Destroy.Core"(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.a76b78461416b4a8, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "call_dedup_ptr.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !{null}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !32, line: 17, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 18, column: 3, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 19, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 19, column: 3, scope: !39)
// CHECK:STDOUT: !44 = !{!36, !36}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 20, column: 3, scope: !39)
// CHECK:STDOUT: !48 = !DILocation(line: 21, column: 5, scope: !39)
// CHECK:STDOUT: !49 = !DILocation(line: 21, column: 3, scope: !39)
// CHECK:STDOUT: !50 = !DILocation(line: 22, column: 3, scope: !39)
// CHECK:STDOUT: !52 = !DILocation(line: 23, column: 5, scope: !39)
// CHECK:STDOUT: !53 = !DILocation(line: 23, column: 3, scope: !39)
// CHECK:STDOUT: !54 = !DILocation(line: 17, column: 1, scope: !39)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.de57ff04fd84f9c8:core.Destroy.Core", scope: null, file: !32, line: 22, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !36)
// CHECK:STDOUT: !57 = !DILocation(line: 22, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.749b51755634ad17:core.Destroy.Core", scope: null, file: !32, line: 20, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !62)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !36)
// CHECK:STDOUT: !61 = !DILocation(line: 20, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !{!60}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !32, line: 18, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !36)
// CHECK:STDOUT: !65 = !DILocation(line: 18, column: 3, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !32, line: 13, type: !45, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !74)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !36)
// CHECK:STDOUT: !73 = !DILocation(line: 14, column: 3, scope: !67)
// CHECK:STDOUT: !74 = !{!68}
// CHECK:STDOUT:
