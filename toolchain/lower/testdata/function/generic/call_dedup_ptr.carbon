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
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !42
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !48
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !52
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !42
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !42
// CHECK:STDOUT:   %.loc19_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !44
// CHECK:STDOUT:   %F.call.loc19 = call ptr @_CF.Main.01c402613389278e(ptr %.loc19_5), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !48
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !48
// CHECK:STDOUT:   %.loc21_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !50
// CHECK:STDOUT:   %F.call.loc21 = call ptr @_CF.Main.01c402613389278e(ptr %.loc21_5), !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !52
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !52
// CHECK:STDOUT:   %.loc23_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !54
// CHECK:STDOUT:   %F.call.loc23 = call ptr @_CF.Main.01c402613389278e(ptr %.loc23_5), !dbg !55
// CHECK:STDOUT:   call void @"_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core"(ptr %ptr_i8.var), !dbg !52
// CHECK:STDOUT:   call void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %ptr_f64.var), !dbg !48
// CHECK:STDOUT:   call void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.01c402613389278e(ptr %x) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.01c402613389278e, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "call_dedup_ptr.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !{null}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !39 = !{null, !38}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !34, line: 17, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 18, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 5, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 3, scope: !41)
// CHECK:STDOUT: !46 = !{!38, !38}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = !DILocation(line: 20, column: 3, scope: !41)
// CHECK:STDOUT: !50 = !DILocation(line: 21, column: 5, scope: !41)
// CHECK:STDOUT: !51 = !DILocation(line: 21, column: 3, scope: !41)
// CHECK:STDOUT: !52 = !DILocation(line: 22, column: 3, scope: !41)
// CHECK:STDOUT: !54 = !DILocation(line: 23, column: 5, scope: !41)
// CHECK:STDOUT: !55 = !DILocation(line: 23, column: 3, scope: !41)
// CHECK:STDOUT: !56 = !DILocation(line: 17, column: 1, scope: !41)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.64a2114a2cb70470:core.Destroy.Core", scope: null, file: !34, line: 22, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !38)
// CHECK:STDOUT: !59 = !DILocation(line: 22, column: 3, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3a7a2e5beb0d0fa7:core.Destroy.Core", scope: null, file: !34, line: 20, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !38)
// CHECK:STDOUT: !63 = !DILocation(line: 20, column: 3, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1feff81df232a20a:core.Destroy.Core", scope: null, file: !34, line: 18, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !38)
// CHECK:STDOUT: !67 = !DILocation(line: 18, column: 3, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.01c402613389278e", scope: null, file: !34, line: 13, type: !47, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !76)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !38)
// CHECK:STDOUT: !75 = !DILocation(line: 14, column: 3, scope: !69)
// CHECK:STDOUT: !76 = !{!70}
// CHECK:STDOUT:
