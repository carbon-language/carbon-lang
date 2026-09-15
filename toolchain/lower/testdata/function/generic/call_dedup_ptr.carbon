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
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !37
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !44
// CHECK:STDOUT:   %ptr_i8.var = alloca ptr, align 8, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !37
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i32.var, align 8, !dbg !37
// CHECK:STDOUT:   %.loc19_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !39
// CHECK:STDOUT:   %F.call.loc19 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc19_5), !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !44
// CHECK:STDOUT:   store ptr poison, ptr %ptr_f64.var, align 8, !dbg !44
// CHECK:STDOUT:   %.loc21_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !46
// CHECK:STDOUT:   %F.call.loc21 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc21_5), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i8.var), !dbg !48
// CHECK:STDOUT:   store ptr poison, ptr %ptr_i8.var, align 8, !dbg !48
// CHECK:STDOUT:   %.loc23_5 = load ptr, ptr %ptr_i8.var, align 8, !dbg !50
// CHECK:STDOUT:   %F.call.loc23 = call ptr @_CF.Main.a76b78461416b4a8(ptr %.loc23_5), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a76b78461416b4a8(ptr %x) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %x, !dbg !59
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
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !32, line: 17, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 18, column: 3, scope: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 19, column: 5, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 3, scope: !36)
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{!41, !41}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 20, column: 3, scope: !36)
// CHECK:STDOUT: !46 = !DILocation(line: 21, column: 5, scope: !36)
// CHECK:STDOUT: !47 = !DILocation(line: 21, column: 3, scope: !36)
// CHECK:STDOUT: !48 = !DILocation(line: 22, column: 3, scope: !36)
// CHECK:STDOUT: !50 = !DILocation(line: 23, column: 5, scope: !36)
// CHECK:STDOUT: !51 = !DILocation(line: 23, column: 3, scope: !36)
// CHECK:STDOUT: !52 = !DILocation(line: 17, column: 1, scope: !36)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a76b78461416b4a8", scope: null, file: !32, line: 13, type: !43, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !60)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !41)
// CHECK:STDOUT: !59 = !DILocation(line: 14, column: 3, scope: !53)
// CHECK:STDOUT: !60 = !{!54}
// CHECK:STDOUT:
