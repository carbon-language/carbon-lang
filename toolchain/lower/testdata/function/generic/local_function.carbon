// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/local_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/local_function.carbon

// Test that calling a local function defined inside a generic function works.
// This previously crashed due to the local function not having a parent name
// scope.

class S {
  fn Make() -> S { return {.n = 0}; }
  var n: i32;
}

fn EnclosingTemplate(template T: type) -> i32 {
  let a: T = T.Make();
  fn Local(y: typeof(a.n)) -> typeof(a.n) {
    return y;
  }
  return Local(a.n);
}

fn F(generic T: Core.Copy, y: T) -> T {
  fn G(x: T) -> T { return x; }
  return G(y);
}

fn Run() -> i32 {
  return F(i32, EnclosingTemplate(S));
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'local_function.carbon'
// CHECK:STDOUT: source_filename = "local_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @S.val.loc18_35 = internal constant { i32 } zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.S.Main(ptr sret({ i32 }) %return) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_34.3.n = getelementptr inbounds nuw { i32 }, ptr %return, i32 0, i32 0, !dbg !45
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @S.val.loc18_35, i64 4, i1 false), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EnclosingTemplate.call = call i32 @_CEnclosingTemplate.Main.d82ccacf0414a2bc(), !dbg !50
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4d6ceefeb7f772f4(i32 %EnclosingTemplate.call), !dbg !51
// CHECK:STDOUT:   ret i32 %F.call, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CEnclosingTemplate.Main.d82ccacf0414a2bc() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.405.temp = alloca { i32 }, align 4, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.405.temp), !dbg !58
// CHECK:STDOUT:   call void @_CMake.S.Main(ptr %.405.temp), !dbg !58
// CHECK:STDOUT:   %.d45.n = getelementptr inbounds nuw { i32 }, ptr %.405.temp, i32 0, i32 0, !dbg !61
// CHECK:STDOUT:   %.5ea = load i32, ptr %.d45.n, align 4, !dbg !61
// CHECK:STDOUT:   %Local.call = call i32 @"_CLocal:enclosed.d82ccacf0414a2bc"(i32 %.5ea), !dbg !62
// CHECK:STDOUT:   ret i32 %Local.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4d6ceefeb7f772f4(i32 %y) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @"_CG:enclosed.4d6ceefeb7f772f4"(i32 %y), !dbg !70
// CHECK:STDOUT:   ret i32 %G.call, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CLocal:enclosed.d82ccacf0414a2bc"(i32 %y) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %y, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CG:enclosed.4d6ceefeb7f772f4"(i32 %x) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "local_function.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{!36}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.S.Main", scope: null, file: !34, line: 18, type: !39, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !45 = !DILocation(line: 18, column: 27, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 18, column: 20, scope: !42)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !34, line: 35, type: !41, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !50 = !DILocation(line: 36, column: 17, scope: !47)
// CHECK:STDOUT: !51 = !DILocation(line: 36, column: 10, scope: !47)
// CHECK:STDOUT: !52 = !{!36, !36}
// CHECK:STDOUT: !53 = !DISubroutineType(types: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 36, column: 3, scope: !47)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "EnclosingTemplate", linkageName: "_CEnclosingTemplate.Main.d82ccacf0414a2bc", scope: null, file: !34, line: 22, type: !41, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !58 = !DILocation(line: 23, column: 14, scope: !55)
// CHECK:STDOUT: !61 = !DILocation(line: 27, column: 16, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 27, column: 10, scope: !55)
// CHECK:STDOUT: !63 = !DILocation(line: 27, column: 3, scope: !55)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4d6ceefeb7f772f4", scope: null, file: !34, line: 30, type: !53, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !72)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !36)
// CHECK:STDOUT: !70 = !DILocation(line: 32, column: 10, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 32, column: 3, scope: !64)
// CHECK:STDOUT: !72 = !{!65}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Local", linkageName: "_CLocal:enclosed.d82ccacf0414a2bc", scope: null, file: !34, line: 24, type: !53, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !80)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !36)
// CHECK:STDOUT: !79 = !DILocation(line: 25, column: 5, scope: !73)
// CHECK:STDOUT: !80 = !{!74}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "G", linkageName: "_CG:enclosed.4d6ceefeb7f772f4", scope: null, file: !34, line: 31, type: !53, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !88)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !36)
// CHECK:STDOUT: !87 = !DILocation(line: 31, column: 21, scope: !81)
// CHECK:STDOUT: !88 = !{!82}
// CHECK:STDOUT:
