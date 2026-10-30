// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/let/global.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/let/global.carbon

// --- empty_class.carbon
library "[[@TEST_NAME]]";

class C {}

let c: C = {};

fn Sink(_: C);

fn F() {
  Sink(c);
}

// --- copy_and_runtime_init.carbon
library "[[@TEST_NAME]]";

fn MakeInt() -> i32;

let a: i32 = 0;
let b: i32 = MakeInt();

fn ReadA() -> i32 { return a; }
fn ReadPackageA() -> i32 { return package.a; }
fn ReadB() -> i32 { return b; }

// --- pointer_and_empty_repr.carbon
library "[[@TEST_NAME]]";

class Pair {
  var x: i32;
  var y: i32;
}

fn MakePair() -> Pair;

let p: Pair = MakePair();
let u: () = ();

fn ReadPairX() -> i32 { return p.x; }
fn ReadUnit() -> () { return u; }

// --- ref_and_class_scope.carbon
library "[[@TEST_NAME]]";

var backing: i32 = 10;
fn GetPtr() -> i32*;

let ref r_const: i32 = backing;
let ref r_runtime: i32 = *GetPtr();

class Holder {
  let k: i32 = 42;
  fn Inside() -> i32 { return k; }
}

fn ReadRefs() -> i32 {
  let _: i32 = r_const;
  return r_runtime;
}

fn ReadHolderOutside() -> i32 {
  return Holder.k;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'empty_class.carbon'
// CHECK:STDOUT: source_filename = "empty_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @_Cc.Main = external global {}
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CSink.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSink.Main(ptr @_Cc.Main), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_13.2.temp = alloca {}, align 1, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_13.2.temp), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "empty_class.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 9, type: !24, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 10, column: 3, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 9, column: 1, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !18, type: !24, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !30 = !DILocation(line: 5, column: 12, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 0, scope: !29)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_and_runtime_init.carbon'
// CHECK:STDOUT: source_filename = "copy_and_runtime_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Ca.Main = external global i32
// CHECK:STDOUT: @_Cb.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeInt.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadA.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.global_ref = load i32, ptr @_Ca.Main, align 4, !dbg !40
// CHECK:STDOUT:   ret i32 %a.global_ref, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadPackageA.Main() #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.global_ref = load i32, ptr @_Ca.Main, align 4, !dbg !45
// CHECK:STDOUT:   ret i32 %a.global_ref, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadB.Main() #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.global_ref = load i32, ptr @_Cb.Main, align 4, !dbg !50
// CHECK:STDOUT:   ret i32 %b.global_ref, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeInt.call = call i32 @_CMakeInt.Main(), !dbg !53
// CHECK:STDOUT:   store i32 %MakeInt.call, ptr @_Cb.Main, align 4, !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !32 = !DIFile(filename: "copy_and_runtime_init.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "ReadA", linkageName: "_CReadA.Main", scope: null, file: !32, line: 8, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !40 = !DILocation(line: 8, column: 28, scope: !37)
// CHECK:STDOUT: !41 = !DILocation(line: 8, column: 21, scope: !37)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "ReadPackageA", linkageName: "_CReadPackageA.Main", scope: null, file: !32, line: 9, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !45 = !DILocation(line: 9, column: 35, scope: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 9, column: 28, scope: !42)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "ReadB", linkageName: "_CReadB.Main", scope: null, file: !32, line: 10, type: !36, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !50 = !DILocation(line: 10, column: 28, scope: !47)
// CHECK:STDOUT: !51 = !DILocation(line: 10, column: 21, scope: !47)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !32, type: !24, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !53 = !DILocation(line: 6, column: 14, scope: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 6, column: 5, scope: !52)
// CHECK:STDOUT: !55 = !DILocation(line: 0, scope: !52)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pointer_and_empty_repr.carbon'
// CHECK:STDOUT: source_filename = "pointer_and_empty_repr.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cp.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakePair.Main(ptr sret({ i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadPairX.Main() #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_33.2 = load i32, ptr @_Cp.Main, align 4, !dbg !63
// CHECK:STDOUT:   ret i32 %.loc13_33.2, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CReadUnit.Main() #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_24.1.temp = alloca { i32, i32 }, align 4, !dbg !71
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_24.1.temp), !dbg !71
// CHECK:STDOUT:   call void @_CMakePair.Main(ptr %.loc10_24.1.temp), !dbg !71
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cp.Main, ptr align 4 %.loc10_24.1.temp, i64 8, i1 false), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!57}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !56 = !DIFile(filename: "pointer_and_empty_repr.carbon", directory: "")
// CHECK:STDOUT: !57 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !56, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "ReadPairX", linkageName: "_CReadPairX.Main", scope: null, file: !56, line: 13, type: !36, spFlags: DISPFlagDefinition, unit: !57)
// CHECK:STDOUT: !63 = !DILocation(line: 13, column: 32, scope: !60)
// CHECK:STDOUT: !64 = !DILocation(line: 13, column: 25, scope: !60)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "ReadUnit", linkageName: "_CReadUnit.Main", scope: null, file: !56, line: 14, type: !24, spFlags: DISPFlagDefinition, unit: !57)
// CHECK:STDOUT: !69 = !DILocation(line: 14, column: 23, scope: !65)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !56, type: !24, spFlags: DISPFlagDefinition, unit: !57)
// CHECK:STDOUT: !71 = !DILocation(line: 10, column: 15, scope: !70)
// CHECK:STDOUT: !72 = !DILocation(line: 10, column: 5, scope: !70)
// CHECK:STDOUT: !73 = !DILocation(line: 0, scope: !70)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_and_class_scope.carbon'
// CHECK:STDOUT: source_filename = "ref_and_class_scope.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cbacking.Main = global i32 0
// CHECK:STDOUT: @_Ck.Holder.Main = external global i32
// CHECK:STDOUT: @_Cr_runtime.Main = global ptr null
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CGetPtr.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CInside.Holder.Main() #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %k.global_ref = load i32, ptr @_Ck.Holder.Main, align 4, !dbg !79
// CHECK:STDOUT:   ret i32 %k.global_ref, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadRefs.Main() #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15 = load i32, ptr @_Cbacking.Main, align 4, !dbg !84
// CHECK:STDOUT:   %r_runtime.global_ref = load ptr, ptr @_Cr_runtime.Main, align 8, !dbg !87
// CHECK:STDOUT:   %.loc16 = load i32, ptr %r_runtime.global_ref, align 4, !dbg !87
// CHECK:STDOUT:   ret i32 %.loc16, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadHolderOutside.Main() #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %k.global_ref = load i32, ptr @_Ck.Holder.Main, align 4, !dbg !92
// CHECK:STDOUT:   ret i32 %k.global_ref, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 10, ptr @_Cbacking.Main, align 4, !dbg !95
// CHECK:STDOUT:   %GetPtr.call = call ptr @_CGetPtr.Main(), !dbg !96
// CHECK:STDOUT:   store ptr %GetPtr.call, ptr @_Cr_runtime.Main, align 8, !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!75}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !74 = !DIFile(filename: "ref_and_class_scope.carbon", directory: "")
// CHECK:STDOUT: !75 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !74, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Inside", linkageName: "_CInside.Holder.Main", scope: null, file: !74, line: 11, type: !36, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 11, column: 31, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 11, column: 24, scope: !76)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "ReadRefs", linkageName: "_CReadRefs.Main", scope: null, file: !74, line: 14, type: !36, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !84 = !DILocation(line: 15, column: 16, scope: !81)
// CHECK:STDOUT: !87 = !DILocation(line: 16, column: 10, scope: !81)
// CHECK:STDOUT: !88 = !DILocation(line: 16, column: 3, scope: !81)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "ReadHolderOutside", linkageName: "_CReadHolderOutside.Main", scope: null, file: !74, line: 19, type: !36, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !92 = !DILocation(line: 20, column: 10, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 20, column: 3, scope: !89)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !74, type: !24, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !95 = !DILocation(line: 3, column: 1, scope: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 7, column: 27, scope: !94)
// CHECK:STDOUT: !98 = !DILocation(line: 7, column: 5, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 0, scope: !94)
// CHECK:STDOUT:
