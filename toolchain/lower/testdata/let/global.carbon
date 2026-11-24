// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
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

// --- for_in_local_fn.carbon
library "[[@TEST_NAME]]";

let n: i32 = fn () -> i32 {
  var a: array(i32, 3) = (1, 2, 3);
  var j: i32 = 0;
  // `i` is not a global `let`.
  for (i: i32 in a) {
    j += i;
  }
  return j;
}();

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'empty_class.carbon'
// CHECK:STDOUT: source_filename = "empty_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @_Cc.Main = constant {} zeroinitializer
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CSink.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSink.Main(ptr @_Cc.Main), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_13.2.temp = alloca {}, align 1, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_13.2.temp), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "empty_class.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !4, line: 9, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 10, column: 3, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 9, column: 1, scope: !11)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !4, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DILocation(line: 5, column: 12, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 0, scope: !15)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_and_runtime_init.carbon'
// CHECK:STDOUT: source_filename = "copy_and_runtime_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Ca.Main = constant i32 0
// CHECK:STDOUT: @_Cb.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CMakeInt.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadA.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.global_ref = load i32, ptr @_Ca.Main, align 4, !dbg !26
// CHECK:STDOUT:   ret i32 %a.global_ref, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadPackageA.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.global_ref = load i32, ptr @_Ca.Main, align 4, !dbg !31
// CHECK:STDOUT:   ret i32 %a.global_ref, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadB.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.global_ref = load i32, ptr @_Cb.Main, align 4, !dbg !36
// CHECK:STDOUT:   ret i32 %b.global_ref, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %MakeInt.call = call i32 @_CMakeInt.Main(), !dbg !39
// CHECK:STDOUT:   store i32 %MakeInt.call, ptr @_Cb.Main, align 4, !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !18 = !DIFile(filename: "copy_and_runtime_init.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "ReadA", linkageName: "_CReadA.Main", scope: null, file: !18, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 8, column: 28, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 8, column: 21, scope: !23)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "ReadPackageA", linkageName: "_CReadPackageA.Main", scope: null, file: !18, line: 9, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 35, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 28, scope: !28)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "ReadB", linkageName: "_CReadB.Main", scope: null, file: !18, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !36 = !DILocation(line: 10, column: 28, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 10, column: 21, scope: !33)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !18, type: !10, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !39 = !DILocation(line: 6, column: 14, scope: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 5, scope: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 0, scope: !38)
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
// CHECK:STDOUT: define i32 @_CReadPairX.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_33.2 = load i32, ptr @_Cp.Main, align 4, !dbg !49
// CHECK:STDOUT:   ret i32 %.loc13_33.2, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CReadUnit.Main() #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_24.1.temp = alloca { i32, i32 }, align 4, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_24.1.temp), !dbg !57
// CHECK:STDOUT:   call void @_CMakePair.Main(ptr %.loc10_24.1.temp), !dbg !57
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cp.Main, ptr align 4 %.loc10_24.1.temp, i64 8, i1 false), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !42 = !DIFile(filename: "pointer_and_empty_repr.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "ReadPairX", linkageName: "_CReadPairX.Main", scope: null, file: !42, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !49 = !DILocation(line: 13, column: 32, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 13, column: 25, scope: !46)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "ReadUnit", linkageName: "_CReadUnit.Main", scope: null, file: !42, line: 14, type: !10, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !55 = !DILocation(line: 14, column: 23, scope: !51)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !42, type: !10, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !57 = !DILocation(line: 10, column: 15, scope: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 10, column: 5, scope: !56)
// CHECK:STDOUT: !59 = !DILocation(line: 0, scope: !56)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ref_and_class_scope.carbon'
// CHECK:STDOUT: source_filename = "ref_and_class_scope.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cbacking.Main = global i32 0
// CHECK:STDOUT: @_Ck.Holder.Main = constant i32 42
// CHECK:STDOUT: @_Cr_runtime.Main = global ptr null
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CGetPtr.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CInside.Holder.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %k.global_ref = load i32, ptr @_Ck.Holder.Main, align 4, !dbg !65
// CHECK:STDOUT:   ret i32 %k.global_ref, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadRefs.Main() #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc15 = load i32, ptr @_Cbacking.Main, align 4, !dbg !70
// CHECK:STDOUT:   %r_runtime.global_ref = load ptr, ptr @_Cr_runtime.Main, align 8, !dbg !73
// CHECK:STDOUT:   %.loc16 = load i32, ptr %r_runtime.global_ref, align 4, !dbg !73
// CHECK:STDOUT:   ret i32 %.loc16, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CReadHolderOutside.Main() #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %k.global_ref = load i32, ptr @_Ck.Holder.Main, align 4, !dbg !78
// CHECK:STDOUT:   ret i32 %k.global_ref, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 10, ptr @_Cbacking.Main, align 4, !dbg !81
// CHECK:STDOUT:   %GetPtr.call = call ptr @_CGetPtr.Main(), !dbg !82
// CHECK:STDOUT:   store ptr %GetPtr.call, ptr @_Cr_runtime.Main, align 8, !dbg !84
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!61}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !60 = !DIFile(filename: "ref_and_class_scope.carbon", directory: "")
// CHECK:STDOUT: !61 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !60, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Inside", linkageName: "_CInside.Holder.Main", scope: null, file: !60, line: 11, type: !22, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 11, column: 31, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 11, column: 24, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "ReadRefs", linkageName: "_CReadRefs.Main", scope: null, file: !60, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !70 = !DILocation(line: 15, column: 16, scope: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 16, column: 10, scope: !67)
// CHECK:STDOUT: !74 = !DILocation(line: 16, column: 3, scope: !67)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "ReadHolderOutside", linkageName: "_CReadHolderOutside.Main", scope: null, file: !60, line: 19, type: !22, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !78 = !DILocation(line: 20, column: 10, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 20, column: 3, scope: !75)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !60, type: !10, spFlags: DISPFlagDefinition, unit: !61)
// CHECK:STDOUT: !81 = !DILocation(line: 3, column: 1, scope: !80)
// CHECK:STDOUT: !82 = !DILocation(line: 7, column: 27, scope: !80)
// CHECK:STDOUT: !84 = !DILocation(line: 7, column: 5, scope: !80)
// CHECK:STDOUT: !85 = !DILocation(line: 0, scope: !80)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'for_in_local_fn.carbon'
// CHECK:STDOUT: source_filename = "for_in_local_fn.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.loc4_3.1 = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT: @_Cn.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_Cfn:enclosed.b0cc8ae043738921"() #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [3 x i32], align 4, !dbg !91
// CHECK:STDOUT:   %j.var = alloca i32, align 4, !dbg !94
// CHECK:STDOUT:   %var = alloca i32, align 4, !dbg !98
// CHECK:STDOUT:   %.loc7_19.5.temp = alloca <{ i32, i1 }>, align 4, !dbg !98
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !91
// CHECK:STDOUT:   %.loc4_34.3.array.index = getelementptr inbounds i8, ptr %a.var, i64 0, !dbg !92
// CHECK:STDOUT:   %.loc4_34.6.array.index = getelementptr inbounds i8, ptr %a.var, i64 4, !dbg !92
// CHECK:STDOUT:   %.loc4_34.9.array.index = getelementptr inbounds i8, ptr %a.var, i64 8, !dbg !92
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @array.loc4_3.1, i64 12, i1 false), !dbg !91
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %j.var), !dbg !94
// CHECK:STDOUT:   store i32 0, ptr %j.var, align 4, !dbg !94
// CHECK:STDOUT:   %Array.as.Iterate.impl.NewCursor.call = call i32 @"_CNewCursor.Array.ef119564f44a13e5.Core:Iterate.Core.979acf9988e393af"(ptr %a.var), !dbg !98
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !98
// CHECK:STDOUT:   store i32 %Array.as.Iterate.impl.NewCursor.call, ptr %var, align 4, !dbg !98
// CHECK:STDOUT:   br label %for.next, !dbg !98
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_19.5.temp), !dbg !98
// CHECK:STDOUT:   call void @"_CNext.Array.ef119564f44a13e5.Core:Iterate.Core.979acf9988e393af"(ptr %.loc7_19.5.temp, ptr %a.var, ptr %var), !dbg !98
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %.loc7_19.5.temp), !dbg !98
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !98
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %.loc7_19.5.temp), !dbg !98
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call = load i32, ptr %j.var, align 4, !dbg !105
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call1 = add i32 %Int.as.AddAssignWith.impl.Op.call, %Optional.Get.call, !dbg !105
// CHECK:STDOUT:   store i32 %Int.as.AddAssignWith.impl.Op.call1, ptr %j.var, align 4, !dbg !105
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %.loc7_19.5.temp), !dbg !98
// CHECK:STDOUT:   br label %for.next, !dbg !107
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %.loc7_19.5.temp), !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9a03059f269926ed(ptr %var), !dbg !98
// CHECK:STDOUT:   %.loc10 = load i32, ptr %j.var, align 4, !dbg !110
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9a03059f269926ed(ptr %j.var), !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c1f9a57dcd9034c0(ptr %a.var), !dbg !91
// CHECK:STDOUT:   ret i32 %.loc10, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CNewCursor.Array.ef119564f44a13e5.Core:Iterate.Core.979acf9988e393af"(ptr %self) #0 !dbg !117 {
// CHECK:STDOUT:   ret i32 0, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.Array.ef119564f44a13e5.Core:Iterate.Core.979acf9988e393af"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !124 {
// CHECK:STDOUT:   %1 = load i32, ptr %cursor, align 4, !dbg !131
// CHECK:STDOUT:   %2 = icmp slt i32 %1, 3, !dbg !131
// CHECK:STDOUT:   br i1 %2, label %3, label %8, !dbg !132
// CHECK:STDOUT:
// CHECK:STDOUT: 3:                                                ; preds = %0
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %cursor), !dbg !134
// CHECK:STDOUT:   %4 = load i32, ptr %cursor, align 4, !dbg !136
// CHECK:STDOUT:   %5 = sub i32 %4, 1, !dbg !136
// CHECK:STDOUT:   %6 = sext i32 %5 to i64, !dbg !135
// CHECK:STDOUT:   %array.index = getelementptr inbounds [4 x i8], ptr %self, i64 %6, !dbg !135
// CHECK:STDOUT:   %7 = load i32, ptr %array.index, align 4, !dbg !135
// CHECK:STDOUT:   call void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr %return, i32 %7), !dbg !137
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT:
// CHECK:STDOUT: 8:                                                ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr %return), !dbg !141
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !145 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !151
// CHECK:STDOUT:   ret i1 %1, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.29b2425ca9ad75a4(ptr %self) #0 !dbg !154 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %self), !dbg !160
// CHECK:STDOUT:   ret i32 %1, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c6daab0befde28a8(ptr %t) #0 !dbg !164 {
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9a03059f269926ed(ptr %t) #0 !dbg !171 {
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c1f9a57dcd9034c0(ptr %t) #0 !dbg !178 {
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %fn.call = call i32 @"_Cfn:enclosed.b0cc8ae043738921"(), !dbg !113
// CHECK:STDOUT:   store i32 %fn.call, ptr @_Cn.Main, align 4, !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !186 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !189
// CHECK:STDOUT:   ret void, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !194 {
// CHECK:STDOUT:   call void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return, i32 %value), !dbg !199
// CHECK:STDOUT:   ret void, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.29b2425ca9ad75a4(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !202 {
// CHECK:STDOUT:   call void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %return), !dbg !205
// CHECK:STDOUT:   ret void, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !207 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %value, i64 4, !dbg !212
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !212
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !212
// CHECK:STDOUT:   ret i1 %2, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr %value) #0 !dbg !215 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw i8, ptr %value, i64 0, !dbg !220
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !220
// CHECK:STDOUT:   ret i32 %1, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #2 !dbg !223 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !227
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !227
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !227
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !227
// CHECK:STDOUT:   ret void, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !231 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw i8, ptr %return, i64 0, !dbg !240
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !240
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %return, i64 4, !dbg !242
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !242
// CHECK:STDOUT:   ret void, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !245 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw i8, ptr %return, i64 4, !dbg !252
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !252
// CHECK:STDOUT:   ret void, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !254 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !259
// CHECK:STDOUT:   ret i32 %1, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !263 {
// CHECK:STDOUT:   ret i32 %self, !dbg !269
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c6daab0befde28a8, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.9a03059f269926ed, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!87}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !44 = !{!6}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !86 = !DIFile(filename: "for_in_local_fn.carbon", directory: "")
// CHECK:STDOUT: !87 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !86, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !88 = distinct !DISubprogram(linkageName: "_Cfn:enclosed.b0cc8ae043738921", scope: null, file: !86, line: 3, type: !22, spFlags: DISPFlagDefinition, unit: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 4, column: 3, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 4, column: 26, scope: !88)
// CHECK:STDOUT: !94 = !DILocation(line: 5, column: 3, scope: !88)
// CHECK:STDOUT: !98 = !DILocation(line: 7, column: 7, scope: !88)
// CHECK:STDOUT: !99 = !{!20, !6}
// CHECK:STDOUT: !100 = !DISubroutineType(types: !99)
// CHECK:STDOUT: !101 = !{!6, !6, !20}
// CHECK:STDOUT: !102 = !DISubroutineType(types: !101)
// CHECK:STDOUT: !103 = !{!6, !6}
// CHECK:STDOUT: !104 = !DISubroutineType(types: !103)
// CHECK:STDOUT: !105 = !DILocation(line: 8, column: 5, scope: !88)
// CHECK:STDOUT: !107 = !DILocation(line: 7, column: 3, scope: !88)
// CHECK:STDOUT: !108 = !{null, !20}
// CHECK:STDOUT: !109 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !110 = !DILocation(line: 10, column: 10, scope: !88)
// CHECK:STDOUT: !111 = !DILocation(line: 10, column: 3, scope: !88)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !86, type: !10, spFlags: DISPFlagDefinition, unit: !87)
// CHECK:STDOUT: !113 = !DILocation(line: 3, column: 14, scope: !112)
// CHECK:STDOUT: !114 = !DILocation(line: 3, column: 5, scope: !112)
// CHECK:STDOUT: !115 = !DILocation(line: 0, scope: !112)
// CHECK:STDOUT: !116 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.Array.ef119564f44a13e5.Core:Iterate.Core.979acf9988e393af", scope: null, file: !116, line: 23, type: !100, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !123)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !6)
// CHECK:STDOUT: !122 = !DILocation(line: 23, column: 38, scope: !117)
// CHECK:STDOUT: !123 = !{!118}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.Array.ef119564f44a13e5.Core:Iterate.Core.979acf9988e393af", scope: null, file: !116, line: 24, type: !102, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !143)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !6)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 2, scope: !124, type: !20)
// CHECK:STDOUT: !131 = !DILocation(line: 25, column: 9, scope: !124)
// CHECK:STDOUT: !132 = !DILocation(line: 25, column: 8, scope: !124)
// CHECK:STDOUT: !134 = !DILocation(line: 26, column: 7, scope: !124)
// CHECK:STDOUT: !135 = !DILocation(line: 27, column: 31, scope: !124)
// CHECK:STDOUT: !136 = !DILocation(line: 27, column: 36, scope: !124)
// CHECK:STDOUT: !137 = !DILocation(line: 27, column: 14, scope: !124)
// CHECK:STDOUT: !138 = !{!6, !20}
// CHECK:STDOUT: !139 = !DISubroutineType(types: !138)
// CHECK:STDOUT: !140 = !DILocation(line: 27, column: 7, scope: !124)
// CHECK:STDOUT: !141 = !DILocation(line: 29, column: 14, scope: !124)
// CHECK:STDOUT: !142 = !DILocation(line: 29, column: 7, scope: !124)
// CHECK:STDOUT: !143 = !{!125, !126}
// CHECK:STDOUT: !144 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.29b2425ca9ad75a4", scope: null, file: !144, line: 36, type: !104, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !153)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !6)
// CHECK:STDOUT: !151 = !DILocation(line: 37, column: 12, scope: !145)
// CHECK:STDOUT: !152 = !DILocation(line: 37, column: 5, scope: !145)
// CHECK:STDOUT: !153 = !{!146}
// CHECK:STDOUT: !154 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.29b2425ca9ad75a4", scope: null, file: !144, line: 39, type: !100, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !162)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 1, scope: !154, type: !6)
// CHECK:STDOUT: !160 = !DILocation(line: 40, column: 12, scope: !154)
// CHECK:STDOUT: !161 = !DILocation(line: 40, column: 5, scope: !154)
// CHECK:STDOUT: !162 = !{!155}
// CHECK:STDOUT: !163 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c6daab0befde28a8", scope: null, file: !163, line: 36, type: !8, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !170)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !6)
// CHECK:STDOUT: !169 = !DILocation(line: 36, column: 1, scope: !164)
// CHECK:STDOUT: !170 = !{!165}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9a03059f269926ed", scope: null, file: !163, line: 36, type: !109, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !177)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !20)
// CHECK:STDOUT: !176 = !DILocation(line: 36, column: 1, scope: !171)
// CHECK:STDOUT: !177 = !{!172}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c1f9a57dcd9034c0", scope: null, file: !163, line: 36, type: !8, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !184)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !6)
// CHECK:STDOUT: !183 = !DILocation(line: 36, column: 1, scope: !178)
// CHECK:STDOUT: !184 = !{!179}
// CHECK:STDOUT: !185 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !185, line: 396, type: !109, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !193)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !20)
// CHECK:STDOUT: !189 = !DILocation(line: 398, column: 5, scope: !186)
// CHECK:STDOUT: !190 = !{null, !20, !20}
// CHECK:STDOUT: !191 = !DISubroutineType(types: !190)
// CHECK:STDOUT: !192 = !DILocation(line: 396, column: 3, scope: !186)
// CHECK:STDOUT: !193 = !{!187}
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.29b2425ca9ad75a4", scope: null, file: !144, line: 33, type: !139, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !201)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !20)
// CHECK:STDOUT: !199 = !DILocation(line: 34, column: 12, scope: !194)
// CHECK:STDOUT: !200 = !DILocation(line: 34, column: 5, scope: !194)
// CHECK:STDOUT: !201 = !{!195}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.29b2425ca9ad75a4", scope: null, file: !144, line: 30, type: !45, spFlags: DISPFlagDefinition, unit: !87)
// CHECK:STDOUT: !205 = !DILocation(line: 31, column: 12, scope: !202)
// CHECK:STDOUT: !206 = !DILocation(line: 31, column: 5, scope: !202)
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !144, line: 141, type: !104, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !214)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !6)
// CHECK:STDOUT: !212 = !DILocation(line: 142, column: 12, scope: !207)
// CHECK:STDOUT: !213 = !DILocation(line: 142, column: 5, scope: !207)
// CHECK:STDOUT: !214 = !{!208}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !144, line: 144, type: !100, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !222)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !6)
// CHECK:STDOUT: !220 = !DILocation(line: 145, column: 12, scope: !215)
// CHECK:STDOUT: !221 = !DILocation(line: 145, column: 5, scope: !215)
// CHECK:STDOUT: !222 = !{!216}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !185, line: 331, type: !191, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !230)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !20)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 2, scope: !223, type: !20)
// CHECK:STDOUT: !227 = !DILocation(line: 331, column: 3, scope: !223)
// CHECK:STDOUT: !228 = !{!20, !20}
// CHECK:STDOUT: !229 = !DISubroutineType(types: !228)
// CHECK:STDOUT: !230 = !{!224, !225}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !144, line: 131, type: !139, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !244)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !20)
// CHECK:STDOUT: !240 = !DILocation(line: 137, column: 5, scope: !231)
// CHECK:STDOUT: !242 = !DILocation(line: 138, column: 5, scope: !231)
// CHECK:STDOUT: !243 = !DILocation(line: 139, column: 5, scope: !231)
// CHECK:STDOUT: !244 = !{!232}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "None", linkageName: "_CNone.c1cc3e41344c17ec:OptionalStorage.Core.b21e3f34d194eaf5", scope: null, file: !144, line: 125, type: !45, spFlags: DISPFlagDefinition, unit: !87)
// CHECK:STDOUT: !252 = !DILocation(line: 128, column: 5, scope: !245)
// CHECK:STDOUT: !253 = !DILocation(line: 129, column: 5, scope: !245)
// CHECK:STDOUT: !254 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !185, line: 64, type: !229, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !262)
// CHECK:STDOUT: !255 = !DILocalVariable(arg: 1, scope: !254, type: !20)
// CHECK:STDOUT: !259 = !DILocation(line: 66, column: 17, scope: !254)
// CHECK:STDOUT: !261 = !DILocation(line: 66, column: 5, scope: !254)
// CHECK:STDOUT: !262 = !{!255}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !185, line: 59, type: !229, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !270)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !20)
// CHECK:STDOUT: !269 = !DILocation(line: 59, column: 28, scope: !263)
// CHECK:STDOUT: !270 = !{!264}
// CHECK:STDOUT:
