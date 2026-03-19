// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/nullptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/nullptr.carbon

// --- simple.carbon

import Cpp inline '''
using nullptr_t = decltype(nullptr);

void TakePtr(int *p);
void TakeNullptr(nullptr_t n);

nullptr_t ReturnNullptr();
''';

fn PassNullptr() {
  Cpp.TakePtr(Cpp.nullptr);

  Cpp.TakeNullptr(Cpp.nullptr);
}

fn ReturnNullptr() -> Cpp.nullptr_t {
  return Cpp.nullptr;
}

fn ReturnNullptrCopy() -> Cpp.nullptr_t {
  var a: Cpp.nullptr_t = Cpp.nullptr;
  return a;
}

fn ReturnConvertedNullptrIndirectly() -> Core.Optional(i32*) {
  var a: Cpp.nullptr_t = Cpp.ReturnNullptr();
  return a;
}

fn ReturnConvertedNullptrDirectly() -> Core.Optional(i32*) {
  return Cpp.ReturnNullptr();
}

fn ConvertNullptrConstant() -> Core.Optional(i32*) {
  return Cpp.nullptr;
}

// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z11TakeNullptrDn.carbon_thunk(ptr noundef %n) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %n, ptr %n.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %n.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn(ptr null)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeNullptrDn(ptr) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z13ReturnNullptrv.carbon_thunk(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %call = call ptr @_Z13ReturnNullptrv()
// CHECK:STDOUT:   store ptr %call, ptr %0, align 8, !tbaa !14
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z13ReturnNullptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNullptr.Main() #2 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_18.2.temp = alloca ptr, align 8, !dbg !19
// CHECK:STDOUT:   %.loc14_22.1.temp = alloca ptr, align 8, !dbg !20
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.83dbfcd16e56a769.Core.b88d1103f417c6d4"(ptr poison), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.2.temp), !dbg !19
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, ptr %.loc12_18.2.temp, align 8, !dbg !19
// CHECK:STDOUT:   %.loc12_18.4 = load ptr, ptr %.loc12_18.2.temp, align 8, !dbg !19
// CHECK:STDOUT:   call void @_Z7TakePtrPi(ptr %.loc12_18.4), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_22.1.temp), !dbg !20
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !20
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, ptr %.loc14_22.1.temp, align 8, !dbg !20
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn.carbon_thunk(ptr %.loc14_22.1.temp), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.6f07019e2bbdfcc4:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrPi(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptr.Main() #2 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !28
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptrCopy.Main() #2 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !31
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !32
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22, ptr %a.var, align 8, !dbg !31
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %a.var, align 8, !dbg !33
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %.loc23), !dbg !33
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrIndirectly.Main() #2 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !36
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk(ptr %a.var), !dbg !37
// CHECK:STDOUT:   %.loc28_10 = load ptr, ptr %a.var, align 8, !dbg !38
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.83dbfcd16e56a769.Core.b88d1103f417c6d4"(ptr %.loc28_10), !dbg !39
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrDirectly.Main() #2 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_28.1.temp = alloca ptr, align 8, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_28.1.temp), !dbg !41
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk(ptr %.loc32_28.1.temp), !dbg !41
// CHECK:STDOUT:   %.loc32_28.4 = load ptr, ptr %.loc32_28.1.temp, align 8, !dbg !41
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.83dbfcd16e56a769.Core.b88d1103f417c6d4"(ptr %.loc32_28.4), !dbg !42
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertNullptrConstant.Main() #2 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.83dbfcd16e56a769.Core.b88d1103f417c6d4"(ptr poison), !dbg !44
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.83dbfcd16e56a769.Core.b88d1103f417c6d4"(ptr %self) #2 !dbg !45 {
// CHECK:STDOUT:   %1 = call ptr @_CNone.Optional.Core.95809cda2a4fffc7(), !dbg !51
// CHECK:STDOUT:   ret ptr %1, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CMake.NullptrT.CppCompat.Core()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.95809cda2a4fffc7() #2 !dbg !53 {
// CHECK:STDOUT:   ret ptr null, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.83dbfcd16e56a769.Core.b88d1103f417c6d4", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 std::nullptr_t", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = !{!15, !15, i64 0}
// CHECK:STDOUT: !15 = !{!"std::nullptr_t", !9, i64 0}
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "PassNullptr", linkageName: "_CPassNullptr.Main", scope: null, file: !6, line: 11, type: !17, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DILocation(line: 12, column: 15, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 14, column: 19, scope: !16)
// CHECK:STDOUT: !21 = !DILocation(line: 12, column: 3, scope: !16)
// CHECK:STDOUT: !22 = !DILocation(line: 14, column: 3, scope: !16)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 1, scope: !16)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "ReturnNullptr", linkageName: "_CReturnNullptr.Main", scope: null, file: !6, line: 17, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !25 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !28 = !DILocation(line: 18, column: 10, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 18, column: 3, scope: !24)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "ReturnNullptrCopy", linkageName: "_CReturnNullptrCopy.Main", scope: null, file: !6, line: 21, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 3, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 26, scope: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 23, column: 10, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 23, column: 3, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "ReturnConvertedNullptrIndirectly", linkageName: "_CReturnConvertedNullptrIndirectly.Main", scope: null, file: !6, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !36 = !DILocation(line: 27, column: 3, scope: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 27, column: 26, scope: !35)
// CHECK:STDOUT: !38 = !DILocation(line: 28, column: 10, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 28, column: 3, scope: !35)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "ReturnConvertedNullptrDirectly", linkageName: "_CReturnConvertedNullptrDirectly.Main", scope: null, file: !6, line: 31, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !41 = !DILocation(line: 32, column: 10, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 32, column: 3, scope: !40)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "ConvertNullptrConstant", linkageName: "_CConvertNullptrConstant.Main", scope: null, file: !6, line: 35, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !44 = !DILocation(line: 36, column: 3, scope: !43)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.NullptrT.CppCompat.Core:ImplicitAs.83dbfcd16e56a769.Core.b88d1103f417c6d4", scope: null, file: !46, line: 46, type: !47, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !46 = !DIFile(filename: "{{.*}}/prelude/types/cpp/nullptr.carbon", directory: "")
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!27, !27}
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !45, type: !27)
// CHECK:STDOUT: !51 = !DILocation(line: 47, column: 14, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 47, column: 7, scope: !45)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.95809cda2a4fffc7", scope: null, file: !54, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !54 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !55 = !DILocation(line: 27, column: 5, scope: !53)
