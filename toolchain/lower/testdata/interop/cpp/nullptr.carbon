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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNullptr.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_18.2.temp = alloca ptr, align 8, !dbg !17
// CHECK:STDOUT:   %.loc14_22.1.temp = alloca ptr, align 8, !dbg !19
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.481e1aec351df623.Core.55456748a8a2eb70"(ptr poison), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.2.temp), !dbg !17
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, ptr %.loc12_18.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   %.loc12_18.4 = load ptr, ptr %.loc12_18.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   call void @_Z7TakePtrPi(ptr %.loc12_18.4), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.632d99dc34484dc8(ptr %.loc12_18.2.temp), !dbg !17
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_22.1.temp), !dbg !19
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, ptr %.loc14_22.1.temp, align 8, !dbg !19
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn.carbon_thunk._(ptr %.loc14_22.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.773050755018f871(ptr %.loc14_22.1.temp), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrPi(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11TakeNullptrDn.carbon_thunk._(ptr noundef %n) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %n, ptr %n.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %0 = load ptr, ptr %n.addr, align 8, !tbaa !87
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn(ptr null)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptr.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !25
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptrCopy.Main() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !30
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !31
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22, ptr %a.var, align 8, !dbg !30
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %a.var, align 8, !dbg !33
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %.loc23), !dbg !33
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.773050755018f871(ptr %a.var), !dbg !30
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrIndirectly.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !38
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk.(ptr %a.var), !dbg !39
// CHECK:STDOUT:   %.loc28_10 = load ptr, ptr %a.var, align 8, !dbg !41
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.481e1aec351df623.Core.55456748a8a2eb70"(ptr %.loc28_10), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.773050755018f871(ptr %a.var), !dbg !38
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13ReturnNullptrv.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %call = call ptr @_Z13ReturnNullptrv()
// CHECK:STDOUT:   store ptr %call, ptr %0, align 8, !tbaa !88
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrDirectly.Main() #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_28.1.temp = alloca ptr, align 8, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_28.1.temp), !dbg !46
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk.(ptr %.loc32_28.1.temp), !dbg !46
// CHECK:STDOUT:   %.loc32_28.4 = load ptr, ptr %.loc32_28.1.temp, align 8, !dbg !46
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.481e1aec351df623.Core.55456748a8a2eb70"(ptr %.loc32_28.4), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.773050755018f871(ptr %.loc32_28.1.temp), !dbg !46
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertNullptrConstant.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.481e1aec351df623.Core.55456748a8a2eb70"(ptr poison), !dbg !51
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.481e1aec351df623.Core.55456748a8a2eb70"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT:   %1 = call ptr @_CNone.Optional.Core.2313f7f09b1ddaea(), !dbg !58
// CHECK:STDOUT:   ret ptr %1, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.632d99dc34484dc8(ptr %t) #0 !dbg !62 {
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.773050755018f871(ptr %t) #0 !dbg !69 {
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CMake.NullptrT.CppCompat.Core()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.2313f7f09b1ddaea() #0 !dbg !77 {
// CHECK:STDOUT:   ret ptr null, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeNullptrDn(ptr) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z13ReturnNullptrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.481e1aec351df623.Core.55456748a8a2eb70", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.773050755018f871, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!89, !90, !91, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!94}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!9, !9}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{!9}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "PassNullptr", linkageName: "_CPassNullptr.Main", scope: null, file: !4, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DILocation(line: 12, column: 15, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 12, column: 3, scope: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 19, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 14, column: 3, scope: !16)
// CHECK:STDOUT: !21 = !DILocation(line: 11, column: 1, scope: !16)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "ReturnNullptr", linkageName: "_CReturnNullptr.Main", scope: null, file: !4, line: 17, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !25 = !DILocation(line: 18, column: 10, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 18, column: 3, scope: !22)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "ReturnNullptrCopy", linkageName: "_CReturnNullptrCopy.Main", scope: null, file: !4, line: 21, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !30 = !DILocation(line: 22, column: 3, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 26, scope: !27)
// CHECK:STDOUT: !33 = !DILocation(line: 23, column: 10, scope: !27)
// CHECK:STDOUT: !34 = !DILocation(line: 23, column: 3, scope: !27)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "ReturnConvertedNullptrIndirectly", linkageName: "_CReturnConvertedNullptrIndirectly.Main", scope: null, file: !4, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !38 = !DILocation(line: 27, column: 3, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 27, column: 26, scope: !35)
// CHECK:STDOUT: !41 = !DILocation(line: 28, column: 10, scope: !35)
// CHECK:STDOUT: !42 = !DILocation(line: 28, column: 3, scope: !35)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "ReturnConvertedNullptrDirectly", linkageName: "_CReturnConvertedNullptrDirectly.Main", scope: null, file: !4, line: 31, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !46 = !DILocation(line: 32, column: 10, scope: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 32, column: 3, scope: !43)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "ConvertNullptrConstant", linkageName: "_CConvertNullptrConstant.Main", scope: null, file: !4, line: 35, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !51 = !DILocation(line: 36, column: 3, scope: !48)
// CHECK:STDOUT: !52 = !DIFile(filename: "{{.*}}/prelude/types/cpp/nullptr.carbon", directory: "")
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.NullptrT.CppCompat.Core:ImplicitAs.481e1aec351df623.Core.55456748a8a2eb70", scope: null, file: !52, line: 51, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !60)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !9)
// CHECK:STDOUT: !58 = !DILocation(line: 52, column: 14, scope: !53)
// CHECK:STDOUT: !59 = !DILocation(line: 52, column: 7, scope: !53)
// CHECK:STDOUT: !60 = !{!54}
// CHECK:STDOUT: !61 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.632d99dc34484dc8", scope: null, file: !61, line: 35, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !9)
// CHECK:STDOUT: !67 = !DILocation(line: 35, column: 1, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.773050755018f871", scope: null, file: !61, line: 35, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !75)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !9)
// CHECK:STDOUT: !74 = !DILocation(line: 35, column: 1, scope: !69)
// CHECK:STDOUT: !75 = !{!70}
// CHECK:STDOUT: !76 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.2313f7f09b1ddaea", scope: null, file: !76, line: 30, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !81 = !DILocation(line: 31, column: 5, scope: !77)
// CHECK:STDOUT: !82 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !83 = !{!"omnipotent char", !82, i64 0}
// CHECK:STDOUT: !84 = !{!"std::nullptr_t", !83, i64 0}
// CHECK:STDOUT: !85 = !{!"any pointer", !83, i64 0}
// CHECK:STDOUT: !86 = !{!"p1 std::nullptr_t", !85, i64 0}
// CHECK:STDOUT: !87 = !{!86, !86, i64 0}
// CHECK:STDOUT: !88 = !{!84, !84, i64 0}
// CHECK:STDOUT: !89 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !90 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !91 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !92 = !{!"int", !83, i64 0}
// CHECK:STDOUT: !93 = !{!"__libc_errno", !92, i64 0}
// CHECK:STDOUT: !94 = !{!93, !92, i64 0}
// CHECK:STDOUT:
