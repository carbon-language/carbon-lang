// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/method_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/method_ptr.carbon

// --- method_ptr.h

class C {
 public:
  void Method(int i) const;
};
using MethodPtrType = void(C::*)() const;

auto GetMethodPtr() -> MethodPtrType;
void TakeMethodPtr(MethodPtrType);

template <typename C>
void TakeGenericMethodPtr(void(C::* ptr)() const);

// --- relay_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

fn G() {
  Cpp.TakeMethodPtr(Cpp.GetMethodPtr());
}

// --- convert_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp;

class C {
  fn Method(self);
}

inline Cpp '''c++
using MethodPtrType = void(Carbon::C::*)() const;
''';

fn G() {
  var unused ptr: Cpp.MethodPtrType = C.Method;
}

// --- pass_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

class C {
  fn Method(self);
}

fn F() {
  Cpp.TakeGenericMethodPtr(C.Method);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'relay_method_ptr.carbon'
// CHECK:STDOUT: source_filename = "relay_method_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_38.1.temp = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   %.loc6_38.5.temp = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_38.1.temp), !dbg !12
// CHECK:STDOUT:   call void @_Z12GetMethodPtrv.carbon_thunk.(ptr %.loc6_38.1.temp), !dbg !12
// CHECK:STDOUT:   %.loc6_38.4 = load ptr, ptr %.loc6_38.1.temp, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc6_38.5.temp), !dbg !12
// CHECK:STDOUT:   store ptr %.loc6_38.4, ptr %.loc6_38.5.temp, align 8, !dbg !12
// CHECK:STDOUT:   call void @_Z13TakeMethodPtrM1CKFvvE.carbon_thunk._(ptr %.loc6_38.5.temp), !dbg !13
// CHECK:STDOUT:   call void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %.loc6_38.5.temp), !dbg !12
// CHECK:STDOUT:   call void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %.loc6_38.1.temp), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z12GetMethodPtrv.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %call = call { i64, i64 } @_Z12GetMethodPtrv()
// CHECK:STDOUT:   store { i64, i64 } %call, ptr %0, align 8, !tbaa !23
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13TakeMethodPtrM1CKFvvE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %coerce = alloca { i64, i64 }, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %2 = load { i64, i64 }, ptr %1, align 8, !tbaa !23
// CHECK:STDOUT:   store { i64, i64 } %2, ptr %coerce, align 8, !tbaa !23
// CHECK:STDOUT:   %3 = getelementptr inbounds nuw { i64, i64 }, ptr %coerce, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i64, ptr %3, align 8
// CHECK:STDOUT:   %5 = getelementptr inbounds nuw { i64, i64 }, ptr %coerce, i32 0, i32 1
// CHECK:STDOUT:   %6 = load i64, ptr %5, align 8
// CHECK:STDOUT:   call void @_Z13TakeMethodPtrM1CKFvvE(i64 %4, i64 %6)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %self) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare { i64, i64 } @_Z12GetMethodPtrv() #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13TakeMethodPtrM1CKFvvE(i64, i64) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!24, !25, !26, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!29}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "relay_method_ptr.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !4, line: 5, type: !7, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DILocation(line: 6, column: 21, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 6, column: 3, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 5, column: 1, scope: !11)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.659d177172b60c17:core.Destroy.Core", scope: null, file: !4, line: 6, type: !10, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !18)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !15, type: !8)
// CHECK:STDOUT: !17 = !DILocation(line: 6, column: 21, scope: !15)
// CHECK:STDOUT: !18 = !{!16}
// CHECK:STDOUT: !19 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !20 = !{!"omnipotent char", !19, i64 0}
// CHECK:STDOUT: !21 = !{!"any pointer", !20, i64 0}
// CHECK:STDOUT: !22 = !{!21, !21, i64 0}
// CHECK:STDOUT: !23 = !{!20, !20, i64 0}
// CHECK:STDOUT: !24 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !25 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !27 = !{!"int", !20, i64 0}
// CHECK:STDOUT: !28 = !{!"__libc_errno", !27, i64 0}
// CHECK:STDOUT: !29 = !{!28, !27, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert_method_ptr.carbon'
// CHECK:STDOUT: source_filename = "convert_method_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMethod.C.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr.var = alloca ptr, align 8, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr.var), !dbg !33
// CHECK:STDOUT:   store ptr @_CMethod.C.Main, ptr %ptr.var, align 8, !dbg !33
// CHECK:STDOUT:   call void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %ptr.var), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.C.Main"(ptr %self) #1 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMethod__carbon_thunk.C.Main(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMethod.C.Main(ptr %self), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!31}
// CHECK:STDOUT: !llvm.module.flags = !{!24, !25, !26, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!29}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !19 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !20 = !{!"omnipotent char", !19, i64 0}
// CHECK:STDOUT: !24 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !25 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !27 = !{!"int", !20, i64 0}
// CHECK:STDOUT: !28 = !{!"__libc_errno", !27, i64 0}
// CHECK:STDOUT: !29 = !{!28, !27, i64 0}
// CHECK:STDOUT: !30 = !DIFile(filename: "convert_method_ptr.carbon", directory: "")
// CHECK:STDOUT: !31 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !30, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !30, line: 13, type: !7, spFlags: DISPFlagDefinition, unit: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 14, column: 3, scope: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 13, column: 1, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.C.Main", scope: null, file: !30, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !39)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !8)
// CHECK:STDOUT: !38 = !DILocation(line: 5, column: 1, scope: !36)
// CHECK:STDOUT: !39 = !{!37}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !30, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !43)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !8)
// CHECK:STDOUT: !42 = !DILocation(line: 5, column: 1, scope: !40)
// CHECK:STDOUT: !43 = !{!41}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !30, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !47)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !8)
// CHECK:STDOUT: !46 = !DILocation(line: 5, column: 1, scope: !44)
// CHECK:STDOUT: !47 = !{!45}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Method__carbon_thunk", linkageName: "_CMethod__carbon_thunk.C.Main", scope: null, file: !30, line: 14, type: !10, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !52)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !8)
// CHECK:STDOUT: !51 = !DILocation(line: 6, column: 3, scope: !48)
// CHECK:STDOUT: !52 = !{!49}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.659d177172b60c17:core.Destroy.Core", scope: null, file: !30, line: 14, type: !10, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !8)
// CHECK:STDOUT: !55 = !DILocation(line: 14, column: 3, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pass_method_ptr.carbon'
// CHECK:STDOUT: source_filename = "pass_method_ptr.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @.b35 = internal constant ptr @_CMethod.C.Main
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMethod.C.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_29.3.temp = alloca ptr, align 8, !dbg !60
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_29.3.temp), !dbg !60
// CHECK:STDOUT:   call void @_Z20TakeGenericMethodPtrIN6Carbon1CEEvMT_KFvvE.carbon_thunk._(ptr @.b35), !dbg !61
// CHECK:STDOUT:   call void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr @.b35), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.C.Main"(ptr %self) #1 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMethod__carbon_thunk.C.Main(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMethod.C.Main(ptr %self), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z20TakeGenericMethodPtrIN6Carbon1CEEvMT_KFvvE.carbon_thunk._(ptr noundef %ptr) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ptr.addr = alloca ptr, align 8
// CHECK:STDOUT:   %coerce = alloca { i64, i64 }, align 8
// CHECK:STDOUT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %0 = load ptr, ptr %ptr.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %1 = load { i64, i64 }, ptr %0, align 8, !tbaa !23
// CHECK:STDOUT:   store { i64, i64 } %1, ptr %coerce, align 8, !tbaa !23
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw { i64, i64 }, ptr %coerce, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i64, ptr %2, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds nuw { i64, i64 }, ptr %coerce, i32 0, i32 1
// CHECK:STDOUT:   %5 = load i64, ptr %4, align 8
// CHECK:STDOUT:   call void @_Z20TakeGenericMethodPtrIN6Carbon1CEEvMT_KFvvE(i64 %3, i64 %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.659d177172b60c17:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z20TakeGenericMethodPtrIN6Carbon1CEEvMT_KFvvE(i64, i64) #4
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!58}
// CHECK:STDOUT: !llvm.module.flags = !{!24, !25, !26, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!29}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !19 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !20 = !{!"omnipotent char", !19, i64 0}
// CHECK:STDOUT: !21 = !{!"any pointer", !20, i64 0}
// CHECK:STDOUT: !22 = !{!21, !21, i64 0}
// CHECK:STDOUT: !23 = !{!20, !20, i64 0}
// CHECK:STDOUT: !24 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !25 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !26 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !27 = !{!"int", !20, i64 0}
// CHECK:STDOUT: !28 = !{!"__libc_errno", !27, i64 0}
// CHECK:STDOUT: !29 = !{!28, !27, i64 0}
// CHECK:STDOUT: !57 = !DIFile(filename: "pass_method_ptr.carbon", directory: "")
// CHECK:STDOUT: !58 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !57, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !57, line: 9, type: !7, spFlags: DISPFlagDefinition, unit: !58)
// CHECK:STDOUT: !60 = !DILocation(line: 10, column: 28, scope: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 10, column: 3, scope: !59)
// CHECK:STDOUT: !62 = !DILocation(line: 9, column: 1, scope: !59)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.C.Main", scope: null, file: !57, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !58, retainedNodes: !66)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !8)
// CHECK:STDOUT: !65 = !DILocation(line: 5, column: 1, scope: !63)
// CHECK:STDOUT: !66 = !{!64}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !57, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !58, retainedNodes: !70)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !8)
// CHECK:STDOUT: !69 = !DILocation(line: 5, column: 1, scope: !67)
// CHECK:STDOUT: !70 = !{!68}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !57, line: 5, type: !10, spFlags: DISPFlagDefinition, unit: !58, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !8)
// CHECK:STDOUT: !73 = !DILocation(line: 5, column: 1, scope: !71)
// CHECK:STDOUT: !74 = !{!72}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Method__carbon_thunk", linkageName: "_CMethod__carbon_thunk.C.Main", scope: null, file: !57, line: 6, type: !10, spFlags: DISPFlagDefinition, unit: !58, retainedNodes: !78)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !8)
// CHECK:STDOUT: !77 = !DILocation(line: 6, column: 3, scope: !75)
// CHECK:STDOUT: !78 = !{!76}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.659d177172b60c17:core.Destroy.Core", scope: null, file: !57, line: 10, type: !10, spFlags: DISPFlagDefinition, unit: !58, retainedNodes: !82)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !8)
// CHECK:STDOUT: !81 = !DILocation(line: 10, column: 28, scope: !79)
// CHECK:STDOUT: !82 = !{!80}
// CHECK:STDOUT:
