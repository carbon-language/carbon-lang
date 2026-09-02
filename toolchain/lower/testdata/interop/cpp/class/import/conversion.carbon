// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/conversion.carbon

// --- class_to_class.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct A {
  int n;
  ~A();
};
struct B {
  operator A() const;
};
''';

fn ConvertToVar(b: Cpp.B) {
  var unused a: Cpp.A = b;
}

fn ConvertToValue(b: Cpp.B) {
  let unused a: Cpp.A = b;
}

// --- class_to_enum.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
enum S : short {};
enum I : short {};
struct EnumB {
  operator S() const;
  operator I() const;
};
''';

fn ConvertToVar(b: Cpp.EnumB) {
  var unused s: Cpp.S = b;
  var unused i: Cpp.I = b;
}

fn ConvertToValue(b: Cpp.EnumB) {
  let unused s: Cpp.S = b;
  let unused i: Cpp.I = b;
}

// --- class_to_int.carbon

library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct IntB {
  operator int() const;
};
''';

fn ConvertToVar(b: Cpp.IntB) {
  var unused a: Cpp.int = b;
}

fn ConvertToValue(b: Cpp.IntB) {
  let unused a: Cpp.int = b;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_class.carbon'
// CHECK:STDOUT: source_filename = "class_to_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !18
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %a.var), !dbg !18
// CHECK:STDOUT:   call void @_ZN1AD1Ev(ptr %a.var), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK1Bcv1AEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !20, !nonnull !25
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4 %0, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca [4 x i8], align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !29
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !29
// CHECK:STDOUT:   call void @_ZN1AD1Ev(ptr %.loc20_25.1.temp), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4, ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "class_to_class.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !1, line: 15, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 16, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 15, column: 1, scope: !12)
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS1B", !22, i64 0}
// CHECK:STDOUT: !22 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !23 = !{!24, !24, i64 0}
// CHECK:STDOUT: !24 = !{!"p1 _ZTS1A", !22, i64 0}
// CHECK:STDOUT: !25 = !{}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !1, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !15)
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 25, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 19, column: 1, scope: !26)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_enum.carbon'
// CHECK:STDOUT: source_filename = "class_to_enum.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca i16, align 2, !dbg !18
// CHECK:STDOUT:   %i.var = alloca i16, align 2, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !18
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %s.var), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !19
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %i.var), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !21, !nonnull !25
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !21, !nonnull !25
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !28
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca i16, align 2, !dbg !33
// CHECK:STDOUT:   %.loc21_25.1.temp = alloca i16, align 2, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !33
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !33
// CHECK:STDOUT:   %.loc20_25.5 = load i16, ptr %.loc20_25.1.temp, align 2, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_25.1.temp), !dbg !34
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %.loc21_25.1.temp), !dbg !34
// CHECK:STDOUT:   %.loc21_25.5 = load i16, ptr %.loc21_25.1.temp, align 2, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "class_to_enum.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !1, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 16, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 14, column: 1, scope: !12)
// CHECK:STDOUT: !21 = !{!22, !22, i64 0}
// CHECK:STDOUT: !22 = !{!"p1 _ZTS5EnumB", !23, i64 0}
// CHECK:STDOUT: !23 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !24 = !{!23, !23, i64 0}
// CHECK:STDOUT: !25 = !{}
// CHECK:STDOUT: !26 = !{!27, !27, i64 0}
// CHECK:STDOUT: !27 = !{!"_ZTS1S", !10, i64 0}
// CHECK:STDOUT: !28 = !{!29, !29, i64 0}
// CHECK:STDOUT: !29 = !{!"_ZTS1I", !10, i64 0}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !1, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !30, type: !15)
// CHECK:STDOUT: !33 = !DILocation(line: 20, column: 25, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 21, column: 25, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 19, column: 1, scope: !30)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_int.carbon'
// CHECK:STDOUT: source_filename = "class_to_int.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !18
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !18
// CHECK:STDOUT:   store i32 %__carbon_thunk.call, ptr %a.var, align 4, !dbg !18
// CHECK:STDOUT:   call void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %a.var), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !20
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !20, !nonnull !23
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "class_to_int.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !1, line: 11, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 12, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 11, column: 1, scope: !12)
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS4IntB", !22, i64 0}
// CHECK:STDOUT: !22 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !23 = !{}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 12, type: !25, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !25 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !26 = !{null, !27}
// CHECK:STDOUT: !27 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !24, type: !27)
// CHECK:STDOUT: !30 = !DILocation(line: 12, column: 3, scope: !24)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !1, line: 15, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !32)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !31, type: !15)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 27, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 15, column: 1, scope: !31)
// CHECK:STDOUT:
