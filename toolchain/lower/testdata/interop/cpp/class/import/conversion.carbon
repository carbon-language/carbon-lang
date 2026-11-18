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
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !29
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %a.var), !dbg !29
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %a.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK1Bcv1AEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !53
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !55
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !55
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !53, !nonnull !57
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4 %0, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca [4 x i8], align 4, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !37
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %.loc20_25.1.temp), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e885b4b34ded440e(ptr %t) #0 !dbg !42 {
// CHECK:STDOUT:   call void @_ZN1AD1Ev(ptr %t), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4, ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.e885b4b34ded440e, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !63, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "class_to_class.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !18, line: 15, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 1, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !18, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 25, scope: !34)
// CHECK:STDOUT: !40 = !DILocation(line: 19, column: 1, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e885b4b34ded440e", scope: null, file: !10, line: 32, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !21)
// CHECK:STDOUT: !46 = !DILocation(line: 34, column: 5, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 32, column: 1, scope: !42)
// CHECK:STDOUT: !48 = !{!43}
// CHECK:STDOUT: !49 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !50 = !{!"omnipotent char", !49, i64 0}
// CHECK:STDOUT: !51 = !{!"any pointer", !50, i64 0}
// CHECK:STDOUT: !52 = !{!"p1 _ZTS1B", !51, i64 0}
// CHECK:STDOUT: !53 = !{!52, !52, i64 0}
// CHECK:STDOUT: !54 = !{!"p1 _ZTS1A", !51, i64 0}
// CHECK:STDOUT: !55 = !{!54, !54, i64 0}
// CHECK:STDOUT: !57 = !{}
// CHECK:STDOUT: !58 = !{!"int", !50, i64 0}
// CHECK:STDOUT: !60 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !61 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !63 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !64 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !65 = !{!64, !58, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_enum.carbon'
// CHECK:STDOUT: source_filename = "class_to_enum.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca i16, align 2, !dbg !72
// CHECK:STDOUT:   %i.var = alloca i16, align 2, !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !72
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %s.var), !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !75
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %i.var), !dbg !75
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b6485bbba53783ee(ptr %i.var), !dbg !75
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9921dfc7dd957f8(ptr %s.var), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !108
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !109
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !109
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !108, !nonnull !57
// CHECK:STDOUT:   %call = call noundef i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !112
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !108
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !109
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !109
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !108, !nonnull !57
// CHECK:STDOUT:   %call = call noundef i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !114
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca i16, align 2, !dbg !85
// CHECK:STDOUT:   %.loc21_25.1.temp = alloca i16, align 2, !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !85
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !85
// CHECK:STDOUT:   %.loc20_25.5 = load i16, ptr %.loc20_25.1.temp, align 2, !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_25.1.temp), !dbg !88
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %.loc21_25.1.temp), !dbg !88
// CHECK:STDOUT:   %.loc21_25.5 = load i16, ptr %.loc21_25.1.temp, align 2, !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b6485bbba53783ee(ptr %.loc21_25.1.temp), !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e9921dfc7dd957f8(ptr %.loc20_25.1.temp), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b6485bbba53783ee(ptr %t) #0 !dbg !93 {
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e9921dfc7dd957f8(ptr %t) #0 !dbg !100 {
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.b6485bbba53783ee, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.e9921dfc7dd957f8, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!67}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !63, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !49 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !50 = !{!"omnipotent char", !49, i64 0}
// CHECK:STDOUT: !51 = !{!"any pointer", !50, i64 0}
// CHECK:STDOUT: !57 = !{}
// CHECK:STDOUT: !58 = !{!"int", !50, i64 0}
// CHECK:STDOUT: !60 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !61 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !63 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !64 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !65 = !{!64, !58, i64 0}
// CHECK:STDOUT: !66 = !DIFile(filename: "class_to_enum.carbon", directory: "")
// CHECK:STDOUT: !67 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !66, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !68 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !66, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !81)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !21)
// CHECK:STDOUT: !72 = !DILocation(line: 15, column: 3, scope: !69)
// CHECK:STDOUT: !75 = !DILocation(line: 16, column: 3, scope: !69)
// CHECK:STDOUT: !78 = !{null, !68}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !80 = !DILocation(line: 14, column: 1, scope: !69)
// CHECK:STDOUT: !81 = !{!70}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !66, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !92)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !21)
// CHECK:STDOUT: !85 = !DILocation(line: 20, column: 25, scope: !82)
// CHECK:STDOUT: !88 = !DILocation(line: 21, column: 25, scope: !82)
// CHECK:STDOUT: !91 = !DILocation(line: 19, column: 1, scope: !82)
// CHECK:STDOUT: !92 = !{!83}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b6485bbba53783ee", scope: null, file: !10, line: 32, type: !79, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !99)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !68)
// CHECK:STDOUT: !98 = !DILocation(line: 32, column: 1, scope: !93)
// CHECK:STDOUT: !99 = !{!94}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e9921dfc7dd957f8", scope: null, file: !10, line: 32, type: !79, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !106)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !68)
// CHECK:STDOUT: !105 = !DILocation(line: 32, column: 1, scope: !100)
// CHECK:STDOUT: !106 = !{!101}
// CHECK:STDOUT: !107 = !{!"p1 _ZTS5EnumB", !51, i64 0}
// CHECK:STDOUT: !108 = !{!107, !107, i64 0}
// CHECK:STDOUT: !109 = !{!51, !51, i64 0}
// CHECK:STDOUT: !110 = !{!"_ZTS1S", !50, i64 0}
// CHECK:STDOUT: !112 = !{!110, !110, i64 0}
// CHECK:STDOUT: !113 = !{!"_ZTS1I", !50, i64 0}
// CHECK:STDOUT: !114 = !{!113, !113, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_int.carbon'
// CHECK:STDOUT: source_filename = "class_to_int.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !122
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !122
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !122
// CHECK:STDOUT:   store i32 %__carbon_thunk.call, ptr %a.var, align 4, !dbg !122
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %a.var), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !145
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !145, !nonnull !57
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !132
// CHECK:STDOUT:   ret void, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !137 {
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!116}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !63, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !49 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !50 = !{!"omnipotent char", !49, i64 0}
// CHECK:STDOUT: !51 = !{!"any pointer", !50, i64 0}
// CHECK:STDOUT: !57 = !{}
// CHECK:STDOUT: !58 = !{!"int", !50, i64 0}
// CHECK:STDOUT: !60 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !61 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !63 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !64 = !{!"__libc_errno", !58, i64 0}
// CHECK:STDOUT: !65 = !{!64, !58, i64 0}
// CHECK:STDOUT: !115 = !DIFile(filename: "class_to_int.carbon", directory: "")
// CHECK:STDOUT: !116 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !115, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !115, line: 11, type: !23, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !128)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !21)
// CHECK:STDOUT: !122 = !DILocation(line: 12, column: 3, scope: !119)
// CHECK:STDOUT: !125 = !{null, !20}
// CHECK:STDOUT: !126 = !DISubroutineType(types: !125)
// CHECK:STDOUT: !127 = !DILocation(line: 11, column: 1, scope: !119)
// CHECK:STDOUT: !128 = !{!120}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !115, line: 15, type: !23, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !136)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !21)
// CHECK:STDOUT: !132 = !DILocation(line: 16, column: 27, scope: !129)
// CHECK:STDOUT: !135 = !DILocation(line: 15, column: 1, scope: !129)
// CHECK:STDOUT: !136 = !{!130}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !126, spFlags: DISPFlagDefinition, unit: !116, retainedNodes: !143)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !20)
// CHECK:STDOUT: !142 = !DILocation(line: 32, column: 1, scope: !137)
// CHECK:STDOUT: !143 = !{!138}
// CHECK:STDOUT: !144 = !{!"p1 _ZTS4IntB", !51, i64 0}
// CHECK:STDOUT: !145 = !{!144, !144, i64 0}
// CHECK:STDOUT:
