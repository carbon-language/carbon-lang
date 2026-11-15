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
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [4 x i8], align 4, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !31
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %a.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK1Bcv1AEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !48
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !50
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !50
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !48, !nonnull !52
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv(ptr dead_on_unwind writable sret(%struct.A) align 4 %0, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca [4 x i8], align 4, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !39
// CHECK:STDOUT:   call void @_ZNK1Bcv1AEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !42
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!55, !56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "class_to_class.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !20, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !35)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 3, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 1, scope: !28)
// CHECK:STDOUT: !35 = !{!29}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !20, line: 19, type: !25, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !43)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !23)
// CHECK:STDOUT: !39 = !DILocation(line: 20, column: 25, scope: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 19, column: 1, scope: !36)
// CHECK:STDOUT: !43 = !{!37}
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"any pointer", !45, i64 0}
// CHECK:STDOUT: !47 = !{!"p1 _ZTS1B", !46, i64 0}
// CHECK:STDOUT: !48 = !{!47, !47, i64 0}
// CHECK:STDOUT: !49 = !{!"p1 _ZTS1A", !46, i64 0}
// CHECK:STDOUT: !50 = !{!49, !49, i64 0}
// CHECK:STDOUT: !52 = !{}
// CHECK:STDOUT: !53 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !55 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !53, i64 0}
// CHECK:STDOUT: !60 = !{!59, !53, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_enum.carbon'
// CHECK:STDOUT: source_filename = "class_to_enum.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca i16, align 2, !dbg !67
// CHECK:STDOUT:   %i.var = alloca i16, align 2, !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !67
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %s.var), !dbg !67
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %i.var), !dbg !70
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %i.var), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !87
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !88
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !88
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !87, !nonnull !52
// CHECK:STDOUT:   %call = call noundef i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !91
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !87
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !88
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !88
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !87, !nonnull !52
// CHECK:STDOUT:   %call = call noundef i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !93
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_25.1.temp = alloca i16, align 2, !dbg !78
// CHECK:STDOUT:   %.loc21_25.1.temp = alloca i16, align 2, !dbg !81
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_25.1.temp), !dbg !78
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1SEv.carbon_thunk._(ptr %b, ptr %.loc20_25.1.temp), !dbg !78
// CHECK:STDOUT:   %.loc20_25.5 = load i16, ptr %.loc20_25.1.temp, align 2, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_25.1.temp), !dbg !81
// CHECK:STDOUT:   call void @_ZNK5EnumBcv1IEv.carbon_thunk._(ptr %b, ptr %.loc21_25.1.temp), !dbg !81
// CHECK:STDOUT:   %.loc21_25.5 = load i16, ptr %.loc21_25.1.temp, align 2, !dbg !81
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i16 @_ZNK5EnumBcv1SEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i16 @_ZNK5EnumBcv1IEv(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!62}
// CHECK:STDOUT: !llvm.module.flags = !{!55, !56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"any pointer", !45, i64 0}
// CHECK:STDOUT: !52 = !{}
// CHECK:STDOUT: !53 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !55 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !53, i64 0}
// CHECK:STDOUT: !60 = !{!59, !53, i64 0}
// CHECK:STDOUT: !61 = !DIFile(filename: "class_to_enum.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !61, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !61, line: 14, type: !25, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !74)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !23)
// CHECK:STDOUT: !67 = !DILocation(line: 15, column: 3, scope: !64)
// CHECK:STDOUT: !70 = !DILocation(line: 16, column: 3, scope: !64)
// CHECK:STDOUT: !73 = !DILocation(line: 14, column: 1, scope: !64)
// CHECK:STDOUT: !74 = !{!65}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !61, line: 19, type: !25, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !85)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !23)
// CHECK:STDOUT: !78 = !DILocation(line: 20, column: 25, scope: !75)
// CHECK:STDOUT: !81 = !DILocation(line: 21, column: 25, scope: !75)
// CHECK:STDOUT: !84 = !DILocation(line: 19, column: 1, scope: !75)
// CHECK:STDOUT: !85 = !{!76}
// CHECK:STDOUT: !86 = !{!"p1 _ZTS5EnumB", !46, i64 0}
// CHECK:STDOUT: !87 = !{!86, !86, i64 0}
// CHECK:STDOUT: !88 = !{!46, !46, i64 0}
// CHECK:STDOUT: !89 = !{!"_ZTS1S", !45, i64 0}
// CHECK:STDOUT: !91 = !{!89, !89, i64 0}
// CHECK:STDOUT: !92 = !{!"_ZTS1I", !45, i64 0}
// CHECK:STDOUT: !93 = !{!92, !92, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'class_to_int.carbon'
// CHECK:STDOUT: source_filename = "class_to_int.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToVar.Main(ptr %b) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca i32, align 4, !dbg !101
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !101
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !101
// CHECK:STDOUT:   store i32 %__carbon_thunk.call, ptr %a.var, align 4, !dbg !101
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !115
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !115, !nonnull !52
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK4IntBcviEv(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertToValue.Main(ptr %b) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK4IntBcviEv.carbon_thunk._(ptr %b), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !112
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!95}
// CHECK:STDOUT: !llvm.module.flags = !{!55, !56, !57, !58, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!60}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !44 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !45 = !{!"omnipotent char", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"any pointer", !45, i64 0}
// CHECK:STDOUT: !52 = !{}
// CHECK:STDOUT: !53 = !{!"int", !45, i64 0}
// CHECK:STDOUT: !55 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !56 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !57 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !58 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !59 = !{!"__libc_errno", !53, i64 0}
// CHECK:STDOUT: !60 = !{!59, !53, i64 0}
// CHECK:STDOUT: !94 = !DIFile(filename: "class_to_int.carbon", directory: "")
// CHECK:STDOUT: !95 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !94, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "ConvertToVar", linkageName: "_CConvertToVar.Main", scope: null, file: !94, line: 11, type: !25, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !105)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !23)
// CHECK:STDOUT: !101 = !DILocation(line: 12, column: 3, scope: !98)
// CHECK:STDOUT: !104 = !DILocation(line: 11, column: 1, scope: !98)
// CHECK:STDOUT: !105 = !{!99}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "ConvertToValue", linkageName: "_CConvertToValue.Main", scope: null, file: !94, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !113)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !23)
// CHECK:STDOUT: !109 = !DILocation(line: 16, column: 27, scope: !106)
// CHECK:STDOUT: !112 = !DILocation(line: 15, column: 1, scope: !106)
// CHECK:STDOUT: !113 = !{!107}
// CHECK:STDOUT: !114 = !{!"p1 _ZTS4IntB", !46, i64 0}
// CHECK:STDOUT: !115 = !{!114, !114, i64 0}
// CHECK:STDOUT:
