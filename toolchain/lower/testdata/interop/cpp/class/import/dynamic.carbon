// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/dynamic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/dynamic.carbon

// --- dynamic_base_from_cpp.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''

struct FurtherBase {
  virtual auto further_base_func() -> void;
};

struct Base: FurtherBase {
  int i;
  virtual auto func() & -> void;
};

__attribute__((optnone)) auto Use(Base& b) {
}

''';

base class Derived {
  extend base: Cpp.Base;
  virtual fn other_func(unused self) {
  }
  override fn func(unused ref self) {
  }
}

fn DoThing() {
  var d: Derived = {.base = Cpp.Base.Base()};
  Cpp.Use(ref d);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'dynamic_base_from_cpp.carbon'
// CHECK:STDOUT: source_filename = "dynamic_base_from_cpp.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN4BaseC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN11FurtherBaseC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTVN6Carbon7DerivedE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTIN6Carbon7DerivedE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTSN6Carbon7DerivedE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV4Base = available_externally constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN11FurtherBase17further_base_funcEv, ptr @_ZNR4Base4funcEv] }, align 8
// CHECK:STDOUT: @_ZTV11FurtherBase = available_externally constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI11FurtherBase, ptr @_ZN11FurtherBase17further_base_funcEv] }, align 8
// CHECK:STDOUT: @_ZTVN6Carbon7DerivedE = linkonce_odr dso_local constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTIN6Carbon7DerivedE, ptr @_ZN11FurtherBase17further_base_funcEv, ptr @_ZNR6Carbon7Derived4funcEv, ptr @_ZNK6Carbon7Derived10other_funcEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTI4Base = external constant ptr
// CHECK:STDOUT: @_ZTI11FurtherBase = external constant ptr
// CHECK:STDOUT: @_ZTIN6Carbon7DerivedE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 16), ptr @_ZTSN6Carbon7DerivedE, ptr @_ZTI4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon7DerivedE = linkonce_odr dso_local constant [18 x i8] c"N6Carbon7DerivedE\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone uwtable
// CHECK:STDOUT: define dso_local void @_Z3UseR4Base(ptr noundef nonnull align 8 dereferenceable(12) %b) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %b, ptr %b.addr, align 8, !tbaa !4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN11FurtherBase17further_base_funcEv(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNR4Base4funcEv(ptr noundef nonnull align 8 dereferenceable(12)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cother_func.Derived.Main(ptr %self) #2 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cfunc.Derived.Main(ptr %self) #2 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_Cfunc:thunk:Base.Cpp:Derived.Main"(ptr %self) #3 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cfunc.Derived.Main(ptr %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.Derived.Main"(ptr %self) #3 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.352b8f386f44a614:core.Destroy.Core"(ptr %self) #2 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.0a0b9df20a60c6f6:core.Destroy.Core"(ptr %self) #2 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22.base = getelementptr inbounds nuw { [16 x i8] }, ptr %self, i32 0, i32 0, !dbg !72
// CHECK:STDOUT:   call void @"_CSelfDestruct.352b8f386f44a614:core.Destroy.Core"(ptr %.loc22.base), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.0a0b9df20a60c6f6:core.Destroy.Core"(ptr %self) #2 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.0a0b9df20a60c6f6:core.Destroy.Core"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #2 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22.base = getelementptr inbounds nuw { [16 x i8] }, ptr %self, i32 0, i32 0, !dbg !80
// CHECK:STDOUT:   call void @"_CSelfDestruct.352b8f386f44a614:core.Destroy.Core"(ptr %.loc22.base), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %self) #2 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core"(ptr %self), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cfunc__carbon_thunk.Derived.Main(ptr %self) #2 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cfunc.Derived.Main(ptr %self), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cother_func__carbon_thunk.Derived.Main(ptr %self) #2 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Derived.other_func.call.vtable = load ptr, ptr %self, align 8, !dbg !92
// CHECK:STDOUT:   %Derived.other_func.call = call ptr @llvm.load.relative.i32(ptr %Derived.other_func.call.vtable, i32 8), !dbg !92
// CHECK:STDOUT:   call void %Derived.other_func.call(ptr %self), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDoThing.Main() #2 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { [16 x i8] }, align 8, !dbg !95
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !95
// CHECK:STDOUT:   %.loc31_44.1.base = getelementptr inbounds nuw { [16 x i8] }, ptr %d.var, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   call void @_ZN4BaseC1Ev.carbon_thunk.(ptr %.loc31_44.1.base), !dbg !97
// CHECK:STDOUT:   %.loc31_44.7.base = getelementptr inbounds nuw { [16 x i8] }, ptr %d.var, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   %.loc31_44.8.base = getelementptr inbounds nuw [16 x i8], ptr %.loc31_44.7.base, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   %.loc31_44.9.vptr = getelementptr inbounds nuw [8 x i8], ptr %.loc31_44.8.base, i32 0, i32 0, !dbg !96
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) (i8, ptr @_ZTVN6Carbon7DerivedE, i64 16), ptr %.loc31_44.9.vptr, align 8, !dbg !96
// CHECK:STDOUT:   %.loc32_11.2.base = getelementptr inbounds nuw { [16 x i8] }, ptr %d.var, i32 0, i32 0, !dbg !100
// CHECK:STDOUT:   call void @_Z3UseR4Base(ptr %.loc32_11.2.base), !dbg !101
// CHECK:STDOUT:   call void @"_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core"(ptr %d.var), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN4BaseC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !4
// CHECK:STDOUT:   call void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) #2
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #6
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNR6Carbon7Derived4funcEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #4 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !108
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cfunc__carbon_thunk.Derived.Main(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNK6Carbon7Derived10other_funcEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #4 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !108
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cother_func__carbon_thunk.Derived.Main(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #7 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN11FurtherBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #2
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 16) (i8, ptr @_ZTV4Base, i64 16), ptr %this1, align 8, !tbaa !104
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11FurtherBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #7 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !106
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) (i8, ptr @_ZTV11FurtherBase, i64 16), ptr %this1, align 8, !tbaa !104
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN11FurtherBase17further_base_funcEv, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress noinline nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #4 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT: attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #7 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!40}
// CHECK:STDOUT: !llvm.module.flags = !{!113, !114, !115, !7, !8}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!117}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !3 = !{!"p1 _ZTS4Base", !2, i64 0}
// CHECK:STDOUT: !4 = !{!3, !3, i64 0}
// CHECK:STDOUT: !7 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !8 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !39 = !DIFile(filename: "dynamic_base_from_cpp.carbon", directory: "")
// CHECK:STDOUT: !40 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !39, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !43 = !{null, !42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !{null}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "other_func", linkageName: "_Cother_func.Derived.Main", scope: null, file: !39, line: 24, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !51)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !42)
// CHECK:STDOUT: !50 = !DILocation(line: 24, column: 3, scope: !47)
// CHECK:STDOUT: !51 = !{!48}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "func", linkageName: "_Cfunc.Derived.Main", scope: null, file: !39, line: 26, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !56)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !42)
// CHECK:STDOUT: !55 = !DILocation(line: 26, column: 3, scope: !52)
// CHECK:STDOUT: !56 = !{!53}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "func", linkageName: "_Cfunc:thunk:Base.Cpp:Derived.Main", scope: null, file: !39, line: 26, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !61)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !42)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 3, scope: !57)
// CHECK:STDOUT: !61 = !{!58}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Derived.Main", scope: null, file: !39, line: 22, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !42)
// CHECK:STDOUT: !64 = !DILocation(line: 22, column: 1, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.352b8f386f44a614:core.Destroy.Core", scope: null, file: !39, line: 22, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !42)
// CHECK:STDOUT: !68 = !DILocation(line: 22, column: 1, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.0a0b9df20a60c6f6:core.Destroy.Core", scope: null, file: !39, line: 22, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !42)
// CHECK:STDOUT: !72 = !DILocation(line: 22, column: 1, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.0a0b9df20a60c6f6:core.Destroy.Core", scope: null, file: !39, line: 22, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !77)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !42)
// CHECK:STDOUT: !76 = !DILocation(line: 22, column: 1, scope: !74)
// CHECK:STDOUT: !77 = !{!75}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !39, line: 22, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !42)
// CHECK:STDOUT: !80 = !DILocation(line: 22, column: 1, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b25525d21a9bd472:core.Destroy.Core", scope: null, file: !39, line: 22, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !42)
// CHECK:STDOUT: !84 = !DILocation(line: 22, column: 1, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "func__carbon_thunk", linkageName: "_Cfunc__carbon_thunk.Derived.Main", scope: null, file: !39, line: 26, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !89)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !42)
// CHECK:STDOUT: !88 = !DILocation(line: 26, column: 3, scope: !86)
// CHECK:STDOUT: !89 = !{!87}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "other_func__carbon_thunk", linkageName: "_Cother_func__carbon_thunk.Derived.Main", scope: null, file: !39, line: 24, type: !44, spFlags: DISPFlagDefinition, unit: !40, retainedNodes: !93)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !42)
// CHECK:STDOUT: !92 = !DILocation(line: 24, column: 3, scope: !90)
// CHECK:STDOUT: !93 = !{!91}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "DoThing", linkageName: "_CDoThing.Main", scope: null, file: !39, line: 30, type: !46, spFlags: DISPFlagDefinition, unit: !40)
// CHECK:STDOUT: !95 = !DILocation(line: 31, column: 3, scope: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 31, column: 20, scope: !94)
// CHECK:STDOUT: !97 = !DILocation(line: 31, column: 29, scope: !94)
// CHECK:STDOUT: !100 = !DILocation(line: 32, column: 11, scope: !94)
// CHECK:STDOUT: !101 = !DILocation(line: 32, column: 3, scope: !94)
// CHECK:STDOUT: !102 = !DILocation(line: 30, column: 1, scope: !94)
// CHECK:STDOUT: !103 = !{!"vtable pointer", !0, i64 0}
// CHECK:STDOUT: !104 = !{!103, !103, i64 0}
// CHECK:STDOUT: !105 = !{!"p1 _ZTS11FurtherBase", !2, i64 0}
// CHECK:STDOUT: !106 = !{!105, !105, i64 0}
// CHECK:STDOUT: !107 = !{!"p1 _ZTSN6Carbon7DerivedE", !2, i64 0}
// CHECK:STDOUT: !108 = !{!107, !107, i64 0}
// CHECK:STDOUT: !110 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !113 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !114 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !115 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !116 = !{!"__libc_errno", !110, i64 0}
// CHECK:STDOUT: !117 = !{!116, !110, i64 0}
// CHECK:STDOUT:
