// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/inherit.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/inherit.carbon

// --- generic_class_inherit_from_cpp.carbon
library "[[@TEST_NAME]]";
import Cpp;

inline Cpp '''
struct Base {
  virtual void f() {}
  int field;
};
''';

class C(T: type) {
  extend base: Cpp.Base;
}

fn Run() -> i32 {
  var c: C(i32) = {
   .base = Cpp.Base.Base(),
  };
  c.f();
  return c.field;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'generic_class_inherit_from_cpp.carbon'
// CHECK:STDOUT: source_filename = "generic_class_inherit_from_cpp.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN4Base1fEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN4BaseC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTV4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTVN6Carbon4Main1CIiEE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTIN6Carbon4Main1CIiEE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTSN6Carbon4Main1CIiEE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV4Base = linkonce_odr dso_local constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN4Base1fEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTVN6Carbon4Main1CIiEE = linkonce_odr dso_local constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTIN6Carbon4Main1CIiEE, ptr @_ZN4Base1fEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTI4Base = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS4Base = linkonce_odr dso_local constant [6 x i8] c"4Base\00", comdat, align 1
// CHECK:STDOUT: @_ZTIN6Carbon4Main1CIiEE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 16), ptr @_ZTSN6Carbon4Main1CIiEE, ptr @_ZTI4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon4Main1CIiEE = linkonce_odr dso_local constant [20 x i8] c"N6Carbon4Main1CIiEE\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN4Base1fEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #0 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #1 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca { [16 x i8] }, align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !29
// CHECK:STDOUT:   %.loc18_3.1.base = getelementptr inbounds nuw { [16 x i8] }, ptr %c.var, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   call void @_ZN4BaseC1Ev.carbon_thunk.(ptr %.loc18_3.1.base), !dbg !31
// CHECK:STDOUT:   %.loc18_3.7.base = getelementptr inbounds nuw { [16 x i8] }, ptr %c.var, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   %.loc18_3.8.vptr = getelementptr inbounds nuw [16 x i8], ptr %.loc18_3.7.base, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) (i8, ptr @_ZTVN6Carbon4Main1CIiEE, i64 16), ptr %.loc18_3.8.vptr, align 8, !dbg !30
// CHECK:STDOUT:   %.loc19_3.1.base = getelementptr inbounds nuw { [16 x i8] }, ptr %c.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %Base.f.call.vtable = load ptr, ptr %.loc19_3.1.base, align 8, !dbg !33
// CHECK:STDOUT:   %Base.f.call = getelementptr ptr, ptr %Base.f.call.vtable, i32 0, !dbg !33
// CHECK:STDOUT:   %Base.f.call.memptr.virtualfn = load ptr, ptr %Base.f.call, align 8, !dbg !33
// CHECK:STDOUT:   call void %Base.f.call.memptr.virtualfn(ptr %.loc19_3.1.base), !dbg !33
// CHECK:STDOUT:   %.loc20_11.1.base = getelementptr inbounds nuw { [16 x i8] }, ptr %c.var, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   %.loc20_11.3.field = getelementptr inbounds nuw [16 x i8], ptr %.loc20_11.1.base, i32 0, i32 8, !dbg !34
// CHECK:STDOUT:   %.loc20_11.4 = load i32, ptr %.loc20_11.3.field, align 4, !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3dc4bd5f34a0e76b(ptr %c.var), !dbg !29
// CHECK:STDOUT:   ret i32 %.loc20_11.4, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN4BaseC1Ev.carbon_thunk.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   call void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3dc4bd5f34a0e76b(ptr %t) #1 !dbg !36 {
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) (i8, ptr @_ZTV4Base, i64 16), ptr %this1, align 8, !tbaa !49
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN4Base1fEv, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!50, !51, !52, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!55}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "generic_class_inherit_from_cpp.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{!20}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 19, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 17, column: 12, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 19, column: 3, scope: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 20, column: 10, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 20, column: 3, scope: !26)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3dc4bd5f34a0e76b", scope: null, file: !10, line: 32, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !21)
// CHECK:STDOUT: !41 = !DILocation(line: 32, column: 1, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"p1 _ZTS4Base", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !48 = !{!"vtable pointer", !43, i64 0}
// CHECK:STDOUT: !49 = !{!48, !48, i64 0}
// CHECK:STDOUT: !50 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !51 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !52 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !53 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !54 = !{!"__libc_errno", !53, i64 0}
// CHECK:STDOUT: !55 = !{!54, !53, i64 0}
// CHECK:STDOUT:
