; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s

; // Bitcode in this test case is reduced version of compiled code below:
;extern "C" {
;#line 1 "/source/dir/foo.h"
;__device__ void foo() {}
;#line 2 "/source/dir/bar.cu"
;__device__ void bar() {}
;}

; CHECK: .target sm_{{[0-9]+$}}

; CHECK: .visible .func foo()
; CHECK: .loc [[FOO:[0-9]+]] 1 31
; CHECK:  ret;
; CHECK: .visible .func bar()
; CHECK: .loc [[BAR:[0-9]+]] 2 31
; CHECK:  ret;

define void @foo() !dbg !4 {
bb:
  ret void, !dbg !10
}

define void @bar() !dbg !7 {
bb:
  ret void, !dbg !11
}

; CHECK-DAG: .file [[FOO]] "{{.*}}foo.h"
; CHECK-DAG: .file [[BAR]] "{{.*}}bar.cu"

; CHECK-NOT: .section .debug{{.*}}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: DebugDirectivesOnly, enums: !2)
!1 = !DIFile(filename: "bar.cu", directory: "/source/dir")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !5, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DIFile(filename: "foo.h", directory: "/source/dir")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DILocation(line: 1, column: 31, scope: !4)
!11 = !DILocation(line: 2, column: 31, scope: !7)
