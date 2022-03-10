; RUN: llc -mtriple powerpc-ibm-aix-xcoff -stop-after=machine-cp < %s | \
; RUN: FileCheck --check-prefix=32BIT %s

; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -stop-after=machine-cp < %s | \
; RUN: FileCheck --check-prefix=64BIT %s

; RUN: llc -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

declare void @foo(...)

define void @test_call() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: BL_NOP <mcsymbol .foo[PR]>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: BL8_NOP <mcsymbol .foo[PR]>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; CHECK-LABEL: test_call
; CHECK: bl .foo
; CHECK-NEXT: nop

  call void bitcast (void (...)* @foo to void ()*)()
  ret void
}

define hidden void @foo_local() {
entry:
  ret void
}

define void @test_local_call() {
entry:
; 32BIT: ADJCALLSTACKDOWN 56, 0, implicit-def dead $r1, implicit $r1
; 32BIT: BL <mcsymbol .foo_local>, csr_aix32, implicit-def dead $lr, implicit $rm, implicit $r2, implicit-def $r1
; 32BIT: ADJCALLSTACKUP 56, 0, implicit-def dead $r1, implicit $r1

; 64BIT: ADJCALLSTACKDOWN 112, 0, implicit-def dead $r1, implicit $r1
; 64BIT: BL8 <mcsymbol .foo_local>, csr_ppc64, implicit-def dead $lr8, implicit $rm, implicit $x2, implicit-def $r1
; 64BIT: ADJCALLSTACKUP 112, 0, implicit-def dead $r1, implicit $r1

; CHECK-LABEL: test_local_call
; CHECK: bl .foo_local
; CHECK-NOT: nop

  call void @foo_local()
  ret void
}
