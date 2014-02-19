; RUN: llc < %s -mtriple=x86_64-apple-macosx10.7 -disable-cfi | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.6 -disable-cfi | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-macosx10.5 -disable-cfi | FileCheck --check-prefix=OLD %s
; RUN: llc < %s -mtriple=i686-apple-macosx10.6 -disable-cfi | FileCheck %s
; RUN: llc < %s -mtriple=i686-apple-macosx10.5 -disable-cfi | FileCheck --check-prefix=OLD %s
; RUN: llc < %s -mtriple=i686-apple-macosx10.4 -disable-cfi | FileCheck --check-prefix=OLD  %s

define private void @foo() {
       ret void
}

define void @bar() {
       call void @foo()
       ret void;
}

; CHECK: l_foo:                                   ## @foo
; CHECK-NEXT: Ltmp0:

; CHECK: _bar:                                   ## @bar
; CHECK-NEXT: Ltmp3:

; CHECK: ## FDE CIE Offset
; CHECK-NEXT: .long
; CHECK-NEXT: Ltmp[[NUM1:[0-9]*]]:
; CHECK-NEXT: Ltmp[[NUM2:[0-9]*]] = Ltmp0-Ltmp[[NUM1]]   ## FDE initial location
; CHECK-NEXT: {{.quad|.long}}   Ltmp[[NUM2]]


; CHECK: ## FDE CIE Offset
; CHECK-NEXT: .long
; CHECK-NEXT: Ltmp[[NUM1:[0-9]*]]:
; CHECK-NEXT: Ltmp[[NUM2:[0-9]*]] = Ltmp3-Ltmp[[NUM1]]   ## FDE initial location
; CHECK-NEXT: {{.quad|.long}}   Ltmp[[NUM2]]


; OLD: l_foo:                                   ## @foo
; OLD-NEXT: Ltmp0:

; OLD: _bar:                                   ## @bar
; OLD-NEXT: Ltmp3:

; OLD: ## FDE CIE Offset
; OLD-NEXT: .long
; OLD-NEXT: Ltmp[[NUM1:[0-9]*]]:
; OLD-NEXT: {{.quad|.long}} Ltmp0-Ltmp[[NUM1]]          ## FDE initial location

; OLD: ## FDE CIE Offset
; OLD-NEXT: .long
; OLD-NEXT: Ltmp[[NUM1:[0-9]*]]:
; OLD-NEXT: {{.quad|.long}} Ltmp3-Ltmp[[NUM1]]          ## FDE initial location
