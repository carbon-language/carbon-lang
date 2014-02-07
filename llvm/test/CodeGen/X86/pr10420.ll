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

; CHECK: L_foo:                                   ## @foo

; CHECK: _bar:                                   ## @bar
; CHECK-NEXT: Ltmp2:

; CHECK: ## FDE CIE Offset
; CHECK-NEXT: .long
; CHECK-NEXT: Ltmp[[NUM1:[0-9]*]]:
; CHECK-NEXT: Ltmp[[NUM2:[0-9]*]] = L_foo-Ltmp[[NUM1]]   ## FDE initial location
; CHECK-NEXT: {{.quad|.long}}   Ltmp[[NUM2]]


; CHECK: ## FDE CIE Offset
; CHECK-NEXT: .long
; CHECK-NEXT: Ltmp[[NUM1:[0-9]*]]:
; CHECK-NEXT: Ltmp[[NUM2:[0-9]*]] = Ltmp2-Ltmp[[NUM1]]   ## FDE initial location
; CHECK-NEXT: {{.quad|.long}}   Ltmp[[NUM2]]


; OLD: L_foo:                                   ## @foo

; OLD: _bar:                                   ## @bar
; OLD-NEXT: Ltmp2:

; OLD: ## FDE CIE Offset
; OLD-NEXT: .long
; OLD-NEXT: Ltmp[[NUM1:[0-9]*]]:
; OLD-NEXT: {{.quad|.long}} L_foo-Ltmp[[NUM1]]          ## FDE initial location

; OLD: ## FDE CIE Offset
; OLD-NEXT: .long
; OLD-NEXT: Ltmp[[NUM1:[0-9]*]]:
; OLD-NEXT: {{.quad|.long}} Ltmp2-Ltmp[[NUM1]]          ## FDE initial location
