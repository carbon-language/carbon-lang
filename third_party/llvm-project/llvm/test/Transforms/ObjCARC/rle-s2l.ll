; RUN: opt -S -basic-aa -objc-arc < %s | FileCheck %s

declare i8* @llvm.objc.loadWeak(i8**)
declare i8* @llvm.objc.loadWeakRetained(i8**)
declare i8* @llvm.objc.storeWeak(i8**, i8*)
declare i8* @llvm.objc.initWeak(i8**, i8*)
declare void @use_pointer(i8*)
declare void @callee()

; Basic redundant @llvm.objc.loadWeak elimination.

; CHECK:      define void @test0(i8** %p) {
; CHECK-NEXT:   %y = call i8* @llvm.objc.loadWeak(i8** %p)
; CHECK-NEXT:   call void @use_pointer(i8* %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test0(i8** %p) {
  %x = call i8* @llvm.objc.loadWeak(i8** %p)
  %y = call i8* @llvm.objc.loadWeak(i8** %p)
  call void @use_pointer(i8* %y)
  ret void
}

; DCE the @llvm.objc.loadWeak.

; CHECK:      define void @test1(i8** %p) {
; CHECK-NEXT:   %y = call i8* @llvm.objc.loadWeakRetained(i8** %p)
; CHECK-NEXT:   call void @use_pointer(i8* %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test1(i8** %p) {
  %x = call i8* @llvm.objc.loadWeak(i8** %p)
  %y = call i8* @llvm.objc.loadWeakRetained(i8** %p)
  call void @use_pointer(i8* %y)
  ret void
}

; Basic redundant @llvm.objc.loadWeakRetained elimination.

; CHECK:      define void @test2(i8** %p) {
; CHECK-NEXT:   %x = call i8* @llvm.objc.loadWeak(i8** %p)
; CHECK-NEXT:   store i8 3, i8* %x
; CHECK-NEXT:   %1 = tail call i8* @llvm.objc.retain(i8* %x)
; CHECK-NEXT:   call void @use_pointer(i8* %x)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test2(i8** %p) {
  %x = call i8* @llvm.objc.loadWeak(i8** %p)
  store i8 3, i8* %x
  %y = call i8* @llvm.objc.loadWeakRetained(i8** %p)
  call void @use_pointer(i8* %y)
  ret void
}

; Basic redundant @llvm.objc.loadWeakRetained elimination, this time
; with a readonly call instead of a store.

; CHECK:      define void @test3(i8** %p) {
; CHECK-NEXT:   %x = call i8* @llvm.objc.loadWeak(i8** %p)
; CHECK-NEXT:   call void @use_pointer(i8* %x) [[RO:#[0-9]+]]
; CHECK-NEXT:   %1 = tail call i8* @llvm.objc.retain(i8* %x)
; CHECK-NEXT:   call void @use_pointer(i8* %x)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test3(i8** %p) {
  %x = call i8* @llvm.objc.loadWeak(i8** %p)
  call void @use_pointer(i8* %x) readonly
  %y = call i8* @llvm.objc.loadWeakRetained(i8** %p)
  call void @use_pointer(i8* %y)
  ret void
}

; A regular call blocks redundant weak load elimination.

; CHECK:      define void @test4(i8** %p) {
; CHECK-NEXT:   %x = call i8* @llvm.objc.loadWeak(i8** %p)
; CHECK-NEXT:   call void @use_pointer(i8* %x) [[RO]]
; CHECK-NEXT:   call void @callee()
; CHECK-NEXT:   %y = call i8* @llvm.objc.loadWeak(i8** %p)
; CHECK-NEXT:   call void @use_pointer(i8* %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test4(i8** %p) {
  %x = call i8* @llvm.objc.loadWeak(i8** %p)
  call void @use_pointer(i8* %x) readonly
  call void @callee()
  %y = call i8* @llvm.objc.loadWeak(i8** %p)
  call void @use_pointer(i8* %y)
  ret void
}

; Store to load forwarding.

; CHECK:      define void @test5(i8** %p, i8* %n) {
; CHECK-NEXT:   %1 = call i8* @llvm.objc.storeWeak(i8** %p, i8* %n)
; CHECK-NEXT:   call void @use_pointer(i8* %n)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test5(i8** %p, i8* %n) {
  call i8* @llvm.objc.storeWeak(i8** %p, i8* %n)
  %y = call i8* @llvm.objc.loadWeak(i8** %p)
  call void @use_pointer(i8* %y)
  ret void
}

; Store to load forwarding with objc_initWeak.

; CHECK:      define void @test6(i8** %p, i8* %n) {
; CHECK-NEXT:   %1 = call i8* @llvm.objc.initWeak(i8** %p, i8* %n)
; CHECK-NEXT:   call void @use_pointer(i8* %n)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test6(i8** %p, i8* %n) {
  call i8* @llvm.objc.initWeak(i8** %p, i8* %n)
  %y = call i8* @llvm.objc.loadWeak(i8** %p)
  call void @use_pointer(i8* %y)
  ret void
}

; Don't forward if there's a may-alias store in the way.

; CHECK:      define void @test7(i8** %p, i8* %n, i8** %q, i8* %m) {
; CHECK-NEXT:   call i8* @llvm.objc.initWeak(i8** %p, i8* %n)
; CHECK-NEXT:   call i8* @llvm.objc.storeWeak(i8** %q, i8* %m)
; CHECK-NEXT:   %y = call i8* @llvm.objc.loadWeak(i8** %p)
; CHECK-NEXT:   call void @use_pointer(i8* %y)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test7(i8** %p, i8* %n, i8** %q, i8* %m) {
  call i8* @llvm.objc.initWeak(i8** %p, i8* %n)
  call i8* @llvm.objc.storeWeak(i8** %q, i8* %m)
  %y = call i8* @llvm.objc.loadWeak(i8** %p)
  call void @use_pointer(i8* %y)
  ret void
}

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes [[RO]] = { readonly }
