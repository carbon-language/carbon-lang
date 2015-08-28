; RUN: llvm-as -o %t.bc %s
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so -plugin-opt=save-temps \
; RUN:    -plugin-opt=O0 -r -o %t.o %t.bc
; RUN: llvm-dis < %t.o.opt.bc -o - | FileCheck --check-prefix=CHECK-O0 %s
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so -plugin-opt=save-temps \
; RUN:    -plugin-opt=O1 -r -o %t.o %t.bc
; RUN: llvm-dis < %t.o.opt.bc -o - | FileCheck --check-prefix=CHECK-O1 %s
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so -plugin-opt=save-temps \
; RUN:    -plugin-opt=O2 -r -o %t.o %t.bc
; RUN: llvm-dis < %t.o.opt.bc -o - | FileCheck --check-prefix=CHECK-O2 %s

; CHECK-O0: define internal void @foo(
; CHECK-O1: define internal void @foo(
; CHECK-O2-NOT: define internal void @foo(
define internal void @foo() {
  ret void
}

; CHECK-O0: define internal i32 @bar(
; CHECK-O1: define internal i32 @bar(
define internal i32 @bar(i1 %p) {
  br i1 %p, label %t, label %f

t:
  br label %end

f:
  br label %end

end:
  ; CHECK-O0: phi
  ; CHECK-O1: select
  %r = phi i32 [ 1, %t ], [ 2, %f ]
  ret i32 %r
}

define void @baz() {
  call void @foo()
  %c = call i32 @bar(i1 true)
  ret void
}

@a = constant i32 1

!0 = !{!"bitset1", i32* @a, i32 0}

; CHECK-O0-NOT: llvm.bitsets
; CHECK-O1-NOT: llvm.bitsets
; CHECK-O2-NOT: llvm.bitsets
!llvm.bitsets = !{ !0 }
