; RUN: opt -verify-memoryssa -loop-rotate %s -S | FileCheck %s
; REQUIRES: asserts

; CHECK-LABEL: @test()
define dso_local void @test() {
entry:
  br label %preheader

preheader:
  br label %l39

l39:
  %v40 = phi float (float)* [ @foo, %preheader ], [ %v43, %crit_edge ]
  %v41 = call float %v40(float undef)
  %v42 = load i32, i32* undef, align 8
  br i1 undef, label %crit_edge, label %loopexit

crit_edge:
  %v43 = load float (float)*, float (float)** undef, align 8
  br label %l39

loopexit:
  unreachable
}

; Function Attrs: readnone
declare dso_local float @foo(float) #0 align 32

attributes #0 = { readnone }
