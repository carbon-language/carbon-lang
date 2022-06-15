; RUN: opt -function-specialization -force-function-specialization -func-specialization-max-iters=2 -inline -instcombine -S < %s | FileCheck %s --check-prefix=ITERS2
; RUN: opt -function-specialization -force-function-specialization -func-specialization-max-iters=3 -inline -instcombine -S < %s | FileCheck %s --check-prefix=ITERS3
; RUN: opt -function-specialization -force-function-specialization -func-specialization-max-iters=4 -inline -instcombine -S < %s | FileCheck %s --check-prefix=ITERS4

@low = internal constant i32 0, align 4
@high = internal constant i32 6, align 4

define internal void @recursiveFunc(i32* nocapture readonly %lo, i32 %step, i32* nocapture readonly %hi) {
  %lo.temp = alloca i32, align 4
  %hi.temp = alloca i32, align 4
  %lo.load = load i32, i32* %lo, align 4
  %hi.load = load i32, i32* %hi, align 4
  %cmp = icmp ne i32 %lo.load, %hi.load
  br i1 %cmp, label %block6, label %ret.block

block6:
  call void @print_val(i32 %lo.load, i32 %hi.load)
  %add = add nsw i32 %lo.load, %step
  %sub = sub nsw i32 %hi.load, %step
  store i32 %add, i32* %lo.temp, align 4
  store i32 %sub, i32* %hi.temp, align 4
  call void @recursiveFunc(i32* nonnull %lo.temp, i32 %step, i32* nonnull %hi.temp)
  br label %ret.block

ret.block:
  ret void
}

; ITERS2:  @funcspec.arg.4 = internal constant i32 2
; ITERS2:  @funcspec.arg.5 = internal constant i32 4

; ITERS3:  @funcspec.arg.7 = internal constant i32 3
; ITERS3:  @funcspec.arg.8 = internal constant i32 3

define i32 @main() {
; ITERS2-LABEL: @main(
; ITERS2-NEXT:    call void @print_val(i32 0, i32 6)
; ITERS2-NEXT:    call void @print_val(i32 1, i32 5)
; ITERS2-NEXT:    call void @recursiveFunc(i32* nonnull @funcspec.arg.4, i32 1, i32* nonnull @funcspec.arg.5)
; ITERS2-NEXT:    ret i32 0
;
; ITERS3-LABEL: @main(
; ITERS3-NEXT:    call void @print_val(i32 0, i32 6)
; ITERS3-NEXT:    call void @print_val(i32 1, i32 5)
; ITERS3-NEXT:    call void @print_val(i32 2, i32 4)
; ITERS3-NEXT:    call void @recursiveFunc(i32* nonnull @funcspec.arg.7, i32 1, i32* nonnull @funcspec.arg.8)
; ITERS3-NEXT:    ret i32 0
;
; ITERS4-LABEL: @main(
; ITERS4-NEXT:    call void @print_val(i32 0, i32 6)
; ITERS4-NEXT:    call void @print_val(i32 1, i32 5)
; ITERS4-NEXT:    call void @print_val(i32 2, i32 4)
; ITERS4-NEXT:    ret i32 0
;
  call void @recursiveFunc(i32* nonnull @low, i32 1, i32* nonnull @high)
  ret i32 0
}

declare dso_local void @print_val(i32, i32)
