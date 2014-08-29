; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -verify-machineinstrs < %s

define void @test() {
  %sext = shl i64 undef, 32
  %1 = ashr exact i64 %sext, 32
  %2 = icmp sgt i64 undef, %1
  br i1 %2, label %3, label %.critedge1

; <label>:3                                       ; preds = %0
  %4 = getelementptr inbounds i32* undef, i64 %1
  %5 = load i32* %4, align 4
  br i1 undef, label %6, label %.critedge1

; <label>:6                                       ; preds = %3
  %7 = and i32 %5, 255
  %8 = icmp eq i32 %7, 255
  br i1 %8, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.lr.ph, %6
  br i1 undef, label %.lr.ph, label %.critedge1

._crit_edge:                                      ; preds = %6
  ret void

.critedge1:                                       ; preds = %.lr.ph, %3, %0
  ret void
}
