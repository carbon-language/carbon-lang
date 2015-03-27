; RUN: llc < %s -march=bpf -show-mc-encoding | FileCheck %s

; Function Attrs: nounwind uwtable
define i32 @ld_b(i64 %foo, i64* nocapture %bar, i8* %ctx, i8* %ctx2) #0 {
  %1 = tail call i64 @llvm.bpf.load.byte(i8* %ctx, i64 123) #2
  %2 = add i64 %1, %foo
  %3 = load volatile i64, i64* %bar, align 8
  %4 = add i64 %2, %3
  %5 = tail call i64 @llvm.bpf.load.byte(i8* %ctx2, i64 %foo) #2
  %6 = add i64 %4, %5
  %7 = load volatile i64, i64* %bar, align 8
  %8 = add i64 %6, %7
  %9 = trunc i64 %8 to i32
  ret i32 %9
; CHECK-LABEL: ld_b:
; CHECK: ldabs_b r0, r6.data + 123
; CHECK: ldind_b r0, r6.data
}

declare i64 @llvm.bpf.load.byte(i8*, i64) #1

; Function Attrs: nounwind uwtable
define i32 @ld_h(i8* %ctx, i8* %ctx2, i32 %foo) #0 {
  %1 = tail call i64 @llvm.bpf.load.half(i8* %ctx, i64 123) #2
  %2 = sext i32 %foo to i64
  %3 = tail call i64 @llvm.bpf.load.half(i8* %ctx2, i64 %2) #2
  %4 = add i64 %3, %1
  %5 = trunc i64 %4 to i32
  ret i32 %5
; CHECK-LABEL: ld_h:
; CHECK: ldind_h r0, r6.data
; CHECK: ldabs_h r0, r6.data + 123
}

declare i64 @llvm.bpf.load.half(i8*, i64) #1

; Function Attrs: nounwind uwtable
define i32 @ld_w(i8* %ctx, i8* %ctx2, i32 %foo) #0 {
  %1 = tail call i64 @llvm.bpf.load.word(i8* %ctx, i64 123) #2
  %2 = sext i32 %foo to i64
  %3 = tail call i64 @llvm.bpf.load.word(i8* %ctx2, i64 %2) #2
  %4 = add i64 %3, %1
  %5 = trunc i64 %4 to i32
  ret i32 %5
; CHECK-LABEL: ld_w:
; CHECK: ldind_w r0, r6.data
; CHECK: ldabs_w r0, r6.data + 123
}

declare i64 @llvm.bpf.load.word(i8*, i64) #1

define i32 @ld_pseudo() #0 {
entry:
  %call = tail call i64 @llvm.bpf.pseudo(i64 2, i64 3)
  tail call void @bar(i64 %call, i32 4) #2
  ret i32 0
; CHECK-LABEL: ld_pseudo:
; CHECK: ld_pseudo r1, 2, 3 # encoding: [0x18,0x21,0x00,0x00,0x03,0x00
}

declare void @bar(i64, i32) #1

declare i64 @llvm.bpf.pseudo(i64, i64) #2
