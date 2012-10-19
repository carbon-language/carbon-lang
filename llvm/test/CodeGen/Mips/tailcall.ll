; RUN: llc -march=mipsel -relocation-model=pic -enable-mips-tail-calls < %s | \
; RUN: FileCheck %s -check-prefix=PIC32
; RUN: llc -march=mipsel -relocation-model=static \
; RUN: -enable-mips-tail-calls < %s | FileCheck %s -check-prefix=STATIC32
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=+n64 -enable-mips-tail-calls \
; RUN: < %s | FileCheck %s -check-prefix=N64

@g0 = common global i32 0, align 4
@g1 = common global i32 0, align 4
@g2 = common global i32 0, align 4
@g3 = common global i32 0, align 4
@g4 = common global i32 0, align 4
@g5 = common global i32 0, align 4
@g6 = common global i32 0, align 4
@g7 = common global i32 0, align 4
@g8 = common global i32 0, align 4
@g9 = common global i32 0, align 4

define i32 @caller1(i32 %a0) nounwind {
entry:
; PIC32-NOT: jalr
; STATIC32-NOT: jal
; N64-NOT: jalr

  %call = tail call i32 @callee1(i32 1, i32 1, i32 1, i32 %a0) nounwind
  ret i32 %call
}

declare i32 @callee1(i32, i32, i32, i32)

define i32 @caller2(i32 %a0, i32 %a1, i32 %a2, i32 %a3) nounwind {
entry:
; PIC32: jalr
; STATIC32: jal
; N64-NOT: jalr

  %call = tail call i32 @callee2(i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3) nounwind
  ret i32 %call
}

declare i32 @callee2(i32, i32, i32, i32, i32)

define i32 @caller3(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) nounwind {
entry:
; PIC32: jalr
; STATIC32: jal
; N64-NOT: jalr

  %call = tail call i32 @callee3(i32 1, i32 1, i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4) nounwind
  ret i32 %call
}

declare i32 @callee3(i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @caller4(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) nounwind {
entry:
; PIC32: jalr
; STATIC32: jal
; N64: jalr

  %call = tail call i32 @callee4(i32 1, i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7) nounwind
  ret i32 %call
}

declare i32 @callee4(i32, i32, i32, i32, i32, i32, i32, i32, i32)

define i32 @caller5() nounwind readonly {
entry:
; PIC32-NOT: jalr
; STATIC32-NOT: jal
; N64-NOT: jalr

  %0 = load i32* @g0, align 4
  %1 = load i32* @g1, align 4
  %2 = load i32* @g2, align 4
  %3 = load i32* @g3, align 4
  %4 = load i32* @g4, align 4
  %5 = load i32* @g5, align 4
  %6 = load i32* @g6, align 4
  %7 = load i32* @g7, align 4
  %8 = load i32* @g8, align 4
  %9 = load i32* @g9, align 4
  %call = tail call fastcc i32 @callee5(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9)
  ret i32 %call
}

define internal fastcc i32 @callee5(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8, i32 %a9) nounwind readnone noinline {
entry:
  %add = add nsw i32 %a1, %a0
  %add1 = add nsw i32 %add, %a2
  %add2 = add nsw i32 %add1, %a3
  %add3 = add nsw i32 %add2, %a4
  %add4 = add nsw i32 %add3, %a5
  %add5 = add nsw i32 %add4, %a6
  %add6 = add nsw i32 %add5, %a7
  %add7 = add nsw i32 %add6, %a8
  %add8 = add nsw i32 %add7, %a9
  ret i32 %add8
}

