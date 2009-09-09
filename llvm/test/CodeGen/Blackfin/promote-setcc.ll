; RUN: llc < %s -march=bfin > %t

; The DAG combiner may sometimes create illegal i16 SETCC operations when run
; after LegalizeOps. Try to tease out all the optimizations in
; TargetLowering::SimplifySetCC.

@x = external global i16
@y = external global i16

declare i16 @llvm.ctlz.i16(i16)

; Case (srl (ctlz x), 5) == const
; Note: ctlz is promoted, so this test does not catch the DAG combiner
define i1 @srl_ctlz_const() {
  %x = load i16* @x
  %c = call i16 @llvm.ctlz.i16(i16 %x)
  %s = lshr i16 %c, 4
  %r = icmp eq i16 %s, 1
  ret i1 %r
}

; Case (zext x) == const
define i1 @zext_const() {
  %x = load i16* @x
  %r = icmp ugt i16 %x, 1
  ret i1 %r
}

; Case (sext x) == const
define i1 @sext_const() {
  %x = load i16* @x
  %y = add i16 %x, 1
  %x2 = sext i16 %y to i32
  %r = icmp ne i32 %x2, -1
  ret i1 %r
}

