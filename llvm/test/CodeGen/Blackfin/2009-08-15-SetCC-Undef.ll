; RUN: llc < %s -march=bfin -verify-machineinstrs
; XFAIL: *

; An undef argument causes a setugt node to escape instruction selection.

define void @bugt() {
cond_next305:
  %tmp306307 = trunc i32 undef to i8              ; <i8> [#uses=1]
  %tmp308 = icmp ugt i8 %tmp306307, 6             ; <i1> [#uses=1]
  br i1 %tmp308, label %bb311, label %bb314

bb311:                                            ; preds = %cond_next305
  unreachable

bb314:                                            ; preds = %cond_next305
  ret void
}
