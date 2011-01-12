;RUN: llc -march=sparc < %s | FileCheck %s


define i8* @frameaddr() nounwind readnone {
entry:
;CHECK: frameaddr
;CHECK: or %g0, %fp, {{.+}}
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @frameaddr2() nounwind readnone {
entry:
;CHECK: frameaddr2
;CHECK: flushw
;CHECK: ld [%fp+56], {{.+}}
;CHECK: ld [{{.+}}+56], {{.+}}
;CHECK: ld [{{.+}}+56], {{.+}}
  %0 = tail call i8* @llvm.frameaddress(i32 3)
  ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone



define i8* @retaddr() nounwind readnone {
entry:
;CHECK: retaddr
;CHECK: or %g0, %i7, {{.+}}
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @retaddr2() nounwind readnone {
entry:
;CHECK: retaddr2
;CHECK: flushw
;CHECK: ld [%fp+56], {{.+}}
;CHECK: ld [{{.+}}+56], {{.+}}
;CHECK: ld [{{.+}}+60], {{.+}}
  %0 = tail call i8* @llvm.returnaddress(i32 3)
  ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
