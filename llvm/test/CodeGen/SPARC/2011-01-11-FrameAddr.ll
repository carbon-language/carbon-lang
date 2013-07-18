;RUN: llc -march=sparc < %s | FileCheck %s -check-prefix=V8
;RUN: llc -march=sparc -mattr=v9 < %s | FileCheck %s -check-prefix=V9
;RUN: llc -march=sparc -regalloc=basic < %s | FileCheck %s -check-prefix=V8
;RUN: llc -march=sparc -regalloc=basic -mattr=v9 < %s | FileCheck %s -check-prefix=V9


define i8* @frameaddr() nounwind readnone {
entry:
;V8-LABEL: frameaddr:
;V8: save %sp, -96, %sp
;V8: jmp %i7+8
;V8: restore %g0, %fp, %o0

;V9-LABEL: frameaddr:
;V9: save %sp, -96, %sp
;V9: jmp %i7+8
;V9: restore %g0, %fp, %o0
  %0 = tail call i8* @llvm.frameaddress(i32 0)
  ret i8* %0
}

define i8* @frameaddr2() nounwind readnone {
entry:
;V8-LABEL: frameaddr2:
;V8: ta 3
;V8: ld [%fp+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}

;V9-LABEL: frameaddr2:
;V9: flushw
;V9: ld [%fp+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}
  %0 = tail call i8* @llvm.frameaddress(i32 3)
  ret i8* %0
}

declare i8* @llvm.frameaddress(i32) nounwind readnone



define i8* @retaddr() nounwind readnone {
entry:
;V8-LABEL: retaddr:
;V8: or %g0, %o7, {{.+}}

;V9-LABEL: retaddr:
;V9: or %g0, %o7, {{.+}}

  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @retaddr2() nounwind readnone {
entry:
;V8-LABEL: retaddr2:
;V8: ta 3
;V8: ld [%fp+56], {{.+}}
;V8: ld [{{.+}}+56], {{.+}}
;V8: ld [{{.+}}+60], {{.+}}

;V9-LABEL: retaddr2:
;V9: flushw
;V9: ld [%fp+56], {{.+}}
;V9: ld [{{.+}}+56], {{.+}}
;V9: ld [{{.+}}+60], {{.+}}

;V8LEAF-LABEL: retaddr2:
;V8LEAF: ta 3
;V8LEAF: ld [%fp+56], %[[R:[goli][0-7]]]
;V8LEAF: ld [%[[R]]+56], %[[R1:[goli][0-7]]]
;V8LEAF: ld [%[[R1]]+60], {{.+}}

;V9LEAF-LABEL: retaddr2:
;V9LEAF: flushw
;V9LEAF: ld [%fp+56], %[[R:[goli][0-7]]]
;V9LEAF: ld [%[[R]]+56], %[[R1:[goli][0-7]]]
;V9LEAF: ld [%[[R1]]+60], {{.+}}

  %0 = tail call i8* @llvm.returnaddress(i32 3)
  ret i8* %0
}

declare i8* @llvm.returnaddress(i32) nounwind readnone
