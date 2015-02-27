; RUN: llc -march=sparc -disable-sparc-leaf-proc=0 < %s | FileCheck %s

; CHECK-LABEL:      func_nobody:
; CHECK:      retl
; CHECK-NEXT: nop
define void @func_nobody() {
entry:
  ret void
}


; CHECK-LABEL:      return_int_const:
; CHECK:      retl
; CHECK-NEXT: mov 1729, %o0
define i32 @return_int_const() {
entry:
  ret i32 1729
}

; CHECK-LABEL:      return_double_const:
; CHECK:      sethi
; CHECK:      retl
; CHECK-NEXT: ldd {{.*}}, %f0

define double @return_double_const() {
entry:
  ret double 0.000000e+00
}

; CHECK-LABEL:      leaf_proc_with_args:
; CHECK:      add {{%o[0-1]}}, {{%o[0-1]}}, [[R:%[go][0-7]]]
; CHECK:      retl
; CHECK-NEXT: add [[R]], %o2, %o0

define i32 @leaf_proc_with_args(i32 %a, i32 %b, i32 %c) {
entry:
  %0 = add nsw i32 %b, %a
  %1 = add nsw i32 %0, %c
  ret i32 %1
}

; CHECK-LABEL:     leaf_proc_with_args_in_stack:
; CHECK-DAG: ld [%sp+92], {{%[go][0-7]}}
; CHECK-DAG: ld [%sp+96], {{%[go][0-7]}}
; CHECK:     retl
; CHECK-NEXT: add {{.*}}, %o0
define i32 @leaf_proc_with_args_in_stack(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f, i32 %g, i32 %h) {
entry:
  %0 = add nsw i32 %b, %a
  %1 = add nsw i32 %0, %c
  %2 = add nsw i32 %1, %d
  %3 = add nsw i32 %2, %e
  %4 = add nsw i32 %3, %f
  %5 = add nsw i32 %4, %g
  %6 = add nsw i32 %5, %h
  ret i32 %6
}

; CHECK-LABEL:      leaf_proc_with_local_array:
; CHECK:      add %sp, -104, %sp
; CHECK:      mov 1, [[R1:%[go][0-7]]]
; CHECK:      st [[R1]], [%sp+96]
; CHECK:      mov 2, [[R2:%[go][0-7]]]
; CHECK:      st [[R2]], [%sp+100]
; CHECK:      ld {{.+}}, %o0
; CHECK:      retl
; CHECK-NEXT: add %sp, 104, %sp

define i32 @leaf_proc_with_local_array(i32 %a, i32 %b, i32 %c) {
entry:
  %array = alloca [2 x i32], align 4
  %0 = sub nsw i32 %b, %c
  %1 = getelementptr inbounds [2 x i32], [2 x i32]* %array, i32 0, i32 0
  store i32 1, i32* %1, align 4
  %2 = getelementptr inbounds [2 x i32], [2 x i32]* %array, i32 0, i32 1
  store i32 2, i32* %2, align 4
  %3 = getelementptr inbounds [2 x i32], [2 x i32]* %array, i32 0, i32 %a
  %4 = load i32, i32* %3, align 4
  ret i32 %4
}
