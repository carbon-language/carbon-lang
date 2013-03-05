; RUN: llc -march=mips64el -mcpu=mips64r2 < %s | FileCheck %s

@gld0 = external global fp128
@gld1 = external global fp128

; CHECK: foo0
; CHECK: sdc1  $f13, 8(${{[0-9]+}})
; CHECK: sdc1  $f12, 0(${{[0-9]+}})

define void @foo0(fp128 %a0) {
entry:
  store fp128 %a0, fp128* @gld0, align 16
  ret void
}

; CHECK: foo1
; CHECK: ldc1  $f13, 8(${{[0-9]+}})
; CHECK: ldc1  $f12, 0(${{[0-9]+}})

define void @foo1() {
entry:
  %0 = load fp128* @gld0, align 16
  tail call void @foo2(fp128 %0)
  ret void
}

declare void @foo2(fp128)

; CHECK: ld   $[[R0:[0-9]+]], %got_disp(gld0)
; CHECK: sdc1 $f2, 8($[[R0]])
; CHECK: sdc1 $f0, 0($[[R0]])
; CHECK: ld   $[[R1:[0-9]+]], %got_disp(gld1)
; CHECK: ldc1 $f0, 0($[[R1]])
; CHECK: ldc1 $f2, 8($[[R1]])

define fp128 @foo3() {
entry:
  %call = tail call fp128 @foo4()
  store fp128 %call, fp128* @gld0, align 16
  %0 = load fp128* @gld1, align 16
  ret fp128 %0
}

declare fp128 @foo4()
