; RUN: opt < %s -instcombine -S | FileCheck %s

;CHECK: @and_consts
;CHECK: and i32 %k, 12
;CHECK: icmp ne i32 %0, 12
;CHECK: ret
define i1 @and_consts(i32 %k, i32 %c1, i32 %c2) {
bb:
  %tmp1 = and i32 4, %k
  %tmp2 = icmp eq i32 %tmp1, 0
  %tmp5 = and i32 8, %k
  %tmp6 = icmp eq i32 %tmp5, 0
  %or = or i1 %tmp2, %tmp6
  ret i1 %or
}

;CHECK: @foo1_and
;CHECK:  shl i32 1, %c1
;CHECK-NEXT:  shl i32 1, %c2
;CHECK-NEXT:  or i32
;CHECK-NEXT:  and i32
;CHECK-NEXT:  icmp ne i32 %1, %0
;CHECK: ret
define i1 @foo1_and(i32 %k, i32 %c1, i32 %c2) {
bb:
  %tmp = shl i32 1, %c1
  %tmp4 = shl i32 1, %c2
  %tmp1 = and i32 %tmp, %k
  %tmp2 = icmp eq i32 %tmp1, 0
  %tmp5 = and i32 %tmp4, %k
  %tmp6 = icmp eq i32 %tmp5, 0
  %or = or i1 %tmp2, %tmp6
  ret i1 %or
}

