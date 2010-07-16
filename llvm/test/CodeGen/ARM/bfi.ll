; RUN: llc -march=arm -mattr=+v6t2 < %s | FileCheck %s

%struct.F = type { [3 x i8], i8 }

@X = common global %struct.F zeroinitializer, align 4 ; <%struct.F*> [#uses=1]

define void @f1([1 x i32] %f.coerce0) nounwind {
entry:
; CHECK: f1
; CHECK: mov r2, #10
; CHECK: bfi r1, r2, #22, #4
  %0 = load i32* bitcast (%struct.F* @X to i32*), align 4 ; <i32> [#uses=1]
  %1 = and i32 %0, -62914561                      ; <i32> [#uses=1]
  %2 = or i32 %1, 41943040                        ; <i32> [#uses=1]
  store i32 %2, i32* bitcast (%struct.F* @X to i32*), align 4
  ret void
}
