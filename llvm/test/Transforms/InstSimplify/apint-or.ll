; RUN: opt < %s -instsimplify -S | FileCheck %s

; Test the case where integer BitWidth <= 64 && BitWidth % 2 != 0.
define i39 @test1(i39 %V, i39 %M) {
    ;; If we have: ((V + N) & C1) | (V & C2)
    ;; .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
    ;; replace with V+N.
    %C1 = xor i39 274877906943, -1 ;; C2 = 274877906943
    %N = and i39 %M, 274877906944
    %A = add i39 %V, %N
    %B = and i39 %A, %C1
    %D = and i39 %V, 274877906943
    %R = or i39 %B, %D
    ret i39 %R
; CHECK-LABEL: @test1
; CHECK-NEXT: and {{.*}}, -274877906944
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

define i7 @test2(i7 %X) {
    %Y = or i7 %X, 0
    ret i7 %Y
; CHECK-LABEL: @test2
; CHECK-NEXT: ret i7 %X
}

define i17 @test3(i17 %X) {
    %Y = or i17 %X, -1
    ret i17 %Y
; CHECK-LABEL: @test3
; CHECK-NEXT: ret i17 -1
}

; Test the case where Integer BitWidth > 64 && BitWidth <= 1024. 
define i399 @test4(i399 %V, i399 %M) {
    ;; If we have: ((V + N) & C1) | (V & C2)
    ;; .. and C2 = ~C1 and C2 is 0+1+ and (N & C2) == 0
    ;; replace with V+N.
    %C1 = xor i399 274877906943, -1 ;; C2 = 274877906943
    %N = and i399 %M, 18446742974197923840
    %A = add i399 %V, %N
    %B = and i399 %A, %C1
    %D = and i399 %V, 274877906943
    %R = or i399 %B, %D
    ret i399 %R
; CHECK-LABEL: @test4
; CHECK-NEXT: and {{.*}}, 18446742974197923840
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

define i777 @test5(i777 %X) {
    %Y = or i777 %X, 0
    ret i777 %Y
; CHECK-LABEL: @test5
; CHECK-NEXT: ret i777 %X
}

define i117 @test6(i117 %X) {
    %Y = or i117 %X, -1
    ret i117 %Y
; CHECK-LABEL: @test6
; CHECK-NEXT: ret i117 -1
}
