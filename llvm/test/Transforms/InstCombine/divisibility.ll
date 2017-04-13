; Test that multiple divisibility checks are merged.

; RUN: opt < %s -instcombine -S | FileCheck %s

define i1 @test1(i32 %A) {
  %B = srem i32 %A, 2
  %C = srem i32 %A, 3
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test1(
; CHECK-NEXT: srem i32 %A, 6
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test2(i32 %A) {
  %B = urem i32 %A, 2
  %C = urem i32 %A, 3
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test2(
; CHECK-NEXT: urem i32 %A, 6
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test3(i32 %A) {
  %B = srem i32 %A, 2
  %C = urem i32 %A, 3
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test3(
; CHECK-NEXT: urem i32 %A, 6
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test4(i32 %A) {
  %B = urem i32 %A, 2
  %C = srem i32 %A, 3
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test4(
; CHECK-NEXT: srem i32 %A, 6
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test5(i32 %A) {
  %B = srem i32 %A, 8
  %C = srem i32 %A, 12
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test5(
; CHECK-NEXT: srem i32 %A, 24
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test6(i32 %A) {
  %B = and i32 %A, 6
  %C = srem i32 %A, 12
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test6(
; CHECK-NEXT: srem i32 %A, 24
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test7(i32 %A) {
  %B = and i32 %A, 8
  %C = srem i32 %A, 12
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test7(
; CHECK-NEXT: and i32 %A, 8
; CHECK-NEXT: srem i32 %A, 12
; CHECK-NEXT: or
; CHECK-NEXT: icmp
; CHECK-NEXT: ret i1
}

define i1 @test8(i32 %A, i32 %B) {
  %C = srem i32 %A, 2
  %D = srem i32 %B, 3
  %E = or i32 %C, %D
  %F = icmp eq i32 %E, 0
  ret i1 %F
; CHECK-LABEL: @test8(
; CHECK-NEXT: srem i32 %B, 3
; CHECK-NEXT: and i32 %A, 1
; CHECK-NEXT: or
; CHECK-NEXT: icmp
; CHECK-NEXT: ret i1
}

define i1 @test9(i32 %A) {
  %B = srem i32 %A, 7589
  %C = srem i32 %A, 395309
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test9(
; CHECK-NEXT: icmp eq i32 %A, 0
; CHECK-NEXT: ret i1 %E
}

define i1 @test10(i32 %A) {
  ; 7589 and 395309 are prime, and
  ; 7589 * 395309 == 3000000001 == -1294967295 (2^32)
  %B = urem i32 %A, 7589
  %C = urem i32 %A, 395309
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test10(
; CHECK-NEXT: urem i32 %A, -1294967295
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test11(i32 %A) {
  %B = urem i32 %A, 65535
  %C = urem i32 %A, 65537
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test11(
; CHECK-NEXT: urem i32 %A, -1
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test12(i32 %A) {
  %B = urem i32 %A, 65536
  %C = urem i32 %A, 65537
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test12(
; CHECK-NEXT: icmp eq i32 %A, 0
; CHECK-NEXT: ret i1
}

define i1 @test13(i32 %A) {
  %B = srem i32 %A, 65536
  %C = urem i32 %A, 65535
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test13(
; CHECK-NEXT: urem i32 %A, -65536
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test14(i32 %A) {
  %B = srem i32 %A, 95
  %C = srem i32 %A, 22605091
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test14(
; CHECK-NEXT: srem i32 %A, 2147483645
; CHECK-NEXT: icmp eq i32 %{{.*}}, 0
; CHECK-NEXT: ret i1
}

define i1 @test15(i32 %A) {
  %B = srem i32 %A, 97
  %C = srem i32 %A, 22605091
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  ret i1 %E
; CHECK-LABEL: @test15(
; CHECK-NEXT: icmp eq i32 %A, 0
; CHECK-NEXT: ret i1
}

define i32 @test16(i32 %A) {
  %B = srem i32 %A, 3
  %C = srem i32 %A, 5
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  %F = zext i1 %E to i32
  %G = add i32 %B, %F
  ret i32 %G
; CHECK-LABEL: @test16(
; CHECK-NEXT:  %B = srem i32 %A, 3
; CHECK-NEXT:  %[[REM:.*]] = srem i32 %A, 15
; CHECK-NEXT:  %E = icmp eq i32 %[[REM]], 0
; CHECK-NEXT:  %F = zext i1 %E to i32
; CHECK-NEXT:  %G = add i32 %B, %F
; CHECK-NEXT:  ret i32 %G
}

define i32 @test17(i32 %A) {
  %B = srem i32 %A, 3
  %C = srem i32 %A, 5
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  %F = zext i1 %E to i32
  %G = add i32 %B, %F
  %H = add i32 %C, %G
  ret i32 %H
; CHECK-LABEL: @test17(
; CHECK-NEXT:  %B = srem i32 %A, 3
; CHECK-NEXT:  %C = srem i32 %A, 5
; CHECK-NOT: srem
; CHECK: ret i32
}

define i32 @test18(i32 %A) {
  %B = srem i32 %A, 3
  %C = and i32 %A, 7
  %D = or i32 %B, %C
  %E = icmp eq i32 %D, 0
  %F = zext i1 %E to i32
  %G = add i32 %C, %F
  ret i32 %G
; CHECK-LABEL: @test18(
; CHECK-NEXT:  %C = and i32 %A, 7
; CHECK-NEXT:  %[[REM:.*]] = srem i32 %A, 24
; CHECK-NEXT:  %E = icmp eq i32 %[[REM]], 0
; CHECK-NEXT:  %F = zext i1 %E to i32
; CHECK-NEXT:  %G = add
; CHECK-NEXT:  ret i32 %G
}

define i1 @test19(i32 %A) {
  %B = srem i32 %A, 6
  %C = srem i32 %A, 10
  %D = icmp eq i32 %B, 0
  %E = icmp eq i32 %C, 0
  %F = and i1 %D, %E
  ret i1 %F
; CHECK-LABEL: @test19(
; CHECK-NEXT:  %[[REM:.*]] = srem i32 %A, 30
; CHECK-NEXT:  icmp eq i32 %[[REM]], 0
; CHECK-NEXT:  ret i1
}

define i1 @test20(i32 %A) {
  %B = and i32 %A, 1
  %C = srem i32 %A, 3
  %D = and i32 %A, 3
  %E = srem i32 %A, 5
  %F = srem i32 %A, 6
  %G = icmp eq i32 %B, 0
  %H = icmp eq i32 %C, 0
  %I = icmp eq i32 %D, 0
  %J = icmp eq i32 %E, 0
  %K = icmp eq i32 %F, 0
  %L = and i1 %G, %H
  %M = and i1 %L, %I
  %N = and i1 %M, %J
  %O = and i1 %N, %K
  ret i1 %O
; CHECK-LABEL: @test20(
; CHECK-NEXT:  srem i32 %A, 60
; CHECK-NEXT:  icmp eq i32
; CHECK-NEXT:  ret i1
}

define i1 @test21(i32 %A) {
  %B = srem i32 %A, -2147483648
  %C = srem i32 %A, 1024
  %D = icmp eq i32 %B, 0
  %E = icmp eq i32 %C, 0
  %F = and i1 %D, %E
  ret i1 %F
; CHECK-LABEL: @test21(
; CHECK-NEXT:  and i32 %A, 2147483647
; CHECK-NEXT:  icmp eq i32
; CHECK-NEXT:  ret i1
}

define i1 @test22(i32 %A) {
  %B = srem i32 %A, 1024
  %C = srem i32 %A, -2147483648
  %D = icmp eq i32 %B, 0
  %E = icmp eq i32 %C, 0
  %F = and i1 %D, %E
  ret i1 %F
; CHECK-LABEL: @test22(
; CHECK-NEXT:  and i32 %A, 2147483647
; CHECK-NEXT:  icmp eq i32
; CHECK-NEXT:  ret i1
}
