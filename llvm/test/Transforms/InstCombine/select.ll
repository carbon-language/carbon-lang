; This test makes sure that these instructions are properly eliminated.
; PR1822

; RUN: opt < %s -instcombine -S | FileCheck %s

define i32 @test1(i32 %A, i32 %B) {
        %C = select i1 false, i32 %A, i32 %B            
        ret i32 %C
; CHECK: @test1
; CHECK: ret i32 %B
}

define i32 @test2(i32 %A, i32 %B) {
        %C = select i1 true, i32 %A, i32 %B             
        ret i32 %C
; CHECK: @test2
; CHECK: ret i32 %A
}


define i32 @test3(i1 %C, i32 %I) {
        ; V = I
        %V = select i1 %C, i32 %I, i32 %I               
        ret i32 %V
; CHECK: @test3
; CHECK: ret i32 %I
}

define i1 @test4(i1 %C) {
        ; V = C
        %V = select i1 %C, i1 true, i1 false            
        ret i1 %V
; CHECK: @test4
; CHECK: ret i1 %C
}

define i1 @test5(i1 %C) {
        ; V = !C
        %V = select i1 %C, i1 false, i1 true            
        ret i1 %V
; CHECK: @test5
; CHECK: xor i1 %C, true
; CHECK: ret i1
}

define i32 @test6(i1 %C) { 
        ; V = cast C to int
        %V = select i1 %C, i32 1, i32 0         
        ret i32 %V
; CHECK: @test6
; CHECK: %V = zext i1 %C to i32
; CHECK: ret i32 %V
}

define i1 @test7(i1 %C, i1 %X) {
        ; R = or C, X       
        %R = select i1 %C, i1 true, i1 %X               
        ret i1 %R
; CHECK: @test7
; CHECK: %R = or i1 %C, %X
; CHECK: ret i1 %R
}

define i1 @test8(i1 %C, i1 %X) {
        ; R = and C, X
        %R = select i1 %C, i1 %X, i1 false              
        ret i1 %R
; CHECK: @test8
; CHECK: %R = and i1 %C, %X
; CHECK: ret i1 %R
}

define i1 @test9(i1 %C, i1 %X) {
        ; R = and !C, X
        %R = select i1 %C, i1 false, i1 %X              
        ret i1 %R
; CHECK: @test9
; CHECK: xor i1 %C, true
; CHECK: %R = and i1
; CHECK: ret i1 %R
}

define i1 @test10(i1 %C, i1 %X) {
        ; R = or !C, X
        %R = select i1 %C, i1 %X, i1 true               
        ret i1 %R
; CHECK: @test10
; CHECK: xor i1 %C, true
; CHECK: %R = or i1
; CHECK: ret i1 %R
}

define i32 @test11(i32 %a) {
        %C = icmp eq i32 %a, 0          
        %R = select i1 %C, i32 0, i32 1         
        ret i32 %R
; CHECK: @test11
; CHECK: icmp ne i32 %a, 0
; CHECK: %R = zext i1
; CHECK: ret i32 %R
}

define i32 @test12(i1 %cond, i32 %a) {
        %b = or i32 %a, 1               
        %c = select i1 %cond, i32 %b, i32 %a            
        ret i32 %c
; CHECK: @test12
; CHECK: %b = zext i1 %cond to i32
; CHECK: %c = or i32 %b, %a
; CHECK: ret i32 %c
}

define i32 @test12a(i1 %cond, i32 %a) {
        %b = ashr i32 %a, 1             
        %c = select i1 %cond, i32 %b, i32 %a            
        ret i32 %c
; CHECK: @test12a
; CHECK: %b = zext i1 %cond to i32
; CHECK: %c = ashr i32 %a, %b
; CHECK: ret i32 %c
}

define i32 @test12b(i1 %cond, i32 %a) {
        %b = ashr i32 %a, 1             
        %c = select i1 %cond, i32 %a, i32 %b            
        ret i32 %c
; CHECK: @test12b
; CHECK: zext i1 %cond to i32
; CHECK: %b = xor i32
; CHECK: %c = ashr i32 %a, %b
; CHECK: ret i32 %c
}

define i32 @test13(i32 %a, i32 %b) {
        %C = icmp eq i32 %a, %b         
        %V = select i1 %C, i32 %a, i32 %b               
        ret i32 %V
; CHECK: @test13
; CHECK: ret i32 %b
}

define i32 @test13a(i32 %a, i32 %b) {
        %C = icmp ne i32 %a, %b         
        %V = select i1 %C, i32 %a, i32 %b               
        ret i32 %V
; CHECK: @test13a
; CHECK: ret i32 %a
}

define i32 @test13b(i32 %a, i32 %b) {
        %C = icmp eq i32 %a, %b         
        %V = select i1 %C, i32 %b, i32 %a               
        ret i32 %V
; CHECK: @test13b
; CHECK: ret i32 %a
}

define i1 @test14a(i1 %C, i32 %X) {
        %V = select i1 %C, i32 %X, i32 0                
        ; (X < 1) | !C
        %R = icmp slt i32 %V, 1         
        ret i1 %R
; CHECK: @test14a
; CHECK: icmp slt i32 %X, 1
; CHECK: xor i1 %C, true
; CHECK: or i1
; CHECK: ret i1 %R
}

define i1 @test14b(i1 %C, i32 %X) {
        %V = select i1 %C, i32 0, i32 %X                
        ; (X < 1) | C
        %R = icmp slt i32 %V, 1         
        ret i1 %R
; CHECK: @test14b
; CHECK: icmp slt i32 %X, 1
; CHECK: or i1
; CHECK: ret i1 %R
}

;; Code sequence for (X & 16) ? 16 : 0
define i32 @test15a(i32 %X) {
        %t1 = and i32 %X, 16            
        %t2 = icmp eq i32 %t1, 0                
        %t3 = select i1 %t2, i32 0, i32 16              
        ret i32 %t3
; CHECK: @test15a
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

;; Code sequence for (X & 32) ? 0 : 24
define i32 @test15b(i32 %X) {
        %t1 = and i32 %X, 32            
        %t2 = icmp eq i32 %t1, 0                
        %t3 = select i1 %t2, i32 32, i32 0              
        ret i32 %t3
; CHECK: @test15b
; CHECK: %t1 = and i32 %X, 32
; CHECK: xor i32 %t1, 32
; CHECK: ret i32
}

;; Alternate code sequence for (X & 16) ? 16 : 0
define i32 @test15c(i32 %X) {
        %t1 = and i32 %X, 16            
        %t2 = icmp eq i32 %t1, 16               
        %t3 = select i1 %t2, i32 16, i32 0              
        ret i32 %t3
; CHECK: @test15c
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

;; Alternate code sequence for (X & 16) ? 16 : 0
define i32 @test15d(i32 %X) {
        %t1 = and i32 %X, 16            
        %t2 = icmp ne i32 %t1, 0                
        %t3 = select i1 %t2, i32 16, i32 0              
        ret i32 %t3
; CHECK: @test15d
; CHECK: %t1 = and i32 %X, 16
; CHECK: ret i32 %t1
}

define i32 @test16(i1 %C, i32* %P) {
        %P2 = select i1 %C, i32* %P, i32* null          
        %V = load i32* %P2              
        ret i32 %V
; CHECK: @test16
; CHECK-NEXT: %V = load i32* %P
; CHECK: ret i32 %V
}

define i1 @test17(i32* %X, i1 %C) {
        %R = select i1 %C, i32* %X, i32* null           
        %RV = icmp eq i32* %R, null             
        ret i1 %RV
; CHECK: @test17
; CHECK: icmp eq i32* %X, null
; CHECK: xor i1 %C, true
; CHECK: %RV = or i1
; CHECK: ret i1 %RV
}

define i32 @test18(i32 %X, i32 %Y, i1 %C) {
        %R = select i1 %C, i32 %X, i32 0                
        %V = sdiv i32 %Y, %R            
        ret i32 %V
; CHECK: @test18
; CHECK: %V = sdiv i32 %Y, %X
; CHECK: ret i32 %V
}

define i32 @test19(i32 %x) {
        %tmp = icmp ugt i32 %x, 2147483647              
        %retval = select i1 %tmp, i32 -1, i32 0         
        ret i32 %retval
; CHECK: @test19
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: ret i32 
}

define i32 @test20(i32 %x) {
        %tmp = icmp slt i32 %x, 0               
        %retval = select i1 %tmp, i32 -1, i32 0         
        ret i32 %retval
; CHECK: @test20
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: ret i32 
}

define i64 @test21(i32 %x) {
        %tmp = icmp slt i32 %x, 0               
        %retval = select i1 %tmp, i64 -1, i64 0         
        ret i64 %retval
; CHECK: @test21
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: sext i32 
; CHECK-NEXT: ret i64
}

define i16 @test22(i32 %x) {
        %tmp = icmp slt i32 %x, 0               
        %retval = select i1 %tmp, i16 -1, i16 0         
        ret i16 %retval
; CHECK: @test22
; CHECK-NEXT: ashr i32 %x, 31
; CHECK-NEXT: trunc i32 
; CHECK-NEXT: ret i16
}

define i1 @test23(i1 %a, i1 %b) {
        %c = select i1 %a, i1 %b, i1 %a         
        ret i1 %c
; CHECK: @test23
; CHECK-NEXT: %c = and i1 %a, %b
; CHECK-NEXT: ret i1 %c
}

define i1 @test24(i1 %a, i1 %b) {
        %c = select i1 %a, i1 %a, i1 %b         
        ret i1 %c
; CHECK: @test24
; CHECK-NEXT: %c = or i1 %a, %b
; CHECK-NEXT: ret i1 %c
}

define i32 @test25(i1 %c)  {
entry:
  br i1 %c, label %jump, label %ret
jump:
  br label %ret 
ret:
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 10, i32 20
  ret i32 %b
; CHECK: @test25
; CHECK: %a = phi i32 [ 10, %jump ], [ 20, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test26(i1 %cond)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  %c = or i1 false, false
  br label %ret 
ret:
  %a = phi i1 [true, %jump], [%c, %entry]
  %b = select i1 %a, i32 10, i32 20
  ret i32 %b
; CHECK: @test26
; CHECK: %a = phi i32 [ 10, %jump ], [ 20, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test27(i1 %c, i32 %A, i32 %B)  {
entry:
  br i1 %c, label %jump, label %ret
jump:
  br label %ret 
ret:
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 %A, i32 %B
  ret i32 %b
; CHECK: @test27
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test28(i1 %cond, i32 %A, i32 %B)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  br label %ret 
ret:
  %c = phi i32 [%A, %jump], [%B, %entry]
  %a = phi i1 [true, %jump], [false, %entry]
  %b = select i1 %a, i32 %A, i32 %c
  ret i32 %b
; CHECK: @test28
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK-NEXT: ret i32 %a
}

define i32 @test29(i1 %cond, i32 %A, i32 %B)  {
entry:
  br i1 %cond, label %jump, label %ret
jump:
  br label %ret 
ret:
  %c = phi i32 [%A, %jump], [%B, %entry]
  %a = phi i1 [true, %jump], [false, %entry]
  br label %next
  
next:
  %b = select i1 %a, i32 %A, i32 %c
  ret i32 %b
; CHECK: @test29
; CHECK: %a = phi i32 [ %A, %jump ], [ %B, %entry ]
; CHECK: ret i32 %a
}


; SMAX(SMAX(x, y), x) -> SMAX(x, y)
define i32 @test30(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y
  
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK: @test30
; CHECK: ret i32 %cond
}

; UMAX(UMAX(x, y), x) -> UMAX(x, y)
define i32 @test31(i32 %x, i32 %y) {
  %cmp = icmp ugt i32 %x, %y 
  %cond = select i1 %cmp, i32 %x, i32 %y
  %cmp5 = icmp ugt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK: @test31
; CHECK: ret i32 %cond
}

; SMIN(SMIN(x, y), x) -> SMIN(x, y)
define i32 @test32(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %y, i32 %x
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %x, i32 %cond
  ret i32 %retval
; CHECK: @test32
; CHECK: ret i32 %cond
}

; MAX(MIN(x, y), x) -> x
define i32 @test33(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %y, i32 %x
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %cond, i32 %x
  ret i32 %retval
; CHECK: @test33
; CHECK: ret i32 %x
}

; MIN(MAX(x, y), x) -> x
define i32 @test34(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  %cond = select i1 %cmp, i32 %x, i32 %y
  %cmp5 = icmp sgt i32 %cond, %x
  %retval = select i1 %cmp5, i32 %x, i32 %cond
  ret i32 %retval
; CHECK: @test34
; CHECK: ret i32 %x
}

define i32 @test35(i32 %x) {
  %cmp = icmp sge i32 %x, 0
  %cond = select i1 %cmp, i32 60, i32 100
  ret i32 %cond
; CHECK: @test35
; CHECK: ashr i32 %x, 31
; CHECK: and i32 {{.*}}, 40
; CHECK: add i32 {{.*}}, 60
; CHECK: ret
}

define i32 @test36(i32 %x) {
  %cmp = icmp slt i32 %x, 0
  %cond = select i1 %cmp, i32 60, i32 100
  ret i32 %cond
; CHECK: @test36
; CHECK: ashr i32 %x, 31
; CHECK: and i32 {{.*}}, -40
; CHECK: add i32 {{.*}}, 100
; CHECK: ret
}

define i32 @test37(i32 %x) {
  %cmp = icmp sgt i32 %x, -1
  %cond = select i1 %cmp, i32 1, i32 -1
  ret i32 %cond
; CHECK: @test37
; CHECK: ashr i32 %x, 31
; CHECK: or i32 {{.*}}, 1
; CHECK: ret
}

define i1 @test38(i1 %cond) {
  %zero = alloca i32
  %one = alloca i32
  %ptr = select i1 %cond, i32* %zero, i32* %one
  %isnull = icmp eq i32* %ptr, null
  ret i1 %isnull
; CHECK: @test38
; CHECK: ret i1 false
}

define i1 @test39(i1 %cond, double %x) {
  %s = select i1 %cond, double %x, double 0x7FF0000000000000 ; RHS = +infty
  %cmp = fcmp ule double %x, %s
  ret i1 %cmp
; CHECK: @test39
; CHECK: ret i1 true
}

define i1 @test40(i1 %cond) {
  %a = alloca i32
  %b = alloca i32
  %c = alloca i32
  %s = select i1 %cond, i32* %a, i32* %b
  %r = icmp eq i32* %s, %c
  ret i1 %r
; CHECK: @test40
; CHECK: ret i1 false
}

define i1 @test41(i1 %cond) {
  %zero = alloca i32
  %one = alloca i32
  br i1 %cond, label %true, label %false
true:
  br label %ret
false:
  br label %ret
ret:
  %ptr = phi i32* [ %zero, %true ] , [ %one, %false ]
  %isnull = icmp eq i32* %ptr, null
  ret i1 %isnull
; CHECK: @test41
; CHECK: ret i1 false
}

define i1 @test42(i1 %cond, double %x) {
  br i1 %cond, label %true, label %false
true:
  br label %ret
false:
  br label %ret
ret:
  %p = phi double [ %x, %true ], [ 0x7FF0000000000000, %false ]; RHS = +infty
  %cmp = fcmp ule double %x, %p
  ret i1 %cmp
; CHECK: @test42
; CHECK: ret i1 true
}

define i1 @test43(i1 %cond) {
  %a = alloca i32
  %b = alloca i32
  %c = alloca i32
  br i1 %cond, label %true, label %false
true:
  br label %ret
false:
  br label %ret
ret:
  %p = phi i32* [ %a, %true ], [ %b, %false ]
  %r = icmp eq i32* %p, %c
  ret i1 %r
; CHECK: @test43
; CHECK: ret i1 false
}
