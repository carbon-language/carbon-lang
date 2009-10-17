; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@X = external global i8
@Y = external global i8
@Z = external global i8

@A = global i1 add (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @A = global i1 xor (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
@B = global i1 sub (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z)), align 2
; CHECK: @B = global i1 xor (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
@C = global i1 mul (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @C = global i1 and (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))

@D = global i1 sdiv (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @D = global i1 icmp ult (i8* @X, i8* @Y)
@E = global i1 udiv (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @E = global i1 icmp ult (i8* @X, i8* @Y)
@F = global i1 srem (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @F = global i1 false ; <i1*> [#uses=0]
@G = global i1 urem (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @G = global i1 false ; <i1*> [#uses=0]

@H = global i1 icmp ule (i32* bitcast (i8* @X to i32*), i32* bitcast (i8* @Y to i32*))
; CHECK: @H = global i1 icmp ule (i8* @X, i8* @Y)

@I = global i1 xor (i1 icmp ult (i8* @X, i8* @Y), i1 false)
; CHECK: @I = global i1 icmp ult (i8* @X, i8* @Y)
@J = global i1 xor (i1 icmp ult (i8* @X, i8* @Y), i1 true)
; CHECK: @J = global i1 icmp uge (i8* @X, i8* @Y)

@K = global i1 icmp eq (i1 icmp ult (i8* @X, i8* @Y), i1 false)
; CHECK: @K = global i1 icmp uge (i8* @X, i8* @Y)
@L = global i1 icmp eq (i1 icmp ult (i8* @X, i8* @Y), i1 true)
; CHECK: @L = global i1 icmp ult (i8* @X, i8* @Y)
@M = global i1 icmp ne (i1 icmp ult (i8* @X, i8* @Y), i1 true)
; CHECK: @M = global i1 icmp uge (i8* @X, i8* @Y)
@N = global i1 icmp ne (i1 icmp ult (i8* @X, i8* @Y), i1 false)
; CHECK: @N = global i1 icmp ult (i8* @X, i8* @Y)

@O = global i1 icmp eq (i32 zext (i1 icmp ult (i8* @X, i8* @Y) to i32), i32 0)
; CHECK: @O = global i1 icmp uge (i8* @X, i8* @Y)

