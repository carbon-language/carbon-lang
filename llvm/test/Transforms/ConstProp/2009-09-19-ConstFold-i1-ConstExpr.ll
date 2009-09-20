; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@X = external global i8
@Y = external global i8
@Z = external global i8

@A = global i1 add (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @A = global i1 xor (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z)) ; <i1*> [#uses=0]
@B = global i1 sub (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @B = global i1 xor (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z)) ; <i1*> [#uses=0]
@C = global i1 mul (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @C = global i1 and (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z)) ; <i1*> [#uses=0]

@D = global i1 sdiv (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @D = global i1 icmp ult (i8* @X, i8* @Y) ; <i1*> [#uses=0]
@E = global i1 udiv (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @E = global i1 icmp ult (i8* @X, i8* @Y) ; <i1*> [#uses=0]
@F = global i1 srem (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @F = global i1 false ; <i1*> [#uses=0]
@G = global i1 urem (i1 icmp ult (i8* @X, i8* @Y), i1 icmp ult (i8* @X, i8* @Z))
; CHECK: @G = global i1 false ; <i1*> [#uses=0]

@H = global i1 icmp ule (i32* bitcast (i8* @X to i32*), i32* bitcast (i8* @Y to i32*))
; CHECK: @H = global i1 icmp ule (i8* @X, i8* @Y) ; <i1*> [#uses=0]

@I = global i1 xor (i1 icmp ult (i8* @X, i8* @Y), i1 false)
; CHECK: @I = global i1 icmp ult (i8* @X, i8* @Y) ; <i1*> [#uses=0]
@J = global i1 xor (i1 icmp ult (i8* @X, i8* @Y), i1 true)
; CHECK: @J = global i1 icmp uge (i8* @X, i8* @Y) ; <i1*> [#uses=0]

@K = global i1 icmp eq (i1 icmp ult (i8* @X, i8* @Y), i1 false)
; CHECK: @K = global i1 icmp uge (i8* @X, i8* @Y) ; <i1*> [#uses=0]
@L = global i1 icmp eq (i1 icmp ult (i8* @X, i8* @Y), i1 true)
; CHECK: @L = global i1 icmp ult (i8* @X, i8* @Y) ; <i1*> [#uses=0]
@M = global i1 icmp ne (i1 icmp ult (i8* @X, i8* @Y), i1 true)
; CHECK: @M = global i1 icmp uge (i8* @X, i8* @Y) ; <i1*> [#uses=0]
@N = global i1 icmp ne (i1 icmp ult (i8* @X, i8* @Y), i1 false)
; CHECK: @N = global i1 icmp ult (i8* @X, i8* @Y) ; <i1*> [#uses=0]
