; "PLAIN" - No optimizations. This tests the target-independent
; constant folder.
; RUN: opt -S -o - %s | FileCheck --check-prefix=PLAIN %s

target datalayout = "e-p:128:128:128-p1:32:32:32-p2:8:8:8-p3:16:16:16-p4:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32"

; The automatic constant folder in opt does not have targetdata access, so
; it can't fold gep arithmetic, in general. However, the constant folder run
; from instcombine and global opt can use targetdata.
; PLAIN: @G8 = global i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 1 to i8 addrspace(1)*), i32 -1)
@G8 = global i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 1 to i8 addrspace(1)*), i32 -1)
; PLAIN: @G1 = global i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i8 1 to i1 addrspace(2)*), i8 -1)
@G1 = global i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i8 1 to i1 addrspace(2)*), i8 -1)
; PLAIN: @F8 = global i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 1 to i8 addrspace(1)*), i32 -2)
@F8 = global i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 1 to i8 addrspace(1)*), i32 -2)
; PLAIN: @F1 = global i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i8 1 to i1 addrspace(2)*), i8 -2)
@F1 = global i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i8 1 to i1 addrspace(2)*), i8 -2)
; PLAIN: @H8 = global i8 addrspace(1)* getelementptr (i8 addrspace(1)* null, i32 -1)
@H8 = global i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 0 to i8 addrspace(1)*), i32 -1)
; PLAIN: @H1 = global i1 addrspace(2)* getelementptr (i1 addrspace(2)* null, i8 -1)
@H1 = global i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i8 0 to i1 addrspace(2)*), i8 -1)


; The target-independent folder should be able to do some clever
; simplifications on sizeof, alignof, and offsetof expressions. The
; target-dependent folder should fold these down to constants.
; PLAIN-X: @a = constant i64 mul (i64 ptrtoint (double addrspace(4)* getelementptr (double addrspace(4)* null, i32 1) to i64), i64 2310)
@a = constant i64 mul (i64 3, i64 mul (i64 ptrtoint ({[7 x double], [7 x double]} addrspace(4)* getelementptr ({[7 x double], [7 x double]} addrspace(4)* null, i64 11) to i64), i64 5))

; PLAIN-X: @b = constant i64 ptrtoint (double addrspace(4)* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64)
@b = constant i64 ptrtoint ([13 x double] addrspace(4)* getelementptr ({i1, [13 x double]} addrspace(4)* null, i64 0, i32 1) to i64)

; PLAIN-X: @c = constant i64 mul nuw (i64 ptrtoint (double addrspace(4)* getelementptr (double addrspace(4)* null, i32 1) to i64), i64 2)
@c = constant i64 ptrtoint (double addrspace(4)* getelementptr ({double, double, double, double} addrspace(4)* null, i64 0, i32 2) to i64)

; PLAIN-X: @d = constant i64 mul nuw (i64 ptrtoint (double addrspace(4)* getelementptr (double addrspace(4)* null, i32 1) to i64), i64 11)
@d = constant i64 ptrtoint (double addrspace(4)* getelementptr ([13 x double] addrspace(4)* null, i64 0, i32 11) to i64)

; PLAIN-X: @e = constant i64 ptrtoint (double addrspace(4)* getelementptr ({ double, float, double, double }* null, i64 0, i32 2) to i64)
@e = constant i64 ptrtoint (double addrspace(4)* getelementptr ({double, float, double, double} addrspace(4)* null, i64 0, i32 2) to i64)

; PLAIN-X: @f = constant i64 1
@f = constant i64 ptrtoint (<{ i16, i128 }> addrspace(4)* getelementptr ({i1, <{ i16, i128 }>} addrspace(4)* null, i64 0, i32 1) to i64)

; PLAIN-X: @g = constant i64 ptrtoint (double addrspace(4)* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64)
@g = constant i64 ptrtoint ({double, double} addrspace(4)* getelementptr ({i1, {double, double}} addrspace(4)* null, i64 0, i32 1) to i64)

; PLAIN-X: @h = constant i64 ptrtoint (i1 addrspace(2)* getelementptr (i1 addrspace(2)* null, i32 1) to i64)
@h = constant i64 ptrtoint (double addrspace(4)* getelementptr (double addrspace(4)* null, i64 1) to i64)

; PLAIN-X: @i = constant i64 ptrtoint (i1 addrspace(2)* getelementptr ({ i1, i1 addrspace(2)* }* null, i64 0, i32 1) to i64)
@i = constant i64 ptrtoint (double addrspace(4)* getelementptr ({i1, double} addrspace(4)* null, i64 0, i32 1) to i64)

; The target-dependent folder should cast GEP indices to integer-sized pointers.

; PLAIN: @M = constant i64 addrspace(4)* getelementptr (i64 addrspace(4)* null, i32 1)
; PLAIN: @N = constant i64 addrspace(4)* getelementptr ({ i64, i64 } addrspace(4)* null, i32 0, i32 1)
; PLAIN: @O = constant i64 addrspace(4)* getelementptr ([2 x i64] addrspace(4)* null, i32 0, i32 1)

@M = constant i64 addrspace(4)* getelementptr (i64 addrspace(4)* null, i32 1)
@N = constant i64 addrspace(4)* getelementptr ({ i64, i64 } addrspace(4)* null, i32 0, i32 1)
@O = constant i64 addrspace(4)* getelementptr ([2 x i64] addrspace(4)* null, i32 0, i32 1)

; Fold GEP of a GEP. Very simple cases are folded.

; PLAIN-X: @Y = global [3 x { i32, i32 }]addrspace(3)* getelementptr inbounds ([3 x { i32, i32 }]addrspace(3)* @ext, i64 2)
@ext = external addrspace(3) global [3 x { i32, i32 }]
@Y = global [3 x { i32, i32 }]addrspace(3)* getelementptr inbounds ([3 x { i32, i32 }]addrspace(3)* getelementptr inbounds ([3 x { i32, i32 }]addrspace(3)* @ext, i64 1), i64 1)

; PLAIN-X: @Z = global i32addrspace(3)* getelementptr inbounds (i32addrspace(3)* getelementptr inbounds ([3 x { i32, i32 }]addrspace(3)* @ext, i64 0, i64 1, i32 0), i64 1)
@Z = global i32addrspace(3)* getelementptr inbounds (i32addrspace(3)* getelementptr inbounds ([3 x { i32, i32 }]addrspace(3)* @ext, i64 0, i64 1, i32 0), i64 1)


; Duplicate all of the above as function return values rather than
; global initializers.

; PLAIN: define i8 addrspace(1)* @goo8() #0 {
; PLAIN:   %t = bitcast i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 1 to i8 addrspace(1)*), i32 -1) to i8 addrspace(1)*
; PLAIN:   ret i8 addrspace(1)* %t
; PLAIN: }
; PLAIN: define i1 addrspace(2)* @goo1() #0 {
; PLAIN:   %t = bitcast i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i32 1 to i1 addrspace(2)*), i32 -1) to i1 addrspace(2)*
; PLAIN:   ret i1 addrspace(2)* %t
; PLAIN: }
; PLAIN: define i8 addrspace(1)* @foo8() #0 {
; PLAIN:   %t = bitcast i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 1 to i8 addrspace(1)*), i32 -2) to i8 addrspace(1)*
; PLAIN:   ret i8 addrspace(1)* %t
; PLAIN: }
; PLAIN: define i1 addrspace(2)* @foo1() #0 {
; PLAIN:   %t = bitcast i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i32 1 to i1 addrspace(2)*), i32 -2) to i1 addrspace(2)*
; PLAIN:   ret i1 addrspace(2)* %t
; PLAIN: }
; PLAIN: define i8 addrspace(1)* @hoo8() #0 {
; PLAIN:   %t = bitcast i8 addrspace(1)* getelementptr (i8 addrspace(1)* null, i32 -1) to i8 addrspace(1)*
; PLAIN:   ret i8 addrspace(1)* %t
; PLAIN: }
; PLAIN: define i1 addrspace(2)* @hoo1() #0 {
; PLAIN:   %t = bitcast i1 addrspace(2)* getelementptr (i1 addrspace(2)* null, i32 -1) to i1 addrspace(2)*
; PLAIN:   ret i1 addrspace(2)* %t
; PLAIN: }
define i8 addrspace(1)* @goo8() #0 {
  %t = bitcast i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 1 to i8 addrspace(1)*), i32 -1) to i8 addrspace(1)*
  ret i8 addrspace(1)* %t
}
define i1 addrspace(2)* @goo1() #0 {
  %t = bitcast i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i32 1 to i1 addrspace(2)*), i32 -1) to i1 addrspace(2)*
  ret i1 addrspace(2)* %t
}
define i8 addrspace(1)* @foo8() #0 {
  %t = bitcast i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 1 to i8 addrspace(1)*), i32 -2) to i8 addrspace(1)*
  ret i8 addrspace(1)* %t
}
define i1 addrspace(2)* @foo1() #0 {
  %t = bitcast i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i32 1 to i1 addrspace(2)*), i32 -2) to i1 addrspace(2)*
  ret i1 addrspace(2)* %t
}
define i8 addrspace(1)* @hoo8() #0 {
  %t = bitcast i8 addrspace(1)* getelementptr (i8 addrspace(1)* inttoptr (i32 0 to i8 addrspace(1)*), i32 -1) to i8 addrspace(1)*
  ret i8 addrspace(1)* %t
}
define i1 addrspace(2)* @hoo1() #0 {
  %t = bitcast i1 addrspace(2)* getelementptr (i1 addrspace(2)* inttoptr (i32 0 to i1 addrspace(2)*), i32 -1) to i1 addrspace(2)*
  ret i1 addrspace(2)* %t
}

; PLAIN-X: define i64 @fa() #0 {
; PLAIN-X:   %t = bitcast i64 mul (i64 ptrtoint (double addrspace(4)* getelementptr (double addrspace(4)* null, i32 1) to i64), i64 2310) to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
; PLAIN-X: define i64 @fb() #0 {
; PLAIN-X:   %t = bitcast i64 ptrtoint (double addrspace(4)* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64) to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
; PLAIN-X: define i64 @fc() #0 {
; PLAIN-X:   %t = bitcast i64 mul nuw (i64 ptrtoint (double addrspace(4)* getelementptr (double addrspace(4)* null, i32 1) to i64), i64 2) to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
; PLAIN-X: define i64 @fd() #0 {
; PLAIN-X:   %t = bitcast i64 mul nuw (i64 ptrtoint (double addrspace(4)* getelementptr (double addrspace(4)* null, i32 1) to i64), i64 11) to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
; PLAIN-X: define i64 @fe() #0 {
; PLAIN-X:   %t = bitcast i64 ptrtoint (double addrspace(4)* getelementptr ({ double, float, double, double }* null, i64 0, i32 2) to i64) to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
; PLAIN-X: define i64 @ff() #0 {
; PLAIN-X:   %t = bitcast i64 1 to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
; PLAIN-X: define i64 @fg() #0 {
; PLAIN-X:   %t = bitcast i64 ptrtoint (double addrspace(4)* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64) to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
; PLAIN-X: define i64 @fh() #0 {
; PLAIN-X:   %t = bitcast i64 ptrtoint (i1 addrspace(2)* getelementptr (i1 addrspace(2)* null, i32 1) to i64) to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
; PLAIN-X: define i64 @fi() #0 {
; PLAIN-X:   %t = bitcast i64 ptrtoint (i1 addrspace(2)* getelementptr ({ i1, i1 addrspace(2)* }* null, i64 0, i32 1) to i64) to i64
; PLAIN-X:   ret i64 %t
; PLAIN-X: }
define i64 @fa() #0 {
  %t = bitcast i64 mul (i64 3, i64 mul (i64 ptrtoint ({[7 x double], [7 x double]}* getelementptr ({[7 x double], [7 x double]}* null, i64 11) to i64), i64 5)) to i64
  ret i64 %t
}
define i64 @fb() #0 {
  %t = bitcast i64 ptrtoint ([13 x double] addrspace(4)* getelementptr ({i1, [13 x double]} addrspace(4)* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fc() #0 {
  %t = bitcast i64 ptrtoint (double addrspace(4)* getelementptr ({double, double, double, double} addrspace(4)* null, i64 0, i32 2) to i64) to i64
  ret i64 %t
}
define i64 @fd() #0 {
  %t = bitcast i64 ptrtoint (double addrspace(4)* getelementptr ([13 x double] addrspace(4)* null, i64 0, i32 11) to i64) to i64
  ret i64 %t
}
define i64 @fe() #0 {
  %t = bitcast i64 ptrtoint (double addrspace(4)* getelementptr ({double, float, double, double} addrspace(4)* null, i64 0, i32 2) to i64) to i64
  ret i64 %t
}
define i64 @ff() #0 {
  %t = bitcast i64 ptrtoint (<{ i16, i128 }> addrspace(4)* getelementptr ({i1, <{ i16, i128 }>} addrspace(4)* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fg() #0 {
  %t = bitcast i64 ptrtoint ({double, double} addrspace(4)* getelementptr ({i1, {double, double}} addrspace(4)* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fh() #0 {
  %t = bitcast i64 ptrtoint (double addrspace(4)* getelementptr (double addrspace(4)* null, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fi() #0 {
  %t = bitcast i64 ptrtoint (double addrspace(4)* getelementptr ({i1, double}addrspace(4)* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}

; PLAIN: define i64* @fM() #0 {
; PLAIN:   %t = bitcast i64* getelementptr (i64* null, i32 1) to i64*
; PLAIN:   ret i64* %t
; PLAIN: }
; PLAIN: define i64* @fN() #0 {
; PLAIN:   %t = bitcast i64* getelementptr ({ i64, i64 }* null, i32 0, i32 1) to i64*
; PLAIN:   ret i64* %t
; PLAIN: }
; PLAIN: define i64* @fO() #0 {
; PLAIN:   %t = bitcast i64* getelementptr ([2 x i64]* null, i32 0, i32 1) to i64*
; PLAIN:   ret i64* %t
; PLAIN: }

define i64* @fM() #0 {
  %t = bitcast i64* getelementptr (i64* null, i32 1) to i64*
  ret i64* %t
}
define i64* @fN() #0 {
  %t = bitcast i64* getelementptr ({ i64, i64 }* null, i32 0, i32 1) to i64*
  ret i64* %t
}
define i64* @fO() #0 {
  %t = bitcast i64* getelementptr ([2 x i64]* null, i32 0, i32 1) to i64*
  ret i64* %t
}

; PLAIN: define i32 addrspace(1)* @fZ() #0 {
; PLAIN:   %t = bitcast i32 addrspace(1)* getelementptr inbounds (i32 addrspace(1)* getelementptr inbounds ([3 x { i32, i32 }] addrspace(1)* @ext2, i64 0, i64 1, i32 0), i64 1) to i32 addrspace(1)*
; PLAIN:   ret i32 addrspace(1)* %t
; PLAIN: }
@ext2 = external addrspace(1) global [3 x { i32, i32 }]
define i32 addrspace(1)* @fZ() #0 {
  %t = bitcast i32 addrspace(1)* getelementptr inbounds (i32 addrspace(1)* getelementptr inbounds ([3 x { i32, i32 }] addrspace(1)* @ext2, i64 0, i64 1, i32 0), i64 1) to i32 addrspace(1)*
  ret i32 addrspace(1)* %t
}

attributes #0 = { nounwind }
