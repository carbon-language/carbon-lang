; "PLAIN" - No optimizations. This tests the target-independent
; constant folder.
; RUN: opt -S -o - < %s | FileCheck --check-prefix=PLAIN %s

; "OPT" - Optimizations but no targetdata. This tests target-independent
; folding in the optimizers.
; RUN: opt -S -o - -instcombine -globalopt < %s | FileCheck --check-prefix=OPT %s

; "TO" - Optimizations and targetdata. This tests target-dependent
; folding in the optimizers.
; RUN: opt -S -o - -instcombine -globalopt -default-data-layout="e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64" < %s | FileCheck --check-prefix=TO %s

; "SCEV" - ScalarEvolution but no targetdata.
; RUN: opt -analyze -scalar-evolution < %s | FileCheck --check-prefix=SCEV %s


; The automatic constant folder in opt does not have targetdata access, so
; it can't fold gep arithmetic, in general. However, the constant folder run
; from instcombine and global opt can use targetdata.

; PLAIN: @G8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1)
; PLAIN: @G1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1)
; PLAIN: @F8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2)
; PLAIN: @F1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2)
; PLAIN: @H8 = global i8* getelementptr (i8* null, i32 -1)
; PLAIN: @H1 = global i1* getelementptr (i1* null, i32 -1)
; OPT: @G8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1)
; OPT: @G1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1)
; OPT: @F8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2)
; OPT: @F1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2)
; OPT: @H8 = global i8* getelementptr (i8* null, i32 -1)
; OPT: @H1 = global i1* getelementptr (i1* null, i32 -1)
; TO: @G8 = global i8* null
; TO: @G1 = global i1* null
; TO: @F8 = global i8* inttoptr (i64 -1 to i8*)
; TO: @F1 = global i1* inttoptr (i64 -1 to i1*)
; TO: @H8 = global i8* inttoptr (i64 -1 to i8*)
; TO: @H1 = global i1* inttoptr (i64 -1 to i1*)

@G8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1)
@G1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1)
@F8 = global i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2)
@F1 = global i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2)
@H8 = global i8* getelementptr (i8* inttoptr (i32 0 to i8*), i32 -1)
@H1 = global i1* getelementptr (i1* inttoptr (i32 0 to i1*), i32 -1)

; The target-independent folder should be able to do some clever
; simplifications on sizeof, alignof, and offsetof expressions. The
; target-dependent folder should fold these down to constants.

; PLAIN: @a = constant i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2310)
; PLAIN: @b = constant i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64)
; PLAIN: @c = constant i64 mul nuw (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2)
; PLAIN: @d = constant i64 mul nuw (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 11)
; PLAIN: @e = constant i64 ptrtoint (double* getelementptr ({ double, float, double, double }* null, i64 0, i32 2) to i64)
; PLAIN: @f = constant i64 1
; PLAIN: @g = constant i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64)
; PLAIN: @h = constant i64 ptrtoint (i1** getelementptr (i1** null, i32 1) to i64)
; PLAIN: @i = constant i64 ptrtoint (i1** getelementptr ({ i1, i1* }* null, i64 0, i32 1) to i64)
; OPT: @a = constant i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2310)
; OPT: @b = constant i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64)
; OPT: @c = constant i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2)
; OPT: @d = constant i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 11)
; OPT: @e = constant i64 ptrtoint (double* getelementptr ({ double, float, double, double }* null, i64 0, i32 2) to i64)
; OPT: @f = constant i64 1
; OPT: @g = constant i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64)
; OPT: @h = constant i64 ptrtoint (i1** getelementptr (i1** null, i32 1) to i64)
; OPT: @i = constant i64 ptrtoint (i1** getelementptr ({ i1, i1* }* null, i64 0, i32 1) to i64)
; TO: @a = constant i64 18480
; TO: @b = constant i64 8
; TO: @c = constant i64 16
; TO: @d = constant i64 88
; TO: @e = constant i64 16
; TO: @f = constant i64 1
; TO: @g = constant i64 8
; TO: @h = constant i64 8
; TO: @i = constant i64 8

@a = constant i64 mul (i64 3, i64 mul (i64 ptrtoint ({[7 x double], [7 x double]}* getelementptr ({[7 x double], [7 x double]}* null, i64 11) to i64), i64 5))
@b = constant i64 ptrtoint ([13 x double]* getelementptr ({i1, [13 x double]}* null, i64 0, i32 1) to i64)
@c = constant i64 ptrtoint (double* getelementptr ({double, double, double, double}* null, i64 0, i32 2) to i64)
@d = constant i64 ptrtoint (double* getelementptr ([13 x double]* null, i64 0, i32 11) to i64)
@e = constant i64 ptrtoint (double* getelementptr ({double, float, double, double}* null, i64 0, i32 2) to i64)
@f = constant i64 ptrtoint (<{ i16, i128 }>* getelementptr ({i1, <{ i16, i128 }>}* null, i64 0, i32 1) to i64)
@g = constant i64 ptrtoint ({double, double}* getelementptr ({i1, {double, double}}* null, i64 0, i32 1) to i64)
@h = constant i64 ptrtoint (double** getelementptr (double** null, i64 1) to i64)
@i = constant i64 ptrtoint (double** getelementptr ({i1, double*}* null, i64 0, i32 1) to i64)

; The target-dependent folder should cast GEP indices to integer-sized pointers.

; PLAIN: @M = constant i64* getelementptr (i64* null, i32 1)
; PLAIN: @N = constant i64* getelementptr ({ i64, i64 }* null, i32 0, i32 1)
; PLAIN: @O = constant i64* getelementptr ([2 x i64]* null, i32 0, i32 1)
; OPT: @M = constant i64* getelementptr (i64* null, i32 1)
; OPT: @N = constant i64* getelementptr ({ i64, i64 }* null, i32 0, i32 1)
; OPT: @O = constant i64* getelementptr ([2 x i64]* null, i32 0, i32 1)
; TO: @M = constant i64* inttoptr (i64 8 to i64*)
; TO: @N = constant i64* inttoptr (i64 8 to i64*)
; TO: @O = constant i64* inttoptr (i64 8 to i64*)

@M = constant i64* getelementptr (i64* null, i32 1)
@N = constant i64* getelementptr ({ i64, i64 }* null, i32 0, i32 1)
@O = constant i64* getelementptr ([2 x i64]* null, i32 0, i32 1)

; Fold GEP of a GEP. Very simple cases are folded without targetdata.

; PLAIN: @Y = global [3 x { i32, i32 }]* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 2)
; PLAIN: @Z = global i32* getelementptr inbounds (i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1)
; OPT: @Y = global [3 x { i32, i32 }]* getelementptr ([3 x { i32, i32 }]* @ext, i64 2)
; OPT: @Z = global i32* getelementptr (i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1)
; TO: @Y = global [3 x { i32, i32 }]* getelementptr ([3 x { i32, i32 }]* @ext, i64 2)
; TO: @Z = global i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 1)

@ext = external global [3 x { i32, i32 }]
@Y = global [3 x { i32, i32 }]* getelementptr inbounds ([3 x { i32, i32 }]* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 1), i64 1)
@Z = global i32* getelementptr inbounds (i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1)

; Duplicate all of the above as function return values rather than
; global initializers.

; PLAIN: define i8* @goo8() #0 {
; PLAIN:   %t = bitcast i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1) to i8*
; PLAIN:   ret i8* %t
; PLAIN: }
; PLAIN: define i1* @goo1() #0 {
; PLAIN:   %t = bitcast i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1) to i1*
; PLAIN:   ret i1* %t
; PLAIN: }
; PLAIN: define i8* @foo8() #0 {
; PLAIN:   %t = bitcast i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2) to i8*
; PLAIN:   ret i8* %t
; PLAIN: }
; PLAIN: define i1* @foo1() #0 {
; PLAIN:   %t = bitcast i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2) to i1*
; PLAIN:   ret i1* %t
; PLAIN: }
; PLAIN: define i8* @hoo8() #0 {
; PLAIN:   %t = bitcast i8* getelementptr (i8* null, i32 -1) to i8*
; PLAIN:   ret i8* %t
; PLAIN: }
; PLAIN: define i1* @hoo1() #0 {
; PLAIN:   %t = bitcast i1* getelementptr (i1* null, i32 -1) to i1*
; PLAIN:   ret i1* %t
; PLAIN: }
; OPT: define i8* @goo8() #0 {
; OPT:   ret i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1)
; OPT: }
; OPT: define i1* @goo1() #0 {
; OPT:   ret i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1)
; OPT: }
; OPT: define i8* @foo8() #0 {
; OPT:   ret i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2)
; OPT: }
; OPT: define i1* @foo1() #0 {
; OPT:   ret i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2)
; OPT: }
; OPT: define i8* @hoo8() #0 {
; OPT:   ret i8* getelementptr (i8* null, i32 -1)
; OPT: }
; OPT: define i1* @hoo1() #0 {
; OPT:   ret i1* getelementptr (i1* null, i32 -1)
; OPT: }
; TO: define i8* @goo8() #0 {
; TO:   ret i8* null
; TO: }
; TO: define i1* @goo1() #0 {
; TO:   ret i1* null
; TO: }
; TO: define i8* @foo8() #0 {
; TO:   ret i8* inttoptr (i64 -1 to i8*)
; TO: }
; TO: define i1* @foo1() #0 {
; TO:   ret i1* inttoptr (i64 -1 to i1*)
; TO: }
; TO: define i8* @hoo8() #0 {
; TO:   ret i8* inttoptr (i64 -1 to i8*)
; TO: }
; TO: define i1* @hoo1() #0 {
; TO:   ret i1* inttoptr (i64 -1 to i1*)
; TO: }
; SCEV: Classifying expressions for: @goo8
; SCEV:   %t = bitcast i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1) to i8*
; SCEV:   -->  ((-1 * sizeof(i8)) + inttoptr (i32 1 to i8*))
; SCEV: Classifying expressions for: @goo1
; SCEV:   %t = bitcast i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1) to i1*
; SCEV:   -->  ((-1 * sizeof(i1)) + inttoptr (i32 1 to i1*))
; SCEV: Classifying expressions for: @foo8
; SCEV:   %t = bitcast i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2) to i8*
; SCEV:   -->  ((-2 * sizeof(i8)) + inttoptr (i32 1 to i8*))
; SCEV: Classifying expressions for: @foo1
; SCEV:   %t = bitcast i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2) to i1*
; SCEV:   -->  ((-2 * sizeof(i1)) + inttoptr (i32 1 to i1*))
; SCEV: Classifying expressions for: @hoo8
; SCEV:   -->  (-1 * sizeof(i8))
; SCEV: Classifying expressions for: @hoo1
; SCEV:   -->  (-1 * sizeof(i1))

define i8* @goo8() nounwind {
  %t = bitcast i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -1) to i8*
  ret i8* %t
}
define i1* @goo1() nounwind {
  %t = bitcast i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -1) to i1*
  ret i1* %t
}
define i8* @foo8() nounwind {
  %t = bitcast i8* getelementptr (i8* inttoptr (i32 1 to i8*), i32 -2) to i8*
  ret i8* %t
}
define i1* @foo1() nounwind {
  %t = bitcast i1* getelementptr (i1* inttoptr (i32 1 to i1*), i32 -2) to i1*
  ret i1* %t
}
define i8* @hoo8() nounwind {
  %t = bitcast i8* getelementptr (i8* inttoptr (i32 0 to i8*), i32 -1) to i8*
  ret i8* %t
}
define i1* @hoo1() nounwind {
  %t = bitcast i1* getelementptr (i1* inttoptr (i32 0 to i1*), i32 -1) to i1*
  ret i1* %t
}

; PLAIN: define i64 @fa() #0 {
; PLAIN:   %t = bitcast i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2310) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fb() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fc() #0 {
; PLAIN:   %t = bitcast i64 mul nuw (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fd() #0 {
; PLAIN:   %t = bitcast i64 mul nuw (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 11) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fe() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (double* getelementptr ({ double, float, double, double }* null, i64 0, i32 2) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @ff() #0 {
; PLAIN:   %t = bitcast i64 1 to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fg() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fh() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (i1** getelementptr (i1** null, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; PLAIN: define i64 @fi() #0 {
; PLAIN:   %t = bitcast i64 ptrtoint (i1** getelementptr ({ i1, i1* }* null, i64 0, i32 1) to i64) to i64
; PLAIN:   ret i64 %t
; PLAIN: }
; OPT: define i64 @fa() #0 {
; OPT:   ret i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2310)
; OPT: }
; OPT: define i64 @fb() #0 {
; OPT:   ret i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64)
; OPT: }
; OPT: define i64 @fc() #0 {
; OPT:   ret i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2)
; OPT: }
; OPT: define i64 @fd() #0 {
; OPT:   ret i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 11)
; OPT: }
; OPT: define i64 @fe() #0 {
; OPT:   ret i64 ptrtoint (double* getelementptr ({ double, float, double, double }* null, i64 0, i32 2) to i64)
; OPT: }
; OPT: define i64 @ff() #0 {
; OPT:   ret i64 1
; OPT: }
; OPT: define i64 @fg() #0 {
; OPT:   ret i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64)
; OPT: }
; OPT: define i64 @fh() #0 {
; OPT:   ret i64 ptrtoint (i1** getelementptr (i1** null, i32 1) to i64)
; OPT: }
; OPT: define i64 @fi() #0 {
; OPT:   ret i64 ptrtoint (i1** getelementptr ({ i1, i1* }* null, i64 0, i32 1) to i64)
; OPT: }
; TO: define i64 @fa() #0 {
; TO:   ret i64 18480
; TO: }
; TO: define i64 @fb() #0 {
; TO:   ret i64 8
; TO: }
; TO: define i64 @fc() #0 {
; TO:   ret i64 16
; TO: }
; TO: define i64 @fd() #0 {
; TO:   ret i64 88
; TO: }
; TO: define i64 @fe() #0 {
; TO:   ret i64 16
; TO: }
; TO: define i64 @ff() #0 {
; TO:   ret i64 1
; TO: }
; TO: define i64 @fg() #0 {
; TO:   ret i64 8
; TO: }
; TO: define i64 @fh() #0 {
; TO:   ret i64 8
; TO: }
; TO: define i64 @fi() #0 {
; TO:   ret i64 8
; TO: }
; SCEV: Classifying expressions for: @fa
; SCEV:   %t = bitcast i64 mul (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2310) to i64 
; SCEV:   -->  (2310 * sizeof(double))
; SCEV: Classifying expressions for: @fb
; SCEV:   %t = bitcast i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64) to i64 
; SCEV:   -->  alignof(double)
; SCEV: Classifying expressions for: @fc
; SCEV:   %t = bitcast i64 mul nuw (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 2) to i64 
; SCEV:   -->  (2 * sizeof(double))
; SCEV: Classifying expressions for: @fd
; SCEV:   %t = bitcast i64 mul nuw (i64 ptrtoint (double* getelementptr (double* null, i32 1) to i64), i64 11) to i64 
; SCEV:   -->  (11 * sizeof(double))
; SCEV: Classifying expressions for: @fe
; SCEV:   %t = bitcast i64 ptrtoint (double* getelementptr ({ double, float, double, double }* null, i64 0, i32 2) to i64) to i64 
; SCEV:   -->  offsetof({ double, float, double, double }, 2)
; SCEV: Classifying expressions for: @ff
; SCEV:   %t = bitcast i64 1 to i64 
; SCEV:   -->  1
; SCEV: Classifying expressions for: @fg
; SCEV:   %t = bitcast i64 ptrtoint (double* getelementptr ({ i1, double }* null, i64 0, i32 1) to i64) to i64
; SCEV:   -->  alignof(double)
; SCEV: Classifying expressions for: @fh
; SCEV:   %t = bitcast i64 ptrtoint (i1** getelementptr (i1** null, i32 1) to i64) to i64
; SCEV:   -->  sizeof(i1*)
; SCEV: Classifying expressions for: @fi
; SCEV:   %t = bitcast i64 ptrtoint (i1** getelementptr ({ i1, i1* }* null, i64 0, i32 1) to i64) to i64
; SCEV:   -->  alignof(i1*)

define i64 @fa() nounwind {
  %t = bitcast i64 mul (i64 3, i64 mul (i64 ptrtoint ({[7 x double], [7 x double]}* getelementptr ({[7 x double], [7 x double]}* null, i64 11) to i64), i64 5)) to i64
  ret i64 %t
}
define i64 @fb() nounwind {
  %t = bitcast i64 ptrtoint ([13 x double]* getelementptr ({i1, [13 x double]}* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fc() nounwind {
  %t = bitcast i64 ptrtoint (double* getelementptr ({double, double, double, double}* null, i64 0, i32 2) to i64) to i64
  ret i64 %t
}
define i64 @fd() nounwind {
  %t = bitcast i64 ptrtoint (double* getelementptr ([13 x double]* null, i64 0, i32 11) to i64) to i64
  ret i64 %t
}
define i64 @fe() nounwind {
  %t = bitcast i64 ptrtoint (double* getelementptr ({double, float, double, double}* null, i64 0, i32 2) to i64) to i64
  ret i64 %t
}
define i64 @ff() nounwind {
  %t = bitcast i64 ptrtoint (<{ i16, i128 }>* getelementptr ({i1, <{ i16, i128 }>}* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fg() nounwind {
  %t = bitcast i64 ptrtoint ({double, double}* getelementptr ({i1, {double, double}}* null, i64 0, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fh() nounwind {
  %t = bitcast i64 ptrtoint (double** getelementptr (double** null, i32 1) to i64) to i64
  ret i64 %t
}
define i64 @fi() nounwind {
  %t = bitcast i64 ptrtoint (double** getelementptr ({i1, double*}* null, i64 0, i32 1) to i64) to i64
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
; OPT: define i64* @fM() #0 {
; OPT:   ret i64* getelementptr (i64* null, i32 1)
; OPT: }
; OPT: define i64* @fN() #0 {
; OPT:   ret i64* getelementptr ({ i64, i64 }* null, i32 0, i32 1)
; OPT: }
; OPT: define i64* @fO() #0 {
; OPT:   ret i64* getelementptr ([2 x i64]* null, i32 0, i32 1)
; OPT: }
; TO: define i64* @fM() #0 {
; TO:   ret i64* inttoptr (i64 8 to i64*)
; TO: }
; TO: define i64* @fN() #0 {
; TO:   ret i64* inttoptr (i64 8 to i64*)
; TO: }
; TO: define i64* @fO() #0 {
; TO:   ret i64* inttoptr (i64 8 to i64*)
; TO: }
; SCEV: Classifying expressions for: @fM
; SCEV:   %t = bitcast i64* getelementptr (i64* null, i32 1) to i64* 
; SCEV:   -->  sizeof(i64)
; SCEV: Classifying expressions for: @fN
; SCEV:   %t = bitcast i64* getelementptr ({ i64, i64 }* null, i32 0, i32 1) to i64* 
; SCEV:   -->  sizeof(i64)
; SCEV: Classifying expressions for: @fO
; SCEV:   %t = bitcast i64* getelementptr ([2 x i64]* null, i32 0, i32 1) to i64* 
; SCEV:   -->  sizeof(i64)

define i64* @fM() nounwind {
  %t = bitcast i64* getelementptr (i64* null, i32 1) to i64*
  ret i64* %t
}
define i64* @fN() nounwind {
  %t = bitcast i64* getelementptr ({ i64, i64 }* null, i32 0, i32 1) to i64*
  ret i64* %t
}
define i64* @fO() nounwind {
  %t = bitcast i64* getelementptr ([2 x i64]* null, i32 0, i32 1) to i64*
  ret i64* %t
}

; PLAIN: define i32* @fZ() #0 {
; PLAIN:   %t = bitcast i32* getelementptr inbounds (i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1) to i32*
; PLAIN:   ret i32* %t
; PLAIN: }
; OPT: define i32* @fZ() #0 {
; OPT:   ret i32* getelementptr (i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1)
; OPT: }
; TO: define i32* @fZ() #0 {
; TO:   ret i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 1)
; TO: }
; SCEV: Classifying expressions for: @fZ
; SCEV:   %t = bitcast i32* getelementptr inbounds (i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1) to i32*
; SCEV:   -->  ((3 * sizeof(i32)) + @ext)

define i32* @fZ() nounwind {
  %t = bitcast i32* getelementptr inbounds (i32* getelementptr inbounds ([3 x { i32, i32 }]* @ext, i64 0, i64 1, i32 0), i64 1) to i32*
  ret i32* %t
}

; PR15262 - Check GEP folding with casts between address spaces.

@p0 = global [4 x i8] zeroinitializer, align 1
@p12 = addrspace(12) global [4 x i8] zeroinitializer, align 1

define i8* @different_addrspace() nounwind noinline {
; OPT: different_addrspace
  %p = getelementptr inbounds i8, i8* addrspacecast ([4 x i8] addrspace(12)* @p12 to i8*),
                                  i32 2
  ret i8* %p
; OPT: ret i8* getelementptr (i8* addrspacecast (i8 addrspace(12)* getelementptr inbounds ([4 x i8] addrspace(12)* @p12, i32 0, i32 0) to i8*), i32 2)
}

define i8* @same_addrspace() nounwind noinline {
; OPT: same_addrspace
  %p = getelementptr inbounds i8, i8* bitcast ([4 x i8] * @p0 to i8*), i32 2
  ret i8* %p
; OPT: ret i8* getelementptr inbounds ([4 x i8]* @p0, i32 0, i32 2)
}

@gv1 = internal global i32 1
@gv2 = internal global [1 x i32] [ i32 2 ]
@gv3 = internal global [1 x i32] [ i32 2 ]

; Handled by TI-independent constant folder
define i1 @gv_gep_vs_gv() {
  ret i1 icmp eq (i32* getelementptr inbounds ([1 x i32]* @gv2, i32 0, i32 0), i32* @gv1)
}
; PLAIN: gv_gep_vs_gv
; PLAIN: ret i1 false

define i1 @gv_gep_vs_gv_gep() {
  ret i1 icmp eq (i32* getelementptr inbounds ([1 x i32]* @gv2, i32 0, i32 0), i32* getelementptr inbounds ([1 x i32]* @gv3, i32 0, i32 0))
}
; PLAIN: gv_gep_vs_gv_gep
; PLAIN: ret i1 false

; CHECK: attributes #0 = { nounwind }
