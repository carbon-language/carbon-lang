; This testcase is for testing expressions constructed from
; constant values, including constant pointers to globals.
;

;;-------------------------------
;; Test constant cast expressions
;;-------------------------------

global ulong u0x00001     ; hexadecimal unsigned integer constants
global long  s0x0012312   ; hexadecimal signed integer constants

%t2 = global int* %t1                             ;; Forward reference without cast
%t3 = global uint* cast (int* %t1 to uint*)       ;; Forward reference with cast
%t1 = global int 4                                ;; int* %0
%t4 = global int** cast (uint** %t3 to int**)     ;; Cast of a previous cast
%t5 = global uint** %t3                           ;; Reference to a previous cast
%t6 = global int*** %t4                           ;; Different ref. to a previous cast
%t7 = global float* cast (int 12345678 to float*) ;; Cast ordinary value to ptr
%t9 = global int cast (float cast (int 8 to float) to int) ;; Nested cast expression

global int* cast (float* %0 to int*)   ;; Forward numeric reference
global float* %0                       ;; Duplicate forward numeric reference
global float 0.0


;;---------------------------------------------------
;; Test constant getelementpr expressions for arrays
;;---------------------------------------------------

%array  = constant [2 x int] [ int 12, int 52 ]
%arrayPtr = global int* getelementptr ([2 x int]* %array, long 0, long 0)    ;; int* &%array[0][0]
%arrayPtr5 = global int** getelementptr (int** %arrayPtr, long 5)    ;; int* &%arrayPtr[5]

%somestr = constant [11x sbyte] c"hello world"
%char5  = global sbyte* getelementptr([11x sbyte]* %somestr, long 0, long 5)

;; cast of getelementptr
%char8a = global int* cast (sbyte* getelementptr([11x sbyte]* %somestr, long 0, long 8) to int*)

;; getelementptr containing casts
%char8b = global sbyte* getelementptr([11x sbyte]* %somestr, long cast (ubyte 0 to long), long cast (sbyte 8 to long))

;;-------------------------------------------------------
;; TODO: Test constant getelementpr expressions for structures
;;-------------------------------------------------------

%SType  = type { int , {float, {ubyte} }, ulong } ;; struct containing struct
%SAType = type { int , {[2x float], ulong} } ;; struct containing array

%S1 = global %SType* null			;; Global initialized to NULL
%S2c = constant %SType { int 1, {float,{ubyte}} {float 2.0, {ubyte} {ubyte 3}}, ulong 4}

%S3c = constant %SAType { int 1, {[2x float], ulong} {[2x float] [float 2.0, float 3.0], ulong 4} }

%S1ptr = global %SType** %S1		    ;; Ref. to global S1
%S2  = global %SType* %S2c		    ;; Ref. to constant S2
%S3  = global %SAType* %S3c		    ;; Ref. to constant S3

					    ;; Pointer to float (**%S1).1.0
%S1fld1a = global float* getelementptr (%SType* %S2c, long 0, ubyte 1, ubyte 0)
					    ;; Another ptr to the same!
%S1fld1b = global float* getelementptr (%SType* %S2c, long 0, ubyte 1, ubyte 0)

%S1fld1bptr = global float** %S1fld1b	    ;; Ref. to previous pointer

					    ;; Pointer to ubyte (**%S2).1.1.0
%S2fld3 = global ubyte* getelementptr (%SType* %S2c, long 0, ubyte 1, ubyte 1, ubyte 0) 

					    ;; Pointer to float (**%S2).1.0[0]
;%S3fld3 = global float* getelementptr (%SAType** %S3, long 0, long 0, ubyte 1, ubyte 0, long 0) 

;;---------------------------------------------------------
;; TODO: Test constant expressions for unary and binary operators
;;---------------------------------------------------------


;;---------------------------------------------------
;; Test duplicate constant expressions
;;---------------------------------------------------

%t4 = global int** cast (uint** %t3 to int**)

%char8a = global int* cast (sbyte* getelementptr([11x sbyte]* %somestr, long 0, long 8) to int*)

;%S3fld3 = global float* getelementptr (%SAType** %S3, long 0, long 0, ubyte 1, ubyte 0, long 0) 


;;---------------------------------------------------

implementation

