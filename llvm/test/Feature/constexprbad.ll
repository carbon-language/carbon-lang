; This testcase is for testing illegal constant expressions.
; Uncomment any code line below to test that the error is caught
; See constexpr.ll in this directory for legal ones.
; 

%somestr = constant [11x sbyte] c"hello world"

;;---------------------------------------------------
;; Illegal cast expressions
;;---------------------------------------------------

;missing attribute (global/constant) or type before operator
;%casterr1 =        cast int 0
;%casterr2 = global cast int 0

;missing or illegal initializer value for constant
;%casterr3 = constant 
;%casterr4 = constant int 4.0

;; 
;;---------------------------------------------------
;; Illegal getelementptr expressions
;;---------------------------------------------------

;; return value must be a pointer to the element
;%geperr1 = global sbyte getelementptr([11x sbyte]* %somestr, long 0, long 8)

;; index types must be valid for pointer type
;%geperr2 = global sbyte* getelementptr([11x sbyte]* %somestr, ubyte 0)
;%geperr3 = global sbyte* getelementptr([11x sbyte]* %somestr, long 0, long 0, long 3)

;; element accessed by index list must match return type
;%geperr4 = global sbyte* getelementptr([11x sbyte]* %somestr)
;%geperr5 = global sbyte* getelementptr([11x sbyte]* %somestr, long 0)
;%geperr6 = global int* getelementptr([11x sbyte]* %somestr, long 0, long 0)

;; Cannot use cast expression in pointer field of getelementptr
;; (unlike the index fields, where it is legal)
;%geperr7 = constant int* getelementptr (int* cast long 0, long 27)


