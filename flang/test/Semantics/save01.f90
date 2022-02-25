! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
MODULE test
SAVE
CONTAINS
PURE FUNCTION pf( )
   IMPLICIT NONE
   INTEGER :: pf
   INTEGER :: mc
   !OK: SAVE statement is not inherited by the function
END FUNCTION

PURE FUNCTION pf2( )
   IMPLICIT NONE
   SAVE
   INTEGER :: pf2
   !ERROR: A pure subprogram may not have a variable with the SAVE attribute
   INTEGER :: mc
END FUNCTION

END MODULE

