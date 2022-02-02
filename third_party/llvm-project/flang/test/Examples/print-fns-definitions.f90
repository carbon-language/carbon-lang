! Check the Flang Print Function Names example plugin prints and counts function/subroutine definitions
! This includes internal and external Function/Subroutines, but not Statement Functions
! This requires that the examples are built (FLANG_BUILD_EXAMPLES=ON) to access flangPrintFunctionNames.so

! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangPrintFunctionNames%pluginext -plugin print-fns %s 2>&1 | FileCheck %s

!-------------------------------------------------
! EXPECTED OUTPUT: Names printed and counts != 0
!-------------------------------------------------
! CHECK: Function: external_func1
! CHECK-NEXT: Function: external_func2
! CHECK-NEXT: Subroutine: external_subr
! CHECK-NEXT: Function: internal_func
! CHECK-NEXT: Subroutine: internal_subr
! CHECK-EMPTY:
! CHECK-NEXT: ==== Functions: 3 ====
! CHECK-NEXT: ==== Subroutines: 2 ====

!--------------------------
! INPUT
!--------------------------
function external_func1()
end function

function external_func2()
end function

subroutine external_subr
end subroutine

program main
contains
    function internal_func()
    end function

    subroutine internal_subr
    end subroutine
end program main
