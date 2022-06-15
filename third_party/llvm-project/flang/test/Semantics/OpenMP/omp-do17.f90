! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Do Loop constructs.

!DEF: /test MainProgram
program test
 !DEF: /test/i ObjectEntity INTEGER(4)
 !DEF: /test/j ObjectEntity INTEGER(4)
 !DEF: /test/k ObjectEntity INTEGER(4)
 integer i, j, k
 !$omp do  collapse(2)
 !DEF: /test/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 foo: do i=0,10
  !DEF: /test/Block1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  foo1: do j=0,10
   !REF: /test/k
   foo2: do k=0,10
    !REF: /test/Block1/i
    select case (i)
    case (5)
     cycle foo1
    case (7)
     cycle foo2
    end select
    !REF: /test/Block1/i
    !REF: /test/Block1/j
    !REF: /test/k
    print *, i, j, k
   end do foo2
  end do foo1
 end do foo

 !$omp do  collapse(2)
 !DEF: /test/Block2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 foo: do i=0,10
  !DEF: /test/Block2/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  foo1: do j=0,10
   !REF: /test/k
   foo2: do k=0,10
    !REF: /test/Block2/i
    if (i<3) then
     cycle foo1
     !REF: /test/Block2/i
    else if (i>8) then
     cycle foo1
    else
     cycle foo2
    end if
    !REF: /test/Block2/i
    !REF: /test/Block2/j
    !REF: /test/k
    print *, i, j, k
   end do foo2
  end do foo1
 end do foo
!$omp end do
end program test
