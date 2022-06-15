! RUN: %python %S/test_errors.py %s %flang_fc1
! Regression test for more than one part-ref with nonzero rank

program m
  interface
    function real_info1(i)
    end
    subroutine real_info2()
    end
    subroutine real_generic()
    end
  end interface
  type mt
    complex :: c, c2(2)
    integer :: x, x2(2)
    character(10) :: s, s2(2)
   contains
    procedure, nopass :: info1 => real_info1
    procedure, nopass :: info2 => real_info2
    procedure, nopass :: real_generic
    generic :: g1 => real_generic
  end type
  type mt2
    type(mt) :: t1(2,2)
  end type
  type mt3
    type(mt2) :: t2(2)
  end type
  type mt4
    type(mt3) :: t3(2)
  end type
  type(mt4) :: t(2)

  print *, t(1)%t3(1)%t2(1)%t1%x ! no error
  print *, t(1)%t3(1)%t2(1)%t1%x2(1) ! no error
  print *, t(1)%t3(1)%t2(1)%t1%s(1:2) ! no error
  print *, t(1)%t3(1)%t2(1)%t1%s2(1)(1:2) ! no error
  print *, t(1)%t3(1)%t2(1)%t1%c%RE ! no error
  print *, t(1)%t3(1)%t2(1)%t1%c%IM ! no error
  print *, t(1)%t3(1)%t2(1)%t1%c2(1)%RE ! no error
  print *, t(1)%t3(1)%t2(1)%t1%c2(1)%IM ! no error

  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t%t3%t2%t1%x
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3%t2%t1%x
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3(1)%t2%t1%x
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3%t2(1)%t1%x
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t%t3%t2%t1%x2(1)
  !ERROR: Reference to whole rank-1 component 'x2' of rank-2 array of derived type is not allowed
  print *, t(1)%t3%t2%t1%x2
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3(1)%t2%t1%x2(1)
  !ERROR: Subscripts of component 'x2' of rank-2 derived type array have rank 1 but must all be scalar
  print *, t(1)%t3(1)%t2(1)%t1%x2(1:)
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t%t3%t2%t1%s(1:2)
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3%t2(1)%t1%s(1:2)
  !ERROR: Subscripts of component 't1' of rank-1 derived type array have rank 1 but must all be scalar
  print *, t%t3%t2%t1(1,:)%s(1:2)
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t%t3%t2%t1%s2(1)(1:2)
  !ERROR: Subscripts of component 's2' of rank-2 derived type array have rank 1 but must all be scalar
  print *, t(1)%t3%t2%t1%s2(1:)(1:2)
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t%t3%t2%t1%c%RE
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3%t2%t1%c%RE
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3(1)%t2%t1%c%RE
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3%t2(1)%t1%c%RE
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t%t3%t2%t1%c%IM
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t%t3%t2%t1%c2(1)%RE
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3%t2%t1%c2(1)%RE
  !ERROR: Subscripts of component 'c2' of rank-2 derived type array have rank 1 but must all be scalar
  print *, t(1)%t3(1)%t2%t1%c2(1:)%RE
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t(1)%t3%t2(1)%t1%c2(1)%RE
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  print *, t%t3%t2%t1%c2(1)%IM

  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  call sub0(t%t3%t2%t1%info1(i))
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  call t%t3%t2%t1%info2
  !ERROR: Reference to whole rank-2 component 't1' of rank-1 array of derived type is not allowed
  call t%t3%t2%t1%g1

  !ERROR: Reference to rank-2 object 't1' has 1 subscripts
  call sub0(t%t3%t2%t1(1)%info1(i))
  !ERROR: Reference to rank-2 object 't1' has 1 subscripts
  call t%t3%t2%t1(1)%info2
  !ERROR: Reference to rank-2 object 't1' has 1 subscripts
  call t%t3%t2%t1(1)%g1

  !ERROR: Reference to rank-2 object 't1' has 1 subscripts
  call sub0(t%t3%t2%t1(1:)%info1(i))
  !ERROR: Reference to rank-2 object 't1' has 1 subscripts
  call t%t3%t2%t1(1:)%info2
  !ERROR: Reference to rank-2 object 't1' has 1 subscripts
  call t%t3%t2%t1(1:)%g1

end
