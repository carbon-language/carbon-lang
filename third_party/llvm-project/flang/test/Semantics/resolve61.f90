! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
program p1
  integer(8) :: a, b, c, d
  pointer(a, b)
  !ERROR: 'b' cannot be a Cray pointer as it is already a Cray pointee
  pointer(b, c)
  !ERROR: 'a' cannot be a Cray pointee as it is already a Cray pointer
  pointer(d, a)
end

program p2
  pointer(a, c)
  !ERROR: 'c' was already declared as a Cray pointee
  pointer(b, c)
end

program p3
  real a
  !ERROR: Cray pointer 'a' must have type INTEGER(8)
  pointer(a, b)
end

program p4
  implicit none
  real b
  !ERROR: No explicit type declared for 'd'
  pointer(a, b), (c, d)
end

program p5
  integer(8) a(10)
  !ERROR: Cray pointer 'a' must be a scalar
  pointer(a, b)
end

program p6
  real b(8)
  !ERROR: Array spec was already declared for 'b'
  pointer(a, b(4))
end

program p7
  !ERROR: Cray pointee 'b' must have must have explicit shape or assumed size
  pointer(a, b(:))
contains
  subroutine s(x, y)
    real :: x(*)  ! assumed size
    !ERROR: Cray pointee 'y' must have must have explicit shape or assumed size
    real :: y(:)  ! assumed shape
    pointer(w, y)
  end
end

program p8
  integer(8), parameter :: k = 2
  type t
  end type
  !ERROR: 't' is not a variable
  pointer(t, a)
  !ERROR: 's' is not a variable
  pointer(s, b)
  !ERROR: 'k' is not a variable
  pointer(k, c)
contains
  subroutine s
  end
end

program p9
  integer(8), parameter :: k = 2
  type t
  end type
  !ERROR: 't' is not a variable
  pointer(a, t)
  !ERROR: 's' is not a variable
  pointer(b, s)
  !ERROR: 'k' is not a variable
  pointer(c, k)
contains
  subroutine s
  end
end

module m10
  integer(8) :: a
  real :: b
end
program p10
  use m10
  !ERROR: 'b' cannot be a Cray pointee as it is use-associated
  pointer(a, c),(d, b)
end

program p11
  pointer(a, b)
  !ERROR: PARAMETER attribute not allowed on 'a'
  parameter(a=2)
  !ERROR: PARAMETER attribute not allowed on 'b'
  parameter(b=3)
end

program p12
  type t1
    sequence
    real c1
  end type
  type t2
    integer c2
  end type
  type(t1) :: x1
  type(t2) :: x2
  pointer(a, x1)
  !ERROR: Type of Cray pointee 'x2' is a non-sequence derived type
  pointer(b, x2)
end
