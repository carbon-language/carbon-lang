! RUN: %python %S/test_errors.py %s %flang_fc1
! Invalid operand types when user-defined operator is available
module m1
  type :: t
  end type
  interface operator(==)
    logical function eq_tt(x, y)
      import :: t
      type(t), intent(in) :: x, y
    end
  end interface
  interface operator(+)
    logical function add_tr(x, y)
      import :: t
      type(t), intent(in) :: x
      real, intent(in) :: y
    end
    logical function plus_t(x)
      import :: t
      type(t), intent(in) :: x
    end
    logical function add_12(x, y)
      real, intent(in) :: x(:), y(:,:)
    end
  end interface
  interface operator(.and.)
    logical function and_tr(x, y)
      import :: t
      type(t), intent(in) :: x
      real, intent(in) :: y
    end
  end interface
  interface operator(//)
    logical function concat_tt(x, y)
      import :: t
      type(t), intent(in) :: x, y
    end
  end interface
  interface operator(.not.)
    logical function not_r(x)
      real, intent(in) :: x
    end
  end interface
  type(t) :: x, y
  real :: r
  logical :: l
  integer :: iVar
  complex :: cvar
  character :: charVar
contains
  subroutine test_relational()
    l = x == y  !OK
    l = x .eq. y  !OK
    l = x .eq. y  !OK
    l = iVar == z'fe' !OK
    l = z'fe' == iVar !OK
    l = r == z'fe' !OK
    l = z'fe' == r !OK
    l = cVar == z'fe' !OK
    l = z'fe' == cVar !OK
    !ERROR: No intrinsic or user-defined OPERATOR(==) matches operand types CHARACTER(KIND=1) and INTEGER(4)
    l = charVar == z'fe'
    !ERROR: No intrinsic or user-defined OPERATOR(==) matches operand types INTEGER(4) and CHARACTER(KIND=1)
    l = z'fe' == charVar
    !ERROR: No intrinsic or user-defined OPERATOR(==) matches operand types LOGICAL(4) and INTEGER(4)
    l = l == z'fe' !OK
    !ERROR: No intrinsic or user-defined OPERATOR(==) matches operand types INTEGER(4) and LOGICAL(4)
    l = z'fe' == l !OK
    !ERROR: No intrinsic or user-defined OPERATOR(==) matches operand types TYPE(t) and REAL(4)
    l = x == r

    lVar = z'a' == b'1010' !OK
  end
  subroutine test_numeric()
    l = x + r  !OK
    !ERROR: No intrinsic or user-defined OPERATOR(+) matches operand types REAL(4) and TYPE(t)
    l = r + x
  end
  subroutine test_logical()
    l = x .and. r  !OK
    !ERROR: No intrinsic or user-defined OPERATOR(.AND.) matches operand types REAL(4) and TYPE(t)
    l = r .and. x
  end
  subroutine test_unary()
    l = +x  !OK
    !ERROR: No intrinsic or user-defined OPERATOR(+) matches operand type LOGICAL(4)
    l = +l
    l = .not. r  !OK
    !ERROR: No intrinsic or user-defined OPERATOR(.NOT.) matches operand type TYPE(t)
    l = .not. x
  end
  subroutine test_concat()
    l = x // y  !OK
    !ERROR: No intrinsic or user-defined OPERATOR(//) matches operand types TYPE(t) and REAL(4)
    l = x // r
  end
  subroutine test_conformability(x, y)
    real :: x(10), y(10,10)
    l = x + y  !OK
    !ERROR: No intrinsic or user-defined OPERATOR(+) matches rank 2 array of REAL(4) and rank 1 array of REAL(4)
    l = y + x
  end
end

! Invalid operand types when user-defined operator is not available
module m2
  intrinsic :: sin
  type :: t
  end type
  type(t) :: x, y
  real :: r
  logical :: l
contains
  subroutine test_relational()
    !ERROR: Operands of .EQ. must have comparable types; have TYPE(t) and REAL(4)
    l = x == r
    !ERROR: Subroutine name is not allowed here
    l = r == test_numeric
    !ERROR: Function call must have argument list
    l = r == sin
  end
  subroutine test_numeric()
    !ERROR: Operands of + must be numeric; have REAL(4) and TYPE(t)
    l = r + x
  end
  subroutine test_logical()
    !ERROR: Operands of .AND. must be LOGICAL; have REAL(4) and TYPE(t)
    l = r .and. x
  end
  subroutine test_unary()
    !ERROR: Operand of unary + must be numeric; have LOGICAL(4)
    l = +l
    !ERROR: Operand of .NOT. must be LOGICAL; have TYPE(t)
    l = .not. x
  end
  subroutine test_concat(a, b)
    character(4,kind=1) :: a
    character(4,kind=2) :: b
    character(4) :: c
    !ERROR: Operands of // must be CHARACTER with the same kind; have CHARACTER(KIND=1) and CHARACTER(KIND=2)
    c = a // b
    !ERROR: Operands of // must be CHARACTER with the same kind; have TYPE(t) and REAL(4)
    l = x // r
  end
  subroutine test_conformability(x, y)
    real :: x(10), y(10,10)
    !ERROR: Operands of + are not conformable; have rank 2 and rank 1
    l = y + x
  end
end

! Invalid untyped operands: user-defined operator doesn't affect errors
module m3
  interface operator(+)
    logical function add(x, y)
      logical, intent(in) :: x
      integer, value :: y
    end
  end interface
contains
  subroutine s1(x, y)
    logical :: x
    integer :: y
    integer, pointer :: px
    logical :: l
    complex :: z
    y = y + z'1'  !OK
    !ERROR: Operands of + must be numeric; have untyped and COMPLEX(4)
    z = z'1' + z
    y = +z'1'  !OK
    !ERROR: Operand of unary - must be numeric; have untyped
    y = -z'1'
    !ERROR: Operands of + must be numeric; have LOGICAL(4) and untyped
    y = x + z'1'
    !ERROR: A NULL() pointer is not allowed as an operand here
    l = x /= null()
    !ERROR: A NULL() pointer is not allowed as a relational operand
    l = null(px) /= null(px)
    !ERROR: A NULL() pointer is not allowed as an operand here
    l = x /= null(px)
    !ERROR: A NULL() pointer is not allowed as an operand here
    l = px /= null()
    !ERROR: A NULL() pointer is not allowed as a relational operand
    l = px /= null(px)
    !ERROR: A NULL() pointer is not allowed as an operand here
    l = null() /= null()
  end
end

! Test alternate operators. They aren't enabled by default so should be
! treated as defined operators, not intrinsic ones.
module m4
contains
  subroutine s1(x, y, z)
    logical :: x
    real :: y, z
    !ERROR: No operator .A. defined for REAL(4) and REAL(4)
    x = y .a. z
    !ERROR: No operator .O. defined for REAL(4) and REAL(4)
    x = y .o. z
    !ERROR: No operator .N. defined for REAL(4)
    x = .n. y
    !ERROR: No operator .XOR. defined for REAL(4) and REAL(4)
    x = y .xor. z
    !ERROR: No operator .X. defined for REAL(4)
    x = .x. y
  end
end

! Like m4 in resolve63 but compiled with different options.
! .A. is a defined operator.
module m5
  interface operator(.A.)
    logical function f1(x, y)
      integer, intent(in) :: x, y
    end
  end interface
  interface operator(.and.)
    logical function f2(x, y)
      real, intent(in) :: x, y
    end
  end interface
contains
  subroutine s1(x, y, z)
    logical :: x
    complex :: y, z
    !ERROR: No intrinsic or user-defined OPERATOR(.AND.) matches operand types COMPLEX(4) and COMPLEX(4)
    x = y .and. z
    !ERROR: No intrinsic or user-defined .A. matches operand types COMPLEX(4) and COMPLEX(4)
    x = y .a. z
  end
end

! Type-bound operators
module m6
  type :: t1
  contains
    procedure, pass(x) :: p1 => f1
    generic :: operator(+) => p1
  end type
  type, extends(t1) :: t2
  contains
    procedure, pass(y) :: p2 => f2
    generic :: operator(+) => p2
  end type
  type :: t3
  contains
    procedure, nopass :: p1 => f1
    !ERROR: OPERATOR(+) procedure 'p1' may not have NOPASS attribute
    generic :: operator(+) => p1
  end type
contains
  integer function f1(x, y)
    class(t1), intent(in) :: x
    integer, intent(in) :: y
  end
  integer function f2(x, y)
    class(t1), intent(in) :: x
    class(t2), intent(in) :: y
  end
  subroutine test(x, y, z)
    class(t1) :: x
    class(t2) :: y
    integer :: i
    i = x + y
    i = x + i
    i = y + i
    !ERROR: No intrinsic or user-defined OPERATOR(+) matches operand types CLASS(t2) and CLASS(t1)
    i = y + x
    !ERROR: No intrinsic or user-defined OPERATOR(+) matches operand types INTEGER(4) and CLASS(t1)
    i = i + x
  end
end

! Some cases where NULL is acceptable - ensure no false errors
module m7
  implicit none
  type :: t1
   contains
    procedure :: s1
    generic :: operator(/) => s1
  end type
  interface operator(-)
    module procedure s2
  end interface
 contains
  integer function s1(x, y)
    class(t1), intent(in) :: x
    class(t1), intent(in), pointer :: y
    s1 = 1
  end
  integer function s2(x, y)
    type(t1), intent(in), pointer :: x, y
    s2 = 2
  end
  subroutine test
    integer :: j
    type(t1), pointer :: x1
    allocate(x1)
    ! These cases are fine.
    j = x1 - x1
    j = x1 - null(mold=x1)
    j = null(mold=x1) - null(mold=x1)
    j = null(mold=x1) - x1
    j = x1 / x1
    j = x1 / null(mold=x1)
    j = null() - null(mold=x1)
    j = null(mold=x1) - null()
    j = null() - null()
    !ERROR: No intrinsic or user-defined OPERATOR(/) matches operand types untyped and TYPE(t1)
    j = null() / null(mold=x1)
    !ERROR: No intrinsic or user-defined OPERATOR(/) matches operand types TYPE(t1) and untyped
    j = null(mold=x1) / null()
    !ERROR: A NULL() pointer is not allowed as an operand here
    j = null() / null()
  end
end

! 16.9.144(6)
module m8
  interface generic
    procedure s1, s2
  end interface
 contains
  subroutine s1(ip1, rp1)
    integer, pointer, intent(in) :: ip1
    real, pointer, intent(in) :: rp1
  end subroutine
  subroutine s2(rp2, ip2)
    real, pointer, intent(in) :: rp2
    integer, pointer, intent(in) :: ip2
  end subroutine
  subroutine test
    integer, pointer :: ip
    real, pointer :: rp
    call generic(ip, rp) ! ok
    call generic(ip, null()) ! ok
    call generic(rp, null()) ! ok
    call generic(null(), rp) ! ok
    call generic(null(), ip) ! ok
    call generic(null(mold=ip), null()) ! ok
    call generic(null(), null(mold=ip)) ! ok
    !ERROR: One or more NULL() actual arguments to the generic procedure 'generic' requires a MOLD= for disambiguation
    call generic(null(), null())
  end subroutine
end

! Ensure no bogus errors for assignments to CLASS(*) allocatable
module m10
  type :: t1
    integer :: n
  end type
 contains
  subroutine test
    class(*), allocatable :: poly
    poly = 1
    poly = 3.14159
    poly = 'Il faut imaginer Sisyphe heureux'
    poly = t1(1)
  end subroutine
end module
