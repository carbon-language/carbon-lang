! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.

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
contains
  subroutine test_relational()
    l = x == y  !OK
    l = x .eq. y  !OK
    !ERROR: No user-defined or intrinsic == operator matches operand types TYPE(t) and REAL(4)
    l = x == r
  end
  subroutine test_numeric()
    l = x + r  !OK
    !ERROR: No user-defined or intrinsic + operator matches operand types REAL(4) and TYPE(t)
    l = r + x
  end
  subroutine test_logical()
    l = x .and. r  !OK
    !ERROR: No user-defined or intrinsic .AND. operator matches operand types REAL(4) and TYPE(t)
    l = r .and. x
  end
  subroutine test_unary()
    l = +x  !OK
    !ERROR: No user-defined or intrinsic + operator matches operand type LOGICAL(4)
    l = +l
    l = .not. r  !OK
    !ERROR: No user-defined or intrinsic .NOT. operator matches operand type TYPE(t)
    l = .not. x
  end
  subroutine test_concat()
    l = x // y  !OK
    !ERROR: No user-defined or intrinsic // operator matches operand types TYPE(t) and REAL(4)
    l = x // r
  end
  subroutine test_conformability(x, y)
    real :: x(10), y(10,10)
    l = x + y  !OK
    !ERROR: No user-defined or intrinsic + operator matches rank 2 array of REAL(4) and rank 1 array of REAL(4)
    l = y + x
  end
end

! Invalid operand types when user-defined operator is not available
module m2
  type :: t
  end type
  type(t) :: x, y
  real :: r
  logical :: l
contains
  subroutine test_relational()
    !ERROR: Operands of .EQ. must have comparable types; have TYPE(t) and REAL(4)
    l = x == r
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
      logical :: x
      integer :: y
    end
  end interface
contains
  subroutine s1(x, y) 
    logical :: x
    integer :: y
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
    !ERROR: Operands of .NE. must have comparable types; have LOGICAL(4) and untyped
    l = x /= null()
  end
end

! Test alternate operators. They aren't enabled by default so should be
! treated as defined operators, not intrinsic ones.
module m4
contains
  subroutine s1(x, y, z)
    logical :: x
    real :: y, z
    !ERROR: Defined operator '.a.' not found
    x = y .a. z
    !ERROR: Defined operator '.o.' not found
    x = y .o. z
    !ERROR: Defined operator '.n.' not found
    x = .n. y
    !ERROR: Defined operator '.xor.' not found
    x = y .xor. z
    !ERROR: Defined operator '.x.' not found
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
    !ERROR: No user-defined or intrinsic .AND. operator matches operand types COMPLEX(4) and COMPLEX(4)
    x = y .and. z
    !ERROR: No specific procedure of generic operator '.a.' matches the actual arguments
    x = y .a. z
  end
end
