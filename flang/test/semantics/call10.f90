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

! Test 15.7 (C1583-C1590, C1592-C1599) constraints and restrictions
! for PURE procedures.
! (C1591 is tested in call11.f90.)

module m

  type :: impureFinal
   contains
    final :: impure
  end type
  type :: t
  end type
  type :: polyAlloc
    class(t), allocatable :: a
  end type

  abstract interface
    subroutine impure
    end subroutine
  end interface

 contains

  subroutine impure(x)
    type(impureFinal) :: x
  end subroutine

  pure real function f01(a)
    real, intent(in) :: a ! ok
  end function
  pure real function f02(a)
    real, value :: a ! ok
  end function
  pure real function f03(a) ! C1583
    ! ERROR: non-POINTER dummy argument of PURE function must be INTENT(IN) or VALUE
    real :: a
  end function
  pure real function f03a(a)
    real, pointer :: a ! ok
  end function
  pure real function f04(a) ! C1583
    ! ERROR: non-POINTER dummy argument of PURE function must be INTENT(IN) or VALUE
    real, intent(out) :: a
  end function
  pure real function f04a(a)
    real, pointer, intent(out) :: a ! ok if pointer
  end function
  pure real function f05(a) ! C1583
    real, intent(out), value :: a ! weird, but ok
  end function
  pure function f06() ! C1584
    ! ERROR: Result of PURE function cannot have an impure FINAL procedure
    type(impureFinal) :: f06
  end function
  pure function f07() ! C1585
    ! ERROR: Result of PURE function cannot be both polymorphic and ALLOCATABLE
    class(t), allocatable :: f07
  end function
  pure function f08() ! C1585
    ! ERROR: Result of PURE function cannot have a polymorphic ALLOCATABLE ultimate component
    type(polyAlloc) :: f08
  end function

  pure subroutine s01(a) ! C1586
    ! ERROR: non-POINTER dummy argument of PURE subroutine must have INTENT() or VALUE attribute
    real :: a
  end subroutine
  pure subroutine s01a(a)
    real, pointer :: a
  end subroutine
  pure subroutine s02(a) ! C1587
    ! ERROR: An INTENT(OUT) dummy argument of a PURE procedure cannot have an impure FINAL procedure
    type(impureFinal), intent(out) :: a
  end subroutine
  pure subroutine s03(a) ! C1588
    ! ERROR: An INTENT(OUT) dummy argument of a PURE procedure cannot be polymorphic
    class(t), intent(out) :: a
  end subroutine
  pure subroutine s04(a) ! C1588
    ! ERROR: An INTENT(OUT) dummy argument of a PURE procedure cannot have a polymorphic ultimate component
    class(polyAlloc), intent(out) :: a
  end subroutine
  pure subroutine s05 ! C1589
    ! ERROR: A PURE subprogram cannot have local variables with the SAVE attribute
    real, save :: v1
    ! ERROR: A PURE subprogram cannot have local variables with the SAVE attribute
    real :: v2 = 0.
    ! ERROR: A PURE subprogram cannot have local variables with the SAVE attribute
    real :: v3
    data v3/0./
    ! ERROR: A PURE subprogram cannot have local variables with the SAVE attribute
    real :: v4
    common /blk/ v4
    block
      ! ERROR: A PURE subprogram cannot have local variables with the SAVE attribute
      real, save :: v5
      ! ERROR: A PURE subprogram cannot have local variables with the SAVE attribute
      real :: v6 = 0.
      ! ERROR: A PURE subprogram cannot have local variables with the SAVE attribute
    end block
  end subroutine
  pure subroutine s06 ! C1589
    ! ERROR: A PURE subprogram cannot have local variables with the VOLATILE attribute
    real, volatile :: v1
    block
      ! ERROR: A PURE subprogram cannot have local variables with the VOLATILE attribute
      real, volatile :: v2
    end block
  end subroutine
  pure subroutine s06(p) ! C1590
    ! ERROR: A dummy procedure of a PURE subprogram must be PURE.
    procedure(impure) :: p
  end subroutine

  ! pmk: Continue with C1592 - C1599

end module
