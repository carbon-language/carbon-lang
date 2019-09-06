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

! Test 15.7 C1591 & others: contexts requiring PURE subprograms

module m

  type :: t
   contains
    procedure :: tbp => pure
  end type
  type, extends(t) :: t2
   contains
    ! ERROR: An overridden PURE type-bound procedure binding must also be PURE
    procedure :: tbp => impure ! 7.5.7.3
  end type

 contains

  pure integer function pure(n)
    integer, value :: n
    pure = n
  end function
  impure integer function impure(n)
    integer, value :: n
    impure = n
  end function

  subroutine test
    real :: a(pure(1)) ! ok
    ! ERROR: A function referenced in a specification expression must be PURE.
    real :: b(impure(1)) ! 10.1.11(4)
    forall (j=1:1)
      ! ERROR: A procedure referenced in a FORALL body must be PURE.
      a(j) = impure(j) ! C1037
    end forall
    ! ERROR: A procedure referenced in a mask expression must be PURE.
    do concurrent (j=1:1, impure(j) /= 0) ! C1121
      ! ERROR: A procedure referenced in a DO CONCURRENT body must be PURE.
      a(j) = impure(j) ! C1139
    end do
  end subroutine
end module
