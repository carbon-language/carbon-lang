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

! Tests for "proc-interface" semantics
! These cases are all valid.

module module1
  abstract interface
    real elemental function abstract1(x)
      real, intent(in) :: x
    end function abstract1
  end interface
  interface
    real elemental function explicit1(x)
      real, intent(in) :: x
    end function explicit1
    integer function logical(x) ! name is ambiguous vs. decl-type-spec
      real, intent(in) :: x
    end function logical
    character(1) function tan(x)
      real, intent(in) :: x
    end function tan
  end interface
  type :: derived1
    procedure(abstract1), pointer, nopass :: p1 => nested1
    procedure(explicit1), pointer, nopass :: p2 => nested1
    procedure(logical), pointer, nopass :: p3 => nested2
    procedure(logical(kind=4)), pointer, nopass :: p4 => nested3
    procedure(complex), pointer, nopass :: p5 => nested4
    procedure(sin), pointer, nopass :: p6 => nested1
    procedure(sin), pointer, nopass :: p7 => cos
    procedure(tan), pointer, nopass :: p8 => nested5
  end type derived1
 contains
  real elemental function nested1(x)
    real, intent(in) :: x
    nested1 = x + 1.
  end function nested1
  integer function nested2(x)
    real, intent(in) :: x
    nested2 = x + 2.
  end function nested2
  logical function nested3(x)
    real, intent(in) :: x
    nested3 = x > 0
  end function nested3
  complex function nested4(x)
    real, intent(in) :: x
    nested4 = cmplx(x + 4., 6.)
  end function nested4
  character function nested5(x)
    real, intent(in) :: x
    nested5 = 'a'
  end function nested5
end module module1

real elemental function explicit1(x)
  real, intent(in) :: x
  explicit1 = -x
end function explicit1

integer function logical(x)
  real, intent(in) :: x
  logical = x + 3.
end function logical

real function tan(x)
  real, intent(in) :: x
  tan = x + 5.
end function tan

program main
  use module1
  type(derived1) :: instance
  if (instance%p1(1.) /= 2.) print *, "p1 failed"
  if (instance%p2(1.) /= 2.) print *, "p2 failed"
  if (instance%p3(1.) /= 3) print *, "p3 failed"
  if (.not. instance%p4(1.)) print *, "p4 failed"
  if (instance%p5(1.) /= (5.,6.)) print *, "p5 failed"
  if (instance%p6(1.) /= 2.) print *, "p6 failed"
  if (instance%p7(0.) /= 1.) print *, "p7 failed"
  if (instance%p8(1.) /= 'a') print *, "p8 failed"
end program main
