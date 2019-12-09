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

! Test resolution of type-bound generics.

module m1
  type :: t
  contains
    procedure, pass(x) :: add1 => add
    procedure, nopass :: add2 => add
    procedure :: add_real
    generic :: g => add1, add2, add_real
  end type
contains
  integer(8) pure function add(x, y)
    class(t), intent(in) :: x, y
  end
  integer(8) pure function add_real(x, y)
    class(t), intent(in) :: x
    real, intent(in) :: y
  end
  subroutine test1(x, y, z)
    type(t) :: x, y
    real :: z(x%add1(y))
  end
  subroutine test2(x, y, z)
    type(t) :: x, y
    real :: z(x%g(y))
  end
  subroutine test3(x, y, z)
    type(t) :: x, y
    real :: z(x%g(y, x))
  end
  subroutine test4(x, y, z)
    type(t) :: x
    real :: y
    real :: z(x%g(y))
  end
end

!Expect: m1.mod
!module m1
! type :: t
! contains
!  procedure, pass(x) :: add1 => add
!  procedure, nopass :: add2 => add
!  procedure :: add_real
!  generic :: g => add1
!  generic :: g => add2
!  generic :: g => add_real
! end type
!contains
! pure function add(x, y)
!  class(t), intent(in) :: x
!  class(t), intent(in) :: y
!  integer(8) :: add
! end
! pure function add_real(x, y)
!  class(t), intent(in) :: x
!  real(4), intent(in) :: y
!  integer(8) :: add_real
! end
! subroutine test1(x, y, z)
!  type(t) :: x
!  type(t) :: y
!  real(4) :: z(1_8:add(x, y))
! end
! subroutine test2(x, y, z)
!  type(t) :: x
!  type(t) :: y
!  real(4) :: z(1_8:x%add1(y))
! end
! subroutine test3(x, y, z)
!  type(t) :: x
!  type(t) :: y
!  real(4) :: z(1_8:x%add2(y, x))
! end
! subroutine test4(x, y, z)
!  type(t) :: x
!  real(4) :: y
!  real(4) :: z(1_8:x%add_real(y))
! end
!end
