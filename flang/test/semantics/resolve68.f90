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
  integer function add(x, y)
    class(t), intent(in) :: x, y
  end
  integer function add_real(x, y)
    class(t), intent(in) :: x
    real, intent(in) :: y
  end
  subroutine test1(x, y, z)
    type(t) :: x
    integer :: y
    integer :: z
    !ERROR: No specific procedure of generic 'g' matches the actual arguments
    z = x%g(y)
  end
  subroutine test2(x, y, z)
    type(t) :: x
    real :: y
    integer :: z
    !ERROR: No specific procedure of generic 'g' matches the actual arguments
    z = x%g(x, y)
  end
end
