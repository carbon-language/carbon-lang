! Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

! modfile with subprograms

module m
contains

  pure subroutine s(x, y) bind(c)
    logical x
    intent(inout) y
    intent(in) x
  end subroutine

  real function f1() result(x)
    x = 1.0
  end function

  function f2(y)
    complex y
    f2 = 2.0
  end function

end

!Expect: m.mod
!module m
!contains
!pure subroutine s(x,y) bind(c)
!logical(4),intent(in)::x
!real(4),intent(inout)::y
!end
!function f1() result(x)
!real(4)::x
!end
!function f2(y)
!real(4)::f2
!complex(4)::y
!end
!end
