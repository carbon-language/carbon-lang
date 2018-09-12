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

! Check modfile generation for external interface
module m
  interface
    integer function f(x)
    end function
    subroutine s(y, z)
      logical y
      complex z
    end subroutine
  end interface
end

!Expect: m.mod
!module m
! interface
!  function f(x)
!   integer(4)::f
!   real(4)::x
!  end
! end interface
! interface
!  subroutine s(y,z)
!   logical(4)::y
!   complex(4)::z
!  end
! end interface
!end
