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

! Check modfile generation for generic interfaces
module m
  interface foo
    subroutine s1(x)
      real x
    end subroutine
    subroutine s2(x)
      complex x
    end subroutine
  end interface
  interface bar
    procedure :: s1
    procedure :: s2
    procedure :: s3
    procedure :: s4
  end interface
contains
  subroutine s3(x)
    logical x
  end
  subroutine s4(x)
    integer x
  end
end

!Expect: m.mod
!module m
! generic::foo=>s1,s2
! interface
!  subroutine s1(x)
!   real(4)::x
!  end
! end interface
! interface
!  subroutine s2(x)
!   complex(4)::x
!  end
! end interface
! generic::bar=>s1,s2,s3,s4
!contains
! subroutine s3(x)
!  logical(4)::x
! end
! subroutine s4(x)
!  integer(4)::x
! end
!end
