! Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

module m
  character(2) :: z
  character(len=3) :: y
  character*4 :: x
  character :: w
  character(len=:), allocatable :: v
contains
  subroutine s(n, a, b, c, d)
    integer :: n
    character(len=n+1,kind=1) :: a
    character(n+2,2) :: b
    character*(n+3) :: c
    character(*) :: d
  end
end

!Expect: m.mod
!module m
!  character(2_4,1)::z
!  character(3_4,1)::y
!  character(4_8,1)::x
!  character(1_8,1)::w
!  character(:,1),allocatable::v
!contains
!  subroutine s(n,a,b,c,d)
!    integer(4)::n
!    character(n+1_4,1)::a
!    character(n+2_4,2)::b
!    character(n+3_4,1)::c
!    character(*,1)::d
!  end
!end
