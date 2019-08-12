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

module m
  implicit complex(8)(z)
  real :: x
  namelist /nl1/ x, y
  namelist /nl2/ y, x
  namelist /nl1/ i, z
  complex(8) :: z
  real :: y
end

!Expect: m.mod
!module m
!  real(4)::x
!  integer(4)::i
!  complex(8)::z
!  real(4)::y
!  namelist/nl1/x,y,i,z
!  namelist/nl2/y,x
!end
