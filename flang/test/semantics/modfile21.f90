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
  logical b
  bind(C) :: /cb2/
  common //t
  common /cb/ x(2:10) /cb2/a,b,c
  common /cb/ y,z
  common w
  common u,v
  complex w
  dimension b(4,4)
  bind(C, name="CB") /cb/
  common /b/ cb
end

!Expect: m.mod
!module m
!  logical(4)::b(1_8:4_8,1_8:4_8)
!  real(4)::t
!  real(4)::x(2_8:10_8)
!  real(4)::a
!  real(4)::c
!  real(4)::y
!  real(4)::z
!  real(4)::u
!  real(4)::v
!  complex(4)::w
!  real(4)::cb
!  common/cb2/a,b,c
!  bind(c)::/cb2/
!  common//t,w,u,v
!  common/cb/x,y,z
!  bind(c, name=1_"CB")::/cb/
!  common/b/cb
!end
