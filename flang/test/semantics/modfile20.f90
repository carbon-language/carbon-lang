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

! Check modfile generation for generic interfaces
module m
  integer, parameter :: k8 = 8
  integer(8), parameter :: k4 = k8/2
  integer, parameter :: k1 = 1
  integer(k8), parameter :: i = 2_k8
  real :: r = 2.0_k4
  character(10, kind=k1) :: c = k1_"asdf"
  complex*16 :: z = (1.0_k8, 2.0_k8)
end

!Expect: m.mod
!module m
!  integer(4),parameter::k8=8_4
!  integer(8),parameter::k4=4_8
!  integer(4),parameter::k1=1_4
!  integer(8),parameter::i=2_8
!  real(4)::r=2._4
!  character(10_4,1)::c=1_"asdf      "
!  complex(8)::z=(1._8,2._8)
!end
