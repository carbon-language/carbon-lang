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

! Test modfiles for entities with initialization
module m
  integer, parameter :: k8 = 8
  integer(8), parameter :: k4 = k8/2
  integer, parameter :: k1 = 1
  integer(k8), parameter :: i = 2_k8
  real :: r = 2.0_k4
  character(10, kind=k1) :: c = k1_"asdf"
  character(10), parameter :: c2 = k1_"qwer"
  complex*16, parameter :: z = (1.0_k8, 2.0_k8)
  type t
    integer :: a = 123
    type(t), pointer :: b => null()
  end type
  type(t), parameter :: x = t(456)
  type(t), parameter :: y = t(789, null())
end

!Expect: m.mod
!module m
!  integer(4),parameter::k8=8_4
!  integer(8),parameter::k4=4_8
!  integer(4),parameter::k1=1_4
!  integer(8),parameter::i=2_8
!  real(4)::r
!  character(10_4,1)::c
!  character(10_4,1),parameter::c2=1_"qwer      "
!  complex(8),parameter::z=(1._8,2._8)
!  type::t
!    integer(4)::a=123_4
!    type(t),pointer::b=>NULL()
!  end type
!  type(t),parameter::x=t(a=456_4,b=NULL())
!  type(t),parameter::y=t(a=789_4,b=NULL())
!end
