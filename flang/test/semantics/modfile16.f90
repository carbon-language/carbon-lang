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
  character(2), parameter :: prefix = 'c_'
  integer, bind(c, name='c_a') :: a
  procedure(sub), bind(c, name=prefix//'b'), pointer :: b
  type, bind(c) :: t
    real :: c
  end type
  real :: d
  external :: d
  bind(c, name='dd') :: d
  real :: e
  bind(c, name='ee') :: e
  external :: e
  bind(c, name='ff') :: f
  real :: f
  external :: f
contains
  subroutine sub() bind(c, name='sub')
  end
end

!Expect: m.mod
!module m
!  character(2_4,1),parameter::prefix=1_"c_"
!  integer(4),bind(c, name=1_"c_a")::a
!  procedure(sub),bind(c, name=1_"c_b"),pointer::b
!  type,bind(c)::t
!    real(4)::c
!  end type
!  procedure(real(4)),bind(c, name=1_"dd")::d
!  procedure(real(4)),bind(c, name=1_"ee")::e
!  procedure(real(4)),bind(c, name=1_"ff")::f
!contains
!  subroutine sub(),bind(c, name=1_"sub")
!  end
!end
