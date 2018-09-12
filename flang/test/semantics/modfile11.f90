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

module m
  type t1(a, b, c)
    integer, kind :: a
    integer(8), len :: b, c
    integer :: d
  end type
  type, extends(t1) :: t2(e)
    integer, len :: e
  end type
  type, extends(t2), bind(c) :: t3
  end type
end

!Expect: m.mod
!module m
!  type::t1(a,b,c)
!    integer(4),kind::a
!    integer(8),len::b
!    integer(8),len::c
!    integer(4)::d
!  end type
!  type,extends(t1)::t2(e)
!    integer(4),len::e
!  end type
!  type,bind(c),extends(t2)::t3
!  end type
!end
