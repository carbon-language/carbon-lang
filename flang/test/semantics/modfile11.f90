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
  type t(a, b, c)
    integer, kind :: a
    integer(8), len :: b, c
    integer :: d
  end type
end

!Expect: m.mod
!module m
!  type::t(a,b,c)
!    integer,kind::a
!    integer(8),len::b
!    integer(8),len::c
!    integer::d
!  end type
!end
