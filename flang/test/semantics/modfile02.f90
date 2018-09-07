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

! Check modfile generation for private type in public API.

module m
  type, private :: t1
    integer :: i
  end type
  type, private :: t2
    integer :: i
  end type
  type(t1) :: x1
  type(t2), private :: x2
end

!Expect: m.mod
!module m
!type,private::t1
!integer::i
!end type
!type,private::t2
!integer::i
!end type
!type(t1)::x1
!type(t2),private::x2
!end
