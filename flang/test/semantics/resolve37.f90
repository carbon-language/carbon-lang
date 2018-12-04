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

integer, parameter :: k = 8
real, parameter :: l = 8.0
integer :: n = 2
!ERROR: Must be a constant value
parameter(m=n)
integer(k) :: x
!ERROR: Must have INTEGER type
integer(l) :: y
!ERROR: Must be a constant value
integer(n) :: z
type t(k)
  integer, kind :: k
end type
!ERROR: Type parameter 'k' lacks a value and has no default
type(t( &
!ERROR: Must have INTEGER type
  .true.)) :: w
!ERROR: Must have INTEGER type
real :: w(l*2)
!ERROR: Must have INTEGER type
character(len=l) :: v
end
