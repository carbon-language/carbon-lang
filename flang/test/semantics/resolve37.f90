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

integer, parameter :: k = 8
real, parameter :: l = 8.0
integer :: n = 2
!ERROR: expression must be constant
parameter(m=n)
integer(k) :: x
!ERROR: expression must be INTEGER
integer(l) :: y
!ERROR: expression must be constant
integer(n) :: z
type t(k)
  integer, kind :: k
end type
!ERROR: expression must be INTEGER
type(t(.true.)) :: w
!ERROR: expression must be INTEGER
real :: w(l*2)
end
