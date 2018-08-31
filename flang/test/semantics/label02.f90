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

! negative test -- invalid labels, out of range

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: label '0' is out of range
! CHECK: label '100000' is out of range
! CHECK: label '123456' is out of range
! CHECK: label '123456' was not found
! CHECK: label '1000' is not distinct

subroutine sub00(a,b,n,m)
  real a(n)
  real :: b(m)
0 print *, "error"
100000 print *, n
  goto 123456
1000 print *, m
1000 print *, m+1
end subroutine sub00
