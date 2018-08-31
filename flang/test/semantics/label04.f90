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
! CHECK: branch into 'do 10 i = 1, m' from another scope
! CHECK: branch into 'do 20 j = 1, n' from another scope

subroutine sub00(a,b,n,m)
  real a(n,m)
  real b(n,m)
  if (n .ne. m) then
     goto 50
  end if
  do 10 i = 1, m
     do 20 j = 1, n
50      a(i,j) = b(i,j) + 2.0
20      continue
10      continue
end subroutine sub00
