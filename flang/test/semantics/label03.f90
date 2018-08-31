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
! CHECK: 'do 10 i = 1, m' doesn't properly nest
! CHECK: label '30' cannot be found
! CHECK: label '40' cannot be found
! CHECK: label '50' doesn't lexically follow DO stmt

subroutine sub00(a,b,n,m)
  real a(n,m)
  real b(n,m)
  do 10 i = 1, m
     do 20 j = 1, n
        a(i,j) = b(i,j) + 2.0
10      continue
20      continue
end subroutine sub00

subroutine sub01(a,b,n,m)
  real a(n,m)
  real b(n,m)
  do 30 i = 1, m
     do 40 j = 1, n
        a(i,j) = b(i,j) + 10.0
35      continue
45      continue
end subroutine sub01

subroutine sub02(a,b,n,m)
  real a(n,m)
  real b(n,m)
50      continue
  do 50 i = 1, m
     do 60 j = 1, n
        a(i,j) = b(i,j) + 20.0
60      continue
end subroutine sub02
