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

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: image control statement not allowed in DO CONCURRENT
! CHECK: SYNC ALL
! CHECK: do-variable must have INTEGER type

subroutine do_concurrent_test1(i,n)
  implicit none
  integer :: i, n
  real :: j
  do 20 j = 1, 20
     do 10 concurrent (i = 1:n)
        SYNC ALL
10   continue
20 enddo
end subroutine do_concurrent_test1
