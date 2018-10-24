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
! CHECK: branch into loop body from outside
! CHECK: the loop branched into

subroutine s(a)
  integer i
  real a(10)
  do 10 i = 1,10
     if (a(i) < 0.0) then
        goto 20
     end if
30   continue
     a(i) = 1.0
10 end do
  goto 40
20 a(i) = -a(i)
  goto 30
40 continue
end subroutine s
