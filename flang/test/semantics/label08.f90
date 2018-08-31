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
! CHECK: IF construct name mismatch
! CHECK: DO construct name mismatch
! CHECK: CYCLE construct-name 'label3' is not in scope

subroutine sub00(a,b,n,m)
  real a(n,m)
  real b(n,m)
  labelone: do i = 1, m
     labeltwo: do j = 1, n
50      a(i,j) = b(i,j) + 2.0
        if (n .eq. m) then
           cycle label3
        end if label3
60   end do labeltwo
  end do label1
end subroutine sub00
