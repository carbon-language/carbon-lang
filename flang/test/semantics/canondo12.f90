! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

! Error test -- DO loop uses obsolete loop termination statement
! See R1131 and C1133

! By default, this is not an error and label do are rewritten to non-label do.
! A warning is generated with -Mstandard

! RUN: ${F18} -funparse-with-symbols -Mstandard %s 2>&1 | ${FileCheck} %s

! CHECK: end do

! The following CHECK-NOT actively uses the fact that the leading zero of labels
! would be removed in the unparse but not the line linked to warnings. We do
! not want to see label do in the unparse only.
! CHECK-NOT: do [1-9]

! CHECK: A DO loop should terminate with an END DO or CONTINUE

subroutine foo4()
  real :: a(10, 10), b(10, 10) = 1.0
  do 01 k=1,4
    block
      real b
      b = a(k, k)
      a(k, k) = k*b
01  end block
end subroutine
