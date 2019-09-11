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

!OPTIONS: -fopenmp

! 2.15.3 Data-Sharing Attribute Clauses
! Both PARALLEL and DO (worksharing) directives need to create new scope,
! so PRIVATE `a` will have new symbol in each region

  !DEF: /MainProgram1/a ObjectEntity REAL(8)
  real*8 a
  !REF: /MainProgram1/a
  a = 3.14
  !$omp parallel  private(a)
  !DEF: /MainProgram1/Block1/a (OmpPrivate) HostAssoc REAL(8)
  a = 2.
  !$omp do  private(a)
  !DEF: /MainProgram1/i (Implicit) ObjectEntity INTEGER(4)
  do i=1,10
     !DEF: /MainProgram1/Block1/Block1/a (OmpPrivate) HostAssoc REAL(8)
     a = 1.
  end do
  !$omp end parallel
  !REF: /MainProgram1/a
  print *, a
end program
