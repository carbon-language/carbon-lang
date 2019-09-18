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

! 1.4.1 Structure of the OpenMP Memory Model

! Test implicit declaration in the OpenMP directive enclosing scope
! through clause; also test to avoid creating multiple symbols for
! the same variable

  !DEF: /MainProgram1/b (Implicit) ObjectEntity REAL(4)
  b = 2
  !DEF: /MainProgram1/c (Implicit) ObjectEntity REAL(4)
  c = 0
  !$omp parallel  private(a,b) shared(c,d)
  !DEF: /MainProgram1/Block1/a (OmpPrivate) HostAssoc REAL(4)
  a = 3.
  !DEF: /MainProgram1/Block1/b (OmpPrivate) HostAssoc REAL(4)
  b = 4
  !REF: /MainProgram1/c
  c = 5
  !DEF: /MainProgram1/d (Implicit) ObjectEntity REAL(4)
  d = 6
  !$omp end parallel
  !DEF: /MainProgram1/a (Implicit) ObjectEntity REAL(4)
  print *, a
end program
