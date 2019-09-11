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
! In the inner OpenMP region, SHARED `a` refers to the `a` in the outer OpenMP
! region; PRIVATE `b` refers to the new `b` in the same OpenMP region

  !DEF: /MainProgram1/b (Implicit) ObjectEntity REAL(4)
  b = 2
  !$omp parallel  private(a) shared(b)
  !DEF: /MainProgram1/Block1/a (OmpPrivate) HostAssoc REAL(4)
  a = 3.
  !REF: /MainProgram1/b
  b = 4
  !$omp parallel  private(b) shared(a)
  !REF: /MainProgram1/Block1/a
  a = 5.
  !DEF: /MainProgram1/Block1/Block1/b (OmpPrivate) HostAssoc REAL(4)
  b = 6
  !$omp end parallel
  !$omp end parallel
  !DEF: /MainProgram1/a (Implicit) ObjectEntity REAL(4)
  !REF: /MainProgram1/b
  print *, a, b
end program
