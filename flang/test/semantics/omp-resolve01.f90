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

! 2.4 An array section designates a subset of the elements in an array. Although
! Substring shares similar syntax but cannot be treated as valid array section.

  character*8 c, b
  character a

  b = "HIFROMPGI"
  c = b(2:7)
  !ERROR: Fortran Substrings are not allowed on OpenMP directives or clauses
  !$omp parallel private(c(1:3))
  a = c(1:1)
  !$omp end parallel
end
