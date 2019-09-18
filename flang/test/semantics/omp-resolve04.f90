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
! A list item that specifies a given variable may not appear in more than
! one clause on the same directive, except that a variable may be specified
! in both firstprivate and lastprivate clauses.

  common /c/ a, b
  integer a(3), b

  A = 1
  B = 2
  !ERROR: 'c' appears in more than one data-sharing clause on the same OpenMP directive
  !$omp parallel shared(/c/,c) private(/c/)
  a(1:2) = 3
  B = 4
  !$omp end parallel
  print *, a, b, c
end
