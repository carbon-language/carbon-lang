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

! 2.15.3 Although variables in common blocks can be accessed by use association
! or host association, common block names cannot. As a result, a common block
! name specified in a data-sharing attribute clause must be declared to be a
! common block in the same scoping unit in which the data-sharing attribute
! clause appears.

  common /c/ a, b
  integer a(3), b

  A = 1
  B = 2
  block
    !ERROR: COMMON block must be declared in the same scoping unit in which the OpenMP directive or clause appears
    !$omp parallel shared(/c/)
    a(1:2) = 3
    B = 4
    !$omp end parallel
  end block
  print *, a, b
end
