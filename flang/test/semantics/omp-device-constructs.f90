! Copyright (c) 2019, Arm Ltd.  All rights reserved.
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

! OPTIONS: -fopenmp
! Check OpenMP clause validity for the following directives:
!     2.10 Device constructs
program main

  real(8) :: arrayA(256), arrayB(256)
  integer :: N

  arrayA = 1.414
  arrayB = 3.14
  N = 256

  !$omp target map(arrayA)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target device(0)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: At most one DEVICE clause can appear on the TARGET directive
  !$omp target device(0) device(1)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: SCHEDULE clause is not allowed on the TARGET directive
  !$omp target schedule(static)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target defaultmap(tofrom:scalar)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: The argument TOFROM:SCALAR must be specified on the DEFAULTMAP clause
  !$omp target defaultmap(tofrom)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: At most one DEFAULTMAP clause can appear on the TARGET directive
  !$omp target defaultmap(tofrom:scalar) defaultmap(tofrom:scalar)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

end program main
