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

! 2.15.2 threadprivate Directive
! The threadprivate directive specifies that variables are replicated,
! with each thread having its own copy. When threadprivate variables are
! referenced in the OpenMP region, we know they are already private to
! their threads, so no new symbol needs to be created.

!DEF: /mm Module
module mm
  !$omp threadprivate (i)
contains
  !DEF: /mm/foo PUBLIC (Subroutine) Subprogram
  subroutine foo
    !DEF: /mm/foo/a ObjectEntity INTEGER(4)
    integer :: a = 3
    !$omp parallel
    !REF: /mm/foo/a
    a = 1
    !DEF: /mm/i PUBLIC (Implicit, OmpThreadprivate) ObjectEntity INTEGER(4)
    !REF: /mm/foo/a
    i = a
    !$omp end parallel
    !REF: /mm/foo/a
    print *, a
    block
      !DEF: /mm/foo/Block2/i ObjectEntity REAL(4)
      real i
      !REF: /mm/foo/Block2/i
      i = 3.14
    end block
  end subroutine foo
end module mm
!DEF: /tt MainProgram
program tt
  !REF: /mm
  use :: mm
  !DEF: /tt/foo (Subroutine) Use
  call foo
end program tt
