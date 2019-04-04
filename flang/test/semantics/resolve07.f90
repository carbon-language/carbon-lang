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

subroutine s1
  integer :: x(2)
  !ERROR: The dimensions of 'x' have already been declared
  allocatable :: x(:)
  real :: y[1:*]
  !ERROR: The codimensions of 'y' have already been declared
  allocatable :: y[:]
end

subroutine s2
  target :: x(1)
  !ERROR: The dimensions of 'x' have already been declared
  integer :: x(2)
  target :: y[1:*]
  !ERROR: The codimensions of 'y' have already been declared
  integer :: y[2:*]
end

subroutine s3
  dimension :: x(4), x2(8)
  !ERROR: The dimensions of 'x' have already been declared
  allocatable :: x(:)
  codimension :: y[*], y2[1:2,2:*]
  !ERROR: The codimensions of 'y' have already been declared
  allocatable :: y[:]
end

subroutine s4
  integer, dimension(10) :: x(2,2), y
end
