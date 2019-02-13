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

module m
  implicit none
  real, parameter :: a = 8.0
  !ERROR: Must have INTEGER type
  integer :: aa = 2_a
  integer :: b = 8
  !ERROR: Must be a constant value
  integer :: bb = 2_b
  !TODO: should get error -- not scalar
  !integer, parameter :: c(10) = 8
  !integer :: cc = 2_c
  integer, parameter :: d = 47
  !ERROR: INTEGER(KIND=47) is not a supported type
  integer :: dd = 2_d
  !ERROR: Parameter 'e' not found
  integer :: ee = 2_e
  !ERROR: Missing initialization for parameter 'f'
  integer, parameter :: f
  integer :: ff = 2_f
  !ERROR: REAL(KIND=23) is not a supported type
  real(d/2) :: g
  !ERROR: REAL*47 is not a supported type
  real*47 :: h
  !ERROR: COMPLEX*47 is not a supported type
  complex*47 :: i
end
