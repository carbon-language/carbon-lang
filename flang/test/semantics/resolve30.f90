! Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
  integer x
  block
    import, none
    !ERROR: 'x' from host scoping unit is not accessible due to IMPORT
    x = 1
  end block
end

subroutine s2
  block
    import, none
    !ERROR: 'y' from host scoping unit is not accessible due to IMPORT
    y = 1
  end block
end

subroutine s3
  implicit none
  integer :: i, j
  block
    import, none
    !ERROR: No explicit type declared for 'i'
    real :: a(16) = [(i, i=1, 16)]
    !ERROR: No explicit type declared for 'j'
    data(a(j), j=1, 16) / 16 * 0.0 /
  end block
end

subroutine s4
  real :: i, j
  !ERROR: Variable 'i' is not integer
  real :: a(16) = [(i, i=1, 16)]
  !ERROR: Variable 'j' is not integer
  data(a(j), j=1, 16) / 16 * 0.0 /
end
