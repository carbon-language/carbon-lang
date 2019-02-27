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

function f1(x, y)
  integer x
  !ERROR: SAVE attribute may not be applied to dummy argument 'x'
  !ERROR: SAVE attribute may not be applied to dummy argument 'y'
  save x,y
  integer y
  !ERROR: SAVE attribute may not be applied to function result 'f1'
  save f1
end

function f2(x, y)
  !ERROR: SAVE attribute may not be applied to function result 'f2'
  real, save :: f2
  !ERROR: SAVE attribute may not be applied to dummy argument 'x'
  complex, save :: x
  allocatable :: y
  !ERROR: SAVE attribute may not be applied to dummy argument 'y'
  integer, save :: y
end

subroutine s3(x)
  !ERROR: SAVE attribute may not be applied to dummy argument 'x'
  procedure(integer), pointer, save :: x
  !ERROR: Procedure 'y' with SAVE attribute must also have POINTER attribute
  procedure(integer), save :: y
end

subroutine s4
  !ERROR: Explicit SAVE of 'z' is redundant due to global SAVE statement
  save z
  save
  procedure(integer), pointer :: x
  !ERROR: Explicit SAVE of 'x' is redundant due to global SAVE statement
  save :: x
  !ERROR: Explicit SAVE of 'y' is redundant due to global SAVE statement
  integer, save :: y
end

subroutine s5
  implicit none
  integer x
  block
    !ERROR: No explicit type declared for 'x'
    save x
  end block
end

subroutine s6
  save x
  save y
  !ERROR: SAVE attribute was already specified on 'y'
  integer, save :: y
  integer, save :: z
  !ERROR: SAVE attribute was already specified on 'x'
  !ERROR: SAVE attribute was already specified on 'z'
  save x,z
end

subroutine s7
  !ERROR: 'x' appears as a COMMON block in a SAVE statement but not in a COMMON statement
  save /x/
end
