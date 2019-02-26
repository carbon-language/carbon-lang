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

module m
  integer :: foo
  !Note: PGI, Intel, and GNU allow this; NAG and Sun do not
  !ERROR: 'foo' is already declared in this scoping unit
  interface foo
  end interface
end module

module m2
  !Note: PGI and GNU allow this; Intel, NAG, and Sun do not
  !ERROR: 's' is already declared in this scoping unit
  interface s
  end interface
contains
  subroutine s
  end subroutine
end module

module m3
  ! This is okay: so is generic and specific
  interface s
    procedure s2
  end interface
  interface s
    procedure s
  end interface
contains
  subroutine s()
  end subroutine
  subroutine s2(x)
  end subroutine
end module
