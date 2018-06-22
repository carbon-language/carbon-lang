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
  !OK: interface followed by type with same name
  interface t
  end interface
  type t
  end type
  type(t) :: x
  x = t()
end subroutine

subroutine s2
  !OK: type followed by interface with same name
  type t
  end type
  interface t
  end interface
  type(t) :: x
  x = t()
end subroutine

subroutine s3
  type t
  end type
  interface t
  end interface
  !ERROR: 't' is already declared in this scoping unit
  type t
  end type
  type(t) :: x
  x = t()
end subroutine
