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

! Test that derived type component does not hide type in host.

!DEF: /m Module
module m
 !DEF: /m/t1 PUBLIC DerivedType
 type :: t1
 end type
 !DEF: /m/t2 PUBLIC DerivedType
 type :: t2
  !DEF: /m/t2/t1 ObjectEntity INTEGER(4)
  integer :: t1
  !DEF: /m/t2/x ObjectEntity TYPE(t1)
  !REF: /m/t1
  type(t1) :: x
 end type
end module
