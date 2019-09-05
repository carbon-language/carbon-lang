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

! Tests -fget-symbols-sources with BLOCK that contains same variable name as 
! another in an outer scope.

program main
  integer :: x
  integer :: y
  block
    integer :: x
    integer :: y
    x = y
  end block
  x = y
end program

!! Inner x
! RUN: ${F18} -fget-definition 24 5 6 -fparse-only -fdebug-semantics %s > %t;
! CHECK:x:.*getdefinition05.f90, 22, 16-17
!! Outer y
! RUN: ${F18} -fget-definition 26 7 8 -fparse-only -fdebug-semantics %s >> %t;
! CHECK:y:.*getdefinition05.f90, 20, 14-15
! RUN: cat %t | ${FileCheck} %s;
