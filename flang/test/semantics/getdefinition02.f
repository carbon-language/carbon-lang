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

! Tests -fget-definition with fixed form.

      module m2
       private :: f
      contains
       pure subroutine s (x, yyy) bind(c)
        intent(in) :: 
     *  x
        intent(inout) :: yyy
       contains
        pure subroutine ss
        end subroutine
       end subroutine
       recursive pure function f() result(x)
        real, allocatable :: x
        x = 1.0
       end function
      end module

! RUN: ${F18} -fget-definition 22 9 10 -fparse-only -fdebug-semantics %s > %t;
! RUN: ${F18} -fget-definition 23 26 29 -fparse-only -fdebug-semantics %s >> %t;
! RUN: ${F18} -fget-definition 30 9 10 -fparse-only -fdebug-semantics %s >> %t;
! RUN: cat %t | ${FileCheck} %s
! CHECK:x:.*getdefinition02.f, 20, 27-28
! CHECK:yyy:.*getdefinition02.f, 20, 30-33
! CHECK:x:.*getdefinition02.f, 29, 30-31
