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

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: Previous declaration of 'i'
! CHECK: Previous declaration of 'j'

subroutine forall
  real :: a(9)
! ERROR: 'i' is already declared in this scoping unit
  forall (i=1:8, i=1:9)  a(i) = i
  forall (j=1:8)
! ERROR: 'j' is already declared in this scoping unit
    forall (j=1:9)
    end forall
  end forall
end subroutine forall
