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

! Construct names

subroutine s1
  real :: foo
  !ERROR: 'foo' is already declared in this scoping unit
  foo: block
  end block foo
end

subroutine s2(x)
  logical :: x
  foo: if (x) then
  end if foo
  !ERROR: 'foo' is already declared in this scoping unit
  foo: do i = 1, 10
  end do foo
end
