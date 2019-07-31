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

! Test DO loop semantics for constraint C1130 --
! "The constraint states that "If the locality-spec DEFAULT ( NONE ) appears in
! a DO CONCURRENT statement; a variable that is a local or construct entity of a
! scope containing the DO CONCURRENT construct; and that appears in the block of
! the construct; shall have its locality explicitly specified by that
! statement."

module m
  real :: mvar
end module m

subroutine s1()
  use m
  integer :: i, ivar, jvar, kvar
  real :: x

  jvar = 5
  
  ! References in this DO CONCURRENT are OK since there's no DEFAULT(NONE)
  ! locality-spec
  do concurrent (i = 1:2:0) shared(jvar)
    ivar = 3
    ivar = ivar + i
    block
      real :: bvar
      x = 3.5
      bvar = 3.5 + i
    end block
    jvar = 5
    mvar = 3.5
  end do
  
  do concurrent (i = 1:2:0) default(none) shared(jvar) local(kvar)
!ERROR: Variable 'ivar' from enclosing scope in a DEFAULT(NONE) DO CONCURRENT, must appear in a locality-spec
    ivar =  &
!ERROR: Variable 'ivar' from enclosing scope in a DEFAULT(NONE) DO CONCURRENT, must appear in a locality-spec
      ivar + i
    block
      real :: bvar
!ERROR: Variable 'x' from enclosing scope in a DEFAULT(NONE) DO CONCURRENT, must appear in a locality-spec
      x = 3.5
      bvar = 3.5 + i ! OK, bvar's scope is within the DO CONCURRENT
    end block
    jvar = 5 ! OK, jvar appears in a locality spec
    kvar = 5 ! OK, kvar appears in a locality spec

!ERROR: Variable 'mvar' from enclosing scope in a DEFAULT(NONE) DO CONCURRENT, must appear in a locality-spec
    mvar = 3.5
  end do

  x = 5.0  ! OK, we're not in a DO CONCURRENT
  
end subroutine s1
