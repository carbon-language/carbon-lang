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

subroutine s3
  real :: a(10,10), b(10,10)
  type y; end type
  integer(8) :: x
  !ERROR: Index name 'y' conflicts with existing identifier
  forall(x=1:10, y=1:10)
    a(x, y) = b(x, y)
  end forall
  !ERROR: Index name 'y' conflicts with existing identifier
  forall(x=1:10, y=1:10) a(x, y) = b(x, y)
end

subroutine s4
  real :: a(10), b(10)
  complex :: x
  !ERROR: Variable 'x' is not scalar integer
  forall(x=1:10)
    a(x) = b(x)
  end forall
  !ERROR: Variable 'y' is not scalar integer
  forall(y=1:10)
    a(y) = b(y)
  end forall
end

subroutine s5
  real :: a(10), b(10)
  !ERROR: 'i' is already declared in this scoping unit
  forall(i=1:10, i=1:10)
    a(i) = b(i)
  end forall
end

subroutine s6
  integer, parameter :: n = 4
  real, dimension(n) :: x
  data(x(i), i=1, n) / n * 0.0 /
  !ERROR: Index name 't' conflicts with existing identifier
  data(x(t), t=1, n) / n * 0.0 /
contains
  subroutine t
  end
end

subroutine s7
  !ERROR: 'i' is already declared in this scoping unit
  do concurrent(integer::i=1:5) local(j, i) &
      !ERROR: 'j' is already declared in this scoping unit
      local_init(k, j) &
      !ERROR: Variable 'a' not found
      shared(a)
    a(i) = j + 1
  end do
end

subroutine s8
  implicit none
  !ERROR: No explicit type declared for 'i'
  do concurrent(i=1:5) &
    !ERROR: No explicit type declared for 'j'
    local(j) &
    !ERROR: No explicit type declared for 'k'
    local_init(k)
  end do
end
