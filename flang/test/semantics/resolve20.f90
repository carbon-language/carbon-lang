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
  abstract interface
    subroutine foo
    end subroutine
  end interface

  procedure() :: a
  procedure(integer) :: b
  procedure(foo) :: c
  procedure(bar) :: d
  !ERROR: 'missing' must be an abstract interface or a procedure with an explicit interface
  procedure(missing) :: e
  !ERROR: 'b' must be an abstract interface or a procedure with an explicit interface
  procedure(b) :: f
  procedure(c) :: g
  external :: h
  !ERROR: 'h' must be an abstract interface or a procedure with an explicit interface
  procedure(h) :: i
  procedure(forward) :: j
  !ERROR: 'bad1' must be an abstract interface or a procedure with an explicit interface
  procedure(bad1) :: k1
  !ERROR: 'bad2' must be an abstract interface or a procedure with an explicit interface
  procedure(bad2) :: k2
  !ERROR: 'bad3' must be an abstract interface or a procedure with an explicit interface
  procedure(bad3) :: k3

  abstract interface
    subroutine forward
    end subroutine
  end interface

  real :: bad1(1)
  real :: bad2
  type :: bad3
  end type

  external :: a, b, c, d
  !ERROR: EXTERNAL attribute not allowed on 'm'
  external :: m
  !ERROR: EXTERNAL attribute not allowed on 'foo'
  external :: foo
  !ERROR: EXTERNAL attribute not allowed on 'bar'
  external :: bar

  !ERROR: PARAMETER attribute not allowed on 'm'
  parameter(m=2)
  !ERROR: PARAMETER attribute not allowed on 'foo'
  parameter(foo=2)
  !ERROR: PARAMETER attribute not allowed on 'bar'
  parameter(bar=2)

contains
  subroutine bar
  end subroutine
end module
