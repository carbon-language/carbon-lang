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

! Error test -- DO loop uses obsolete loop termination statement
! See R1131 and C1131

! RUN: ${F18} -funparse-with-symbols -Mstandard %s 2>&1 | ${FileCheck} %s

module iso_fortran_env
  type :: team_type
  end type
end

subroutine foo0()
  do 1 j=1,2
    if (.true.) then
! CHECK: A DO loop should terminate with an END DO or CONTINUE
1   end if
  do 2 k=1,2
    do i=3,4
      print*, i+k
! CHECK: A DO loop should terminate with an END DO or CONTINUE
2    end do
  do 3 l=1,2
    do 3 m=1,2
      select case (l)
      case default
        print*, "default", m, l
      case (1)
        print*, "start"
! CHECK: A DO loop should terminate with an END DO or CONTINUE
! CHECK: A DO loop should terminate with an END DO or CONTINUE
3     end select
end subroutine

subroutine foo1()
  real :: a(10, 10), b(10, 10) = 1.0
  do 4 k=1,2
    forall (i = 1:10, j = 1:10, b(i, j) /= 0.0)
      a(i, j) = real (i + j - k)
      b(i, j) = a(i, j) + b(i, j) * real (i * j)
! CHECK: A DO loop should terminate with an END DO or CONTINUE
4   end forall
end subroutine

subroutine foo2()
  real :: a(10, 10), b(10, 10) = 1.0
  do 4 k=1,4
    where (a<k)
      a = a + b
      b = a - b
    elsewhere
      a = a*2
! CHECK: A DO loop should terminate with an END DO or CONTINUE
4   end where
end subroutine

subroutine foo3()
  real :: a(10, 10), b(10, 10) = 1.0
  do 4 k=1,4
    associate (x=>a(k+1, 2*k), y=>b(k, 2*k-1))
      x = 4*x*x + x*y -2*y
! CHECK: A DO loop should terminate with an END DO or CONTINUE
4   end associate
end subroutine

subroutine foo4()
  real :: a(10, 10), b(10, 10) = 1.0
  do 4 k=1,4
    block
      real b
      b = a(k, k)
      a(k, k) = k*b
! CHECK: A DO loop should terminate with an END DO or CONTINUE
4   end block
end subroutine

subroutine foo5()
  real :: a(10, 10), b(10, 10) = 1.0
  do 4 k=1,4
    critical
      b(k+1, k) = a(k, k+1)
! CHECK: A DO loop should terminate with an END DO or CONTINUE
4   end critical
end subroutine

subroutine foo6(a)
  type whatever
    class(*), allocatable :: x
  end type
  type(whatever) :: a(10)
  do 4 k=1,10
    select type (ax => a(k)%x)
      type is (integer)
        print*, "integer: ", ax
      class default
        print*, "not useable"
! CHECK: A DO loop should terminate with an END DO or CONTINUE
4   end select
end subroutine

subroutine foo7(a)
  integer :: a(..)
  do 4 k=1,10
    select rank (a)
      rank (0)
        a = a+k
      rank (1)
        a(k) = a(k)+k
      rank default
        print*, "error"
! CHECK: A DO loop should terminate with an END DO or CONTINUE
4   end select
end subroutine

subroutine foo8()
  use  :: iso_fortran_env, only : team_type
  type(team_type) :: odd_even
  do 1 k=1,10
    change team (odd_even)
! CHECK: A DO loop should terminate with an END DO or CONTINUE
1   end team
end subroutine

program endDo
  do 10 i = 1, 5
! CHECK: A DO loop should terminate with an END DO or CONTINUE
10  print *, "in loop"
end program endDo
