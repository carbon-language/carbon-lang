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

! Confirm enforcement of constraints and restrictions in 15.6.2.1

non_recursive function f01(n) result(res)
  integer, value :: n
  integer :: res
  if (n <= 0) then
    res = n
  else
    !ERROR: non recursive function can't recurse
    res = n * f01(n-1) ! 15.6.2.1(3)
  end if
end function

non_recursive function f02(n) result(res)
  integer, value :: n
  integer :: res
  if (n <= 0) then
    res = n
  else
    res = nested()
  end if
 contains
  integer function nested
    !ERROR: non recursive function can't recurse
    nested = n * f02(n-1) ! 15.6.2.1(3)
  end function nested
end function

! ERROR: assumed-length character function cannot be RECURSIVE
recursive character(*) function f03(n) ! C723
  integer, value :: n
  f03 = ''
end function

recursive function f04(n) result(res) ! C723
  integer, value :: n
  ! ERROR: assumed-length character function cannot be RECURSIVE
  character(*) :: res
  res = ''
end function

character(*) function f05()
  ! ERROR: assumed-length character function cannot return an array
  dimension :: f05(1) ! C723
  f05(1) = ''
end function

function f06()
  ! ERROR: assumed-length character function cannot return an array
  character(*) :: f06(1) ! C723
  f06(1) = ''
end function

character(*) function f07()
  ! ERROR: assumed-length character function cannot return a POINTER
  pointer :: f07 ! C723
  character, target :: a = ' '
  f07 => a
end function

function f08()
  ! ERROR: assumed-length character function cannot return a POINTER
  character(*), pointer :: f08 ! C723
  character, target :: a = ' '
  f08 => a
end function

! ERROR: assumed-length character function cannot be declared PURE
pure character(*) function f09() ! C723
  f09 = ''
end function

pure function f10()
  ! ERROR: assumed-length character function cannot be declared PURE
  character(*) :: f10 ! C723
  f10 = ''
end function

! ERROR: assumed-length character function cannot be declared ELEMENTAL
elemental character(*) function f11(n) ! C723
  integer, value :: n
  f11 = ''
end function

elemental function f12(n)
  ! ERROR: assumed-length character function cannot be declared ELEMENTAL
  character(*) :: f12 ! C723
  integer, value :: n
  f12 = ''
end function

function f13(n) result(res)
  integer, value :: n
  character(*) :: res
  if (n <= 0) then
    res = ''
  else
    !ERROR: assumed-length CHARACTER(*) function can't recurse
    res = f13(n-1) ! 15.6.2.1(3)
  end if
end function

function f14(n) result(res)
  integer, value :: n
  character(*) :: res
  if (n <= 0) then
    res = ''
  else
    res = nested()
  end if
 contains
  character(1) function nested
    !ERROR: assumed-length CHARACTER(*) function can't recurse
    nested = f14(n-1) ! 15.6.2.1(3)
  end function nested
end function
