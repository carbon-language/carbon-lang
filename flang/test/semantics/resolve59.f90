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

! Testing 15.6.2.2 point 4 (What function-name refers to depending on the
! presence of RESULT).


module m_no_result
! Without RESULT, it refers to the result object (no recursive
! calls possible)
contains
  ! testing with data object results
  function f1()
    real :: x, f1
    !ERROR: Reference to array 'f1' with empty subscript list
    x = acos(f1())
    f1 = x
    x = acos(f1) !OK
  end function
  function f2(i)
    integer i
    real :: x, f2
    !ERROR: Reference to rank-0 object 'f2' has 1 subscripts
    x = acos(f2(i+1))
    f2 = x
    x = acos(f2) !OK
  end function
  function f3(i)
    integer i
    real :: x, f3(1)
    ! OK reference to array result f1
    x = acos(f3(i+1))
    f3 = x
    x = sum(acos(f3)) !OK
  end function

  ! testing with function pointer results
  function rf()
    real :: rf
  end function
  function f4()
    procedure(rf), pointer :: f4
    f4 => rf
    ! OK call to f4 pointer (rf)
    x = acos(f4())
    !ERROR: Typeless item not allowed for 'x=' argument
    x = acos(f4)
  end function
  function f5(x)
    real :: x
    procedure(acos), pointer :: f5
    f5 => acos
    ! OK call to f5 pointer (acos)
    x = acos(f5(x+1))
    !ERROR: Typeless item not allowed for 'x=' argument
    x = acos(f5)
  end function
end module

module m_with_result
! With RESULT, it refers to the function (recursive calls possible)
contains
  ! testing with data object results
  function f1() result(r)
    real :: r
    r = acos(f1()) !OK, recursive call
    !ERROR: Typeless item not allowed for 'x=' argument
    x = acos(f1)
  end function
  function f2(i) result(r)
    integer i
    real :: r
    r = acos(f2(i+1)) ! OK, recursive call
    !ERROR: Typeless item not allowed for 'x=' argument
    r = acos(f2)
  end function
  function f3(i) result(r)
    integer i
    real :: r(1)
    r = acos(f3(i+1)) !OK recursive call
    !ERROR: Typeless item not allowed for 'x=' argument
    r = sum(acos(f3))
  end function

  ! testing with function pointer results
  function rf()
    real :: rf
  end function
  function f4() result(r)
    real :: x
    procedure(rf), pointer :: r
    r => rf
    !ERROR: Typeless item not allowed for 'x=' argument
    x = acos(f4()) ! recursive call
    !ERROR: Typeless item not allowed for 'x=' argument
    x = acos(f4)
    x = acos(r()) ! OK
  end function
  function f5(x) result(r)
    real :: x
    procedure(acos), pointer :: r
    r => acos
    !ERROR: Typeless item not allowed for 'x=' argument
    x = acos(f5(x+1)) ! recursive call
    !ERROR: Typeless item not allowed for 'x=' argument
    x = acos(f5)
    x = acos(r(x+1)) ! OK
  end function

  ! testing that calling the result is also caught
  function f6() result(r)
    real :: x, r
    !ERROR: Reference to array 'r' with empty subscript list
    x = r()
  end function
end module
