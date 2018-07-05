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

subroutine test1
  !ERROR: Generic interface 'foo' has both a function and a subroutine
  interface foo
    subroutine s1(x)
    end subroutine
    subroutine s2(x, y)
    end subroutine
    function f()
    end function
  end interface
end subroutine

subroutine test2
  !ERROR: Generic interface 'foo' has both a function and a subroutine
  interface foo
    function f1(x)
    end function
    subroutine s()
    end subroutine
    function f2(x, y)
    end function
  end interface
end subroutine

module test3
  !ERROR: Generic interface 'foo' has both a function and a subroutine
  interface foo
    module procedure s
    module procedure f
  end interface
contains
  subroutine s(x)
  end subroutine
  function f()
  end function
end module

subroutine test4
  type foo
  end type
  !ERROR: Generic interface 'foo' may only contain functions due to derived type with same name
  interface foo
    subroutine s()
    end subroutine
  end interface
end subroutine

subroutine test5
  interface foo
    function f1()
    end function
  end interface
  interface bar
    subroutine s1()
    end subroutine
    subroutine s2(x)
    end subroutine
  end interface
  !ERROR: Cannot call function 'foo' like a subroutine
  call foo()
  !ERROR: Cannot call subroutine 'bar' like a function
  x = bar()
end subroutine
