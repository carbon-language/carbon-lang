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

integer :: y
procedure() :: a
procedure(real) :: b
call a  ! OK - can be function or subroutine
!ERROR: Cannot call subroutine 'a' like a function
c = a()
!ERROR: Cannot call function 'b' like a subroutine
call b
!ERROR: Cannot call function 'y' like a subroutine
call y
call x
!ERROR: Cannot call subroutine 'x' like a function
z = x()
end

subroutine s
  !ERROR: Cannot call function 'f' like a subroutine
  call f
  !ERROR: Cannot call subroutine 's' like a function
  i = s()
contains
  function f()
  end
end

subroutine s2
  ! subroutine vs. function is determined by use
  external :: a, b
  call a()
  !ERROR: Cannot call subroutine 'a' like a function
  x = a()
  x = b()
  !ERROR: Cannot call function 'b' like a subroutine
  call b()
end

subroutine s3
  ! subroutine vs. function is determined by use, even in internal subprograms
  external :: a
  procedure() :: b
contains
  subroutine s3a()
    x = a()
    call b()
  end
  subroutine s3b()
    !ERROR: Cannot call function 'a' like a subroutine
    call a()
    !ERROR: Cannot call subroutine 'b' like a function
    x = b()
  end
end

module m
  ! subroutine vs. function is determined at end of specification part
  external :: a
  procedure() :: b
contains
  subroutine s()
    call a()
    !ERROR: Cannot call subroutine 'b' like a function
    x = b()
  end
end

! Call to entity in global scope, even with IMPORT, NONE
subroutine s4
  block
    import, none
    integer :: i
    !ERROR: Use of 'm' as a procedure conflicts with its declaration
    i = m()
    !ERROR: Use of 'm' as a procedure conflicts with its declaration
    call m()
  end block
end

! Call to entity in global scope, even with IMPORT, NONE
subroutine s5
  block
    import, none
    integer :: i
    i = foo()
    !ERROR: Cannot call function 'foo' like a subroutine
    call foo()
  end block
end
