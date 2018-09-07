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
!ERROR: Use of 'y' as a procedure conflicts with its declaration
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
