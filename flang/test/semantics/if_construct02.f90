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

! Check that if constructs only accept scalar logical expressions.
! TODO: expand the test to check this restriction for more types.

INTEGER :: I
LOGICAL, DIMENSION (2) :: B

!ERROR: Expected a scalar LOGICAL expression
if ( B ) then
  a = 1
end if

!ERROR: Expected a scalar LOGICAL expression
if ( B ) then
  a = 2
else
  a = 3
endif

!ERROR: Expected a scalar LOGICAL expression
if ( B ) then
  a = 4
!ERROR: Expected a scalar LOGICAL expression
else if( B ) then
  a = 5
end if

!ERROR: Expected a scalar LOGICAL expression
if ( B ) then
  a = 6
!ERROR: Expected a scalar LOGICAL expression
else if( B ) then
  a = 7
!ERROR: Expected a scalar LOGICAL expression
elseif( B ) then
  a = 8
end if

!ERROR: Expected a scalar LOGICAL expression
if ( B ) then
  a = 9
!ERROR: Expected a scalar LOGICAL expression
else if( B ) then
  a = 10
else
  a = 11
end if

!ERROR: Expected a scalar LOGICAL expression
if ( B ) then
  a = 12
!ERROR: Expected a scalar LOGICAL expression
else if( B ) then
  a = 13
!ERROR: Expected a scalar LOGICAL expression
else if( B ) then
  a = 14
end if


!ERROR: Expected a LOGICAL expression
if ( I ) then
  a = 1
end if

!ERROR: Expected a LOGICAL expression
if ( I ) then
  a = 2
else
  a = 3
endif

!ERROR: Expected a LOGICAL expression
if ( I ) then
  a = 4
!ERROR: Expected a LOGICAL expression
else if( I ) then
  a = 5
end if

!ERROR: Expected a LOGICAL expression
if ( I ) then
  a = 6
!ERROR: Expected a LOGICAL expression
else if( I ) then
  a = 7
!ERROR: Expected a LOGICAL expression
elseif( I ) then
  a = 8
end if

!ERROR: Expected a LOGICAL expression
if ( I ) then
  a = 9
!ERROR: Expected a LOGICAL expression
else if( I ) then
  a = 10
else
  a = 11
end if

!ERROR: Expected a LOGICAL expression
if ( I ) then
  a = 12
!ERROR: Expected a LOGICAL expression
else if( I ) then
  a = 13
!ERROR: Expected a LOGICAL expression
else if( I ) then
  a = 14
end if

end
