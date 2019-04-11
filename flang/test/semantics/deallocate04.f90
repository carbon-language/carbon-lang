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

! Check for semantic errors in DEALLOCATE statements

INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt) :: p
End Type

Type(t),Allocatable :: x(:)

Real :: r
Integer :: s
Integer :: e
Integer :: pi
Character(256) :: ee
Procedure(Real) :: prp

Allocate(x(3))

!ERROR: component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
Deallocate(x(2)%p)

!ERROR: name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
Deallocate(pi)

!ERROR: component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
!ERROR: name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
Deallocate(x(2)%p, pi)

!ERROR: name in DEALLOCATE statement must be a variable name
Deallocate(prp)

!ERROR: name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
!ERROR: name in DEALLOCATE statement must be a variable name
Deallocate(pi, prp)

!ERROR: name in DEALLOCATE statement must be a variable name
Deallocate(maxvalue)

!ERROR: component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
Deallocate(x%p)

!ERROR: component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
!ERROR: Must have default CHARACTER type
Deallocate(x%p, stat=s, errmsg=e)

!ERROR: component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute
!ERROR: Must have INTEGER type
!ERROR: Must have default CHARACTER type
Deallocate(x%p, stat=r, errmsg=e)

End Program
