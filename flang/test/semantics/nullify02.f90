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

! Check for semantic errors in NULLIFY statements

INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt) :: p
End Type

Type(t),Allocatable :: x(:)

Integer :: pi
Procedure(Real) :: prp

Allocate(x(3))
!ERROR: component in NULLIFY statement must have the POINTER attribute
Nullify(x(2)%p)

!ERROR: name in NULLIFY statement must have the POINTER attribute
Nullify(pi)

!ERROR: name in NULLIFY statement must have the POINTER attribute
Nullify(prp)

!ERROR: name in NULLIFY statement must be a variable or procedure pointer name
Nullify(maxvalue)

End Program
