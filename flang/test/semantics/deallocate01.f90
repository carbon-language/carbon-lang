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

! Test that DEALLOCATE works

INTEGER, PARAMETER :: maxvalue=1024

Type dt
  Integer :: l = 3
End Type
Type t
  Type(dt),Pointer :: p
End Type

Type(t),Allocatable :: x(:)
Type(t),Pointer :: y(:)
Type(t),Pointer :: z
Integer :: s
CHARACTER(256) :: e

Integer, Pointer :: pi

Allocate(p)
Allocate(x(3))

Deallocate(x(2)%p)

Deallocate(y(2)%p)

Deallocate(pi)

Deallocate(z%p)

Deallocate(x%p, stat=s, errmsg=e)
Deallocate(x%p, errmsg=e)
Deallocate(x%p, stat=s)

Deallocate(y%p, stat=s, errmsg=e)
Deallocate(y%p, errmsg=e)
Deallocate(y%p, stat=s)

Deallocate(z, stat=s, errmsg=e)
Deallocate(z, errmsg=e)
Deallocate(z, stat=s)

Deallocate(z, y%p, stat=s, errmsg=e)
Deallocate(z, y%p, errmsg=e)
Deallocate(z, y%p, stat=s)

End Program
