! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Check for semantic errors in DEALLOCATE statements

Module share
  Real, Pointer :: rp
End Module share

Program deallocatetest
Use share

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
Integer, Parameter :: const_s = 13
Integer :: e
Integer :: pi
Character(256) :: ee
Procedure(Real) :: prp

Allocate(rp)
Deallocate(rp)

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

!ERROR: STAT may not be duplicated in a DEALLOCATE statement
Deallocate(x, stat=s, stat=s)
!ERROR: STAT variable 'const_s' must be definable
Deallocate(x, stat=const_s)
!ERROR: ERRMSG may not be duplicated in a DEALLOCATE statement
Deallocate(x, errmsg=ee, errmsg=ee)
!ERROR: STAT may not be duplicated in a DEALLOCATE statement
Deallocate(x, stat=s, errmsg=ee, stat=s)
!ERROR: ERRMSG may not be duplicated in a DEALLOCATE statement
Deallocate(x, stat=s, errmsg=ee, errmsg=ee)

End Program deallocatetest
