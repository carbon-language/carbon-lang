implicit none(external)
external x
call x
!ERROR: 'y' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
call y
end
