block data foo
  real :: pi = asin(-1.0) ! ok
  !ERROR: An initialized variable in BLOCK DATA must be in a COMMON block
  integer :: notInCommon = 1
  integer :: uninitialized ! ok
  !ERROR: 'p' may not appear in a BLOCK DATA subprogram
  procedure(sin), pointer :: p => cos
  !ERROR: 'p' is already declared as a procedure
  common /block/ pi, p
  real :: inBlankCommon
  data inBlankCommon / 1.0 /
  common inBlankCommon
end block data
