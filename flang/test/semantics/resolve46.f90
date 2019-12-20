program main
  intrinsic :: cos ! a specific & generic intrinsic name
  intrinsic :: alog10 ! a specific intrinsic name, not generic
  intrinsic :: null ! a weird special case
  intrinsic :: bessel_j0 ! generic intrinsic, not specific
  !ERROR: 'haltandcatchfire' is not a known intrinsic procedure
  intrinsic :: haltandcatchfire
  procedure(sin), pointer :: p
  p => alog ! valid use of an unrestricted specific intrinsic
  p => alog10 ! ditto, but already declared intrinsic
  p => cos ! ditto, but also generic
  p => tan ! a generic & an unrestricted specific, not already declared
  !TODO ERROR: a restricted specific, to be caught in ass't semantics
  p => amin1
  !TODO ERROR: a generic, to be caught in ass't semantics
  p => bessel_j0
end program main
