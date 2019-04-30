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

  character(kind=1,len=50) internal_file
  character(kind=1,len=100) msg
  character(20) sign
  integer*1 stat1, id1
  integer*2 stat2
  integer*4 stat4
  integer*8 stat8
  integer :: iunit = 10
  integer, parameter :: junit = 11

  namelist /nnn/ nn1, nn2

  sign = 'suppress'

  open(10)

  write(*)
  write(*, *)
  write(*)
  write(*, *)
  write(unit=*) 'Ok'
  write(unit=iunit)
  write(unit=junit)
  write(unit=iunit, *)
  write(unit=junit, *)
  write(10)
  write(unit=10) 'Ok'
  write(*, nnn)
  write(10, nnn)
  write(internal_file)
  write(internal_file, *)
  write(internal_file, fmt=*)
  write(internal_file, fmt=1) 'Ok'
  write(internal_file, nnn)
  write(internal_file, nml=nnn)
  write(unit=internal_file, *)
  write(fmt=*, unit=internal_file)
  write(10, advance='yes', fmt=1) 'Ok'
  write(10, *, delim='quote', sign='plus') jj
  write(10, '(A)', advance='no', asynchronous='yes', decimal='comma', &
      err=9, id=id, iomsg=msg, iostat=stat2, round='processor_defined', &
      sign=sign) 'Ok'

  print*
  print*, 'Ok'

  !ERROR: duplicate UNIT specifier
  write(internal_file, unit=*)

  !ERROR: WRITE statement must have a UNIT specifier
  write(nml=nnn)

  !ERROR: WRITE statement must not have a BLANK specifier
  !ERROR: WRITE statement must not have a END specifier
  !ERROR: WRITE statement must not have a EOR specifier
  !ERROR: WRITE statement must not have a PAD specifier
  write(*, eor=9, blank='zero', end=9, pad='no')

  !ERROR: if NML appears, REC must not appear
  !ERROR: if NML appears, FMT must not appear
  !ERROR: if NML appears, a data list must not appear
  write(10, nnn, rec=40, fmt=1) 'Ok'

  !ERROR: if UNIT=* appears, POS must not appear
  write(*, pos=n, nml=nnn)

  !ERROR: if UNIT=* appears, REC must not appear
  write(*, rec=n)

  !ERROR: if UNIT=internal-file appears, POS must not appear
  write(internal_file, err=9, pos=n, nml=nnn)

  !ERROR: if UNIT=internal-file appears, REC must not appear
  write(internal_file, rec=n, err=9)

  !ERROR: if UNIT=* appears, REC must not appear
  write(*, rec=13) 'Ok'

  !ERROR: if ADVANCE appears, UNIT=internal-file must not appear
  write(internal_file, advance='yes', fmt=1) 'Ok'

  !ERROR: if ADVANCE appears, an explicit format must also appear
  write(10, advance='yes') 'Ok'

  !ERROR: invalid ASYNCHRONOUS value 'non'
  write(*, asynchronous='non')

  !ERROR: if ASYNCHRONOUS='YES' appears, UNIT=number must also appear
  write(*, asynchronous='yes')

  !ERROR: if ASYNCHRONOUS='YES' appears, UNIT=number must also appear
  write(internal_file, asynchronous='yes')

  !ERROR: if ID appears, ASYNCHRONOUS='YES' must also appear
  write(10, *, id=id) "Ok"

  !ERROR: if ID appears, ASYNCHRONOUS='YES' must also appear
  write(10, *, id=id, asynchronous='no') "Ok"

  !ERROR: if POS appears, REC must not appear
  write(10, pos=13, rec=13) 'Ok'

  !ERROR: if DECIMAL appears, FMT or NML must also appear
  !ERROR: if ROUND appears, FMT or NML must also appear
  !ERROR: if SIGN appears, FMT or NML must also appear
  !ERROR: invalid DECIMAL value 'Komma'
  write(10, decimal='Komma', sign='plus', round='down') jj

  !ERROR: if DELIM appears, FMT=* or NML must also appear
  !ERROR: invalid DELIM value 'Nix'
  write(delim='Nix', fmt='(A)', unit=10) 'Ok' !C1228

  !ERROR: ID kind (1) is smaller than default INTEGER kind (4)
  write(id=id1, unit=10, asynchronous='Yes') 'Ok'

1 format (A)
9 continue
end
