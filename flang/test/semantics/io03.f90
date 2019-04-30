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
  character(kind=2,len=50) internal_file2
  character(kind=4,len=50) internal_file4
  character(kind=1,len=111) msg
  character(20) advance
  integer*1 stat1
  integer*2 stat2, id2
  integer*8 stat8
  integer :: iunit = 10
  integer, parameter :: junit = 11

  namelist /mmm/ mm1, mm2
  namelist /nnn/ nn1, nn2

  advance='no'

  open(10)

  read*
  read(*)
  read*, jj
  read(*, *) jj
  read(unit=*, *) jj
  read(*, fmt=*) jj
  read(*, '(I4)') jj
  read(*, fmt='(I4)') jj
  read(fmt='(I4)', unit=*) jj
  read(iunit, *) jj
  read(junit, *) jj
  read(10, *) jj
  read(internal_file, *) jj
  read(10, nnn)
  read(internal_file, nnn)
  read(internal_file, nml=nnn)
  read(fmt=*, unit=internal_file)
  read(nml=nnn, unit=internal_file)
  read(iunit, nnn)
  read(10, nml=nnn)
  read(10, asynchronous='no') jj
  read(10, asynchronous='yes') jj
  read(10, eor=9, advance='no', fmt='(I4)') jj
  read(10, eor=9, advance='no', fmt='(I4)') jj
  read(10, asynchronous='yes', id=id) jj
  read(10, '(I4)', advance='no', asynchronous='yes', blank='null', &
      decimal='comma', end=9, eor=9, err=9, id=id, iomsg=msg, iostat=stat2, &
      pad='no', round='processor_defined', size=kk) jj

  !ERROR: invalid character kind for an internal file variable
  read(internal_file2, *) jj

  !ERROR: invalid character kind for an internal file variable
  read(internal_file4, *) jj

  !ERROR: duplicate IOSTAT specifier
  read(11, pos=ipos, iostat=stat1, iostat=stat2)

  !ERROR: duplicate END specifier
  read(11, end=9, pos=ipos, end=9)

  !ERROR: duplicate NML specifier
  read(10, nml=mmm, nml=nnn)

  !ERROR: READ statement must have a UNIT specifier
  read(err=9, iostat=stat8) jj

  !ERROR: READ statement must not have a DELIM specifier
  !ERROR: READ statement must not have a SIGN specifier
  read(10, delim='quote', sign='plus') jj

  !ERROR: if NML appears, REC must not appear
  read(10, nnn, rec=nn)

  !ERROR: if NML appears, FMT must not appear
  !ERROR: if NML appears, a data list must not appear
  read(10, fmt=*, nml=nnn) jj

  !ERROR: if UNIT=* appears, REC must not appear
  read(*, rec=13)

  !ERROR: if UNIT=* appears, POS must not appear
  read(*, pos=13)

  !ERROR: if UNIT=internal-file appears, REC must not appear
  read(internal_file, rec=13)

  !ERROR: if UNIT=internal-file appears, POS must not appear
  read(internal_file, pos=13)

  !ERROR: if REC appears, END must not appear
  read(10, fmt='(I4)', end=9, rec=13) jj

  !ERROR: if REC appears, FMT=* must not appear
  read(10, *, rec=13) jj

  !ERROR: if ADVANCE appears, UNIT=internal-file must not appear
  read(internal_file, '(I4)', eor=9, advance='no') jj

  !ERROR: if ADVANCE appears, an explicit format must also appear
  !ERROR: if EOR appears, ADVANCE with value 'NO' must also appear
  read(10, eor=9, advance='yes')

  !ERROR: if EOR appears, ADVANCE with value 'NO' must also appear
  read(10, eor=9)

  !ERROR: invalid ASYNCHRONOUS value 'nay'
  read(10, asynchronous='nay') ! prog req

  !ERROR: if ASYNCHRONOUS='YES' appears, UNIT=number must also appear
  read(*, asynchronous='yes')

  !ERROR: if ASYNCHRONOUS='YES' appears, UNIT=number must also appear
  read(internal_file, asynchronous='y'//'es')

  !ERROR: if ID appears, ASYNCHRONOUS='YES' must also appear
  read(10, id=id)

  !ERROR: if ID appears, ASYNCHRONOUS='YES' must also appear
  read(10, asynchronous='n'//'o', id=id)

  !ERROR: if POS appears, REC must not appear
  read(10, pos=13, rec=13) jj

  !ERROR: if DECIMAL appears, FMT or NML must also appear
  !ERROR: if BLANK appears, FMT or NML must also appear
  !ERROR: invalid DECIMAL value 'Punkt'
  read(10, decimal='Punkt', blank='null') jj

  !ERROR: if ROUND appears, FMT or NML must also appear
  !ERROR: if PAD appears, FMT or NML must also appear
  read(10, pad='no', round='nearest') jj

  !ERROR: ID kind (2) is smaller than default INTEGER kind (4)
  read(10, id=id2, asynchronous='yes') jj

9 continue
end
