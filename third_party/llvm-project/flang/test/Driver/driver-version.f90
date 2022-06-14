
!-----------
! RUN LINES
!-----------
! RUN: %flang --version 2>&1 | FileCheck %s --check-prefix=VERSION
! RUN: not %flang --versions 2>&1 | FileCheck %s --check-prefix=ERROR
! RUN: %flang_fc1 -version 2>&1 | FileCheck %s --check-prefix=VERSION-FC1
! RUN: not %flang_fc1 --version 2>&1 | FileCheck %s --check-prefix=ERROR-FC1

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! VERSION: flang-new version
! VERSION-NEXT: Target:
! VERSION-NEXT: Thread model:
! VERSION-NEXT: InstalledDir:

! ERROR: flang-new: error: unsupported option '--versions'; did you mean '--version'?

! VERSION-FC1: LLVM version

! ERROR-FC1: error: unknown argument '--version'; did you mean '-version'?
