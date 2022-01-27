! Test that the driver correctly reports diagnostics from the prescanner, no
! matter what driver action/phase is run. We need this test as Flang currently
! has no central API for managing the diagnostics. For this reason the driver
! needs to make sure that the diagnostics are indeed issued (rather that relying
! on some DiagnosticsEngine).

!-----------
! RUN LINES
!-----------
! Test with -E (i.e. PrintPreprocessedAction, stops after prescanning)
! RUN: %flang -E -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -E -I %S/Inputs/ %s 2>&1 | FileCheck %s

! Test with -fsyntax-only (i.e. ParseSyntaxOnlyAction, stops after semantic checks)
! RUN: %flang -fsyntax-only -I %S/Inputs/ %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fsyntax-only -I %S/Inputs/ %s 2>&1 | FileCheck %s

!-----------------------
! EXPECTED OUTPUT
!-----------------------
! CHECK: prescanner-diag.f90:27:20: #include: extra stuff ignored after file name
! CHECK: prescanner-diag.f90:28:20: #include: extra stuff ignored after file name

!-------
! INPUT
!-------
#include <empty.h> comment
#include "empty.h" comment
end
