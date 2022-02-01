! Ensure argument -fdebug-dump-provenance works as expected.

!----------
! RUN LINE
!----------
! RUN: %flang_fc1 -fdebug-dump-provenance %s  2>&1 | FileCheck %s

!----------------
! EXPECTED OUTPUT
!----------------
! CHECK: AllSources:
! CHECK-NEXT: AllSources range_ [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes)
! CHECK-NEXT:    [1..1] (1 bytes) -> compiler '?'(0x3f)
! CHECK-NEXT:    [2..2] (1 bytes) -> compiler ' '(0x20)
! CHECK-NEXT:    [3..3] (1 bytes) -> compiler '\'(0x5c)
! CHECK-NEXT:    [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes) -> file {{.*[/\\]}}debug-provenance.f90
! CHECK-NEXT:    [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes) -> compiler '(after end of source)'
! CHECK-NEXT: CookedSource::provenanceMap_:
! CHECK-NEXT: offsets [{{[0-9]*}}..{{[0-9]*}}] -> provenances [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes)
! CHECK-NEXT: CookedSource::invertedMap_:
! CHECK-NEXT: provenances [{{[0-9]*}}..{{[0-9]*}}] ({{[0-9]*}} bytes) -> offsets [{{[0-9]*}}..{{[0-9]*}}]
! CHECK-EMPTY:

!-------------
! TEST INPUT
!------------
program A
end
