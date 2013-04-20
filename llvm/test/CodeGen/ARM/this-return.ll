; RUN: llc < %s -mtriple=armv6-linux-gnueabi -arm-tail-calls | FileCheck %s -check-prefix=CHECKELF
; RUN: llc < %s -mtriple=thumbv7-apple-ios -arm-tail-calls | FileCheck %s -check-prefix=CHECKT2D

%struct.A = type { i8 }
%struct.B = type { i32 }
%struct.C = type { %struct.B }
%struct.D = type { %struct.B }

declare %struct.A* @A_ctor_base(%struct.A* returned)
declare %struct.B* @B_ctor_base(%struct.B* returned, i32)
declare %struct.B* @B_ctor_complete(%struct.B* returned, i32)

declare %struct.A* @A_ctor_base_nothisret(%struct.A*)
declare %struct.B* @B_ctor_base_nothisret(%struct.B*, i32)
declare %struct.B* @B_ctor_complete_nothisret(%struct.B*, i32)

define %struct.C* @C_ctor_base(%struct.C* returned %this, i32 %x) {
entry:
; CHECKELF: C_ctor_base:
; CHECKELF-NOT: mov {{r[0-9]+}}, r0
; CHECKELF: bl A_ctor_base
; CHECKELF-NOT: mov r0, {{r[0-9]+}}
; CHECKELF: b B_ctor_base
; CHECKT2D: C_ctor_base:
; CHECKT2D-NOT: mov {{r[0-9]+}}, r0
; CHECKT2D: blx _A_ctor_base
; CHECKT2D-NOT: mov r0, {{r[0-9]+}}
; CHECKT2D: b.w _B_ctor_base
  %0 = bitcast %struct.C* %this to %struct.A*
  %call = tail call %struct.A* @A_ctor_base(%struct.A* %0)
  %1 = getelementptr inbounds %struct.C* %this, i32 0, i32 0
  %call2 = tail call %struct.B* @B_ctor_base(%struct.B* %1, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_base_nothisret(%struct.C* %this, i32 %x) {
entry:
; CHECKELF: C_ctor_base_nothisret:
; CHECKELF: mov [[SAVETHIS:r[0-9]+]], r0
; CHECKELF: bl A_ctor_base_nothisret
; CHECKELF: mov r0, [[SAVETHIS]]
; CHECKELF-NOT: b B_ctor_base_nothisret
; CHECKT2D: C_ctor_base_nothisret:
; CHECKT2D: mov [[SAVETHIS:r[0-9]+]], r0
; CHECKT2D: blx _A_ctor_base_nothisret
; CHECKT2D: mov r0, [[SAVETHIS]]
; CHECKT2D-NOT: b.w _B_ctor_base_nothisret
  %0 = bitcast %struct.C* %this to %struct.A*
  %call = tail call %struct.A* @A_ctor_base_nothisret(%struct.A* %0)
  %1 = getelementptr inbounds %struct.C* %this, i32 0, i32 0
  %call2 = tail call %struct.B* @B_ctor_base_nothisret(%struct.B* %1, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_complete(%struct.C* %this, i32 %x) {
entry:
; CHECKELF: C_ctor_complete:
; CHECKELF: b C_ctor_base
; CHECKT2D: C_ctor_complete:
; CHECKT2D: b.w _C_ctor_base
  %call = tail call %struct.C* @C_ctor_base(%struct.C* %this, i32 %x)
  ret %struct.C* %this
}

define %struct.C* @C_ctor_complete_nothisret(%struct.C* %this, i32 %x) {
entry:
; CHECKELF: C_ctor_complete_nothisret:
; CHECKELF-NOT: b C_ctor_base_nothisret
; CHECKT2D: C_ctor_complete_nothisret:
; CHECKT2D-NOT: b.w _C_ctor_base_nothisret
  %call = tail call %struct.C* @C_ctor_base_nothisret(%struct.C* %this, i32 %x)
  ret %struct.C* %this
}

define %struct.D* @D_ctor_base(%struct.D* %this, i32 %x) {
entry:
; CHECKELF: D_ctor_base:
; CHECKELF-NOT: mov {{r[0-9]+}}, r0
; CHECKELF: bl B_ctor_complete
; CHECKELF-NOT: mov r0, {{r[0-9]+}}
; CHECKELF: b B_ctor_complete
; CHECKT2D: D_ctor_base:
; CHECKT2D-NOT: mov {{r[0-9]+}}, r0
; CHECKT2D: blx _B_ctor_complete
; CHECKT2D-NOT: mov r0, {{r[0-9]+}}
; CHECKT2D: b.w _B_ctor_complete
  %b = getelementptr inbounds %struct.D* %this, i32 0, i32 0
  %call = tail call %struct.B* @B_ctor_complete(%struct.B* %b, i32 %x)
  %call2 = tail call %struct.B* @B_ctor_complete(%struct.B* %b, i32 %x)
  ret %struct.D* %this
}
