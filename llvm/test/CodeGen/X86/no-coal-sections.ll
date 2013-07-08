; RUN: llc < %s -mtriple x86_64-apple-macosx10 | FileCheck %s
; <rdar://problem/14265330>

; CHECK:     .section __TEXT,__text
; CHECK-NOT: .section __TEXT,__textcoal_nt
; CHECK:     .globl __ZN6TrickyIiLi0EEC1Ev

; CHECK:     .section __DATA,__data
; CHECK-NOT: .section __DATA,__datacoal_nt
; CHECK:     .globl __ZTV6TrickyIiLi0EE

; CHECK:     .section __TEXT,__const
; CHECK-NOT: .section __TEXT,__const_coal
; CHECK:     .globl __ZTS6TrickyIiLi0EE

; CHECK:     .section __DATA,__data
; CHECK-NOT: .section __DATA,__datacoal_nt
; CHECK:     .globl __ZTI6TrickyIiLi0EE

%class.Tricky = type { i32 (...)**, %union.anon }
%union.anon = type { i32 }

@_ZTV6TrickyIiLi0EE = linkonce_odr unnamed_addr constant [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI6TrickyIiLi0EE to i8*), i8* bitcast (void (%class.Tricky*)* @_ZN6TrickyIiLi0EED1Ev to i8*), i8* bitcast (void (%class.Tricky*)* @_ZN6TrickyIiLi0EED0Ev to i8*)]
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS6TrickyIiLi0EE = linkonce_odr constant [15 x i8] c"6TrickyIiLi0EE\00"
@_ZTI6TrickyIiLi0EE = linkonce_odr unnamed_addr constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([15 x i8]* @_ZTS6TrickyIiLi0EE, i32 0, i32 0) }

; Function Attrs: nounwind ssp uwtable
define i32 @main() {
entry:
  %ok = alloca %class.Tricky, align 8
  call void @_ZN6TrickyIiLi0EEC1Ev(%class.Tricky* %ok)
  ret i32 0
}

; Function Attrs: inlinehint nounwind ssp uwtable
define linkonce_odr void @_ZN6TrickyIiLi0EEC1Ev(%class.Tricky* nocapture %this) unnamed_addr align 2 {
entry:
  tail call void @_ZN6TrickyIiLi0EEC2Ev(%class.Tricky* %this)
  ret void
}

; Function Attrs: nounwind readnone ssp uwtable
define linkonce_odr void @_ZN6TrickyIiLi0EED1Ev(%class.Tricky* nocapture %this) unnamed_addr align 2 {
entry:
  ret void
}

; Function Attrs: nounwind readnone ssp uwtable
define linkonce_odr void @_ZN6TrickyIiLi0EED2Ev(%class.Tricky* nocapture %this) unnamed_addr align 2 {
entry:
  ret void
}

; Function Attrs: inlinehint nounwind ssp uwtable
define linkonce_odr void @_ZN6TrickyIiLi0EEC2Ev(%class.Tricky* nocapture %this) unnamed_addr align 2 {
entry:
  %0 = getelementptr inbounds %class.Tricky* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ([4 x i8*]* @_ZTV6TrickyIiLi0EE, i64 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret void
}

; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @_ZN6TrickyIiLi0EED0Ev(%class.Tricky* %this) unnamed_addr align 2 {
invoke.cont:
  %0 = bitcast %class.Tricky* %this to i8*
  tail call void @_ZdlPv(i8* %0)
  ret void
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare void @_ZdlPv(i8*)
