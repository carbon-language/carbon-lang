; RUN: llc < %s -mtriple x86_64-apple-darwin10 -march x86 | not grep {{.globl\[\[:space:\]\]*__Z4funcv.eh}}
; RUN: llc < %s -mtriple x86_64-apple-darwin9  -march x86 | FileCheck %s -check-prefix=DARWIN9

%struct.__pointer_type_info_pseudo = type { %struct.__type_info_pseudo, i32, %"struct.std::type_info"* }
%struct.__type_info_pseudo = type { i8*, i8* }
%"struct.std::type_info" = type opaque

@.str = private constant [12 x i8] c"hello world\00", align 1
@_ZTIPc = external constant %struct.__pointer_type_info_pseudo

define void @_Z4funcv() noreturn optsize ssp {
entry:
  %0 = tail call i8* @__cxa_allocate_exception(i64 8) nounwind
  %1 = bitcast i8* %0 to i8**
  store i8* getelementptr inbounds ([12 x i8]* @.str, i64 0, i64 0), i8** %1, align 8
  tail call void @__cxa_throw(i8* %0, i8* bitcast (%struct.__pointer_type_info_pseudo* @_ZTIPc to i8*), void (i8*)* null) noreturn
  unreachable
}

; DARWIN9: .globl __Z4funcv.eh

declare i8* @__cxa_allocate_exception(i64) nounwind

declare void @__cxa_throw(i8*, i8*, void (i8*)*) noreturn
