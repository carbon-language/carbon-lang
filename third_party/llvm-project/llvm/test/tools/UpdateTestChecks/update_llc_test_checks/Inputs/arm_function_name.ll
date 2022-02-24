; Check that we accept functions with '$' in the name.
; TODO: This is not handled correcly on 32bit ARM and needs to be fixed.
;
; RUN: llc -mtriple=armv7-unknown-linux < %s | FileCheck --prefix=LINUX %s
; RUN: llc -mtriple=armv7-apple-darwin < %s | FileCheck --prefix=DARWIN %s
; RUN: llc -mtriple=armv7-apple-ios < %s | FileCheck --prefix=IOS %s
;
define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}
