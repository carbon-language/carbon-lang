; The purpose of this test is to verify that we do not produce unneeded
; relocations when symbols are in the same section and we know their offset.

; RUN: llc -filetype=obj -mtriple i686-pc-win32 %s -o - | coff-dump.py | FileCheck %s
; RUN: llc -filetype=obj -mtriple x86_64-pc-win32 %s -o - | coff-dump.py | FileCheck %s

define void @foo() {
e:
  br label %i
i:
  br label %i
}

define void @bar() {
e:
  br label %i
i:
  br label %i
}

define void @baz() {
e:
  call void @baz()
  ret void
}

; CHECK:     Sections = [
; CHECK-NOT: NumberOfRelocations = {{[^0]}}
; CHECK:     Symbols = [
