; The purpose of this test is to verify that symbol aliases
; (@foo = alias <type> @bar) generate the correct entries in the symbol table.
; They should be identical except for the name.

; RUN: llc -filetype=obj -mtriple i686-pc-win32 %s -o - | coff-dump.py | FileCheck %s
; RUN: llc -filetype=obj -mtriple x86_64-pc-win32 %s -o - | coff-dump.py | FileCheck %s

define void @foo() {
entry:
  ret void
}

@bar = global i32 zeroinitializer

@foo_alias = alias void ()* @foo
@bar_alias = alias i32* @bar

; CHECK:      Name               = {{_?}}foo
; CHECK-NEXT: Value              = [[FOO_VALUE:.*$]]
; CHECK-NEXT: SectionNumber      = [[FOO_SECTION_NUMBER:.*$]]
; CHECK-NEXT: SimpleType         = [[FOO_SIMPLE_TYPE:.*$]]
; CHECK-NEXT: ComplexType        = [[FOO_COMPLEX_TYPE:.*$]]
; CHECK-NEXT: StorageClass       = [[FOO_STORAGE_CLASS:.*$]]
; CHECK-NEXT: NumberOfAuxSymbols = [[FOO_NUMBER_OF_AUX_SYMBOLS:.*$]]

; CHECK:      Name               = {{_?}}bar
; CHECK-NEXT: Value              = [[BAR_VALUE:.*$]]
; CHECK-NEXT: SectionNumber      = [[BAR_SECTION_NUMBER:.*$]]
; CHECK-NEXT: SimpleType         = [[BAR_SIMPLE_TYPE:.*$]]
; CHECK-NEXT: ComplexType        = [[BAR_COMPLEX_TYPE:.*$]]
; CHECK-NEXT: StorageClass       = [[BAR_STORAGE_CLASS:.*$]]
; CHECK-NEXT: NumberOfAuxSymbols = [[BAR_NUMBER_OF_AUX_SYMBOLS:.*$]]

; CHECK:      Name               = {{_?}}foo_alias
; CHECK-NEXT: Value              = [[FOO_VALUE]]
; CHECK-NEXT: SectionNumber      = [[FOO_SECTION_NUMBER]]
; CHECK-NEXT: SimpleType         = [[FOO_SIMPLE_TYPE]]
; CHECK-NEXT: ComplexType        = [[FOO_COMPLEX_TYPE]]
; CHECK-NEXT: StorageClass       = [[FOO_STORAGE_CLASS]]
; CHECK-NEXT: NumberOfAuxSymbols = [[FOO_NUMBER_OF_AUX_SYMBOLS]]

; CHECK:      Name               = {{_?}}bar_alias
; CHECK-NEXT: Value              = [[BAR_VALUE]]
; CHECK-NEXT: SectionNumber      = [[BAR_SECTION_NUMBER]]
; CHECK-NEXT: SimpleType         = [[BAR_SIMPLE_TYPE]]
; CHECK-NEXT: ComplexType        = [[BAR_COMPLEX_TYPE]]
; CHECK-NEXT: StorageClass       = [[BAR_STORAGE_CLASS]]
; CHECK-NEXT: NumberOfAuxSymbols = [[BAR_NUMBER_OF_AUX_SYMBOLS]]
