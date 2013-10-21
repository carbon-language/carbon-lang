; No arguments means internalize everything
; RUN: opt < %s -internalize -S | FileCheck --check-prefix=ALL %s

; Non existent files should be treated as if they were empty (so internalize
; everything)
; RUN: opt < %s -internalize -internalize-public-api-file /nonexistent/file 2> /dev/null -S | FileCheck --check-prefix=ALL %s

; Internalize all but foo and j
; RUN: opt < %s -internalize -internalize-public-api-list foo -internalize-public-api-list j -S | FileCheck --check-prefix=FOO_AND_J %s

; RUN: opt < %s -S -internalize -internalize-public-api-list bar -internalize-public-api-list foo -internalize-public-api-file /nonexistent/file  2> /dev/null | FileCheck --check-prefix=FOO_AND_BAR %s

; -file and -list options should be merged, the apifile contains foo and j
; RUN: opt < %s -internalize -internalize-public-api-list bar -internalize-public-api-file %S/apifile -S | FileCheck --check-prefix=FOO_J_AND_BAR %s

; Put zed1 and zed2 in the symbol table. If the address is not relevant, we
; internalize them.
; RUN: opt < %s -internalize -internalize-dso-list zed1,zed2,zed3 -S | FileCheck --check-prefix=ZEDS %s

; ALL: @i = internal global
; FOO_AND_J: @i = internal global
; FOO_AND_BAR: @i = internal global
; FOO_J_AND_BAR: @i = internal global
@i = global i32 0

; ALL: @j = internal global
; FOO_AND_J: @j = global
; FOO_AND_BAR: @j = internal global
; FOO_J_AND_BAR: @j = global
@j = global i32 0

; ZEDS: @zed1 = internal global i32 42
@zed1 = linkonce_odr global i32 42

; ZEDS: @zed2 = internal unnamed_addr global i32 42
@zed2 = linkonce_odr unnamed_addr global i32 42

; ZEDS: @zed3 = linkonce_odr global i32 42
@zed3 = linkonce_odr global i32 42
define i32* @get_zed3() {
       ret i32* @zed3
}

; ALL: define internal void @main() {
; FOO_AND_J: define internal void @main() {
; FOO_AND_BAR: define internal void @main() {
; FOO_J_AND_BAR: define internal void @main() {
define void @main() {
        ret void
}

; ALL: define internal void @foo() {
; FOO_AND_J: define void @foo() {
; FOO_AND_BAR: define void @foo() {
; FOO_J_AND_BAR: define void @foo() {
define void @foo() {
        ret void
}

; ALL: define available_externally void @bar() {
; FOO_AND_J: define available_externally void @bar() {
; FOO_AND_BAR: define available_externally void @bar() {
; FOO_J_AND_BAR: define available_externally void @bar() {
define available_externally void @bar() {
  ret void
}
