; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck --check-prefix=ELF %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck --check-prefix=MACHO %s

@foo = linker_private global i32 42
;ELF: {{^}}.Lfoo:
;MACHO: {{^}}l_foo:

define i32* @f() {
  ret i32* @foo
}
