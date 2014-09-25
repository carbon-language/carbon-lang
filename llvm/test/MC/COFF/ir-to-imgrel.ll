; RUN: llc -mtriple=x86_64-pc-win32 %s -o - | FileCheck %s --check-prefix=X64

@__ImageBase = external global i8

; X64: .quad   "?x@@3HA"@IMGREL
@"\01?x@@3HA" = global i64 sub nsw (i64 ptrtoint (i64* @"\01?x@@3HA" to i64), i64 ptrtoint (i8* @__ImageBase to i64)), align 8
