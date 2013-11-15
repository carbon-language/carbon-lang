; RUN: not llc -mtriple=i686-pc-linux-gnu %s -o /dev/null 2>&1 | FileCheck %s

@a = external global i32
@b = alias i32* @a
; CHECK: b: Target doesn't support aliases to declarations
