; RUN: llc < %s -march=arm | grep align.*1 | count 1
; RUN: llc < %s -mtriple=arm-linux-gnueabi | \
; RUN:   grep align.*2 | count 2
; RUN: llc < %s -mtriple=arm-linux-gnueabi | \
; RUN:   grep align.*3 | count 2
; RUN: llc < %s -mtriple=arm-apple-darwin | \
; RUN:   grep align.*2 | count 4

@a = global i1 true
@b = global i8 1
@c = global i16 2
@d = global i32 3
@e = global i64 4
@f = global float 5.0
@g = global double 6.0
