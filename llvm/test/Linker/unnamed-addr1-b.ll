; This file is for use with unnamed-addr1-a.ll
; RUN: true

@global-c = common unnamed_addr global i32 42
@global-d = unnamed_addr global i32 42
@global-e = unnamed_addr global i32 42
@global-f = unnamed_addr global i32 42

@global-g = common global i32 42
@global-h = global i32 42
@global-i = global i32 42
@global-j = global i32 42
