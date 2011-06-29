; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 | FileCheck %s
; CHECK: .zerofill __DATA,__bss,__MergedGlobals,16,2

%struct.config = type { i16, i16, i16, i16 }

@prev = external global [0 x i16]
@max_lazy_match = internal unnamed_addr global i32 0, align 4
@read_buf = external global i32 (i8*, i32)*
@window = external global [0 x i8]
@lookahead = internal unnamed_addr global i32 0, align 4
@eofile.b = internal unnamed_addr global i1 false
@ins_h = internal unnamed_addr global i32 0, align 4
