; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin | \
; RUN:   grep {\\.mod_init_func}
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin | \
; RUN:   grep {\\.mod_term_func} 
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnu | \
; RUN:   grep {\\.section \\.ctors,"aw",.progbits}
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnu | \
; RUN:   grep {\\.section \\.dtors,"aw",.progbits}
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnueabi | \
; RUN:   grep {\\.section \\.init_array,"aw",.init_array}
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnueabi | \
; RUN:   grep {\\.section \\.fini_array,"aw",.fini_array}

%llvm.global_ctors = appending global [1 x { int, void ()* }] [ { int, void ()* } { int 65535, void ()* %__mf_init } ]		; <[1 x { int, void ()* }]*> [#uses=0]
%llvm.global_dtors = appending global [1 x { int, void ()* }] [ { int, void ()* } { int 65535, void ()* %__mf_fini } ]		; <[1 x { int, void ()* }]*> [#uses=0]

void %__mf_init() {
entry:
	ret void
}

void %__mf_fini() {
entry:
	ret void
}
