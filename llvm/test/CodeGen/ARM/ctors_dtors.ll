; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin | grep '\.mod_init_func' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-apple-darwin | grep '\.mod_term_func' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnu | grep '\.section \.ctors,"aw",.progbits' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnu | grep '\.section \.dtors,"aw",.progbits' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnueabi | grep '\.section \.init_array,"aw",.init_array' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux-gnueabi  | grep '\.section \.fini_array,"aw",.fini_array'

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
