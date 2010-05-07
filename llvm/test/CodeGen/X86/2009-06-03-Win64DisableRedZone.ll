; RUN: llc < %s | grep "subq.*\\\$40, \\\%rsp"
target triple = "x86_64-pc-mingw64"

define x86_fp80 @a(i64 %x) nounwind readnone {
entry:
	%conv = sitofp i64 %x to x86_fp80		; <x86_fp80> [#uses=1]
	ret x86_fp80 %conv
}

