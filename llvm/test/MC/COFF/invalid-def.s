# RUN: not llvm-mc -triple i686-windows -filetype obj -o /dev/null %s
# REQUIRES: asserts

	.def first
	.def second

