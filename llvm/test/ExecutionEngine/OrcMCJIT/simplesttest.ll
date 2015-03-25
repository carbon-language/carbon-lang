; RUN: %lli -jit-kind=orc-mcjit %s > /dev/null

define i32 @main() {
	ret i32 0
}

