; RUN: %lli -use-mcjit %s > /dev/null

define i32 @main() {
	ret i32 0
}

