// RUN: %clang_cc1 -g -emit-llvm -o - %s | FileCheck %s
// Radar 9199234

int bar();
int foo(int i) {
	int j = 0;
	if (i) {
		j = bar();
//CHECK:  store i32 %call, i32* %j, align 4, !dbg 
//CHECK-NOT:  br label %if.end, !dbg 
	} 
	else
	{
		j = bar() + 2;
	}
	return j;
}
