// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// radar 7607781

typedef struct {
	int a;
	int b;
} mystruct;
	
void g(int (^block)(mystruct s)) {
	mystruct x;
	int v = block(x);
}

void f(const void **arg) {
	__block const void **q = arg;
	g(^(mystruct s){
		*q++ = (void*)s.a;
		return 314;
		});
}

// CHECK-LP: (struct __Block_byref_q_0 *)&q
