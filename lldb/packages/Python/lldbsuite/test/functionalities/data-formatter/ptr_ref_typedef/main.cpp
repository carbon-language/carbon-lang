//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

typedef int Foo;

int main() {
	int lval = 1;
	Foo* x = &lval;
	Foo& y = lval;
	Foo&& z = 1;
	return 0; // Set breakpoint here
}

