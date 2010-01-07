GXX := g++
CLANGXX := clang++

all: one

test.cc: gen.cc
	g++ gen.cc -o gen
	gen >test.cc

test-gcc.sum: test.cc
	time $(GXX) test.cc -o test-gcc
	test-gcc >test-gcc.sum

test-clang.sum: test.cc
	time $(CLANGXX) test.cc -o test-clang
	test-clang >test-clang.sum

one: test-gcc.sum test-clang.sum
	cmp test-gcc.sum test-clang.sum
