all: toy-mcjit

toy-mcjit : toy.cpp
	clang++ toy.cpp -g -O3 -rdynamic -fno-rtti `llvm-config --cppflags --ldflags --libs core mcjit native` -o toy-mcjit
