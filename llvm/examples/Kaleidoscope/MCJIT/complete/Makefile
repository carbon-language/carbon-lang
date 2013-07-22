all: toy

toy : toy.cpp
	clang++ toy.cpp -g -O3 -rdynamic -fno-rtti `llvm-config --cppflags --ldflags --libs core jit mcjit native irreader` -o toy
