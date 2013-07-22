all: toy-mcjit toy-jit

toy-mcjit : toy.cpp
	clang++ toy.cpp -g -O3 -rdynamic -fno-rtti `llvm-config --cppflags --ldflags --libs core mcjit native` -o toy-mcjit

toy-jit : toy-jit.cpp
	clang++ toy-jit.cpp -g -O3 -rdynamic `llvm-config --cppflags --ldflags --libs core jit native` -o toy-jit
