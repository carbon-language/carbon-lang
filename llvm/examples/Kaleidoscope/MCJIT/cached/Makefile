all: toy-mcjit toy-jit toy-ir-gen

toy-mcjit : toy.cpp
	clang++ toy.cpp -g -O3 -rdynamic -fno-rtti `llvm-config --cppflags --ldflags --libs core mcjit native irreader` -o toy-mcjit

toy-jit : toy-jit.cpp
	clang++ toy-jit.cpp -g -O3 -rdynamic -fno-rtti `llvm-config --cppflags --ldflags --libs core jit native irreader` -o toy-jit

# This is a special build for the purpose of converting Kaleidoscope input to an IR file
toy-ir-gen : toy-jit.cpp
	clang++ toy-jit.cpp -g -O3 -rdynamic -fno-rtti -DDUMP_FINAL_MODULE `llvm-config --cppflags --ldflags --libs core jit native irreader` -o toy-ir-gen
