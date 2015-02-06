.PHONY: all
all: toy

toy: toy.cpp
	clang++ -std=c++11 toy.cpp -g -O0 -rdynamic -fno-rtti `llvm-config --cppflags --ldflags --system-libs --libs core orcjit native` -o toy

.PHONY: clean
clean:
	rm -f toy
