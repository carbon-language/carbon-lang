UNAME := $(shell uname -s)

ifeq ($(UNAME),Darwin)
	CXX := xcrun --sdk macosx clang++
else
	CXX := clang++
endif

LLVM_CXXFLAGS := $(shell llvm-config --cppflags --ldflags --system-libs --libs core orcjit native)

toy: toy.cpp
	$(CXX) $(LLVM_CXXFLAGS) -Wall -std=c++14 -g -O0 -rdynamic -fno-rtti -o toy toy.cpp

.PHONY: clean
clean:
	rm -f toy
