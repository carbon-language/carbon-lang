UNAME := $(shell uname -s)

ifeq ($(UNAME),Darwin)
	CXX := xcrun --sdk macosx clang++
else
	CXX := clang++
endif

LLVM_CXXFLAGS := $(shell llvm-config --cxxflags)
LLVM_LDFLAGS := $(shell llvm-config  --ldflags --system-libs --libs core orcjit native)

toy: toy.cpp
	$(CXX) $(LLVM_CXXFLAGS) -Wall -std=c++11 -g -O0 -rdynamic -fno-rtti -o toy toy.cpp $(LLVM_LDFLAGS)

.PHONY: clean
clean:
	rm -f toy
