; RUN: llvm-as < %s | llc -march=c


%X = type { int, float }

void %test() {
  getelementptr %X* null, long 0, ubyte 1
  ret void
}
