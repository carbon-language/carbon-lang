
%X = type { int, float }

void %test() {
  getelementptr %X* null, long 0, ubyte 1
  ret void
}
