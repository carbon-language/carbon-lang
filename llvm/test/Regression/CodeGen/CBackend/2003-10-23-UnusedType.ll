; RUN: llvm-as < %s | llc -march=c


%A = type { uint, sbyte*, { uint, uint, uint, uint, uint, uint, uint, uint }*, ushort }

void %test(%A *) { ret void }
