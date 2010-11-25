// llvm-gcc -O1+ should run simplify libcalls, O0 shouldn't
// and -fno-builtins shouldn't.
// -fno-math-errno should emit an llvm intrinsic, -fmath-errno should not.
// RUN: %llvmgcc %s -S -fno-math-errno -O0 -o - | grep {call.*exp2\\.f32}
// RUN: %llvmgcc %s -S -fmath-errno -O0 -o - | grep {call.*exp2f}
// RUN: %llvmgcc %s -S -O1 -o - | grep {call.*ldexp}
// RUN: %llvmgcc %s -S -O3 -fno-builtin -o - | grep {call.*exp2f}

float exp2f(float);

float t4(unsigned char x) {
  return exp2f(x);
}

