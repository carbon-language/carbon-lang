; RUN: as < %s | opt -funcresolve -disable-output 2>&1 | not grep WARNING

declare int %foo(int *%X)
declare int %foo(float *%X)

implementation

void %test() {
  call int %foo(int* null)
  call int %foo(float* null)
  ret void
}
