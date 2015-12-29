// RUN: %clang_profgen -O3 -o %t %s
// RUN: %run %t %t.out.1 %t.out.2 %t.out.3 %t.out.4
// RUN: cat %t.out.1 | FileCheck %s
// RUN: diff %t.out.1 %t.out.2
// RUN: diff %t.out.2 %t.out.3
// RUN: diff %t.out.3 %t.out.4

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ProfBufferIO ProfBufferIO;
ProfBufferIO *llvmCreateBufferIOInternal(FILE *File, uint32_t DefaultBufferSz);
void llvmDeleteBufferIO(ProfBufferIO *BufferIO);

int llvmBufferIOWrite(ProfBufferIO *BufferIO, const char *Data, uint32_t Size);
int llvmBufferIOFlush(ProfBufferIO *BufferIO);

int __llvm_profile_runtime = 0;

const char *SmallData = "ABC\n";
const char *MediumData =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789\n";
char LargeData[10 * 1024];
int main(int argc, const char *argv[]) {
  ProfBufferIO *BufferIO;
  FILE *File[4];
  uint32_t IOBufferSize[4] = {8, 128, 8 * 1024, 11 * 1024};
  int I, J;
  if (argc < 5)
    return 1;

  for (I = 0; I < 10 * 1024 - 2; I++)
    LargeData[I] = 'A';

  LargeData[I++] = '\n';
  LargeData[I++] = '\0';

  for (J = 0; J < 4; J++) {
    File[J] = fopen(argv[1 + J], "w");
    if (!File[J])
      return 1;

    BufferIO = llvmCreateBufferIOInternal(File[J], IOBufferSize[J]);

    llvmBufferIOWrite(BufferIO, "Short Strings:\n", strlen("Short Strings:\n"));
    for (I = 0; I < 1024; I++) {
      llvmBufferIOWrite(BufferIO, SmallData, strlen(SmallData));
    }
    llvmBufferIOWrite(BufferIO, "Long Strings:\n", strlen("Long Strings:\n"));
    for (I = 0; I < 1024; I++) {
      llvmBufferIOWrite(BufferIO, MediumData, strlen(MediumData));
    }
    llvmBufferIOWrite(BufferIO, "Extra Long Strings:\n",
                      strlen("Extra Long Strings:\n"));
    for (I = 0; I < 10; I++) {
      llvmBufferIOWrite(BufferIO, LargeData, strlen(LargeData));
    }
    llvmBufferIOWrite(BufferIO, "Mixed Strings:\n", strlen("Mixed Strings:\n"));
    for (I = 0; I < 1024; I++) {
      llvmBufferIOWrite(BufferIO, MediumData, strlen(MediumData));
      llvmBufferIOWrite(BufferIO, SmallData, strlen(SmallData));
    }
    llvmBufferIOWrite(BufferIO, "Endings:\n", strlen("Endings:\n"));
    llvmBufferIOWrite(BufferIO, "END\n", strlen("END\n"));
    llvmBufferIOWrite(BufferIO, "ENDEND\n", strlen("ENDEND\n"));
    llvmBufferIOWrite(BufferIO, "ENDENDEND\n", strlen("ENDENDEND\n"));
    llvmBufferIOWrite(BufferIO, "ENDENDENDEND\n", strlen("ENDENDENDEND\n"));
    llvmBufferIOFlush(BufferIO);

    llvmDeleteBufferIO(BufferIO);

    fclose(File[J]);
  }
  return 0;
}

// CHECK-LABEL: Short Strings:
// CHECK: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-NEXT: ABC
// CHECK-LABEL: Long Strings:
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-LABEL: Mixed Strings:
// CHECK: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABC
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABC
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABC
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABC
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABC
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABC
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABC
// CHECK-NEXT: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
// CHECK-NEXT: ABC
// CHECK-LABEL: Endings:
// CHECK: END
// CHECK-NEXT: ENDEND
// CHECK-NEXT: ENDENDEND
// CHECK-NEXT: ENDENDENDEND
