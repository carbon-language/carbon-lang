// RUN: %clangxx_asan -DWAIT -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT -m64 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT -m32 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT -m32 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s

// RUN: %clangxx_asan -DWAITPID -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAITPID -m64 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAITPID -m32 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAITPID -m32 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s

// RUN: %clangxx_asan -DWAITID -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAITID -m64 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAITID -m32 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAITID -m32 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s

// RUN: %clangxx_asan -DWAIT3 -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT3 -m64 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT3 -m32 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT3 -m32 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s

// RUN: %clangxx_asan -DWAIT4 -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT4 -m64 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT4 -m32 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT4 -m32 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s

// RUN: %clangxx_asan -DWAIT3_RUSAGE -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT3_RUSAGE -m64 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT3_RUSAGE -m32 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT3_RUSAGE -m32 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s

// RUN: %clangxx_asan -DWAIT4_RUSAGE -m64 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT4_RUSAGE -m64 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT4_RUSAGE -m32 -O0 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s
// RUN: %clangxx_asan -DWAIT4_RUSAGE -m32 -O3 %s -o %t && %t 2>&1 | %symbolize | FileCheck %s


#include <assert.h>
#include <unistd.h>
#include <wait.h>

int main(int argc, char **argv) {
  pid_t pid = fork();
  if (pid) { // parent
    int x[3];
    int *status = x + argc * 3;
    int res;
#if defined(WAIT)
    res = wait(status);
#elif defined(WAITPID)
    res = waitpid(pid, status, WNOHANG);
#elif defined(WAITID)
    siginfo_t *si = (siginfo_t*)(x + argc * 3);
    res = waitid(P_ALL, 0, si, WEXITED | WNOHANG);
#elif defined(WAIT3)
    res = wait3(status, WNOHANG, NULL);
#elif defined(WAIT4)
    res = wait4(pid, status, WNOHANG, NULL);
#elif defined(WAIT3_RUSAGE) || defined(WAIT4_RUSAGE)
    struct rusage *ru = (struct rusage*)(x + argc * 3);
    int good_status;
# if defined(WAIT3_RUSAGE)
    res = wait3(&good_status, WNOHANG, ru);
# elif defined(WAIT4_RUSAGE)
    res = wait4(pid, &good_status, WNOHANG, ru);
# endif
#endif
    // CHECK: stack-buffer-overflow
    // CHECK: {{WRITE of size .* at 0x.* thread T0}}
    // CHECK: {{in .*wait}}
    // CHECK: {{in main .*wait.cc:}}
    // CHECK: is located in stack of thread T0 at offset
    // CHECK: in main
    return res != -1;
  }
  // child
  return 0;
}
