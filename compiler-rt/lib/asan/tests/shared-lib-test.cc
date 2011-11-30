//===-- asan_rtl.cc ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include <string>

using std::string;

const char kMyName[] = "shared-lib-test";
const char kSoName[] = "shared-lib-test-so";

typedef void (fun_t)(int x);

int main(int argc, char *argv[]) {
  string path = strdup(argv[0]);
  size_t start = path.find(kMyName);
  if (start == string::npos) return 1;
  path.replace(start, strlen(kMyName), kSoName);
  path += ".so";
  // printf("opening %s ... ", path.c_str());
  void *lib = dlopen(path.c_str(), RTLD_NOW);
  if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  fun_t *inc = (fun_t*)dlsym(lib, "inc");
  if (!inc) return 1;
  // printf("ok\n");
  inc(1);
  inc(-1);
  return 0;
}
