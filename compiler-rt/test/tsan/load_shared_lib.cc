// Check that if the list of shared libraries changes between the two race
// reports, the second report occurring in a new shared library is still
// symbolized correctly.

// RUN: %clangxx_tsan -O1 %p/SharedLibs/load_shared_lib-so.cc \
// RUN:     -fPIC -shared -o %t-so.so
// RUN: %clangxx_tsan -O1 %s -o %t && not %t 2>&1 | FileCheck %s

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>

#include <string>

int GLOB = 0;

void *write_glob(void *unused) {
  GLOB++;
  return NULL;
}

void race_two_threads(void *(*access_callback)(void *unused)) {
  pthread_t t1, t2;
  pthread_create(&t1, NULL, access_callback, NULL);
  pthread_create(&t2, NULL, access_callback, NULL);
  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
}

int main(int argc, char *argv[]) {
  std::string path = std::string(argv[0]) + std::string("-so.so");
  race_two_threads(write_glob);
  // CHECK: write_glob
  void *lib = dlopen(path.c_str(), RTLD_NOW);
    if (!lib) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  void *(*write_from_so)(void *unused);
  *(void **)&write_from_so = dlsym(lib, "write_from_so");
  race_two_threads(write_from_so);
  // CHECK: write_from_so
  return 0;
}
