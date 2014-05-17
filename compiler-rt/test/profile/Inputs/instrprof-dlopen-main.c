#ifdef DLOPEN_FUNC_DIR
#include <dlfcn.h>
#else
void func(int K);
void func2(int K);
#endif

int main(int argc, char *argv[]) {
#ifdef DLOPEN_FUNC_DIR
  void *f1_handle = dlopen(DLOPEN_FUNC_DIR"/func.shared", DLOPEN_FLAGS);
  void (*func)(int) = (void (*)(int))dlsym(f1_handle, "func");
  void *f2_handle = dlopen(DLOPEN_FUNC_DIR"/func2.shared", DLOPEN_FLAGS);
  void (*func2)(int) = (void (*)(int))dlsym(f2_handle, "func2");
#endif
  func(1);
  func2(0);
  return 0;
}
