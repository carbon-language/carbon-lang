#include <thread>

void llvm::call_once(once_flag& flag, void (*fptr)(void)) {
  std::call_once(flag, fptr);
}
