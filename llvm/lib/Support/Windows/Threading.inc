#include <winbase.h>

#ifdef MemoryFence
// WinNT.h seems to define a MemoryFence macro.
#undef MemoryFence
#endif

void llvm::call_once(once_flag &flag, void (*fptr)(void)) {
  while (flag != Done) {
    if (flag == Wait) {
      ::Sleep(1);
      continue;
    }

    sys::cas_flag old_val = sys::CompareAndSwap(&flag, Wait, Uninitialized);
    if (old_val == Uninitialized) {
      fptr();
      sys::MemoryFence();
      flag = Done;
      return;
    }
  }
  sys::MemoryFence();
}
