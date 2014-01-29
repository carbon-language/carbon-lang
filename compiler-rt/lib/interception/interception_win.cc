//===-- interception_linux.cc -----------------------------------*- C++ -*-===//
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
// Windows-specific interception methods.
//===----------------------------------------------------------------------===//

#ifdef _WIN32

#include "interception.h"
#include <windows.h>

namespace __interception {

bool GetRealFunctionAddress(const char *func_name, uptr *func_addr) {
  const char *DLLS[] = {
    "msvcr80.dll",
    "msvcr90.dll",
    "kernel32.dll",
    NULL
  };
  *func_addr = 0;
  for (size_t i = 0; *func_addr == 0 && DLLS[i]; ++i) {
    *func_addr = (uptr)GetProcAddress(GetModuleHandleA(DLLS[i]), func_name);
  }
  return (*func_addr != 0);
}

// FIXME: internal_str* and internal_mem* functions should be moved from the
// ASan sources into interception/.

static void _memset(void *p, int value, size_t sz) {
  for (size_t i = 0; i < sz; ++i)
    ((char*)p)[i] = (char)value;
}

static void _memcpy(void *dst, void *src, size_t sz) {
  char *dst_c = (char*)dst,
       *src_c = (char*)src;
  for (size_t i = 0; i < sz; ++i)
    dst_c[i] = src_c[i];
}

static void WriteJumpInstruction(char *jmp_from, char *to) {
  // jmp XXYYZZWW = E9 WW ZZ YY XX, where XXYYZZWW is an offset fromt jmp_from
  // to the next instruction to the destination.
  ptrdiff_t offset = to - jmp_from - 5;
  *jmp_from = '\xE9';
  *(ptrdiff_t*)(jmp_from + 1) = offset;
}

bool OverrideFunction(uptr old_func, uptr new_func, uptr *orig_old_func) {
#ifdef _WIN64
# error OverrideFunction was not tested on x64
#endif
  // Basic idea:
  // We write 5 bytes (jmp-to-new_func) at the beginning of the 'old_func'
  // to override it. We want to be able to execute the original 'old_func' from
  // the wrapper, so we need to keep the leading 5+ bytes ('head') of the
  // original instructions somewhere with a "jmp old_func+head".
  // We call these 'head'+5 bytes of instructions a "trampoline".

  // Trampolines are allocated from a common pool.
  const int POOL_SIZE = 1024;
  static char *pool = NULL;
  static size_t pool_used = 0;
  if (pool == NULL) {
    pool = (char*)VirtualAlloc(NULL, POOL_SIZE,
                               MEM_RESERVE | MEM_COMMIT,
                               PAGE_EXECUTE_READWRITE);
    // FIXME: set PAGE_EXECUTE_READ access after setting all interceptors?
    if (pool == NULL)
      return false;
    _memset(pool, 0xCC /* int 3 */, POOL_SIZE);
  }

  char* old_bytes = (char*)old_func;
  char* trampoline = pool + pool_used;

  // Find out the number of bytes of the instructions we need to copy to the
  // island and store it in 'head'.
  size_t head = 0;
  while (head < 5) {
    switch (old_bytes[head]) {
      case '\x51':  // push ecx
      case '\x52':  // push edx
      case '\x53':  // push ebx
      case '\x54':  // push esp
      case '\x55':  // push ebp
      case '\x56':  // push esi
      case '\x57':  // push edi
      case '\x5D':  // pop ebp
        head++;
        continue;
      case '\x6A':  // 6A XX = push XX
        head += 2;
        continue;
      case '\xE9':  // E9 XX YY ZZ WW = jmp WWZZYYXX
        head += 5;
        continue;
    }
    switch (*(unsigned short*)(old_bytes + head)) {  // NOLINT
      case 0xFF8B:  // 8B FF = mov edi, edi
      case 0xEC8B:  // 8B EC = mov ebp, esp
      case 0xC033:  // 33 C0 = xor eax, eax
        head += 2;
        continue;
      case 0x458B:  // 8B 45 XX = mov eax, dword ptr [ebp+XXh]
      case 0x5D8B:  // 8B 5D XX = mov ebx, dword ptr [ebp+XXh]
      case 0xEC83:  // 83 EC XX = sub esp, XX
        head += 3;
        continue;
      case 0xC1F7:  // F7 C1 XX YY ZZ WW = test ecx, WWZZYYXX
        head += 6;
        continue;
      case 0x3D83:  // 83 3D XX YY ZZ WW TT = cmp TT, WWZZYYXX
        head += 7;
        continue;
    }
    switch (0x00FFFFFF & *(unsigned int*)(old_bytes + head)) {
      case 0x24448A:  // 8A 44 24 XX = mov eal, dword ptr [esp+XXh]
      case 0x244C8B:  // 8B 4C 24 XX = mov ecx, dword ptr [esp+XXh]
      case 0x24548B:  // 8B 54 24 XX = mov edx, dword ptr [esp+XXh]
      case 0x24748B:  // 8B 74 24 XX = mov esi, dword ptr [esp+XXh]
      case 0x247C8B:  // 8B 7C 24 XX = mov edi, dword ptr [esp+XXh]
        head += 4;
        continue;
    }

    // Unknown instruction!
    // FIXME: Unknown instruction failures might happen when we add a new
    // interceptor or a new compiler version. In either case, they should result
    // in visible and readable error messages. However, merely calling abort()
    // or __debugbreak() leads to an infinite recursion in CheckFailed.
    // Do we have a good way to abort with an error message here?
    return false;
  }

  if (pool_used + head + 5 > POOL_SIZE)
    return false;

  // Now put the "jump to trampoline" instruction into the original code.
  DWORD old_prot, unused_prot;
  if (!VirtualProtect((void*)old_func, head, PAGE_EXECUTE_READWRITE,
                      &old_prot))
    return false;

  // Put the needed instructions into the trampoline bytes.
  _memcpy(trampoline, old_bytes, head);
  WriteJumpInstruction(trampoline + head, old_bytes + head);
  *orig_old_func = (uptr)trampoline;
  pool_used += head + 5;

  // Intercept the 'old_func'.
  WriteJumpInstruction(old_bytes, (char*)new_func);
  _memset(old_bytes + 5, 0xCC /* int 3 */, head - 5);

  if (!VirtualProtect((void*)old_func, head, old_prot, &unused_prot))
    return false;  // not clear if this failure bothers us.

  return true;
}

}  // namespace __interception

#endif  // _WIN32
