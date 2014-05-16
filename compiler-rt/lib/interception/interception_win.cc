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

static char *GetMemoryForTrampoline(size_t size) {
  // Trampolines are allocated from a common pool.
  const int POOL_SIZE = 1024;
  static char *pool = NULL;
  static size_t pool_used = 0;
  if (!pool) {
    pool = (char *)VirtualAlloc(NULL, POOL_SIZE, MEM_RESERVE | MEM_COMMIT,
                                PAGE_EXECUTE_READWRITE);
    // FIXME: Might want to apply PAGE_EXECUTE_READ access after all the
    // interceptors are in place.
    if (!pool)
      return NULL;
    _memset(pool, 0xCC /* int 3 */, POOL_SIZE);
  }

  if (pool_used + size > POOL_SIZE)
    return NULL;

  char *ret = pool + pool_used;
  pool_used += size;
  return ret;
}

// Returns 0 on error.
static size_t RoundUpToInstrBoundary(size_t size, char *code) {
  size_t cursor = 0;
  while (cursor < size) {
    switch (code[cursor]) {
      case '\x51':  // push ecx
      case '\x52':  // push edx
      case '\x53':  // push ebx
      case '\x54':  // push esp
      case '\x55':  // push ebp
      case '\x56':  // push esi
      case '\x57':  // push edi
      case '\x5D':  // pop ebp
        cursor++;
        continue;
      case '\x6A':  // 6A XX = push XX
        cursor += 2;
        continue;
      case '\xE9':  // E9 XX YY ZZ WW = jmp WWZZYYXX
        cursor += 5;
        continue;
    }
    switch (*(unsigned short*)(code + cursor)) {  // NOLINT
      case 0xFF8B:  // 8B FF = mov edi, edi
      case 0xEC8B:  // 8B EC = mov ebp, esp
      case 0xC033:  // 33 C0 = xor eax, eax
        cursor += 2;
        continue;
      case 0x458B:  // 8B 45 XX = mov eax, dword ptr [ebp+XXh]
      case 0x5D8B:  // 8B 5D XX = mov ebx, dword ptr [ebp+XXh]
      case 0xEC83:  // 83 EC XX = sub esp, XX
        cursor += 3;
        continue;
      case 0xC1F7:  // F7 C1 XX YY ZZ WW = test ecx, WWZZYYXX
        cursor += 6;
        continue;
      case 0x3D83:  // 83 3D XX YY ZZ WW TT = cmp TT, WWZZYYXX
        cursor += 7;
        continue;
    }
    switch (0x00FFFFFF & *(unsigned int*)(code + cursor)) {
      case 0x24448A:  // 8A 44 24 XX = mov eal, dword ptr [esp+XXh]
      case 0x244C8B:  // 8B 4C 24 XX = mov ecx, dword ptr [esp+XXh]
      case 0x24548B:  // 8B 54 24 XX = mov edx, dword ptr [esp+XXh]
      case 0x24748B:  // 8B 74 24 XX = mov esi, dword ptr [esp+XXh]
      case 0x247C8B:  // 8B 7C 24 XX = mov edi, dword ptr [esp+XXh]
        cursor += 4;
        continue;
    }

    // Unknown instruction!
    // FIXME: Unknown instruction failures might happen when we add a new
    // interceptor or a new compiler version. In either case, they should result
    // in visible and readable error messages. However, merely calling abort()
    // or __debugbreak() leads to an infinite recursion in CheckFailed.
    // Do we have a good way to abort with an error message here?
    return 0;
  }

  return cursor;
}

bool OverrideFunction(uptr old_func, uptr new_func, uptr *orig_old_func) {
#ifdef _WIN64
#error OverrideFunction is not yet supported on x64
#endif
  // Function overriding works basically like this:
  // We write "jmp <new_func>" (5 bytes) at the beginning of the 'old_func'
  // to override it.
  // We might want to be able to execute the original 'old_func' from the
  // wrapper, in this case we need to keep the leading 5+ bytes ('head')
  // of the original code somewhere with a "jmp <old_func+head>".
  // We call these 'head'+5 bytes of instructions a "trampoline".
  char *old_bytes = (char *)old_func;

  // We'll need at least 5 bytes for a 'jmp'.
  size_t head = 5;
  if (orig_old_func) {
    // Find out the number of bytes of the instructions we need to copy
    // to the trampoline and store it in 'head'.
    head = RoundUpToInstrBoundary(head, old_bytes);
    if (!head)
      return false;

    // Put the needed instructions into the trampoline bytes.
    char *trampoline = GetMemoryForTrampoline(head + 5);
    if (!trampoline)
      return false;
    _memcpy(trampoline, old_bytes, head);
    WriteJumpInstruction(trampoline + head, old_bytes + head);
    *orig_old_func = (uptr)trampoline;
  }

  // Now put the "jmp <new_func>" instruction at the original code location.
  // We should preserve the EXECUTE flag as some of our own code might be
  // located in the same page (sic!).  FIXME: might consider putting the
  // __interception code into a separate section or something?
  DWORD old_prot, unused_prot;
  if (!VirtualProtect((void *)old_bytes, head, PAGE_EXECUTE_READWRITE,
                      &old_prot))
    return false;

  WriteJumpInstruction(old_bytes, (char *)new_func);
  _memset(old_bytes + 5, 0xCC /* int 3 */, head - 5);

  // Restore the original permissions.
  if (!VirtualProtect((void *)old_bytes, head, old_prot, &unused_prot))
    return false;  // not clear if this failure bothers us.

  return true;
}

}  // namespace __interception

#endif  // _WIN32
