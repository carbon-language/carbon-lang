//===-- sanitizer_symbolizer_win.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
// Windows-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_WINDOWS
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

#include "sanitizer_symbolizer.h"

namespace __sanitizer {

class WinSymbolizer : public Symbolizer {
 public:
  WinSymbolizer() : initialized_(false) {}

  uptr SymbolizePC(uptr addr, AddressInfo *frames, uptr max_frames) {
    if (max_frames == 0)
      return 0;

    BlockingMutexLock l(&dbghelp_mu_);
    if (!initialized_) {
      if (!TrySymInitialize()) {
        // OK, maybe the client app has called SymInitialize already.
        // That's a bit unfortunate for us as all the DbgHelp functions are
        // single-threaded and we can't coordinate with the app.
        // FIXME: Can we stop the other threads at this point?
        // Anyways, we have to reconfigure stuff to make sure that SymInitialize
        // has all the appropriate options set.
        // Cross our fingers and reinitialize DbgHelp.
        Report("*** WARNING: Failed to initialize DbgHelp!              ***\n");
        Report("*** Most likely this means that the app is already      ***\n");
        Report("*** using DbgHelp, possibly with incompatible flags.    ***\n");
        Report("*** Due to technical reasons, symbolization might crash ***\n");
        Report("*** or produce wrong results.                           ***\n");
        SymCleanup(GetCurrentProcess());
        TrySymInitialize();
      }
      initialized_ = true;
    }

    // See http://msdn.microsoft.com/en-us/library/ms680578(VS.85).aspx
    char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(CHAR)];
    PSYMBOL_INFO symbol = (PSYMBOL_INFO)buffer;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    DWORD64 offset = 0;
    BOOL got_objname = SymFromAddr(GetCurrentProcess(),
                                   (DWORD64)addr, &offset, symbol);
    if (!got_objname)
      return 0;

    DWORD unused;
    IMAGEHLP_LINE64 line_info;
    line_info.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
    BOOL got_fileline = SymGetLineFromAddr64(GetCurrentProcess(), (DWORD64)addr,
                                             &unused, &line_info);
    AddressInfo *info = &frames[0];
    info->Clear();
    info->function = internal_strdup(symbol->Name);
    info->function_offset = (uptr)offset;
    if (got_fileline) {
      info->file = internal_strdup(line_info.FileName);
      info->line = line_info.LineNumber;
    }

    IMAGEHLP_MODULE64 mod_info;
    internal_memset(&mod_info, 0, sizeof(mod_info));
    mod_info.SizeOfStruct = sizeof(mod_info);
    if (SymGetModuleInfo64(GetCurrentProcess(), addr, &mod_info))
      info->FillAddressAndModuleInfo(addr, mod_info.ImageName,
                                     addr - (uptr)mod_info.BaseOfImage);
    return 1;
  }

  bool CanReturnFileLineInfo() {
    return true;
  }

  const char *Demangle(const char *name) {
    CHECK(initialized_);
    static char demangle_buffer[1000];
    if (name[0] == '\01' &&
        UnDecorateSymbolName(name + 1, demangle_buffer, sizeof(demangle_buffer),
                             UNDNAME_NAME_ONLY))
      return demangle_buffer;
    else
      return name;
  }

  // FIXME: Implement GetModuleNameAndOffsetForPC().

 private:
  bool TrySymInitialize() {
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME | SYMOPT_LOAD_LINES);
    return SymInitialize(GetCurrentProcess(), 0, TRUE);
    // FIXME: We don't call SymCleanup() on exit yet - should we?
  }

  // All DbgHelp functions are single threaded, so we should use a mutex to
  // serialize accesses.
  BlockingMutex dbghelp_mu_;
  bool initialized_;
};

Symbolizer *Symbolizer::PlatformInit() {
  static bool called_once = false;
  CHECK(!called_once && "Shouldn't create more than one symbolizer");
  called_once = true;
  return new(symbolizer_allocator_) WinSymbolizer();
}

}  // namespace __sanitizer

#endif  // _WIN32
