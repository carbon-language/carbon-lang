//===-- sanitizer_symbolizer_libbacktrace.cc ------------------------------===//
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
// Libbacktrace implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer.h"
#include "sanitizer_symbolizer_libbacktrace.h"

#if SANITIZER_LIBBACKTRACE
# include "backtrace-supported.h"
# if SANITIZER_POSIX && BACKTRACE_SUPPORTED && !BACKTRACE_USES_MALLOC
#  include "backtrace.h"
# else
#  define SANITIZER_LIBBACKTRACE 0
# endif
#endif

namespace __sanitizer {

#if SANITIZER_LIBBACKTRACE

namespace {

struct SymbolizeCodeData {
  AddressInfo *frames;
  uptr n_frames;
  uptr max_frames;
  const char *module_name;
  uptr module_offset;
};

extern "C" {
static int SymbolizeCodePCInfoCallback(void *vdata, uintptr_t addr,
                                       const char *filename, int lineno,
                                       const char *function) {
  SymbolizeCodeData *cdata = (SymbolizeCodeData *)vdata;
  if (function) {
    AddressInfo *info = &cdata->frames[cdata->n_frames++];
    info->Clear();
    info->FillAddressAndModuleInfo(addr, cdata->module_name,
                                   cdata->module_offset);
    info->function = internal_strdup(function);
    if (filename)
      info->file = internal_strdup(filename);
    info->line = lineno;
    if (cdata->n_frames == cdata->max_frames)
      return 1;
  }
  return 0;
}

static void SymbolizeCodeCallback(void *vdata, uintptr_t addr,
                                  const char *symname, uintptr_t, uintptr_t) {
  SymbolizeCodeData *cdata = (SymbolizeCodeData *)vdata;
  if (symname) {
    AddressInfo *info = &cdata->frames[0];
    info->Clear();
    info->FillAddressAndModuleInfo(addr, cdata->module_name,
                                   cdata->module_offset);
    info->function = internal_strdup(symname);
    cdata->n_frames = 1;
  }
}

static void SymbolizeDataCallback(void *vdata, uintptr_t, const char *symname,
                                  uintptr_t symval, uintptr_t symsize) {
  DataInfo *info = (DataInfo *)vdata;
  if (symname && symval) {
    info->name = internal_strdup(symname);
    info->start = symval;
    info->size = symsize;
  }
}

static void ErrorCallback(void *, const char *, int) {}
}  // extern "C"

}  // namespace

LibbacktraceSymbolizer *LibbacktraceSymbolizer::get(LowLevelAllocator *alloc) {
  // State created in backtrace_create_state is leaked.
  void *state = (void *)(backtrace_create_state("/proc/self/exe", 0,
                                                ErrorCallback, NULL));
  if (!state)
    return 0;
  return new(*alloc) LibbacktraceSymbolizer(state);
}

uptr LibbacktraceSymbolizer::SymbolizeCode(uptr addr, AddressInfo *frames,
                                           uptr max_frames,
                                           const char *module_name,
                                           uptr module_offset) {
  SymbolizeCodeData data;
  data.frames = frames;
  data.n_frames = 0;
  data.max_frames = max_frames;
  data.module_name = module_name;
  data.module_offset = module_offset;
  backtrace_pcinfo((backtrace_state *)state_, addr, SymbolizeCodePCInfoCallback,
                   ErrorCallback, &data);
  if (data.n_frames)
    return data.n_frames;
  backtrace_syminfo((backtrace_state *)state_, addr, SymbolizeCodeCallback,
                    ErrorCallback, &data);
  return data.n_frames;
}

bool LibbacktraceSymbolizer::SymbolizeData(DataInfo *info) {
  backtrace_syminfo((backtrace_state *)state_, info->address,
                    SymbolizeDataCallback, ErrorCallback, info);
  return true;
}

#else  // SANITIZER_LIBBACKTRACE

LibbacktraceSymbolizer *LibbacktraceSymbolizer::get(LowLevelAllocator *alloc) {
  return 0;
}

uptr LibbacktraceSymbolizer::SymbolizeCode(uptr addr, AddressInfo *frames,
                                           uptr max_frames,
                                           const char *module_name,
                                           uptr module_offset) {
  (void)state_;
  return 0;
}

bool LibbacktraceSymbolizer::SymbolizeData(DataInfo *info) {
  return false;
}

#endif  // SANITIZER_LIBBACKTRACE

}  // namespace __sanitizer
