//===-- sanitizer_libignore.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_LINUX

#include "sanitizer_libignore.h"
#include "sanitizer_flags.h"
#include "sanitizer_procmaps.h"

namespace __sanitizer {

LibIgnore::LibIgnore(LinkerInitialized) {
}

void LibIgnore::Init(const SuppressionContext &supp) {
  BlockingMutexLock lock(&mutex_);
  CHECK_EQ(count_, 0);
  const uptr n = supp.SuppressionCount();
  for (uptr i = 0; i < n; i++) {
    const Suppression *s = supp.SuppressionAt(i);
    if (s->type != SuppressionLib)
      continue;
    if (count_ >= kMaxLibs) {
      Report("%s: too many called_from_lib suppressions (max: %d)\n",
             SanitizerToolName, kMaxLibs);
      Die();
    }
    Lib *lib = &libs_[count_++];
    lib->templ = internal_strdup(s->templ);
    lib->name = 0;
    lib->loaded = false;
  }
}

void LibIgnore::OnLibraryLoaded(const char *name) {
  BlockingMutexLock lock(&mutex_);
  // Try to match suppressions with symlink target.
  InternalScopedBuffer<char> buf(4096);
  if (name != 0 && internal_readlink(name, buf.data(), buf.size() - 1) > 0 &&
      buf.data()[0]) {
    for (uptr i = 0; i < count_; i++) {
      Lib *lib = &libs_[i];
      if (!lib->loaded && lib->real_name == 0 &&
          TemplateMatch(lib->templ, name))
        lib->real_name = internal_strdup(buf.data());
    }
  }

  // Scan suppressions list and find newly loaded and unloaded libraries.
  MemoryMappingLayout proc_maps(/*cache_enabled*/false);
  InternalScopedBuffer<char> module(4096);
  for (uptr i = 0; i < count_; i++) {
    Lib *lib = &libs_[i];
    bool loaded = false;
    proc_maps.Reset();
    uptr b, e, off, prot;
    while (proc_maps.Next(&b, &e, &off, module.data(), module.size(), &prot)) {
      if ((prot & MemoryMappingLayout::kProtectionExecute) == 0)
        continue;
      if (TemplateMatch(lib->templ, module.data()) ||
          (lib->real_name != 0 &&
          internal_strcmp(lib->real_name, module.data()) == 0)) {
        if (loaded) {
          Report("%s: called_from_lib suppression '%s' is matched against"
                 " 2 libraries: '%s' and '%s'\n",
                 SanitizerToolName, lib->templ, lib->name, module.data());
          Die();
        }
        loaded = true;
        if (lib->loaded)
          continue;
        VReport(1,
                "Matched called_from_lib suppression '%s' against library"
                " '%s'\n",
                lib->templ, module.data());
        lib->loaded = true;
        lib->name = internal_strdup(module.data());
        const uptr idx = atomic_load(&loaded_count_, memory_order_relaxed);
        code_ranges_[idx].begin = b;
        code_ranges_[idx].end = e;
        atomic_store(&loaded_count_, idx + 1, memory_order_release);
      }
    }
    if (lib->loaded && !loaded) {
      Report("%s: library '%s' that was matched against called_from_lib"
             " suppression '%s' is unloaded\n",
             SanitizerToolName, lib->name, lib->templ);
      Die();
    }
  }
}

void LibIgnore::OnLibraryUnloaded() {
  OnLibraryLoaded(0);
}

}  // namespace __sanitizer

#endif  // #if SANITIZER_LINUX
