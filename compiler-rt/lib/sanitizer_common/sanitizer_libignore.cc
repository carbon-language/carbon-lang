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
  const char *real_name = 0;
  InternalScopedBuffer<char> buf(4096);
  if (name != 0 && internal_readlink(name, buf.data(), buf.size() - 1) > 0)
    real_name = buf.data();

  BlockingMutexLock lock(&mutex_);
  MemoryMappingLayout proc_maps(/*cache_enabled*/false);
  InternalScopedBuffer<char> fn(4096);
  for (uptr i = 0; i < count_; i++) {
    Lib *lib = &libs_[i];
    bool loaded = false;
    proc_maps.Reset();
    uptr b, e, off, prot;
    while (proc_maps.Next(&b, &e, &off, fn.data(), fn.size(), &prot)) {
      bool symlink = false;
      if (((prot & MemoryMappingLayout::kProtectionExecute) != 0) &&
          (TemplateMatch(lib->templ, fn.data()) ||
          // Resolve symlinks.
          (real_name != 0 && real_name[0] != 0 &&
          TemplateMatch(lib->templ, name) &&
          internal_strcmp(real_name, fn.data()) == 0 &&
          (symlink = true)))) {
        if (loaded) {
          Report("%s: called_from_lib suppression '%s' is matched against"
                 " 2 libraries: '%s' and '%s'\n",
                 SanitizerToolName, lib->templ, lib->name, fn.data());
          Die();
        }
        loaded = true;
        if (!lib->loaded) {
          lib->loaded = true;
          lib->name = internal_strdup(fn.data());
          if (symlink)
            lib->real_name = internal_strdup(real_name);
          const uptr idx = atomic_load(&loaded_count_, memory_order_relaxed);
          code_ranges_[idx].begin = b;
          code_ranges_[idx].end = e;
          atomic_store(&loaded_count_, idx + 1, memory_order_release);
        }
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
