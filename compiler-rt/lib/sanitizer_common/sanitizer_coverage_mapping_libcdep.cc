//===-- sanitizer_coverage_mapping.cc -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Mmap-based implementation of sanitizer coverage.
//
// This is part of the implementation of code coverage that does not require
// __sanitizer_cov_dump() call. Data is stored in 2 files per process.
//
// $pid.sancov.map describes process memory layout in the following text-based
// format:
// <pointer size in bits>  // 1 line, 32 or 64
// <mapping start> <mapping end> <base address> <dso name> // repeated
// ...
// Mapping lines are NOT sorted. This file is updated every time memory layout
// is changed (i.e. in dlopen() and dlclose() interceptors).
//
// $pid.sancov.raw is a binary dump of PC values, sizeof(uptr) each. Again, not
// sorted. This file is extended by 64Kb at a time and mapped into memory. It
// contains one or more 0 words at the end, up to the next 64Kb aligned offset.
//
// To convert these 2 files to the usual .sancov format, run sancov.py rawunpack
// $pid.sancov.raw.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator_internal.h"
#include "sanitizer_libc.h"
#include "sanitizer_procmaps.h"

namespace __sanitizer {

static const uptr kMaxNumberOfModules = 1 << 14;

void CovUpdateMapping() {
  if (!common_flags()->coverage || !common_flags()->coverage_direct) return;

  int err;
  InternalScopedString tmp_path(64);
  internal_snprintf((char *)tmp_path.data(), tmp_path.size(),
                    "%zd.sancov.map.tmp", internal_getpid());
  uptr map_fd = OpenFile(tmp_path.data(), true);
  if (internal_iserror(map_fd)) {
    Report(" Coverage: failed to open %s for writing\n", tmp_path.data());
    Die();
  }

  InternalScopedBuffer<char> modules_data(kMaxNumberOfModules *
                                          sizeof(LoadedModule));
  LoadedModule *modules = (LoadedModule *)modules_data.data();
  CHECK(modules);
  int n_modules = GetListOfModules(modules, kMaxNumberOfModules,
                                   /* filter */ 0);

  InternalScopedString line(4096);
  line.append("%d\n", sizeof(uptr) * 8);
  uptr res = internal_write(map_fd, line.data(), line.length());
  if (internal_iserror(res, &err)) {
    Printf("sancov.map write failed: %d\n", err);
    Die();
  }
  line.clear();

  for (int i = 0; i < n_modules; ++i) {
    char *module_name = StripModuleName(modules[i].full_name());
    for (unsigned j = 0; j < modules[i].n_ranges(); ++j) {
      line.append("%zx %zx %zx %s\n", modules[i].address_range_start(j),
                  modules[i].address_range_end(j), modules[i].base_address(),
                  module_name);
      res = internal_write(map_fd, line.data(), line.length());
      if (internal_iserror(res, &err)) {
        Printf("sancov.map write failed: %d\n", err);
        Die();
      }
      line.clear();
    }
    InternalFree(module_name);
  }

  internal_close(map_fd);

  InternalScopedString path(64);
  internal_snprintf((char *)path.data(), path.size(), "%zd.sancov.map",
                    internal_getpid());
  res = internal_rename(tmp_path.data(), path.data());
  if (internal_iserror(res, &err)) {
    Printf("sancov.map rename failed: %d\n", err);
    Die();
  }
}

}  // namespace __sanitizer
