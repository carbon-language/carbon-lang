//===-- sanitizer_coverage.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Sanitizer Coverage.
// This file implements run-time support for a poor man's coverage tool.
//
// Compiler instrumentation:
// For every interesting basic block the compiler injects the following code:
// if (*Guard) {
//    __sanitizer_cov();
//    *Guard = 1;
// }
// It's fine to call __sanitizer_cov more than once for a given block.
//
// Run-time:
//  - __sanitizer_cov(): record that we've executed the PC (GET_CALLER_PC).
//  - __sanitizer_cov_dump: dump the coverage data to disk.
//  For every module of the current process that has coverage data
//  this will create a file module_name.PID.sancov. The file format is simple:
//  it's just a sorted sequence of 4-byte offsets in the module.
//
// Eventually, this coverage implementation should be obsoleted by a more
// powerful general purpose Clang/LLVM coverage instrumentation.
// Consider this implementation as prototype.
//
// FIXME: support (or at least test with) dlclose.
//===----------------------------------------------------------------------===//

#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_flags.h"

atomic_uint32_t dump_once_guard;  // Ensure that CovDump runs only once.

// pc_array is the array containing the covered PCs.
// To make the pc_array thread- and async-signal-safe it has to be large enough.
// 128M counters "ought to be enough for anybody" (4M on 32-bit).
// pc_array is allocated with MmapNoReserveOrDie and so it uses only as
// much RAM as it really needs.
static const uptr kPcArraySize = FIRST_32_SECOND_64(1 << 22, 1 << 27);
static uptr *pc_array;
static atomic_uintptr_t pc_array_index;

static bool cov_sandboxed = false;
static int cov_fd = kInvalidFd;
static unsigned int cov_max_block_size = 0;

namespace __sanitizer {

// Simply add the pc into the vector under lock. If the function is called more
// than once for a given PC it will be inserted multiple times, which is fine.
static void CovAdd(uptr pc) {
  if (!pc_array) return;
  uptr idx = atomic_fetch_add(&pc_array_index, 1, memory_order_relaxed);
  CHECK_LT(idx, kPcArraySize);
  pc_array[idx] = pc;
}

void CovInit() {
  pc_array = reinterpret_cast<uptr *>(
      MmapNoReserveOrDie(sizeof(uptr) * kPcArraySize, "CovInit"));
}

static inline bool CompareLess(const uptr &a, const uptr &b) {
  return a < b;
}

// Block layout for packed file format: header, followed by module name (no
// trailing zero), followed by data blob.
struct CovHeader {
  int pid;
  unsigned int module_name_length;
  unsigned int data_length;
};

static void CovWritePacked(int pid, const char *module, const void *blob,
                           unsigned int blob_size) {
  if (cov_fd < 0) return;
  unsigned module_name_length = internal_strlen(module);
  CovHeader header = {pid, module_name_length, blob_size};

  if (cov_max_block_size == 0) {
    // Writing to a file. Just go ahead.
    internal_write(cov_fd, &header, sizeof(header));
    internal_write(cov_fd, module, module_name_length);
    internal_write(cov_fd, blob, blob_size);
  } else {
    // Writing to a socket. We want to split the data into appropriately sized
    // blocks.
    InternalScopedBuffer<char> block(cov_max_block_size);
    CHECK_EQ((uptr)block.data(), (uptr)(CovHeader *)block.data());
    uptr header_size_with_module = sizeof(header) + module_name_length;
    CHECK_LT(header_size_with_module, cov_max_block_size);
    unsigned int max_payload_size =
        cov_max_block_size - header_size_with_module;
    char *block_pos = block.data();
    internal_memcpy(block_pos, &header, sizeof(header));
    block_pos += sizeof(header);
    internal_memcpy(block_pos, module, module_name_length);
    block_pos += module_name_length;
    char *block_data_begin = block_pos;
    char *blob_pos = (char *)blob;
    while (blob_size > 0) {
      unsigned int payload_size = Min(blob_size, max_payload_size);
      blob_size -= payload_size;
      internal_memcpy(block_data_begin, blob_pos, payload_size);
      blob_pos += payload_size;
      ((CovHeader *)block.data())->data_length = payload_size;
      internal_write(cov_fd, block.data(),
                     header_size_with_module + payload_size);
    }
  }
}

// If packed = false: <name>.<pid>.<sancov> (name = module name).
// If packed = true and name == 0: <pid>.<sancov>.<packed>.
// If packed = true and name != 0: <name>.<sancov>.<packed> (name is
// user-supplied).
static int CovOpenFile(bool packed, const char* name) {
  InternalScopedBuffer<char> path(1024);
  if (!packed) {
    CHECK(name);
    internal_snprintf((char *)path.data(), path.size(), "%s.%zd.sancov",
                      name, internal_getpid());
  } else {
    if (!name)
      internal_snprintf((char *)path.data(), path.size(), "%zd.sancov.packed",
                        internal_getpid());
    else
      internal_snprintf((char *)path.data(), path.size(), "%s.sancov.packed",
                        name);
  }
  uptr fd = OpenFile(path.data(), true);
  if (internal_iserror(fd)) {
    Report(" SanitizerCoverage: failed to open %s for writing\n", path.data());
    return -1;
  }
  return fd;
}

// Dump the coverage on disk.
static void CovDump() {
  if (!common_flags()->coverage) return;
#if !SANITIZER_WINDOWS
  if (atomic_fetch_add(&dump_once_guard, 1, memory_order_relaxed))
    return;
  uptr size = atomic_load(&pc_array_index, memory_order_relaxed);
  InternalSort(&pc_array, size, CompareLess);
  InternalMmapVector<u32> offsets(size);
  const uptr *vb = pc_array;
  const uptr *ve = vb + size;
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr mb, me, off, prot;
  InternalScopedBuffer<char> module(4096);
  InternalScopedBuffer<char> path(4096 * 2);
  for (int i = 0;
       proc_maps.Next(&mb, &me, &off, module.data(), module.size(), &prot);
       i++) {
    if ((prot & MemoryMappingLayout::kProtectionExecute) == 0)
      continue;
    while (vb < ve && *vb < mb) vb++;
    if (vb >= ve) break;
    if (*vb < me) {
      offsets.clear();
      const uptr *old_vb = vb;
      CHECK_LE(off, *vb);
      for (; vb < ve && *vb < me; vb++) {
        uptr diff = *vb - (i ? mb : 0) + off;
        CHECK_LE(diff, 0xffffffffU);
        offsets.push_back(static_cast<u32>(diff));
      }
      char *module_name = StripModuleName(module.data());
      if (cov_sandboxed) {
        if (cov_fd >= 0) {
          CovWritePacked(internal_getpid(), module_name, offsets.data(),
                         offsets.size() * sizeof(u32));
          VReport(1, " CovDump: %zd PCs written to packed file\n", vb - old_vb);
        }
      } else {
        // One file per module per process.
        internal_snprintf((char *)path.data(), path.size(), "%s.%zd.sancov",
                          module_name, internal_getpid());
        int fd = CovOpenFile(false /* packed */, module_name);
        if (fd > 0) {
          internal_write(fd, offsets.data(), offsets.size() * sizeof(u32));
          internal_close(fd);
          VReport(1, " CovDump: %s: %zd PCs written\n", path.data(),
                  vb - old_vb);
        }
      }
      InternalFree(module_name);
    }
  }
  if (cov_fd >= 0)
    internal_close(cov_fd);
#endif  // !SANITIZER_WINDOWS
}

void CovPrepareForSandboxing(__sanitizer_sandbox_arguments *args) {
  if (!args) return;
  if (!common_flags()->coverage) return;
  cov_sandboxed = args->coverage_sandboxed;
  if (!cov_sandboxed) return;
  cov_fd = args->coverage_fd;
  cov_max_block_size = args->coverage_max_block_size;
  if (cov_fd < 0)
    // Pre-open the file now. The sandbox won't allow us to do it later.
    cov_fd = CovOpenFile(true /* packed */, 0);
}

int MaybeOpenCovFile(const char *name) {
  CHECK(name);
  if (!common_flags()->coverage) return -1;
  return CovOpenFile(true /* packed */, name);
}
}  // namespace __sanitizer

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov() {
  CovAdd(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_dump() { CovDump(); }
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_init() { CovInit(); }
SANITIZER_INTERFACE_ATTRIBUTE
sptr __sanitizer_maybe_open_cov_file(const char *name) {
  return MaybeOpenCovFile(name);
}
}  // extern "C"
