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

// With coverage_direct=1 in ASAN_OPTIONS, pc_array memory is mapped to a file.
// In this mode, __sanitizer_cov_dump does nothing, and CovUpdateMapping()
// dump current memory layout to another file.

static bool cov_sandboxed = false;
static int cov_fd = kInvalidFd;
static unsigned int cov_max_block_size = 0;

namespace __sanitizer {

class CoverageData {
 public:
  void Init();
  void Extend(uptr npcs);
  void Add(uptr pc);

  uptr *data();
  uptr size();

 private:
  // Maximal size pc array may ever grow.
  // We MmapNoReserve this space to ensure that the array is contiguous.
  static const uptr kPcArrayMaxSize = FIRST_32_SECOND_64(1 << 22, 1 << 27);
  // The amount file mapping for the pc array is grown by.
  static const uptr kPcArrayMmapSize = 64 * 1024;

  // pc_array is allocated with MmapNoReserveOrDie and so it uses only as
  // much RAM as it really needs.
  uptr *pc_array;
  // Index of the first available pc_array slot.
  atomic_uintptr_t pc_array_index;
  // Array size.
  atomic_uintptr_t pc_array_size;
  // Current file mapped size of the pc array.
  uptr pc_array_mapped_size;
  // Descriptor of the file mapped pc array.
  int pc_fd;
  StaticSpinMutex mu;

  void DirectInit();
};

static CoverageData coverage_data;

void CoverageData::DirectInit() {
  InternalScopedString path(64);
  internal_snprintf((char *)path.data(), path.size(), "%zd.sancov.raw",
                    internal_getpid());
  pc_fd = OpenFile(path.data(), true);
  if (internal_iserror(pc_fd)) {
    Report(" Coverage: failed to open %s for writing\n", path.data());
    Die();
  }

  atomic_store(&pc_array_size, 0, memory_order_relaxed);
  pc_array_mapped_size = 0;

  CovUpdateMapping();
}

void CoverageData::Init() {
  pc_array = reinterpret_cast<uptr *>(
      MmapNoReserveOrDie(sizeof(uptr) * kPcArrayMaxSize, "CovInit"));
  if (common_flags()->coverage_direct) {
    DirectInit();
  } else {
    pc_fd = 0;
    atomic_store(&pc_array_size, kPcArrayMaxSize, memory_order_relaxed);
  }
}

// Extend coverage PC array to fit additional npcs elements.
void CoverageData::Extend(uptr npcs) {
  // If pc_fd=0, pc array is a huge anonymous mapping that does not need to be
  // resized.
  if (!pc_fd) return;
  SpinMutexLock l(&mu);

  uptr size = atomic_load(&pc_array_size, memory_order_relaxed);
  size += npcs * sizeof(uptr);

  if (size > pc_array_mapped_size) {
    uptr new_mapped_size = pc_array_mapped_size;
    while (size > new_mapped_size) new_mapped_size += kPcArrayMmapSize;

    // Extend the file and map the new space at the end of pc_array.
    uptr res = internal_ftruncate(pc_fd, new_mapped_size);
    int err;
    if (internal_iserror(res, &err)) {
      Printf("failed to extend raw coverage file: %d\n", err);
      Die();
    }
    void *p = MapWritableFileToMemory(pc_array + pc_array_mapped_size,
                                      new_mapped_size - pc_array_mapped_size,
                                      pc_fd, pc_array_mapped_size);
    CHECK_EQ(p, pc_array + pc_array_mapped_size);
    pc_array_mapped_size = new_mapped_size;
  }

  atomic_store(&pc_array_size, size, memory_order_release);
}

// Simply add the pc into the vector under lock. If the function is called more
// than once for a given PC it will be inserted multiple times, which is fine.
void CoverageData::Add(uptr pc) {
  if (!pc_array) return;
  uptr idx = atomic_fetch_add(&pc_array_index, 1, memory_order_relaxed);
  CHECK_LT(idx * sizeof(uptr),
           atomic_load(&pc_array_size, memory_order_acquire));
  pc_array[idx] = pc;
}

uptr *CoverageData::data() {
  return pc_array;
}

uptr CoverageData::size() {
  return atomic_load(&pc_array_index, memory_order_relaxed);
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
  if (!common_flags()->coverage || common_flags()->coverage_direct) return;
#if !SANITIZER_WINDOWS
  if (atomic_fetch_add(&dump_once_guard, 1, memory_order_relaxed))
    return;
  uptr size = coverage_data.size();
  InternalMmapVector<u32> offsets(size);
  uptr *vb = coverage_data.data();
  uptr *ve = vb + size;
  SortArray(vb, size);
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
  coverage_data.Add(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()));
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_dump() { CovDump(); }
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_init() {
  coverage_data.Init();
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_module_init(uptr npcs) {
  coverage_data.Extend(npcs);
}
SANITIZER_INTERFACE_ATTRIBUTE
sptr __sanitizer_maybe_open_cov_file(const char *name) {
  return MaybeOpenCovFile(name);
}
}  // extern "C"
