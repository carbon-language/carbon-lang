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
// if (Guard < 0) {
//    __sanitizer_cov(&Guard);
// }
// At the module start up time __sanitizer_cov_module_init sets the guards
// to consecutive negative numbers (-1, -2, -3, ...).
// It's fine to call __sanitizer_cov more than once for a given block.
//
// Run-time:
//  - __sanitizer_cov(): record that we've executed the PC (GET_CALLER_PC).
//    and atomically set Guard to -Guard.
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
#include "sanitizer_symbolizer.h"
#include "sanitizer_flags.h"

static atomic_uint32_t dump_once_guard;  // Ensure that CovDump runs only once.

static atomic_uintptr_t coverage_counter;

// pc_array is the array containing the covered PCs.
// To make the pc_array thread- and async-signal-safe it has to be large enough.
// 128M counters "ought to be enough for anybody" (4M on 32-bit).

// With coverage_direct=1 in ASAN_OPTIONS, pc_array memory is mapped to a file.
// In this mode, __sanitizer_cov_dump does nothing, and CovUpdateMapping()
// dump current memory layout to another file.

static bool cov_sandboxed = false;
static int cov_fd = kInvalidFd;
static unsigned int cov_max_block_size = 0;
static bool coverage_enabled = false;
static const char *coverage_dir;

namespace __sanitizer {

class CoverageData {
 public:
  void Init();
  void Enable();
  void Disable();
  void ReInit();
  void BeforeFork();
  void AfterFork(int child_pid);
  void Extend(uptr npcs);
  void Add(uptr pc, u32 *guard);
  void IndirCall(uptr caller, uptr callee, uptr callee_cache[],
                 uptr cache_size);
  void DumpCallerCalleePairs();
  void DumpTrace();

  ALWAYS_INLINE
  void TraceBasicBlock(s32 *id);

  void InitializeGuardArray(s32 *guards);
  void InitializeGuards(s32 *guards, uptr n);
  void ReinitializeGuards();

  uptr *data();
  uptr size();

 private:
  // Maximal size pc array may ever grow.
  // We MmapNoReserve this space to ensure that the array is contiguous.
  static const uptr kPcArrayMaxSize = FIRST_32_SECOND_64(1 << 24, 1 << 27);
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

  // Vector of coverage guard arrays, protected by mu.
  InternalMmapVectorNoCtor<s32*> guard_array_vec;

  // Caller-Callee (cc) array, size and current index.
  static const uptr kCcArrayMaxSize = FIRST_32_SECOND_64(1 << 18, 1 << 24);
  uptr **cc_array;
  atomic_uintptr_t cc_array_index;
  atomic_uintptr_t cc_array_size;

  // Tracing event array, size and current pointer.
  // We record all events (basic block entries) in a global buffer of u32
  // values. Each such value is the index in pc_array.
  // So far the tracing is highly experimental:
  //   - not thread-safe;
  //   - does not support long traces;
  //   - not tuned for performance.
  static const uptr kTrEventArrayMaxSize = FIRST_32_SECOND_64(1 << 22, 1 << 30);
  u32 *tr_event_array;
  uptr tr_event_array_size;
  u32 *tr_event_pointer;
  static const uptr kTrPcArrayMaxSize    = FIRST_32_SECOND_64(1 << 22, 1 << 27);

  StaticSpinMutex mu;

  void DirectOpen();
};

static CoverageData coverage_data;

void CovUpdateMapping(const char *path, uptr caller_pc = 0);

void CoverageData::DirectOpen() {
  InternalScopedString path(kMaxPathLength);
  internal_snprintf((char *)path.data(), path.size(), "%s/%zd.sancov.raw",
                    coverage_dir, internal_getpid());
  pc_fd = OpenFile(path.data(), true);
  if (internal_iserror(pc_fd)) {
    Report(" Coverage: failed to open %s for writing\n", path.data());
    Die();
  }

  pc_array_mapped_size = 0;
  CovUpdateMapping(coverage_dir);
}

void CoverageData::Init() {
  pc_fd = kInvalidFd;
}

void CoverageData::Enable() {
  if (pc_array)
    return;
  pc_array = reinterpret_cast<uptr *>(
      MmapNoReserveOrDie(sizeof(uptr) * kPcArrayMaxSize, "CovInit"));
  atomic_store(&pc_array_index, 0, memory_order_relaxed);
  if (common_flags()->coverage_direct) {
    atomic_store(&pc_array_size, 0, memory_order_relaxed);
  } else {
    atomic_store(&pc_array_size, kPcArrayMaxSize, memory_order_relaxed);
  }

  cc_array = reinterpret_cast<uptr **>(MmapNoReserveOrDie(
      sizeof(uptr *) * kCcArrayMaxSize, "CovInit::cc_array"));
  atomic_store(&cc_array_size, kCcArrayMaxSize, memory_order_relaxed);
  atomic_store(&cc_array_index, 0, memory_order_relaxed);

  // Allocate tr_event_array with a guard page at the end.
  tr_event_array = reinterpret_cast<u32 *>(MmapNoReserveOrDie(
      sizeof(tr_event_array[0]) * kTrEventArrayMaxSize + GetMmapGranularity(),
      "CovInit::tr_event_array"));
  Mprotect(reinterpret_cast<uptr>(&tr_event_array[kTrEventArrayMaxSize]),
           GetMmapGranularity());
  tr_event_array_size = kTrEventArrayMaxSize;
  tr_event_pointer = tr_event_array;
}

void CoverageData::InitializeGuardArray(s32 *guards) {
  Enable();  // Make sure coverage is enabled at this point.
  s32 n = guards[0];
  for (s32 j = 1; j <= n; j++) {
    uptr idx = atomic_fetch_add(&pc_array_index, 1, memory_order_relaxed);
    guards[j] = -static_cast<s32>(idx + 1);
  }
}

void CoverageData::Disable() {
  if (pc_array) {
    internal_munmap(pc_array, sizeof(uptr) * kPcArrayMaxSize);
    pc_array = nullptr;
  }
  if (cc_array) {
    internal_munmap(cc_array, sizeof(uptr *) * kCcArrayMaxSize);
    cc_array = nullptr;
  }
  if (tr_event_array) {
    internal_munmap(tr_event_array,
                    sizeof(tr_event_array[0]) * kTrEventArrayMaxSize +
                        GetMmapGranularity());
    tr_event_array = nullptr;
    tr_event_pointer = nullptr;
  }
  if (pc_fd != kInvalidFd) {
    internal_close(pc_fd);
    pc_fd = kInvalidFd;
  }
}

void CoverageData::ReinitializeGuards() {
  // Assuming single thread.
  atomic_store(&pc_array_index, 0, memory_order_relaxed);
  for (uptr i = 0; i < guard_array_vec.size(); i++)
    InitializeGuardArray(guard_array_vec[i]);
}

void CoverageData::ReInit() {
  Disable();
  if (coverage_enabled) {
    if (common_flags()->coverage_direct) {
      // In memory-mapped mode we must extend the new file to the known array
      // size.
      uptr size = atomic_load(&pc_array_size, memory_order_relaxed);
      Enable();
      if (size) Extend(size);
      if (coverage_enabled) CovUpdateMapping(coverage_dir);
    } else {
      Enable();
    }
  }
  // Re-initialize the guards.
  // We are single-threaded now, no need to grab any lock.
  CHECK_EQ(atomic_load(&pc_array_index, memory_order_relaxed), 0);
  ReinitializeGuards();
}

void CoverageData::BeforeFork() {
  mu.Lock();
}

void CoverageData::AfterFork(int child_pid) {
  // We are single-threaded so it's OK to release the lock early.
  mu.Unlock();
  if (child_pid == 0) ReInit();
}

// Extend coverage PC array to fit additional npcs elements.
void CoverageData::Extend(uptr npcs) {
  if (!common_flags()->coverage_direct) return;
  SpinMutexLock l(&mu);

  uptr size = atomic_load(&pc_array_size, memory_order_relaxed);
  size += npcs * sizeof(uptr);

  if (coverage_enabled && size > pc_array_mapped_size) {
    if (pc_fd == kInvalidFd) DirectOpen();
    CHECK_NE(pc_fd, kInvalidFd);

    uptr new_mapped_size = pc_array_mapped_size;
    while (size > new_mapped_size) new_mapped_size += kPcArrayMmapSize;
    CHECK_LE(new_mapped_size, sizeof(uptr) * kPcArrayMaxSize);

    // Extend the file and map the new space at the end of pc_array.
    uptr res = internal_ftruncate(pc_fd, new_mapped_size);
    int err;
    if (internal_iserror(res, &err)) {
      Printf("failed to extend raw coverage file: %d\n", err);
      Die();
    }

    uptr next_map_base = ((uptr)pc_array) + pc_array_mapped_size;
    void *p = MapWritableFileToMemory((void *)next_map_base,
                                      new_mapped_size - pc_array_mapped_size,
                                      pc_fd, pc_array_mapped_size);
    CHECK_EQ((uptr)p, next_map_base);
    pc_array_mapped_size = new_mapped_size;
  }

  atomic_store(&pc_array_size, size, memory_order_release);
}

void CoverageData::InitializeGuards(s32 *guards, uptr n) {
  // The array 'guards' has n+1 elements, we use the element zero
  // to store 'n'.
  CHECK_LT(n, 1 << 30);
  guards[0] = static_cast<s32>(n);
  InitializeGuardArray(guards);
  SpinMutexLock l(&mu);
  guard_array_vec.push_back(guards);
}

// If guard is negative, atomically set it to -guard and store the PC in
// pc_array.
void CoverageData::Add(uptr pc, u32 *guard) {
  atomic_uint32_t *atomic_guard = reinterpret_cast<atomic_uint32_t*>(guard);
  s32 guard_value = atomic_load(atomic_guard, memory_order_relaxed);
  if (guard_value >= 0) return;

  atomic_store(atomic_guard, -guard_value, memory_order_relaxed);
  if (!pc_array) return;

  uptr idx = -guard_value - 1;
  if (idx >= atomic_load(&pc_array_index, memory_order_acquire))
    return;  // May happen after fork when pc_array_index becomes 0.
  CHECK_LT(idx * sizeof(uptr),
           atomic_load(&pc_array_size, memory_order_acquire));
  pc_array[idx] = pc;
  atomic_fetch_add(&coverage_counter, 1, memory_order_relaxed);
}

// Registers a pair caller=>callee.
// When a given caller is seen for the first time, the callee_cache is added
// to the global array cc_array, callee_cache[0] is set to caller and
// callee_cache[1] is set to cache_size.
// Then we are trying to add callee to callee_cache [2,cache_size) if it is
// not there yet.
// If the cache is full we drop the callee (may want to fix this later).
void CoverageData::IndirCall(uptr caller, uptr callee, uptr callee_cache[],
                             uptr cache_size) {
  if (!cc_array) return;
  atomic_uintptr_t *atomic_callee_cache =
      reinterpret_cast<atomic_uintptr_t *>(callee_cache);
  uptr zero = 0;
  if (atomic_compare_exchange_strong(&atomic_callee_cache[0], &zero, caller,
                                     memory_order_seq_cst)) {
    uptr idx = atomic_fetch_add(&cc_array_index, 1, memory_order_relaxed);
    CHECK_LT(idx * sizeof(uptr),
             atomic_load(&cc_array_size, memory_order_acquire));
    callee_cache[1] = cache_size;
    cc_array[idx] = callee_cache;
  }
  CHECK_EQ(atomic_load(&atomic_callee_cache[0], memory_order_relaxed), caller);
  for (uptr i = 2; i < cache_size; i++) {
    uptr was = 0;
    if (atomic_compare_exchange_strong(&atomic_callee_cache[i], &was, callee,
                                       memory_order_seq_cst)) {
      atomic_fetch_add(&coverage_counter, 1, memory_order_relaxed);
      return;
    }
    if (was == callee)  // Already have this callee.
      return;
  }
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
    const char *blob_pos = (const char *)blob;
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
static int CovOpenFile(bool packed, const char *name,
                       const char *extension = "sancov") {
  InternalScopedString path(kMaxPathLength);
  if (!packed) {
    CHECK(name);
    path.append("%s/%s.%zd.%s", coverage_dir, name, internal_getpid(),
                extension);
  } else {
    if (!name)
      path.append("%s/%zd.%s.packed", coverage_dir, internal_getpid(),
                  extension);
    else
      path.append("%s/%s.%s.packed", coverage_dir, name, extension);
  }
  uptr fd = OpenFile(path.data(), true);
  if (internal_iserror(fd)) {
    Report(" SanitizerCoverage: failed to open %s for writing\n", path.data());
    return -1;
  }
  return fd;
}

// Dump trace PCs and trace events into two separate files.
void CoverageData::DumpTrace() {
  uptr max_idx = tr_event_pointer - tr_event_array;
  if (!max_idx) return;
  auto sym = Symbolizer::GetOrInit();
  if (!sym)
    return;
  InternalScopedString out(32 << 20);
  for (uptr i = 0, n = size(); i < n; i++) {
    const char *module_name = "<unknown>";
    uptr module_address = 0;
    sym->GetModuleNameAndOffsetForPC(pc_array[i], &module_name,
                                     &module_address);
    out.append("%s 0x%zx\n", module_name, module_address);
  }
  int fd = CovOpenFile(false, "trace-points");
  if (fd < 0) return;
  internal_write(fd, out.data(), out.length());
  internal_close(fd);

  fd = CovOpenFile(false, "trace-events");
  if (fd < 0) return;
  uptr bytes_to_write = max_idx * sizeof(tr_event_array[0]);
  u8 *event_bytes = reinterpret_cast<u8*>(tr_event_array);
  // The trace file could be huge, and may not be written with a single syscall.
  while (bytes_to_write) {
    uptr actually_written = internal_write(fd, event_bytes, bytes_to_write);
    if (actually_written <= bytes_to_write) {
      bytes_to_write -= actually_written;
      event_bytes += actually_written;
    } else {
      break;
    }
  }
  internal_close(fd);
  VReport(1, " CovDump: Trace: %zd PCs written\n", size());
  VReport(1, " CovDump: Trace: %zd Events written\n", max_idx);
}

// This function dumps the caller=>callee pairs into a file as a sequence of
// lines like "module_name offset".
void CoverageData::DumpCallerCalleePairs() {
  uptr max_idx = atomic_load(&cc_array_index, memory_order_relaxed);
  if (!max_idx) return;
  auto sym = Symbolizer::GetOrInit();
  if (!sym)
    return;
  InternalScopedString out(32 << 20);
  uptr total = 0;
  for (uptr i = 0; i < max_idx; i++) {
    uptr *cc_cache = cc_array[i];
    CHECK(cc_cache);
    uptr caller = cc_cache[0];
    uptr n_callees = cc_cache[1];
    const char *caller_module_name = "<unknown>";
    uptr caller_module_address = 0;
    sym->GetModuleNameAndOffsetForPC(caller, &caller_module_name,
                                     &caller_module_address);
    for (uptr j = 2; j < n_callees; j++) {
      uptr callee = cc_cache[j];
      if (!callee) break;
      total++;
      const char *callee_module_name = "<unknown>";
      uptr callee_module_address = 0;
      sym->GetModuleNameAndOffsetForPC(callee, &callee_module_name,
                                       &callee_module_address);
      out.append("%s 0x%zx\n%s 0x%zx\n", caller_module_name,
                 caller_module_address, callee_module_name,
                 callee_module_address);
    }
  }
  int fd = CovOpenFile(false, "caller-callee");
  if (fd < 0) return;
  internal_write(fd, out.data(), out.length());
  internal_close(fd);
  VReport(1, " CovDump: %zd caller-callee pairs written\n", total);
}

// Record the current PC into the event buffer.
// Every event is a u32 value (index in tr_pc_array_index) so we compute
// it once and then cache in the provided 'cache' storage.
//
// This function will eventually be inlined by the compiler.
void CoverageData::TraceBasicBlock(s32 *id) {
  // Will trap here if
  //  1. coverage is not enabled at run-time.
  //  2. The array tr_event_array is full.
  *tr_event_pointer = static_cast<u32>(*id - 1);
  tr_event_pointer++;
}

static void CovDumpAsBitSet() {
  if (!common_flags()->coverage_bitset) return;
  if (!coverage_data.size()) return;
  int fd = CovOpenFile(/* packed */false, "combined", "bitset-sancov");
  if (fd < 0) return;
  uptr n = coverage_data.size();
  uptr n_set_bits = 0;
  InternalScopedBuffer<char> out(n);
  for (uptr i = 0; i < n; i++) {
    uptr pc = coverage_data.data()[i];
    out[i] = pc ? '1' : '0';
    if (pc)
      n_set_bits++;
  }
  internal_write(fd, out.data(), n);
  internal_close(fd);
  VReport(1, " CovDump: bitset of %zd bits written, %zd bits are set\n", n,
          n_set_bits);
}

// Dump the coverage on disk.
static void CovDump() {
  if (!coverage_enabled || common_flags()->coverage_direct) return;
#if !SANITIZER_WINDOWS
  if (atomic_fetch_add(&dump_once_guard, 1, memory_order_relaxed))
    return;
  CovDumpAsBitSet();
  coverage_data.DumpTrace();
  if (!common_flags()->coverage_pcs) return;
  uptr size = coverage_data.size();
  InternalMmapVector<u32> offsets(size);
  uptr *vb = coverage_data.data();
  uptr *ve = vb + size;
  SortArray(vb, size);
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr mb, me, off, prot;
  InternalScopedString module(kMaxPathLength);
  InternalScopedString path(kMaxPathLength);
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
      const char *module_name = StripModuleName(module.data());
      if (cov_sandboxed) {
        if (cov_fd >= 0) {
          CovWritePacked(internal_getpid(), module_name, offsets.data(),
                         offsets.size() * sizeof(u32));
          VReport(1, " CovDump: %zd PCs written to packed file\n", vb - old_vb);
        }
      } else {
        // One file per module per process.
        path.clear();
        path.append("%s/%s.%zd.sancov", coverage_dir, module_name,
                    internal_getpid());
        int fd = CovOpenFile(false /* packed */, module_name);
        if (fd > 0) {
          internal_write(fd, offsets.data(), offsets.size() * sizeof(u32));
          internal_close(fd);
          VReport(1, " CovDump: %s: %zd PCs written\n", path.data(),
                  vb - old_vb);
        }
      }
    }
  }
  if (cov_fd >= 0)
    internal_close(cov_fd);
  coverage_data.DumpCallerCalleePairs();
#endif  // !SANITIZER_WINDOWS
}

void CovPrepareForSandboxing(__sanitizer_sandbox_arguments *args) {
  if (!args) return;
  if (!coverage_enabled) return;
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
  if (!coverage_enabled) return -1;
  return CovOpenFile(true /* packed */, name);
}

void CovBeforeFork() {
  coverage_data.BeforeFork();
}

void CovAfterFork(int child_pid) {
  coverage_data.AfterFork(child_pid);
}

void InitializeCoverage(bool enabled, const char *dir) {
  if (coverage_enabled)
    return;  // May happen if two sanitizer enable coverage in the same process.
  coverage_enabled = enabled;
  coverage_dir = dir;
  coverage_data.Init();
  if (enabled) coverage_data.Enable();
#if !SANITIZER_WINDOWS
  if (!common_flags()->coverage_direct) Atexit(__sanitizer_cov_dump);
#endif
}

void ReInitializeCoverage(bool enabled, const char *dir) {
  coverage_enabled = enabled;
  coverage_dir = dir;
  coverage_data.ReInit();
}

void CoverageUpdateMapping() {
  if (coverage_enabled)
    CovUpdateMapping(coverage_dir);
}

}  // namespace __sanitizer

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov(u32 *guard) {
  coverage_data.Add(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()),
                    guard);
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_with_check(u32 *guard) {
  atomic_uint32_t *atomic_guard = reinterpret_cast<atomic_uint32_t*>(guard);
  if (__sanitizer::atomic_load(atomic_guard, memory_order_relaxed))
    __sanitizer_cov(guard);
}
SANITIZER_INTERFACE_ATTRIBUTE void
__sanitizer_cov_indir_call16(uptr callee, uptr callee_cache16[]) {
  coverage_data.IndirCall(StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()),
                          callee, callee_cache16, 16);
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_init() {
  coverage_enabled = true;
  coverage_dir = common_flags()->coverage_dir;
  coverage_data.Init();
}
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_dump() { CovDump(); }
SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_cov_module_init(s32 *guards,
                                                               uptr npcs) {
  coverage_data.InitializeGuards(guards, npcs);
  if (!common_flags()->coverage_direct) return;
  if (SANITIZER_ANDROID && coverage_enabled) {
    // dlopen/dlclose interceptors do not work on Android, so we rely on
    // Extend() calls to update .sancov.map.
    CovUpdateMapping(coverage_dir, GET_CALLER_PC());
  }
  coverage_data.Extend(npcs);
}
SANITIZER_INTERFACE_ATTRIBUTE
sptr __sanitizer_maybe_open_cov_file(const char *name) {
  return MaybeOpenCovFile(name);
}
SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_total_unique_coverage() {
  return atomic_load(&coverage_counter, memory_order_relaxed);
}

SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_cov_trace_func_enter(s32 *id) {
  coverage_data.TraceBasicBlock(id);
}
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_cov_trace_basic_block(s32 *id) {
  coverage_data.TraceBasicBlock(id);
}
SANITIZER_INTERFACE_ATTRIBUTE
void __sanitizer_reset_coverage() {
  coverage_data.ReinitializeGuards();
  internal_bzero_aligned16(
      coverage_data.data(),
      RoundUpTo(coverage_data.size() * sizeof(coverage_data.data()[0]), 16));
}
SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_coverage_guards(uptr **data) {
  *data = coverage_data.data();
  return coverage_data.size();
}
}  // extern "C"
