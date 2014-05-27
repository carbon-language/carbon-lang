//===-- sanitizer_common.cc -----------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_libc.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

const char *SanitizerToolName = "SanitizerTool";

uptr GetPageSizeCached() {
  static uptr PageSize;
  if (!PageSize)
    PageSize = GetPageSize();
  return PageSize;
}


// By default, dump to stderr. If |log_to_file| is true and |report_fd_pid|
// isn't equal to the current PID, try to obtain file descriptor by opening
// file "report_path_prefix.<PID>".
fd_t report_fd = kStderrFd;

// Set via __sanitizer_set_report_path.
bool log_to_file = false;
char report_path_prefix[sizeof(report_path_prefix)];

// PID of process that opened |report_fd|. If a fork() occurs, the PID of the
// child thread will be different from |report_fd_pid|.
uptr report_fd_pid = 0;

// PID of the tracer task in StopTheWorld. It shares the address space with the
// main process, but has a different PID and thus requires special handling.
uptr stoptheworld_tracer_pid = 0;
// Cached pid of parent process - if the parent process dies, we want to keep
// writing to the same log file.
uptr stoptheworld_tracer_ppid = 0;

static DieCallbackType DieCallback;
void SetDieCallback(DieCallbackType callback) {
  DieCallback = callback;
}

DieCallbackType GetDieCallback() {
  return DieCallback;
}

void NORETURN Die() {
  if (DieCallback) {
    DieCallback();
  }
  internal__exit(1);
}

static CheckFailedCallbackType CheckFailedCallback;
void SetCheckFailedCallback(CheckFailedCallbackType callback) {
  CheckFailedCallback = callback;
}

void NORETURN CheckFailed(const char *file, int line, const char *cond,
                          u64 v1, u64 v2) {
  if (CheckFailedCallback) {
    CheckFailedCallback(file, line, cond, v1, v2);
  }
  Report("Sanitizer CHECK failed: %s:%d %s (%lld, %lld)\n", file, line, cond,
                                                            v1, v2);
  Die();
}

uptr ReadFileToBuffer(const char *file_name, char **buff,
                      uptr *buff_size, uptr max_len) {
  uptr PageSize = GetPageSizeCached();
  uptr kMinFileLen = PageSize;
  uptr read_len = 0;
  *buff = 0;
  *buff_size = 0;
  // The files we usually open are not seekable, so try different buffer sizes.
  for (uptr size = kMinFileLen; size <= max_len; size *= 2) {
    uptr openrv = OpenFile(file_name, /*write*/ false);
    if (internal_iserror(openrv)) return 0;
    fd_t fd = openrv;
    UnmapOrDie(*buff, *buff_size);
    *buff = (char*)MmapOrDie(size, __func__);
    *buff_size = size;
    // Read up to one page at a time.
    read_len = 0;
    bool reached_eof = false;
    while (read_len + PageSize <= size) {
      uptr just_read = internal_read(fd, *buff + read_len, PageSize);
      if (just_read == 0) {
        reached_eof = true;
        break;
      }
      read_len += just_read;
    }
    internal_close(fd);
    if (reached_eof)  // We've read the whole file.
      break;
  }
  return read_len;
}

typedef bool UptrComparisonFunction(const uptr &a, const uptr &b);

template<class T>
static inline bool CompareLess(const T &a, const T &b) {
  return a < b;
}

void SortArray(uptr *array, uptr size) {
  InternalSort<uptr*, UptrComparisonFunction>(&array, size, CompareLess);
}

// We want to map a chunk of address space aligned to 'alignment'.
// We do it by maping a bit more and then unmaping redundant pieces.
// We probably can do it with fewer syscalls in some OS-dependent way.
void *MmapAlignedOrDie(uptr size, uptr alignment, const char *mem_type) {
// uptr PageSize = GetPageSizeCached();
  CHECK(IsPowerOfTwo(size));
  CHECK(IsPowerOfTwo(alignment));
  uptr map_size = size + alignment;
  uptr map_res = (uptr)MmapOrDie(map_size, mem_type);
  uptr map_end = map_res + map_size;
  uptr res = map_res;
  if (res & (alignment - 1))  // Not aligned.
    res = (map_res + alignment) & ~(alignment - 1);
  uptr end = res + size;
  if (res != map_res)
    UnmapOrDie((void*)map_res, res - map_res);
  if (end != map_end)
    UnmapOrDie((void*)end, map_end - end);
  return (void*)res;
}

const char *StripPathPrefix(const char *filepath,
                            const char *strip_path_prefix) {
  if (filepath == 0) return 0;
  if (strip_path_prefix == 0) return filepath;
  const char *pos = internal_strstr(filepath, strip_path_prefix);
  if (pos == 0) return filepath;
  pos += internal_strlen(strip_path_prefix);
  if (pos[0] == '.' && pos[1] == '/')
    pos += 2;
  return pos;
}

void PrintSourceLocation(InternalScopedString *buffer, const char *file,
                         int line, int column) {
  CHECK(file);
  buffer->append("%s",
                 StripPathPrefix(file, common_flags()->strip_path_prefix));
  if (line > 0) {
    buffer->append(":%d", line);
    if (column > 0)
      buffer->append(":%d", column);
  }
}

void PrintModuleAndOffset(InternalScopedString *buffer, const char *module,
                          uptr offset) {
  buffer->append("(%s+0x%zx)",
                 StripPathPrefix(module, common_flags()->strip_path_prefix),
                 offset);
}

void ReportErrorSummary(const char *error_message) {
  if (!common_flags()->print_summary)
    return;
  InternalScopedBuffer<char> buff(kMaxSummaryLength);
  internal_snprintf(buff.data(), buff.size(),
                    "SUMMARY: %s: %s", SanitizerToolName, error_message);
  __sanitizer_report_error_summary(buff.data());
}

void ReportErrorSummary(const char *error_type, const char *file,
                        int line, const char *function) {
  if (!common_flags()->print_summary)
    return;
  InternalScopedBuffer<char> buff(kMaxSummaryLength);
  internal_snprintf(
      buff.data(), buff.size(), "%s %s:%d %s", error_type,
      file ? StripPathPrefix(file, common_flags()->strip_path_prefix) : "??",
      line, function ? function : "??");
  ReportErrorSummary(buff.data());
}

void ReportErrorSummary(const char *error_type, StackTrace *stack) {
  if (!common_flags()->print_summary)
    return;
  AddressInfo ai;
#if !SANITIZER_GO
  if (stack->size > 0 && Symbolizer::Get()->CanReturnFileLineInfo()) {
    // Currently, we include the first stack frame into the report summary.
    // Maybe sometimes we need to choose another frame (e.g. skip memcpy/etc).
    uptr pc = StackTrace::GetPreviousInstructionPc(stack->trace[0]);
    Symbolizer::Get()->SymbolizePC(pc, &ai, 1);
  }
#endif
  ReportErrorSummary(error_type, ai.file, ai.line, ai.function);
}

LoadedModule::LoadedModule(const char *module_name, uptr base_address) {
  full_name_ = internal_strdup(module_name);
  base_address_ = base_address;
  n_ranges_ = 0;
}

void LoadedModule::addAddressRange(uptr beg, uptr end) {
  CHECK_LT(n_ranges_, kMaxNumberOfAddressRanges);
  ranges_[n_ranges_].beg = beg;
  ranges_[n_ranges_].end = end;
  n_ranges_++;
}

bool LoadedModule::containsAddress(uptr address) const {
  for (uptr i = 0; i < n_ranges_; i++) {
    if (ranges_[i].beg <= address && address < ranges_[i].end)
      return true;
  }
  return false;
}

char *StripModuleName(const char *module) {
  if (module == 0)
    return 0;
  const char *short_module_name = internal_strrchr(module, '/');
  if (short_module_name)
    short_module_name += 1;
  else
    short_module_name = module;
  return internal_strdup(short_module_name);
}

static atomic_uintptr_t g_total_mmaped;

void IncreaseTotalMmap(uptr size) {
  if (!common_flags()->mmap_limit_mb) return;
  uptr total_mmaped =
      atomic_fetch_add(&g_total_mmaped, size, memory_order_relaxed) + size;
  if ((total_mmaped >> 20) > common_flags()->mmap_limit_mb) {
    // Since for now mmap_limit_mb is not a user-facing flag, just CHECK.
    uptr mmap_limit_mb = common_flags()->mmap_limit_mb;
    common_flags()->mmap_limit_mb = 0;  // Allow mmap in CHECK.
    RAW_CHECK(total_mmaped >> 20 < mmap_limit_mb);
  }
}

void DecreaseTotalMmap(uptr size) {
  if (!common_flags()->mmap_limit_mb) return;
  atomic_fetch_sub(&g_total_mmaped, size, memory_order_relaxed);
}

}  // namespace __sanitizer

using namespace __sanitizer;  // NOLINT

extern "C" {
void __sanitizer_set_report_path(const char *path) {
  if (!path)
    return;
  uptr len = internal_strlen(path);
  if (len > sizeof(report_path_prefix) - 100) {
    Report("ERROR: Path is too long: %c%c%c%c%c%c%c%c...\n",
           path[0], path[1], path[2], path[3],
           path[4], path[5], path[6], path[7]);
    Die();
  }
  if (report_fd != kStdoutFd &&
      report_fd != kStderrFd &&
      report_fd != kInvalidFd)
    internal_close(report_fd);
  report_fd = kInvalidFd;
  log_to_file = false;
  if (internal_strcmp(path, "stdout") == 0) {
    report_fd = kStdoutFd;
  } else if (internal_strcmp(path, "stderr") == 0) {
    report_fd = kStderrFd;
  } else {
    internal_strncpy(report_path_prefix, path, sizeof(report_path_prefix));
    report_path_prefix[len] = '\0';
    log_to_file = true;
  }
}

void __sanitizer_report_error_summary(const char *error_summary) {
  Printf("%s\n", error_summary);
}
}  // extern "C"
