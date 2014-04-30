//===-- msan_report.cc ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// Error reporting.
//===----------------------------------------------------------------------===//

#include "msan.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_symbolizer.h"

using namespace __sanitizer;

namespace __msan {

class Decorator: private __sanitizer::AnsiColorDecorator {
 public:
  Decorator() : __sanitizer::AnsiColorDecorator(PrintsToTtyCached()) { }
  const char *Warning()    { return Red(); }
  const char *Origin()     { return Magenta(); }
  const char *Name()   { return Green(); }
  const char *End()    { return Default(); }
};

static void DescribeStackOrigin(const char *so, uptr pc) {
  Decorator d;
  char *s = internal_strdup(so);
  char *sep = internal_strchr(s, '@');
  CHECK(sep);
  *sep = '\0';
  Printf("%s", d.Origin());
  Printf(
      "  %sUninitialized value was created by an allocation of '%s%s%s'"
      " in the stack frame of function '%s%s%s'%s\n",
      d.Origin(), d.Name(), s, d.Origin(), d.Name(),
      Symbolizer::Get()->Demangle(sep + 1), d.Origin(), d.End());
  InternalFree(s);

  if (pc) {
    // For some reason function address in LLVM IR is 1 less then the address
    // of the first instruction.
    pc += 1;
    StackTrace::PrintStack(&pc, 1);
  }
}

static void DescribeOrigin(u32 origin) {
  VPrintf(1, "  raw origin id: %d\n", origin);
  uptr pc;
  while (true) {
    if (const char *so = GetOriginDescrIfStack(origin, &pc)) {
      DescribeStackOrigin(so, pc);
      break;
    }
    Decorator d;
    uptr size = 0;
    const uptr *trace = StackDepotGet(origin, &size);
    CHECK_GT(size, 0);
    if (TRACE_IS_CHAINED(trace[size - 1])) {
      // Linked origin.
      // FIXME: copied? modified? passed through? observed?
      Printf("  %sUninitialized value was stored to memory at%s\n", d.Origin(),
             d.End());
      StackTrace::PrintStack(trace, size - 1);
      origin = TRACE_TO_CHAINED_ID(trace[size - 1]);
    } else {
      Printf("  %sUninitialized value was created by a heap allocation%s\n",
             d.Origin(), d.End());
      StackTrace::PrintStack(trace, size);
      break;
    }
  }
}

void ReportUMR(StackTrace *stack, u32 origin) {
  if (!__msan::flags()->report_umrs) return;

  SpinMutexLock l(&CommonSanitizerReportMutex);

  Decorator d;
  Printf("%s", d.Warning());
  Report(" WARNING: MemorySanitizer: use-of-uninitialized-value\n");
  Printf("%s", d.End());
  stack->Print();
  if (origin) {
    DescribeOrigin(origin);
  }
  ReportErrorSummary("use-of-uninitialized-value", stack);
}

void ReportExpectedUMRNotFound(StackTrace *stack) {
  SpinMutexLock l(&CommonSanitizerReportMutex);

  Printf(" WARNING: Expected use of uninitialized value not found\n");
  stack->Print();
}

void ReportAtExitStatistics() {
  SpinMutexLock l(&CommonSanitizerReportMutex);

  if (msan_report_count > 0) {
    Decorator d;
    Printf("%s", d.Warning());
    Printf("MemorySanitizer: %d warnings reported.\n", msan_report_count);
    Printf("%s", d.End());
  }

  StackDepotStats *stack_depot_stats = StackDepotGetStats();
  // FIXME: we want this at normal exit, too!
  // FIXME: but only with verbosity=1 or something
  Printf("Unique heap origins: %zu\n", stack_depot_stats->n_uniq_ids);
  Printf("Stack depot mapped bytes: %zu\n", stack_depot_stats->mapped);
}

class OriginSet {
 public:
  OriginSet() : next_id_(0) {}
  int insert(u32 o) {
    // Scan from the end for better locality.
    for (int i = next_id_ - 1; i >= 0; --i)
      if (origins_[i] == o) return i;
    if (next_id_ == kMaxSize_) return OVERFLOW;
    int id = next_id_++;
    origins_[id] = o;
    return id;
  }
  int size() { return next_id_; }
  u32 get(int id) { return origins_[id]; }
  static char asChar(int id) {
    switch (id) {
      case MISSING:
        return '.';
      case OVERFLOW:
        return '*';
      default:
        return 'A' + id;
    }
  }
  static const int OVERFLOW = -1;
  static const int MISSING = -2;

 private:
  static const int kMaxSize_ = 'Z' - 'A' + 1;
  u32 origins_[kMaxSize_];
  int next_id_;
};

void DescribeMemoryRange(const void *x, uptr size) {
  // Real limits.
  uptr start = MEM_TO_SHADOW(x);
  uptr end = start + size;
  // Scan limits: align start down to 4; align size up to 16.
  uptr s = start & ~3UL;
  size = end - s;
  size = (size + 15) & ~15UL;
  uptr e = s + size;

  // Single letter names to origin id mapping.
  OriginSet origin_set;

  uptr pos = 0;  // Offset from aligned start.
  bool with_origins = __msan_get_track_origins();
  // True if there is at least 1 poisoned bit in the last 4-byte group.
  bool last_quad_poisoned;
  int origin_ids[4];  // Single letter origin ids for the current line.

  Decorator d;
  Printf("%s", d.Warning());
  Printf("Shadow map of [%p, %p), %zu bytes:\n", start, end, end - start);
  Printf("%s", d.End());
  while (s < e) {
    // Line start.
    if (pos % 16 == 0) {
      for (int i = 0; i < 4; ++i) origin_ids[i] = -1;
      Printf("%p:", s);
    }
    // Group start.
    if (pos % 4 == 0) {
      Printf(" ");
      last_quad_poisoned = false;
    }
    // Print shadow byte.
    if (s < start || s >= end) {
      Printf("..");
    } else {
      unsigned char v = *(unsigned char *)s;
      if (v) last_quad_poisoned = true;
      Printf("%02x", v);
    }
    // Group end.
    if (pos % 4 == 3 && with_origins) {
      int id = OriginSet::MISSING;
      if (last_quad_poisoned) {
        u32 o = *(u32 *)SHADOW_TO_ORIGIN(s - 3);
        id = origin_set.insert(o);
      }
      origin_ids[(pos % 16) / 4] = id;
    }
    // Line end.
    if (pos % 16 == 15) {
      if (with_origins) {
        Printf("  |");
        for (int i = 0; i < 4; ++i) {
          char c = OriginSet::asChar(origin_ids[i]);
          Printf("%c", c);
          if (i != 3) Printf(" ");
        }
        Printf("|");
      }
      Printf("\n");
    }
    size--;
    s++;
    pos++;
  }

  Printf("\n");

  for (int i = 0; i < origin_set.size(); ++i) {
    u32 o = origin_set.get(i);
    Printf("Origin %c (origin_id %x):\n", OriginSet::asChar(i), o);
    DescribeOrigin(o);
  }
}

}  // namespace __msan
