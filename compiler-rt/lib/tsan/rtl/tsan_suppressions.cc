//===-- tsan_suppressions.cc ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//

#include "tsan_suppressions.h"
#include "tsan_rtl.h"
#include "tsan_flags.h"
#include "tsan_mman.h"
#include "tsan_platform.h"

namespace __tsan {

static Suppression *g_suppressions;

static char *ReadFile(const char *filename) {
  if (filename == 0 || filename[0] == 0)
    return 0;
  InternalScopedBuf<char> tmp(4*1024);
  if (filename[0] == '/')
    Snprintf(tmp, tmp.Size(), "%s", filename);
  else
    Snprintf(tmp, tmp.Size(), "%s/%s", internal_getpwd(), filename);
  fd_t fd = internal_open(tmp, false);
  if (fd == kInvalidFd) {
    Printf("ThreadSanitizer: failed to open suppressions file '%s'\n",
        tmp.Ptr());
    Die();
  }
  const uptr fsize = internal_filesize(fd);
  if (fsize == (uptr)-1) {
    Printf("ThreadSanitizer: failed to stat suppressions file '%s'\n",
        tmp.Ptr());
    Die();
  }
  char *buf = (char*)internal_alloc(MBlockSuppression, fsize + 1);
  if (fsize != internal_read(fd, buf, fsize)) {
    Printf("ThreadSanitizer: failed to read suppressions file '%s'\n",
        tmp.Ptr());
    Die();
  }
  internal_close(fd);
  buf[fsize] = 0;
  return buf;
}

bool SuppressionMatch(char *templ, const char *str) {
  char *tpos;
  const char *spos;
  while (templ && templ[0]) {
    if (templ[0] == '*') {
      templ++;
      continue;
    }
    if (str[0] == 0)
      return false;
    tpos = (char*)internal_strchr(templ, '*');
    if (tpos != 0)
      tpos[0] = 0;
    spos = internal_strstr(str, templ);
    str = spos + internal_strlen(templ);
    templ = tpos;
    if (tpos)
      tpos[0] = '*';
    if (spos == 0)
      return false;
  }
  return true;
}

Suppression *SuppressionParse(const char* supp) {
  Suppression *head = 0;
  const char *line = supp;
  while (line) {
    while (line[0] == ' ' || line[0] == '\t')
      line++;
    const char *end = internal_strchr(line, '\n');
    if (end == 0)
      end = line + internal_strlen(line);
    if (line != end && line[0] != '#') {
      const char *end2 = end;
      while (line != end2 && (end2[-1] == ' ' || end2[-1] == '\t'))
        end2--;
      SuppressionType stype;
      if (0 == internal_strncmp(line, "race:", sizeof("race:") - 1)) {
        stype = SuppressionRace;
        line += sizeof("race:") - 1;
      } else if (0 == internal_strncmp(line, "thread:",
          sizeof("thread:") - 1)) {
        stype = SuppressionThread;
        line += sizeof("thread:") - 1;
      } else if (0 == internal_strncmp(line, "mutex:",
          sizeof("mutex:") - 1)) {
        stype = SuppressionMutex;
        line += sizeof("mutex:") - 1;
      } else if (0 == internal_strncmp(line, "signal:",
          sizeof("signal:") - 1)) {
        stype = SuppressionSignal;
        line += sizeof("signal:") - 1;
      } else {
        Printf("ThreadSanitizer: failed to parse suppressions file\n");
        Die();
      }
      Suppression *s = (Suppression*)internal_alloc(MBlockSuppression,
          sizeof(Suppression));
      s->next = head;
      head = s;
      s->type = stype;
      s->func = (char*)internal_alloc(MBlockSuppression, end2 - line + 1);
      internal_memcpy(s->func, line, end2 - line);
      s->func[end2 - line] = 0;
    }
    if (end[0] == 0)
      break;
    line = end + 1;
  }
  return head;
}

void SuppressionFree(Suppression *supp) {
  while (supp) {
    Suppression *tmp = supp;
    supp = tmp->next;
    internal_free(tmp->func);
    internal_free(tmp);
  }
}

void InitializeSuppressions() {
  char *supp = ReadFile(flags()->suppressions);
  g_suppressions = SuppressionParse(supp);
}

void FinalizeSuppressions() {
  SuppressionFree(g_suppressions);
  g_suppressions = 0;
}

bool IsSuppressed(ReportType typ, const ReportStack *stack) {
  if (g_suppressions == 0 || stack == 0)
    return false;
  SuppressionType stype;
  if (typ == ReportTypeRace)
    stype = SuppressionRace;
  else if (typ == ReportTypeThreadLeak)
    stype = SuppressionThread;
  else if (typ == ReportTypeMutexDestroyLocked)
    stype = SuppressionMutex;
  else if (typ == ReportTypeSignalUnsafe)
    stype = SuppressionSignal;
  else
    return false;
  for (const ReportStack *frame = stack; frame; frame = frame->next) {
    if (frame->func == 0)
      continue;
    for (Suppression *supp = g_suppressions; supp; supp = supp->next) {
      if (stype == supp->type && SuppressionMatch(supp->func, frame->func)) {
        DPrintf("ThreadSanitizer: matched suppression '%s'\n", supp->func);
        return true;
      }
    }
  }
  return false;
}
}  // namespace __tsan
