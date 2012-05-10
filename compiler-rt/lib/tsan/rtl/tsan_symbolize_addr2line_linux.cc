//===-- tsan_symbolize_addr2line.cc -----------------------------*- C++ -*-===//
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
#include "tsan_symbolize.h"
#include "tsan_mman.h"
#include "tsan_rtl.h"

#include <unistd.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <link.h>
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/wait.h>

namespace __tsan {

static bool GetSymbolizerFd(int *infdp, int *outfdp) {
  static int outfd[2];
  static int infd[2];
  static int pid = -1;
  static int inited = 0;
  if (inited == 0) {
    inited = -1;
    if (pipe(outfd)) {
      Printf("ThreadSanitizer: pipe() failed (%d)\n", errno);
      Die();
    }
    if (pipe(infd)) {
      Printf("ThreadSanitizer: pipe() failed (%d)\n", errno);
      Die();
    }
    pid = fork();
    if (pid == 0) {
      close(STDOUT_FILENO);
      close(STDIN_FILENO);
      dup2(outfd[0], STDIN_FILENO);
      dup2(infd[1], STDOUT_FILENO);
      close(outfd[0]);
      close(outfd[1]);
      close(infd[0]);
      close(infd[1]);
      InternalScopedBuf<char> exe(PATH_MAX);
      ssize_t len = readlink("/proc/self/exe", exe, exe.Size() - 1);
      exe.Ptr()[len] = 0;
      execl("/usr/bin/addr2line", "/usr/bin/addr2line", "-Cfe", exe.Ptr(),
          NULL);
      _exit(0);
    } else if (pid < 0) {
      Printf("ThreadSanitizer: failed to fork symbolizer\n");
      Die();
    }
    close(outfd[0]);
    close(infd[1]);
    inited = 1;
  } else if (inited > 0) {
    int status = 0;
    if (pid == waitpid(pid, &status, WNOHANG)) {
      Printf("ThreadSanitizer: symbolizer died with status %d\n",
          WEXITSTATUS(status));
      Die();
    }
  }
  *infdp = infd[0];
  *outfdp = outfd[1];
  return inited > 0;
}

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *ctx) {
  *(uptr*)ctx = (uptr)info->dlpi_addr;
  return 1;
}

static uptr GetImageBase() {
  static uptr base = 0;
  if (base == 0)
    dl_iterate_phdr(dl_iterate_phdr_cb, &base);
  return base;
}

ReportStack *SymbolizeCode(uptr addr) {
  uptr base = GetImageBase();
  uptr offset = addr - base;
  int infd = -1;
  int outfd = -1;
  if (!GetSymbolizerFd(&infd, &outfd))
    return 0;
  char addrstr[32];
  Snprintf(addrstr, sizeof(addrstr), "%p\n", (void*)offset);
  if (0 >= write(outfd, addrstr, internal_strlen(addrstr))) {
    Printf("ThreadSanitizer: can't write from symbolizer\n");
    Die();
  }
  InternalScopedBuf<char> func(1024);
  ssize_t len = read(infd, func, func.Size() - 1);
  if (len <= 0) {
    Printf("ThreadSanitizer: can't read from symbolizer\n");
    Die();
  }
  func.Ptr()[len] = 0;
  ReportStack *res = (ReportStack*)internal_alloc(MBlockReportStack,
                                                  sizeof(ReportStack));
  internal_memset(res, 0, sizeof(*res));
  res->module = (char*)internal_alloc(MBlockReportStack, 4);
  internal_memcpy(res->module, "exe", 4);
  res->offset = offset;
  res->pc = addr;

  char *pos = strchr(func, '\n');
  if (pos && func[0] != '?') {
    res->func = (char*)internal_alloc(MBlockReportStack, pos - func + 1);
    internal_memcpy(res->func, func, pos - func);
    res->func[pos - func] = 0;
    char *pos2 = strchr(pos, ':');
    if (pos2) {
      res->file = (char*)internal_alloc(MBlockReportStack, pos2 - pos - 1 + 1);
      internal_memcpy(res->file, pos + 1, pos2 - pos - 1);
      res->file[pos2 - pos - 1] = 0;
      res->line = atoi(pos2 + 1);
     }
  }
  return res;
}

ReportStack *SymbolizeData(uptr addr) {
  return 0;
  /*
  if (base == 0)
    base = GetImageBase();
  int res = 0;
  InternalScopedBuf<char> cmd(1024);
  Snprintf(cmd, cmd.Size(),
  "nm -alC %s|grep \"%lx\"|awk '{printf(\"%%s\\n%%s\", $3, $4)}' > tsan.tmp2",
    exe, (addr - base));
  if (system(cmd))
    return 0;
  FILE* f3 = fopen("tsan.tmp2", "rb");
  if (f3) {
    InternalScopedBuf<char> tmp(1024);
    if (fread(tmp, 1, tmp.Size(), f3) <= 0)
      return 0;
    char *pos = strchr(tmp, '\n');
    if (pos && tmp[0] != '?') {
      res = 1;
      symb[0].module = 0;
      symb[0].offset = addr;
      symb[0].name = alloc->Alloc<char>(pos - tmp + 1);
      internal_memcpy(symb[0].name, tmp, pos - tmp);
      symb[0].name[pos - tmp] = 0;
      symb[0].file = 0;
      symb[0].line = 0;
      char *pos2 = strchr(pos, ':');
      if (pos2) {
        symb[0].file = alloc->Alloc<char>(pos2 - pos - 1 + 1);
        internal_memcpy(symb[0].file, pos + 1, pos2 - pos - 1);
        symb[0].file[pos2 - pos - 1] = 0;
        symb[0].line = atoi(pos2 + 1);
      }
    }
    fclose(f3);
  }
  return res;
  */
}

}  // namespace __tsan
