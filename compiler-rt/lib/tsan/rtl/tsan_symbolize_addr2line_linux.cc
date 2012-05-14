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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <link.h>
#include <linux/limits.h>
#include <sys/types.h>

namespace __tsan {

struct ModuleDesc {
  ModuleDesc *next;
  const char *fullname;
  const char *name;
  uptr base;
  uptr end;
  int inp_fd;
  int out_fd;
};

struct DlIteratePhdrCtx {
  ModuleDesc *modules;
  bool is_first;
};

static void InitModule(ModuleDesc *m) {
  int outfd[2];
  if (pipe(outfd)) {
    Printf("ThreadSanitizer: pipe() failed (%d)\n", errno);
    Die();
  }
  int infd[2];
  if (pipe(infd)) {
    Printf("ThreadSanitizer: pipe() failed (%d)\n", errno);
    Die();
  }
  int pid = fork();
  if (pid == 0) {
    close(STDOUT_FILENO);
    close(STDIN_FILENO);
    dup2(outfd[0], STDIN_FILENO);
    dup2(infd[1], STDOUT_FILENO);
    close(outfd[0]);
    close(outfd[1]);
    close(infd[0]);
    close(infd[1]);
    execl("/usr/bin/addr2line", "/usr/bin/addr2line", "-Cfe", m->fullname, 0);
    _exit(0);
  } else if (pid < 0) {
    Printf("ThreadSanitizer: failed to fork symbolizer\n");
    Die();
  }
  close(outfd[0]);
  close(infd[1]);
  m->inp_fd = infd[0];
  m->out_fd = outfd[1];
}

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrCtx *ctx = (DlIteratePhdrCtx*)arg;
  InternalScopedBuf<char> tmp(128);
  if (ctx->is_first) {
    Snprintf(tmp.Ptr(), tmp.Size(), "/proc/%d/exe", (int)getpid());
    info->dlpi_name = tmp.Ptr();
  }
  ctx->is_first = false;
  if (info->dlpi_name == 0 || info->dlpi_name[0] == 0)
    return 0;
  ModuleDesc *m = (ModuleDesc*)internal_alloc(MBlockReportStack,
                                              sizeof(ModuleDesc));
  m->next = ctx->modules;
  ctx->modules = m;
  m->fullname = internal_strdup(info->dlpi_name);
  m->name = strrchr(m->fullname, '/');  // FIXME: internal_strrchr
  if (m->name)
    m->name += 1;
  else
    m->name = m->fullname;
  m->base = (uptr)-1;
  m->end = 0;
  m->inp_fd = -1;
  m->out_fd = -1;
  for (int i = 0; i < info->dlpi_phnum; i++) {
    uptr base1 = info->dlpi_addr + info->dlpi_phdr[i].p_vaddr;
    uptr end1 = base1 + info->dlpi_phdr[i].p_memsz;
    if (m->base > base1)
      m->base = base1;
    if (m->end < end1)
      m->end = end1;
  }
  DPrintf("Module %s %lx-%lx\n", m->name, m->base, m->end);
  return 0;
}

static ModuleDesc *InitModules() {
  DlIteratePhdrCtx ctx = {0, true};
  dl_iterate_phdr(dl_iterate_phdr_cb, &ctx);
  return ctx.modules;
}

static ModuleDesc *GetModuleDesc(uptr addr) {
  static ModuleDesc *modules = 0;
  if (modules == 0)
    modules = InitModules();
  for (ModuleDesc *m = modules; m; m = m->next) {
    if (addr >= m->base && addr < m->end) {
      if (m->inp_fd == -1)
        InitModule(m);
      return m;
    }
  }
  return 0;
}

static ReportStack *NewFrame(uptr addr) {
  ReportStack *ent = (ReportStack*)internal_alloc(MBlockReportStack,
                                                  sizeof(ReportStack));
  internal_memset(ent, 0, sizeof(*ent));
  ent->pc = addr;
  return ent;
}

ReportStack *SymbolizeCode(uptr addr) {
  ModuleDesc *m = GetModuleDesc(addr);
  if (m == 0)
    NewFrame(addr);
  uptr offset = addr - m->base;
  char addrstr[32];
  Snprintf(addrstr, sizeof(addrstr), "%p\n", (void*)offset);
  if (0 >= write(m->out_fd, addrstr, internal_strlen(addrstr))) {
    Printf("ThreadSanitizer: can't write from symbolizer\n");
    Die();
  }
  InternalScopedBuf<char> func(1024);
  ssize_t len = read(m->inp_fd, func, func.Size() - 1);
  if (len <= 0) {
    Printf("ThreadSanitizer: can't read from symbolizer\n");
    Die();
  }
  func.Ptr()[len] = 0;
  ReportStack *res = NewFrame(addr);
  res->module = internal_strdup(m->name);
  res->offset = offset;
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
