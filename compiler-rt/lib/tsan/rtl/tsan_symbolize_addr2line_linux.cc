//===-- tsan_symbolize_addr2line.cc ---------------------------------------===//
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
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "tsan_symbolize.h"
#include "tsan_mman.h"
#include "tsan_rtl.h"
#include "tsan_platform.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <link.h>
#include <linux/limits.h>
#include <sys/types.h>

namespace __tsan {

struct ModuleDesc {
  const char *fullname;
  const char *name;
  uptr base;
  int inp_fd;
  int out_fd;
};

struct SectionDesc {
  SectionDesc *next;
  ModuleDesc *module;
  uptr base;
  uptr end;
};

struct DlIteratePhdrCtx {
  SectionDesc *sections;
  bool is_first;
};

static void NOINLINE InitModule(ModuleDesc *m) {
  int outfd[2] = {};
  if (pipe(&outfd[0])) {
    Printf("ThreadSanitizer: outfd pipe() failed (%d)\n", errno);
    Die();
  }
  int infd[2] = {};
  if (pipe(&infd[0])) {
    Printf("ThreadSanitizer: infd pipe() failed (%d)\n", errno);
    Die();
  }
  int pid = fork();
  if (pid == 0) {
    __sanitizer_set_report_fd(STDERR_FILENO);
    internal_close(STDOUT_FILENO);
    internal_close(STDIN_FILENO);
    internal_dup2(outfd[0], STDIN_FILENO);
    internal_dup2(infd[1], STDOUT_FILENO);
    internal_close(outfd[0]);
    internal_close(outfd[1]);
    internal_close(infd[0]);
    internal_close(infd[1]);
    for (int fd = getdtablesize(); fd > 2; fd--)
      internal_close(fd);
    execl("/usr/bin/addr2line", "/usr/bin/addr2line", "-Cfe", m->fullname, 0);
    _exit(0);
  } else if (pid < 0) {
    Printf("ThreadSanitizer: failed to fork symbolizer\n");
    Die();
  }
  internal_close(outfd[0]);
  internal_close(infd[1]);
  m->inp_fd = infd[0];
  m->out_fd = outfd[1];
}

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrCtx *ctx = (DlIteratePhdrCtx*)arg;
  InternalScopedBuffer<char> tmp(128);
  if (ctx->is_first) {
    internal_snprintf(tmp.data(), tmp.size(), "/proc/%d/exe", GetPid());
    info->dlpi_name = tmp.data();
  }
  ctx->is_first = false;
  if (info->dlpi_name == 0 || info->dlpi_name[0] == 0)
    return 0;
  ModuleDesc *m = (ModuleDesc*)internal_alloc(MBlockReportStack,
                                              sizeof(ModuleDesc));
  m->fullname = internal_strdup(info->dlpi_name);
  m->name = internal_strrchr(m->fullname, '/');
  if (m->name)
    m->name += 1;
  else
    m->name = m->fullname;
  m->base = (uptr)info->dlpi_addr;
  m->inp_fd = -1;
  m->out_fd = -1;
  DPrintf("Module %s %zx\n", m->name, m->base);
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const Elf64_Phdr *s = &info->dlpi_phdr[i];
    DPrintf("  Section p_type=%zx p_offset=%zx p_vaddr=%zx p_paddr=%zx"
            " p_filesz=%zx p_memsz=%zx p_flags=%zx p_align=%zx\n",
            (uptr)s->p_type, (uptr)s->p_offset, (uptr)s->p_vaddr,
            (uptr)s->p_paddr, (uptr)s->p_filesz, (uptr)s->p_memsz,
            (uptr)s->p_flags, (uptr)s->p_align);
    if (s->p_type != PT_LOAD)
      continue;
    SectionDesc *sec = (SectionDesc*)internal_alloc(MBlockReportStack,
                                                    sizeof(SectionDesc));
    sec->module = m;
    sec->base = info->dlpi_addr + s->p_vaddr;
    sec->end = sec->base + s->p_memsz;
    sec->next = ctx->sections;
    ctx->sections = sec;
    DPrintf("  Section %zx-%zx\n", sec->base, sec->end);
  }
  return 0;
}

static SectionDesc *InitSections() {
  DlIteratePhdrCtx ctx = {0, true};
  dl_iterate_phdr(dl_iterate_phdr_cb, &ctx);
  return ctx.sections;
}

static SectionDesc *GetSectionDesc(uptr addr) {
  static SectionDesc *sections = 0;
  if (sections == 0)
    sections = InitSections();
  for (SectionDesc *s = sections; s; s = s->next) {
    if (addr >= s->base && addr < s->end) {
      if (s->module->inp_fd == -1)
        InitModule(s->module);
      return s;
    }
  }
  return 0;
}

ReportStack *SymbolizeCodeAddr2Line(uptr addr) {
  SectionDesc *s = GetSectionDesc(addr);
  if (s == 0)
    return NewReportStackEntry(addr);
  ModuleDesc *m = s->module;
  uptr offset = addr - m->base;
  char addrstr[32];
  internal_snprintf(addrstr, sizeof(addrstr), "%p\n", (void*)offset);
  if (0 >= internal_write(m->out_fd, addrstr, internal_strlen(addrstr))) {
    Printf("ThreadSanitizer: can't write from symbolizer (%d, %d)\n",
        m->out_fd, errno);
    Die();
  }
  InternalScopedBuffer<char> func(1024);
  ssize_t len = internal_read(m->inp_fd, func.data(), func.size() - 1);
  if (len <= 0) {
    Printf("ThreadSanitizer: can't read from symbolizer (%d, %d)\n",
        m->inp_fd, errno);
    Die();
  }
  func.data()[len] = 0;
  ReportStack *res = NewReportStackEntry(addr);
  res->module = internal_strdup(m->name);
  res->offset = offset;
  char *pos = (char*)internal_strchr(func.data(), '\n');
  if (pos && func[0] != '?') {
    res->func = (char*)internal_alloc(MBlockReportStack, pos - func.data() + 1);
    internal_memcpy(res->func, func.data(), pos - func.data());
    res->func[pos - func.data()] = 0;
    char *pos2 = (char*)internal_strchr(pos, ':');
    if (pos2) {
      res->file = (char*)internal_alloc(MBlockReportStack, pos2 - pos - 1 + 1);
      internal_memcpy(res->file, pos + 1, pos2 - pos - 1);
      res->file[pos2 - pos - 1] = 0;
      res->line = atoi(pos2 + 1);
     }
  }
  return res;
}

ReportStack *SymbolizeDataAddr2Line(uptr addr) {
  return 0;
}

}  // namespace __tsan
