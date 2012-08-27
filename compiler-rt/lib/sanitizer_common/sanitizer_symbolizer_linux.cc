//===-- sanitizer_symbolizer_linux.cc -------------------------------------===//
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
// Linux-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//
#ifdef __linux__
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_symbolizer.h"

#include <elf.h>
#include <errno.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#if !defined(__ANDROID__) && !defined(ANDROID)
#include <link.h>
#endif

namespace __sanitizer {

bool StartSymbolizerSubprocess(const char *path_to_symbolizer,
                               int *input_fd, int *output_fd) {
  int *infd = NULL;
  int *outfd = NULL;
  // The client program may close its stdin and/or stdout and/or stderr
  // thus allowing socketpair to reuse file descriptors 0, 1 or 2.
  // In this case the communication between the forked processes may be
  // broken if either the parent or the child tries to close or duplicate
  // these descriptors. The loop below produces two pairs of file
  // descriptors, each greater than 2 (stderr).
  int sock_pair[5][2];
  for (int i = 0; i < 5; i++) {
    if (pipe(sock_pair[i]) == -1) {
      for (int j = 0; j < i; j++) {
        internal_close(sock_pair[j][0]);
        internal_close(sock_pair[j][1]);
      }
      Report("WARNING: Can't create a socket pair to start "
             "external symbolizer (errno: %d)\n", errno);
      return false;
    } else if (sock_pair[i][0] > 2 && sock_pair[i][1] > 2) {
      if (infd == NULL) {
        infd = sock_pair[i];
      } else {
        outfd = sock_pair[i];
        for (int j = 0; j < i; j++) {
          if (sock_pair[j] == infd) continue;
          internal_close(sock_pair[j][0]);
          internal_close(sock_pair[j][1]);
        }
        break;
      }
    }
  }
  CHECK(infd);
  CHECK(outfd);

  int pid = fork();
  if (pid == -1) {
    // Fork() failed.
    internal_close(infd[0]);
    internal_close(infd[1]);
    internal_close(outfd[0]);
    internal_close(outfd[1]);
    Report("WARNING: failed to fork external symbolizer "
           " (errno: %d)\n", errno);
    return false;
  } else if (pid == 0) {
    // Child subprocess.
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
    execl(path_to_symbolizer, path_to_symbolizer, (char*)0);
    Exit(1);
  }

  // Continue execution in parent process.
  internal_close(outfd[0]);
  internal_close(infd[1]);
  *input_fd = infd[0];
  *output_fd = outfd[1];
  return true;
}

#if defined(__ANDROID__) || defined(ANDROID)
uptr GetListOfModules(LoadedModule *modules, uptr max_modules) {
  UNIMPLEMENTED();
}
#else  // ANDROID
typedef ElfW(Phdr) Elf_Phdr;

struct DlIteratePhdrData {
  LoadedModule *modules;
  uptr current_n;
  uptr max_n;
};

static const uptr kMaxPathLength = 512;

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrData *data = (DlIteratePhdrData*)arg;
  if (data->current_n == data->max_n)
    return 0;
  char *module_name = 0;
  if (data->current_n == 0) {
    // First module is the binary itself.
    module_name = (char*)InternalAlloc(kMaxPathLength);
    uptr module_name_len = readlink("/proc/self/exe",
                                    module_name, kMaxPathLength);
    CHECK_NE(module_name_len, (uptr)-1);
    CHECK_LT(module_name_len, kMaxPathLength);
    module_name[module_name_len] = '\0';
  } else if (info->dlpi_name) {
    module_name = internal_strdup(info->dlpi_name);
  }
  if (module_name == 0 || module_name[0] == '\0')
    return 0;
  void *mem = &data->modules[data->current_n];
  LoadedModule *cur_module = new(mem) LoadedModule(module_name,
                                                   info->dlpi_addr);
  data->current_n++;
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const Elf_Phdr *phdr = &info->dlpi_phdr[i];
    if (phdr->p_type == PT_LOAD) {
      uptr cur_beg = info->dlpi_addr + phdr->p_vaddr;
      uptr cur_end = cur_beg + phdr->p_memsz;
      cur_module->addAddressRange(cur_beg, cur_end);
    }
  }
  InternalFree(module_name);
  return 0;
}

uptr GetListOfModules(LoadedModule *modules, uptr max_modules) {
  CHECK(modules);
  DlIteratePhdrData data = {modules, 0, max_modules};
  dl_iterate_phdr(dl_iterate_phdr_cb, &data);
  return data.current_n;
}
#endif  // ANDROID

}  // namespace __sanitizer

#endif  // __linux__
