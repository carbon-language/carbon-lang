//===-- sanitizer_symbolizer_posix_libcdep.cc -----------------------------===//
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
// POSIX-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_POSIX
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer.h"

#include <errno.h>
#include <sys/wait.h>
#include <unistd.h>

namespace __sanitizer {

#if defined(__x86_64__)
static const char* const kSymbolizerArch = "--default-arch=x86_64";
#elif defined(__i386__)
static const char* const kSymbolizerArch = "--default-arch=i386";
#else
static const char* const kSymbolizerArch = "";
#endif

bool StartSymbolizerSubprocess(const char *path_to_symbolizer,
                               int *input_fd, int *output_fd) {
  if (!FileExists(path_to_symbolizer)) {
    Report("WARNING: invalid path to external symbolizer!\n");
    return false;
  }

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
    execl(path_to_symbolizer, path_to_symbolizer, kSymbolizerArch, (char*)0);
    internal__exit(1);
  }

  // Continue execution in parent process.
  internal_close(outfd[0]);
  internal_close(infd[1]);
  *input_fd = infd[0];
  *output_fd = outfd[1];

  // Check that symbolizer subprocess started successfully.
  int pid_status;
  SleepForMillis(kSymbolizerStartupTimeMillis);
  int exited_pid = waitpid(pid, &pid_status, WNOHANG);
  if (exited_pid != 0) {
    // Either waitpid failed, or child has already exited.
    Report("WARNING: external symbolizer didn't start up correctly!\n");
    return false;
  }

  return true;
}


}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
