//===-- sanitizer_symbolizer_process_libcdep.cc ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of SymbolizerProcess used by external symbolizers.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_POSIX
#include "sanitizer_posix.h"
#include "sanitizer_symbolizer_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#if SANITIZER_MAC
#include <util.h>  // for forkpty()
#endif  // SANITIZER_MAC

namespace __sanitizer {

SymbolizerProcess::SymbolizerProcess(const char *path, bool use_forkpty)
    : path_(path),
      input_fd_(kInvalidFd),
      output_fd_(kInvalidFd),
      times_restarted_(0),
      failed_to_start_(false),
      reported_invalid_path_(false),
      use_forkpty_(use_forkpty) {
  CHECK(path_);
  CHECK_NE(path_[0], '\0');
}

const char *SymbolizerProcess::SendCommand(const char *command) {
  for (; times_restarted_ < kMaxTimesRestarted; times_restarted_++) {
    // Start or restart symbolizer if we failed to send command to it.
    if (const char *res = SendCommandImpl(command))
      return res;
    Restart();
  }
  if (!failed_to_start_) {
    Report("WARNING: Failed to use and restart external symbolizer!\n");
    failed_to_start_ = true;
  }
  return 0;
}

bool SymbolizerProcess::Restart() {
  if (input_fd_ != kInvalidFd)
    internal_close(input_fd_);
  if (output_fd_ != kInvalidFd)
    internal_close(output_fd_);
  return StartSymbolizerSubprocess();
}

const char *SymbolizerProcess::SendCommandImpl(const char *command) {
  if (input_fd_ == kInvalidFd || output_fd_ == kInvalidFd)
      return 0;
  if (!WriteToSymbolizer(command, internal_strlen(command)))
      return 0;
  if (!ReadFromSymbolizer(buffer_, kBufferSize))
      return 0;
  return buffer_;
}

bool SymbolizerProcess::ReadFromSymbolizer(char *buffer, uptr max_length) {
  if (max_length == 0)
    return true;
  uptr read_len = 0;
  while (true) {
    uptr just_read = internal_read(input_fd_, buffer + read_len,
                                   max_length - read_len - 1);
    // We can't read 0 bytes, as we don't expect external symbolizer to close
    // its stdout.
    if (just_read == 0 || just_read == (uptr)-1) {
      Report("WARNING: Can't read from symbolizer at fd %d\n", input_fd_);
      return false;
    }
    read_len += just_read;
    if (ReachedEndOfOutput(buffer, read_len))
      break;
  }
  buffer[read_len] = '\0';
  return true;
}

bool SymbolizerProcess::WriteToSymbolizer(const char *buffer, uptr length) {
  if (length == 0)
    return true;
  uptr write_len = internal_write(output_fd_, buffer, length);
  if (write_len == 0 || write_len == (uptr)-1) {
    Report("WARNING: Can't write to symbolizer at fd %d\n", output_fd_);
    return false;
  }
  return true;
}

bool SymbolizerProcess::StartSymbolizerSubprocess() {
  if (!FileExists(path_)) {
    if (!reported_invalid_path_) {
      Report("WARNING: invalid path to external symbolizer!\n");
      reported_invalid_path_ = true;
    }
    return false;
  }

  int pid;
  if (use_forkpty_) {
#if SANITIZER_MAC
    fd_t fd = kInvalidFd;
    // Use forkpty to disable buffering in the new terminal.
    pid = forkpty(&fd, 0, 0, 0);
    if (pid == -1) {
      // forkpty() failed.
      Report("WARNING: failed to fork external symbolizer (errno: %d)\n",
             errno);
      return false;
    } else if (pid == 0) {
      // Child subprocess.
      ExecuteWithDefaultArgs(path_);
      internal__exit(1);
    }

    // Continue execution in parent process.
    input_fd_ = output_fd_ = fd;

    // Disable echo in the new terminal, disable CR.
    struct termios termflags;
    tcgetattr(fd, &termflags);
    termflags.c_oflag &= ~ONLCR;
    termflags.c_lflag &= ~ECHO;
    tcsetattr(fd, TCSANOW, &termflags);
#else  // SANITIZER_MAC
    UNIMPLEMENTED();
#endif  // SANITIZER_MAC
  } else {
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

    // Real fork() may call user callbacks registered with pthread_atfork().
    pid = internal_fork();
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
      for (int fd = sysconf(_SC_OPEN_MAX); fd > 2; fd--)
        internal_close(fd);
      ExecuteWithDefaultArgs(path_);
      internal__exit(1);
    }

    // Continue execution in parent process.
    internal_close(outfd[0]);
    internal_close(infd[1]);
    input_fd_ = infd[0];
    output_fd_ = outfd[1];
  }

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
