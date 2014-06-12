//===- llvm/Support/Program.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Program class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PROGRAM_H
#define LLVM_SUPPORT_PROGRAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Path.h"
#include <system_error>

namespace llvm {
namespace sys {

  /// This is the OS-specific separator for PATH like environment variables:
  // a colon on Unix or a semicolon on Windows.
#if defined(LLVM_ON_UNIX)
  const char EnvPathSeparator = ':';
#elif defined (LLVM_ON_WIN32)
  const char EnvPathSeparator = ';';
#endif

/// @brief This struct encapsulates information about a process.
struct ProcessInfo {
#if defined(LLVM_ON_UNIX)
  typedef pid_t ProcessId;
#elif defined(LLVM_ON_WIN32)
  typedef unsigned long ProcessId; // Must match the type of DWORD on Windows.
  typedef void * HANDLE; // Must match the type of HANDLE on Windows.
  /// The handle to the process (available on Windows only).
  HANDLE ProcessHandle;
#else
#error "ProcessInfo is not defined for this platform!"
#endif

  /// The process identifier.
  ProcessId Pid;

  /// The return code, set after execution.
  int ReturnCode;

  ProcessInfo();
};

  /// This function attempts to locate a program in the operating
  /// system's file system using some pre-determined set of locations to search
  /// (e.g. the PATH on Unix). Paths with slashes are returned unmodified.
  ///
  /// It does not perform hashing as a shell would but instead stats each PATH
  /// entry individually so should generally be avoided. Core LLVM library
  /// functions and options should instead require fully specified paths.
  ///
  /// @returns A string containing the path of the program or an empty string if
  /// the program could not be found.
  std::string FindProgramByName(const std::string& name);

  // These functions change the specified standard stream (stdin or stdout) to
  // binary mode. They return errc::success if the specified stream
  // was changed. Otherwise a platform dependent error is returned.
  std::error_code ChangeStdinToBinary();
  std::error_code ChangeStdoutToBinary();

  /// This function executes the program using the arguments provided.  The
  /// invoked program will inherit the stdin, stdout, and stderr file
  /// descriptors, the environment and other configuration settings of the
  /// invoking program.
  /// This function waits for the program to finish, so should be avoided in
  /// library functions that aren't expected to block. Consider using
  /// ExecuteNoWait() instead.
  /// @returns an integer result code indicating the status of the program.
  /// A zero or positive value indicates the result code of the program.
  /// -1 indicates failure to execute
  /// -2 indicates a crash during execution or timeout
  int ExecuteAndWait(
      StringRef Program, ///< Path of the program to be executed. It is
      /// presumed this is the result of the FindProgramByName method.
      const char **args, ///< A vector of strings that are passed to the
      ///< program.  The first element should be the name of the program.
      ///< The list *must* be terminated by a null char* entry.
      const char **env = nullptr, ///< An optional vector of strings to use for
      ///< the program's environment. If not provided, the current program's
      ///< environment will be used.
      const StringRef **redirects = nullptr, ///< An optional array of pointers
      ///< to paths. If the array is null, no redirection is done. The array
      ///< should have a size of at least three. The inferior process's
      ///< stdin(0), stdout(1), and stderr(2) will be redirected to the
      ///< corresponding paths.
      ///< When an empty path is passed in, the corresponding file
      ///< descriptor will be disconnected (ie, /dev/null'd) in a portable
      ///< way.
      unsigned secondsToWait = 0, ///< If non-zero, this specifies the amount
      ///< of time to wait for the child process to exit. If the time
      ///< expires, the child is killed and this call returns. If zero,
      ///< this function will wait until the child finishes or forever if
      ///< it doesn't.
      unsigned memoryLimit = 0, ///< If non-zero, this specifies max. amount
      ///< of memory can be allocated by process. If memory usage will be
      ///< higher limit, the child is killed and this call returns. If zero
      ///< - no memory limit.
      std::string *ErrMsg = nullptr, ///< If non-zero, provides a pointer to a
      ///< string instance in which error messages will be returned. If the
      ///< string is non-empty upon return an error occurred while invoking the
      ///< program.
      bool *ExecutionFailed = nullptr);

  /// Similar to ExecuteAndWait, but returns immediately.
  /// @returns The \see ProcessInfo of the newly launced process.
  /// \note On Microsoft Windows systems, users will need to either call \see
  /// Wait until the process finished execution or win32 CloseHandle() API on
  /// ProcessInfo.ProcessHandle to avoid memory leaks.
  ProcessInfo
  ExecuteNoWait(StringRef Program, const char **args, const char **env = nullptr,
                const StringRef **redirects = nullptr, unsigned memoryLimit = 0,
                std::string *ErrMsg = nullptr, bool *ExecutionFailed = nullptr);

  /// Return true if the given arguments fit within system-specific
  /// argument length limits.
  bool argumentsFitWithinSystemLimits(ArrayRef<const char*> Args);

  /// This function waits for the process specified by \p PI to finish.
  /// \returns A \see ProcessInfo struct with Pid set to:
  /// \li The process id of the child process if the child process has changed
  /// state.
  /// \li 0 if the child process has not changed state.
  /// \note Users of this function should always check the ReturnCode member of
  /// the \see ProcessInfo returned from this function.
  ProcessInfo Wait(
      const ProcessInfo &PI, ///< The child process that should be waited on.
      unsigned SecondsToWait, ///< If non-zero, this specifies the amount of
      ///< time to wait for the child process to exit. If the time expires, the
      ///< child is killed and this function returns. If zero, this function
      ///< will perform a non-blocking wait on the child process.
      bool WaitUntilTerminates, ///< If true, ignores \p SecondsToWait and waits
      ///< until child has terminated.
      std::string *ErrMsg = nullptr ///< If non-zero, provides a pointer to a
      ///< string instance in which error messages will be returned. If the
      ///< string is non-empty upon return an error occurred while invoking the
      ///< program.
      );
  }
}

#endif
