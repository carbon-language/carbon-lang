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
#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_linux.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_symbolizer.h"

#include <errno.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

// C++ demangling function, as required by Itanium C++ ABI. This is weak,
// because we do not require a C++ ABI library to be linked to a program
// using sanitizers; if it's not present, we'll just use the mangled name.
namespace __cxxabiv1 {
  extern "C" SANITIZER_WEAK_ATTRIBUTE
  char *__cxa_demangle(const char *mangled, char *buffer,
                                  size_t *length, int *status);
}

namespace __sanitizer {

// Attempts to demangle the name via __cxa_demangle from __cxxabiv1.
static const char *DemangleCXXABI(const char *name) {
  // FIXME: __cxa_demangle aggressively insists on allocating memory.
  // There's not much we can do about that, short of providing our
  // own demangler (libc++abi's implementation could be adapted so that
  // it does not allocate). For now, we just call it anyway, and we leak
  // the returned value.
  if (__cxxabiv1::__cxa_demangle)
    if (const char *demangled_name =
          __cxxabiv1::__cxa_demangle(name, 0, 0, 0))
      return demangled_name;

  return name;
}

#if defined(__x86_64__)
static const char* const kSymbolizerArch = "--default-arch=x86_64";
#elif defined(__i386__)
static const char* const kSymbolizerArch = "--default-arch=i386";
#elif defined(__powerpc64__)
static const char* const kSymbolizerArch = "--default-arch=powerpc64";
#else
static const char* const kSymbolizerArch = "--default-arch=unknown";
#endif

static const int kSymbolizerStartupTimeMillis = 10;

// Creates external symbolizer connected via pipe, user should write
// to output_fd and read from input_fd.
static bool StartSymbolizerSubprocess(const char *path_to_symbolizer,
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

// Extracts the prefix of "str" that consists of any characters not
// present in "delims" string, and copies this prefix to "result", allocating
// space for it.
// Returns a pointer to "str" after skipping extracted prefix and first
// delimiter char.
static const char *ExtractToken(const char *str, const char *delims,
                                char **result) {
  uptr prefix_len = internal_strcspn(str, delims);
  *result = (char*)InternalAlloc(prefix_len + 1);
  internal_memcpy(*result, str, prefix_len);
  (*result)[prefix_len] = '\0';
  const char *prefix_end = str + prefix_len;
  if (*prefix_end != '\0') prefix_end++;
  return prefix_end;
}

// Same as ExtractToken, but converts extracted token to integer.
static const char *ExtractInt(const char *str, const char *delims,
                              int *result) {
  char *buff;
  const char *ret = ExtractToken(str, delims, &buff);
  if (buff != 0) {
    *result = (int)internal_atoll(buff);
  }
  InternalFree(buff);
  return ret;
}

static const char *ExtractUptr(const char *str, const char *delims,
                               uptr *result) {
  char *buff;
  const char *ret = ExtractToken(str, delims, &buff);
  if (buff != 0) {
    *result = (uptr)internal_atoll(buff);
  }
  InternalFree(buff);
  return ret;
}

// ExternalSymbolizer encapsulates communication between the tool and
// external symbolizer program, running in a different subprocess,
// For now we assume the following protocol:
// For each request of the form
//   <module_name> <module_offset>
// passed to STDIN, external symbolizer prints to STDOUT response:
//   <function_name>
//   <file_name>:<line_number>:<column_number>
//   <function_name>
//   <file_name>:<line_number>:<column_number>
//   ...
//   <empty line>
// ExternalSymbolizer may not be used from two threads simultaneously.
class ExternalSymbolizer {
 public:
  explicit ExternalSymbolizer(const char *path)
      : path_(path),
        input_fd_(kInvalidFd),
        output_fd_(kInvalidFd),
        times_restarted_(0),
        failed_to_start_(false) {
    CHECK(path_);
    CHECK_NE(path[0], '\0');
  }

  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    for (; times_restarted_ < kMaxTimesRestarted; times_restarted_++) {
      // Start or restart symbolizer if we failed to send command to it.
      if (char *res = SendCommandImpl(is_data, module_name, module_offset))
        return res;
      Restart();
    }
    if (!failed_to_start_) {
      Report("WARNING: Failed to use and restart external symbolizer!\n");
      failed_to_start_ = true;
    }
    return 0;
  }

  void Flush() {
  }

 private:
  bool Restart() {
    if (input_fd_ != kInvalidFd)
      internal_close(input_fd_);
    if (output_fd_ != kInvalidFd)
      internal_close(output_fd_);
    return StartSymbolizerSubprocess(path_, &input_fd_, &output_fd_);
  }

  char *SendCommandImpl(bool is_data, const char *module_name,
                        uptr module_offset) {
    if (input_fd_ == kInvalidFd || output_fd_ == kInvalidFd)
      return 0;
    CHECK(module_name);
    internal_snprintf(buffer_, kBufferSize, "%s\"%s\" 0x%zx\n",
                      is_data ? "DATA " : "", module_name, module_offset);
    if (!writeToSymbolizer(buffer_, internal_strlen(buffer_)))
      return 0;
    if (!readFromSymbolizer(buffer_, kBufferSize))
      return 0;
    return buffer_;
  }

  bool readFromSymbolizer(char *buffer, uptr max_length) {
    if (max_length == 0)
      return true;
    uptr read_len = 0;
    while (true) {
      uptr just_read = internal_read(input_fd_, buffer + read_len,
                                     max_length - read_len);
      // We can't read 0 bytes, as we don't expect external symbolizer to close
      // its stdout.
      if (just_read == 0 || just_read == (uptr)-1) {
        Report("WARNING: Can't read from symbolizer at fd %d\n", input_fd_);
        return false;
      }
      read_len += just_read;
      // Empty line marks the end of symbolizer output.
      if (read_len >= 2 && buffer[read_len - 1] == '\n' &&
                           buffer[read_len - 2] == '\n') {
        break;
      }
    }
    return true;
  }

  bool writeToSymbolizer(const char *buffer, uptr length) {
    if (length == 0)
      return true;
    uptr write_len = internal_write(output_fd_, buffer, length);
    if (write_len == 0 || write_len == (uptr)-1) {
      Report("WARNING: Can't write to symbolizer at fd %d\n", output_fd_);
      return false;
    }
    return true;
  }

  const char *path_;
  int input_fd_;
  int output_fd_;

  static const uptr kBufferSize = 16 * 1024;
  char buffer_[kBufferSize];

  static const uptr kMaxTimesRestarted = 5;
  uptr times_restarted_;
  bool failed_to_start_;
};

#if SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
bool __sanitizer_symbolize_code(const char *ModuleName, u64 ModuleOffset,
                                char *Buffer, int MaxLength);
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
bool __sanitizer_symbolize_data(const char *ModuleName, u64 ModuleOffset,
                                char *Buffer, int MaxLength);
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void __sanitizer_symbolize_flush();
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
int __sanitizer_symbolize_demangle(const char *Name, char *Buffer,
                                   int MaxLength);
}  // extern "C"

class InternalSymbolizer {
 public:
  typedef bool (*SanitizerSymbolizeFn)(const char*, u64, char*, int);

  static InternalSymbolizer *get(LowLevelAllocator *alloc) {
    if (__sanitizer_symbolize_code != 0 &&
        __sanitizer_symbolize_data != 0) {
      return new(*alloc) InternalSymbolizer();
    }
    return 0;
  }

  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    SanitizerSymbolizeFn symbolize_fn = is_data ? __sanitizer_symbolize_data
                                                : __sanitizer_symbolize_code;
    if (symbolize_fn(module_name, module_offset, buffer_, kBufferSize))
      return buffer_;
    return 0;
  }

  void Flush() {
    if (__sanitizer_symbolize_flush)
      __sanitizer_symbolize_flush();
  }

  const char *Demangle(const char *name) {
    if (__sanitizer_symbolize_demangle) {
      for (uptr res_length = 1024;
           res_length <= InternalSizeClassMap::kMaxSize;) {
        char *res_buff = static_cast<char*>(InternalAlloc(res_length));
        uptr req_length =
            __sanitizer_symbolize_demangle(name, res_buff, res_length);
        if (req_length > res_length) {
          res_length = req_length + 1;
          InternalFree(res_buff);
          continue;
        }
        return res_buff;
      }
    }
    return name;
  }

 private:
  InternalSymbolizer() { }

  static const int kBufferSize = 16 * 1024;
  static const int kMaxDemangledNameSize = 1024;
  char buffer_[kBufferSize];
};
#else  // SANITIZER_SUPPORTS_WEAK_HOOKS

class InternalSymbolizer {
 public:
  static InternalSymbolizer *get(LowLevelAllocator *alloc) { return 0; }
  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    return 0;
  }
  void Flush() { }
  const char *Demangle(const char *name) { return name; }
};

#endif  // SANITIZER_SUPPORTS_WEAK_HOOKS

class POSIXSymbolizer : public Symbolizer {
 public:
  POSIXSymbolizer(ExternalSymbolizer *external_symbolizer,
                  InternalSymbolizer *internal_symbolizer)
      : Symbolizer(),
        external_symbolizer_(external_symbolizer),
        internal_symbolizer_(internal_symbolizer) {}

  uptr SymbolizePC(uptr addr, AddressInfo *frames, uptr max_frames) {
    BlockingMutexLock l(&mu_);
    if (max_frames == 0)
      return 0;
    LoadedModule *module = FindModuleForAddress(addr);
    if (module == 0)
      return 0;
    const char *module_name = module->full_name();
    uptr module_offset = addr - module->base_address();
    const char *str = SendCommand(false, module_name, module_offset);
    if (str == 0) {
      // External symbolizer was not initialized or failed. Fill only data
      // about module name and offset.
      AddressInfo *info = &frames[0];
      info->Clear();
      info->FillAddressAndModuleInfo(addr, module_name, module_offset);
      return 1;
    }
    uptr frame_id = 0;
    for (frame_id = 0; frame_id < max_frames; frame_id++) {
      AddressInfo *info = &frames[frame_id];
      char *function_name = 0;
      str = ExtractToken(str, "\n", &function_name);
      CHECK(function_name);
      if (function_name[0] == '\0') {
        // There are no more frames.
        break;
      }
      info->Clear();
      info->FillAddressAndModuleInfo(addr, module_name, module_offset);
      info->function = function_name;
      // Parse <file>:<line>:<column> buffer.
      char *file_line_info = 0;
      str = ExtractToken(str, "\n", &file_line_info);
      CHECK(file_line_info);
      const char *line_info = ExtractToken(file_line_info, ":", &info->file);
      line_info = ExtractInt(line_info, ":", &info->line);
      line_info = ExtractInt(line_info, "", &info->column);
      InternalFree(file_line_info);

      // Functions and filenames can be "??", in which case we write 0
      // to address info to mark that names are unknown.
      if (0 == internal_strcmp(info->function, "??")) {
        InternalFree(info->function);
        info->function = 0;
      }
      if (0 == internal_strcmp(info->file, "??")) {
        InternalFree(info->file);
        info->file = 0;
      }
    }
    if (frame_id == 0) {
      // Make sure we return at least one frame.
      AddressInfo *info = &frames[0];
      info->Clear();
      info->FillAddressAndModuleInfo(addr, module_name, module_offset);
      frame_id = 1;
    }
    return frame_id;
  }

  bool SymbolizeData(uptr addr, DataInfo *info) {
    BlockingMutexLock l(&mu_);
    LoadedModule *module = FindModuleForAddress(addr);
    if (module == 0)
      return false;
    const char *module_name = module->full_name();
    uptr module_offset = addr - module->base_address();
    internal_memset(info, 0, sizeof(*info));
    info->address = addr;
    info->module = internal_strdup(module_name);
    info->module_offset = module_offset;
    const char *str = SendCommand(true, module_name, module_offset);
    if (str == 0)
      return true;
    str = ExtractToken(str, "\n", &info->name);
    str = ExtractUptr(str, " ", &info->start);
    str = ExtractUptr(str, "\n", &info->size);
    info->start += module->base_address();
    return true;
  }

  bool IsAvailable() {
    return internal_symbolizer_ != 0 || external_symbolizer_ != 0;
  }

  bool IsExternalAvailable() {
    return external_symbolizer_ != 0;
  }

  void Flush() {
    BlockingMutexLock l(&mu_);
    if (internal_symbolizer_ != 0) {
      SymbolizerScope sym_scope(this);
      internal_symbolizer_->Flush();
    }
    if (external_symbolizer_ != 0)
      external_symbolizer_->Flush();
  }

  const char *Demangle(const char *name) {
    BlockingMutexLock l(&mu_);
    // Run hooks even if we don't use internal symbolizer, as cxxabi
    // demangle may call system functions.
    SymbolizerScope sym_scope(this);
    if (internal_symbolizer_ != 0)
      return internal_symbolizer_->Demangle(name);
    return DemangleCXXABI(name);
  }

  void PrepareForSandboxing() {
#if SANITIZER_LINUX && !SANITIZER_ANDROID
    BlockingMutexLock l(&mu_);
    // Cache /proc/self/exe on Linux.
    CacheBinaryName();
#endif
  }

 private:
  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    mu_.CheckLocked();
    // First, try to use internal symbolizer.
    if (internal_symbolizer_) {
      SymbolizerScope sym_scope(this);
      return internal_symbolizer_->SendCommand(is_data, module_name,
                                               module_offset);
    }
    // Otherwise, fall back to external symbolizer.
    if (external_symbolizer_) {
      return external_symbolizer_->SendCommand(is_data, module_name,
                                               module_offset);
    }
    return 0;
  }

  LoadedModule *FindModuleForAddress(uptr address) {
    mu_.CheckLocked();
    bool modules_were_reloaded = false;
    if (modules_ == 0 || !modules_fresh_) {
      modules_ = (LoadedModule*)(symbolizer_allocator_.Allocate(
          kMaxNumberOfModuleContexts * sizeof(LoadedModule)));
      CHECK(modules_);
      n_modules_ = GetListOfModules(modules_, kMaxNumberOfModuleContexts,
                                    /* filter */ 0);
      // FIXME: Return this check when GetListOfModules is implemented on Mac.
      // CHECK_GT(n_modules_, 0);
      CHECK_LT(n_modules_, kMaxNumberOfModuleContexts);
      modules_fresh_ = true;
      modules_were_reloaded = true;
    }
    for (uptr i = 0; i < n_modules_; i++) {
      if (modules_[i].containsAddress(address)) {
        return &modules_[i];
      }
    }
    // Reload the modules and look up again, if we haven't tried it yet.
    if (!modules_were_reloaded) {
      // FIXME: set modules_fresh_ from dlopen()/dlclose() interceptors.
      // It's too aggressive to reload the list of modules each time we fail
      // to find a module for a given address.
      modules_fresh_ = false;
      return FindModuleForAddress(address);
    }
    return 0;
  }

  // 16K loaded modules should be enough for everyone.
  static const uptr kMaxNumberOfModuleContexts = 1 << 14;
  LoadedModule *modules_;  // Array of module descriptions is leaked.
  uptr n_modules_;
  // If stale, need to reload the modules before looking up addresses.
  bool modules_fresh_;
  BlockingMutex mu_;

  ExternalSymbolizer *external_symbolizer_;        // Leaked.
  InternalSymbolizer *const internal_symbolizer_;  // Leaked.
};

Symbolizer *Symbolizer::PlatformInit(const char *path_to_external) {
  InternalSymbolizer* internal_symbolizer =
      InternalSymbolizer::get(&symbolizer_allocator_);
  ExternalSymbolizer *external_symbolizer = 0;

  if (!internal_symbolizer) {
    // Find path to llvm-symbolizer if it's not provided.
    if (!path_to_external)
      path_to_external = FindPathToBinary("llvm-symbolizer");
    if (path_to_external && path_to_external[0] != '\0')
      external_symbolizer = new(symbolizer_allocator_)
          ExternalSymbolizer(path_to_external);
  }

  return new(symbolizer_allocator_)
      POSIXSymbolizer(external_symbolizer, internal_symbolizer);
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
