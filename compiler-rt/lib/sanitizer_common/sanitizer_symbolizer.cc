//===-- sanitizer_symbolizer.cc -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries. See sanitizer_symbolizer.h for details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

void AddressInfo::Clear() {
  InternalFree(module);
  InternalFree(function);
  InternalFree(file);
  internal_memset(this, 0, sizeof(AddressInfo));
}

LoadedModule::LoadedModule(const char *module_name, uptr base_address) {
  full_name_ = internal_strdup(module_name);
  short_name_ = internal_strrchr(module_name, '/');
  if (short_name_ == 0) {
    short_name_ = full_name_;
  } else {
    short_name_++;
  }
  base_address_ = base_address;
  n_ranges_ = 0;
}

void LoadedModule::addAddressRange(uptr beg, uptr end) {
  CHECK_LT(n_ranges_, kMaxNumberOfAddressRanges);
  ranges_[n_ranges_].beg = beg;
  ranges_[n_ranges_].end = end;
  n_ranges_++;
}

bool LoadedModule::containsAddress(uptr address) const {
  for (uptr i = 0; i < n_ranges_; i++) {
    if (ranges_[i].beg <= address && address < ranges_[i].end)
      return true;
  }
  return false;
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
    *result = internal_atoll(buff);
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
//   <empty line>
class ExternalSymbolizer {
 public:
  ExternalSymbolizer(const char *path, int input_fd, int output_fd)
      : path_(path),
        input_fd_(input_fd),
        output_fd_(output_fd),
        times_restarted_(0) {
    CHECK(path_);
    CHECK_NE(input_fd_, kInvalidFd);
    CHECK_NE(output_fd_, kInvalidFd);
  }
  bool getFileLineInfo(const char *module, uptr module_offset,
                       AddressInfo *info) {
    CHECK(module);
    // FIXME: Make sure this buffer always has sufficient size to hold
    // large debug info.
    static const int kMaxBufferSize = 1024;
    InternalScopedBuffer<char> buffer(kMaxBufferSize);
    internal_snprintf(buffer, kMaxBufferSize, "%s %zu\n",
                      module, module_offset);
    // FIXME: If read/write fails, we should try to respawn symbolizer
    // subprocess.
    if (!writeToSymbolizer(buffer, internal_strlen(buffer)))
      return false;
    if (!readFromSymbolizer(buffer, kMaxBufferSize))
      return false;
    const char *str = buffer.data();
    str = ExtractToken(str, "\n", &info->function);
    str = ExtractToken(str, ":\n", &info->file);
    str = ExtractInt(str, ":\n", &info->line);
    str = ExtractInt(str, ":\n", &info->column);
    // Functions and filenames can be "??", in which case we write 0 to address
    // info to mark that names are unknown.
    if (0 == internal_strcmp(info->function, "??")) {
      InternalFree(info->function);
      info->function = 0;
    }
    if (0 == internal_strcmp(info->file, "??")) {
      InternalFree(info->file);
      info->file = 0;
    }
    return true;
  }
  bool Restart() {
    if (times_restarted_ >= kMaxTimesRestarted) return false;
    times_restarted_++;
    internal_close(input_fd_);
    internal_close(output_fd_);
    return StartSymbolizerSubprocess(path_, &input_fd_, &output_fd_);
  }

 private:
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

  static const uptr kMaxTimesRestarted = 5;
  uptr times_restarted_;
};

class Symbolizer {
 public:
  uptr SymbolizeCode(uptr addr, AddressInfo *frames, uptr max_frames) {
    if (max_frames == 0)
      return 0;
    AddressInfo *info = &frames[0];
    info->Clear();
    info->address = addr;
    LoadedModule *module = FindModuleForAddress(addr);
    if (module) {
      info->module = internal_strdup(module->full_name());
      info->module_offset = info->address - module->base_address();
      if (external_symbolizer_ == 0) {
        ReportExternalSymbolizerError(
            "WARNING: Trying to symbolize code, but external "
            "symbolizer is not initialized!\n");
      } else {
        while (!external_symbolizer_->getFileLineInfo(
            info->module, info->module_offset, info)) {
          // Try to restart symbolizer subprocess. If we don't succeed, forget
          // about it and don't try to use it later.
          if (!external_symbolizer_->Restart()) {
            ReportExternalSymbolizerError(
                "WARNING: Failed to use and restart external symbolizer!\n");
            InternalFree(external_symbolizer_);
            external_symbolizer_ = 0;
            break;
          }
        }
      }
      return 1;
    }
    return 0;
  }
  bool InitializeExternalSymbolizer(const char *path_to_symbolizer) {
    int input_fd, output_fd;
    if (!StartSymbolizerSubprocess(path_to_symbolizer, &input_fd, &output_fd))
      return false;
    void *mem = InternalAlloc(sizeof(ExternalSymbolizer));
    external_symbolizer_ = new(mem) ExternalSymbolizer(path_to_symbolizer,
                                                       input_fd, output_fd);
    return true;
  }

 private:
  LoadedModule *FindModuleForAddress(uptr address) {
    if (modules_ == 0) {
      modules_ = (LoadedModule*)InternalAlloc(
          kMaxNumberOfModuleContexts * sizeof(LoadedModule));
      CHECK(modules_);
      n_modules_ = GetListOfModules(modules_, kMaxNumberOfModuleContexts);
      CHECK_GT(n_modules_, 0);
      CHECK_LT(n_modules_, kMaxNumberOfModuleContexts);
    }
    for (uptr i = 0; i < n_modules_; i++) {
      if (modules_[i].containsAddress(address)) {
        return &modules_[i];
      }
    }
    return 0;
  }
  void ReportExternalSymbolizerError(const char *msg) {
    // Don't use atomics here for now, as SymbolizeCode can't be called
    // from multiple threads anyway.
    static bool reported;
    if (!reported) {
      Report(msg);
      reported = true;
    }
  }

  static const uptr kMaxNumberOfModuleContexts = 4096;
  // Array of module descriptions is leaked.
  LoadedModule *modules_;
  uptr n_modules_;

  ExternalSymbolizer *external_symbolizer_;
};

static Symbolizer symbolizer;  // Linker initialized.

uptr SymbolizeCode(uptr address, AddressInfo *frames, uptr max_frames) {
  return symbolizer.SymbolizeCode(address, frames, max_frames);
}

bool InitializeExternalSymbolizer(const char *path_to_symbolizer) {
  return symbolizer.InitializeExternalSymbolizer(path_to_symbolizer);
}

}  // namespace __sanitizer
