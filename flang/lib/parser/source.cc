#include "char-buffer.h"
#include "idioms.h"
#include "source.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// TODO: Port to Windows &c.

namespace Fortran {
namespace parser {

SourceFile::~SourceFile() { Close(); }

bool SourceFile::Open(std::string path, std::stringstream *error) {
  Close();
  path_ = path;
  std::string error_path;
  errno = 0;
  if (path == "-") {
    error_path = "standard input";
    fileDescriptor_ = 0;
  } else {
    error_path = "'"s + path + "'";
    fileDescriptor_ = open(path.c_str(), O_RDONLY);
    if (fileDescriptor_ < 0) {
      *error << "could not open '" << error_path
             << "': " << std::strerror(errno);
      return false;
    }
  }
  struct stat statbuf;
  if (fstat(fileDescriptor_, &statbuf) != 0) {
    *error << "fstat failed on " << error_path << ": " << std::strerror(errno);
    Close();
    return false;
  }
  if (S_ISDIR(statbuf.st_mode)) {
    *error << error_path << " is a directory";
    Close();
    return false;
  }

  // Try to map the the source file into the process' address space.
  if (S_ISREG(statbuf.st_mode)) {
    bytes_ = static_cast<size_t>(statbuf.st_size);
    if (bytes_ > 0) {
      auto vp = mmap(0, bytes_, PROT_READ, MAP_SHARED, fileDescriptor_, 0);
      if (vp != MAP_FAILED) {
        content_ = reinterpret_cast<const char *>(vp);
        if (content_[bytes_ - 1] == '\n' &&
            std::memchr(vp, '\r', bytes_) == nullptr) {
          isMemoryMapped_ = true;
          return true;
        }
        // The file needs normalizing.
        munmap(vp, bytes_);
        content_ = nullptr;
      }
    }
  }

  // Couldn't map the file, or its content needs line ending normalization.
  // Read it into an expandable buffer, then marshal its content into a single
  // contiguous block.
  CharBuffer buffer;
  while (true) {
    size_t count;
    char *to{buffer.FreeSpace(&count)};
    ssize_t got{read(fileDescriptor_, to, count)};
    if (got < 0) {
      *error << "could not read " << error_path << ": " << std::strerror(errno);
      Close();
      return false;
    }
    if (got == 0) {
      break;
    }
    buffer.Claim(got);
  }
  close(fileDescriptor_);
  fileDescriptor_ = -1;
  bytes_ = buffer.bytes();
  if (bytes_ == 0) {
    // empty file
    content_ = nullptr;
    return true;
  }

  char *contig{new char[bytes_ + 1 /* for extra newline if needed */]};
  content_ = contig;
  char *to{contig};
  for (char ch : buffer) {
    if (ch != '\r') {
      *to++ = ch;
    }
  }
  if (to == contig || to[-1] != '\n') {
    *to++ = '\n';  // supply a missing terminal newline
  }
  bytes_ = to - contig;
  return true;
}

void SourceFile::Close() {
  if (isMemoryMapped_) {
    munmap(reinterpret_cast<void *>(const_cast<char *>(content_)), bytes_);
    isMemoryMapped_ = false;
  } else if (content_ != nullptr) {
    delete[] content_;
  }
  content_ = nullptr;
  bytes_ = 0;
  if (fileDescriptor_ >= 0) {
    close(fileDescriptor_);
    fileDescriptor_ = -1;
  }
  path_.clear();
}
}  // namespace parser
}  // namespace Fortran
