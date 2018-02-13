#include "source.h"
#include "char-buffer.h"
#include "idioms.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>  // TODO pmk rm
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

// TODO: Port to Windows &c.

namespace Fortran {
namespace parser {

SourceFile::~SourceFile() { Close(); }

static std::vector<size_t> FindLineStarts(const char *source, size_t bytes) {
  if (bytes == 0) {
    return {};
  }
  CHECK(source[bytes - 1] == '\n' && "missing ultimate newline");
  std::vector<size_t> result;
  size_t at{0};
  do {
    result.push_back(at);
    const void *vp{static_cast<const void *>(&source[at])};
    const void *vnl{std::memchr(vp, '\n', bytes - at)};
    const char *nl{static_cast<const char *>(vnl)};
    at = nl + 1 - source;
  } while (at < bytes);
  result.shrink_to_fit();
  return result;
}

std::string DirectoryName(std::string path) {
  auto lastSlash = path.rfind("/");
  return lastSlash == std::string::npos ? path : path.substr(0, lastSlash);
}

std::string LocateSourceFile(
    std::string name, const std::vector<std::string> &searchPath) {
  if (name.empty() || name == "-" || name[0] == '/') {
    return name;
  }
  for (const std::string &dir : searchPath) {
    std::string path{dir + '/' + name};
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) == 0 && !S_ISDIR(statbuf.st_mode)) {
      return path;
    }
  }
  return name;
}

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
      *error << "could not open " << error_path << ": " << std::strerror(errno);
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
      void *vp = mmap(0, bytes_, PROT_READ, MAP_SHARED, fileDescriptor_, 0);
      if (vp != MAP_FAILED) {
        content_ = static_cast<const char *>(const_cast<const void *>(vp));
        if (content_[bytes_ - 1] == '\n' &&
            std::memchr(vp, '\r', bytes_) == nullptr) {
          isMemoryMapped_ = true;
          lineStart_ = FindLineStarts(content_, bytes_);
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
  bytes_ = buffer.size();
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
  lineStart_ = FindLineStarts(content_, bytes_);
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

std::pair<int, int> SourceFile::FindOffsetLineAndColumn(size_t at) const {
  CHECK(at < bytes_);
  if (lineStart_.empty()) {
    return {1, static_cast<int>(at + 1)};
  }
  size_t low{0}, count{lineStart_.size()};
  while (count > 1) {
    size_t mid{low + (count >> 1)};
    if (lineStart_[mid] > at) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  return {
      static_cast<int>(low + 1), static_cast<int>(at - lineStart_[low] + 1)};
}
}  // namespace parser
}  // namespace Fortran
