#include "char-buffer.h"
#include "idioms.h"
#include <algorithm>
#include <cstring>

namespace Fortran {

char *CharBuffer::FreeSpace(size_t *n) {
  int offset{LastBlockOffset()};
  if (blocks_.empty()) {
    blocks_.emplace_front();
    last_ = blocks_.begin();
    lastBlockEmpty_ = true;
  } else if (offset == 0 && !lastBlockEmpty_) {
    last_ = blocks_.emplace_after(last_);
    lastBlockEmpty_ = true;
  }
  *n = Block::capacity - offset;
  return last_->data + offset;
}

void CharBuffer::Claim(size_t n) {
  if (n > 0) {
    bytes_ += n;
    lastBlockEmpty_ = false;
  }
}

void CharBuffer::Put(const char *data, size_t n) {
  size_t chunk;
  for (size_t at{0}; at < n; at += chunk) {
    char *to{FreeSpace(&chunk)};
    chunk = std::min(n - at, chunk);
    Claim(chunk);
    std::memcpy(to, data + at, chunk);
  }
}

void CharBuffer::Put(const std::string &str) {
  Put(str.data(), str.size());
}

void CharBuffer::Put(const std::vector<char> &data) {
  size_t n{data.size()};
  size_t chunk;
  for (size_t at{0}; at < n; at += chunk) {
    char *to{FreeSpace(&chunk)};
    chunk = std::min(n - at, chunk);
    Claim(chunk);
    std::memcpy(to, &data[at], chunk);
  }
}

void CharBuffer::CopyToContiguous(char *data) {
  char *to{data};
  for (char ch : *this) {
    *to++ = ch;
  }
  CHECK(to == data + bytes_);
}
}  // namespace Fortran
