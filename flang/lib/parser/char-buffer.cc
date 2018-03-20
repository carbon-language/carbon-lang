#include "char-buffer.h"
#include "idioms.h"
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace Fortran {
namespace parser {

char *CharBuffer::FreeSpace(std::size_t *n) {
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

void CharBuffer::Claim(std::size_t n) {
  if (n > 0) {
    bytes_ += n;
    lastBlockEmpty_ = false;
  }
}

void CharBuffer::Put(const char *data, std::size_t n) {
  std::size_t chunk;
  for (std::size_t at{0}; at < n; at += chunk) {
    char *to{FreeSpace(&chunk)};
    chunk = std::min(n - at, chunk);
    Claim(chunk);
    std::memcpy(to, data + at, chunk);
  }
}

void CharBuffer::Put(const std::string &str) { Put(str.data(), str.size()); }
}  // namespace parser
}  // namespace Fortran
