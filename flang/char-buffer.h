#ifndef FORTRAN_CHAR_BUFFER_H_
#define FORTRAN_CHAR_BUFFER_H_

// Defines a simple expandable buffer suitable for efficiently accumulating
// a stream of bytes.

#include <forward_list>
#include <string>
#include <utility>
#include <vector>

namespace Fortran {

class CharBuffer {
 public:
  CharBuffer() {}
  CharBuffer(CharBuffer &&that)
    : blocks_(std::move(that.blocks_)), last_{that.last_},
      bytes_{that.bytes_}, lastBlockEmpty_{that.lastBlockEmpty_} {
    that.clear();
  }
  CharBuffer &operator=(CharBuffer &&that) {
    blocks_ = std::move(that.blocks_);
    last_ = that.last_;
    bytes_ = that.bytes_;
    lastBlockEmpty_ = that.lastBlockEmpty_;
    that.clear();
    return *this;
  }

  size_t bytes() const { return bytes_; }

  void clear() {
    blocks_.clear();
    last_ = blocks_.end();
    bytes_ = 0;
    lastBlockEmpty_ = false;
  }

  char *FreeSpace(size_t *);
  void Claim(size_t);
  void Put(const char *data, size_t n);
  void Put(const std::string &);
  void Put(char x) { Put(&x, 1); }
  void CopyToContiguous(char *data);

 private:
  struct Block {
    static constexpr size_t capacity{1 << 20};
    char data[capacity];
  };

 public:
  class iterator {
   public:
    iterator() {}
    iterator(std::forward_list<Block>::const_iterator block, int offset)
      : block_{block}, offset_{offset} {}
    iterator(const iterator &that)
      : block_{that.block_}, offset_{that.offset_} {}
    iterator &operator=(const iterator &that) {
      block_ = that.block_;
      offset_ = that.offset_;
      return *this;
    }
    const char &operator*() const { return block_->data[offset_]; }
    iterator &operator++() {
      if (++offset_ == Block::capacity) {
        ++block_;
        offset_ = 0;
      }
      return *this;
    }
    iterator operator++(int) {
      iterator result{*this};
      ++*this;
      return result;
    }
    iterator &operator+=(size_t n) {
      while (n >= Block::capacity - offset_) {
        n -= Block::capacity - offset_;
        offset_ = 0;
        ++block_;
      }
      offset_ += n;
      return *this;
    }
    bool operator==(const iterator &that) const {
      return block_ == that.block_ && offset_ == that.offset_;
    }
    bool operator!=(const iterator &that) const {
      return block_ != that.block_ || offset_ != that.offset_;
    }
   private:
    std::forward_list<Block>::const_iterator block_;
    int offset_;
  };

  iterator begin() const { return iterator(blocks_.begin(), 0); }
  iterator end() const {
    int offset = LastBlockOffset();
    if (offset != 0 || lastBlockEmpty_) {
      return iterator(last_, offset);
    }
    return iterator(blocks_.end(), 0);
  }

 private:
  int LastBlockOffset() const { return bytes_ % Block::capacity; }
  std::forward_list<Block> blocks_;
  std::forward_list<Block>::iterator last_{blocks_.end()};
  size_t bytes_{0};
  bool lastBlockEmpty_{false};
};
}  // namespace Fortran
#endif  // FORTRAN_CHAR_BUFFER_H_
