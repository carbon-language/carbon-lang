//===--- Endian.h - utility functions for endian-aware reads/writes -----*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_ENDIAN_H
#define LLD_CORE_ENDIAN_H

#include "llvm/Support/Endian.h"

namespace lld {

inline uint16_t read16le(const void *p) {
  return *reinterpret_cast<const llvm::support::ulittle16_t *>(p);
}

inline uint32_t read32le(const void *p) {
  return *reinterpret_cast<const llvm::support::ulittle32_t *>(p);
}

inline uint64_t read64le(const void *p) {
  return *reinterpret_cast<const llvm::support::ulittle64_t *>(p);
}

inline void write16le(void *p, uint16_t v) {
  *reinterpret_cast<llvm::support::ulittle16_t *>(p) = v;
}

inline void write32le(void *p, uint32_t v) {
  *reinterpret_cast<llvm::support::ulittle32_t *>(p) = v;
}

inline void write64le(void *p, uint64_t v) {
  *reinterpret_cast<llvm::support::ulittle64_t *>(p) = v;
}

inline uint16_t read16be(const void *p) {
  return *reinterpret_cast<const llvm::support::ubig16_t *>(p);
}

inline uint32_t read32be(const void *p) {
  return *reinterpret_cast<const llvm::support::ubig32_t *>(p);
}

inline uint64_t read64be(const void *p) {
  return *reinterpret_cast<const llvm::support::ubig64_t *>(p);
}

inline void write16be(void *p, uint16_t v) {
  *reinterpret_cast<llvm::support::ubig16_t *>(p) = v;
}

inline void write32be(void *p, uint32_t v) {
  *reinterpret_cast<llvm::support::ubig32_t *>(p) = v;
}

inline void write64be(void *p, uint64_t v) {
  *reinterpret_cast<llvm::support::ubig64_t *>(p) = v;
}

} // namespace lld

#endif
