//===- EndianStream.h - Stream ops with endian specific data ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities for operating on streams that have endian
// specific data.
//
//===----------------------------------------------------------------------===//

#ifndef _LLVM_SUPPORT_ENDIAN_STREAM_H_
#define _LLVM_SUPPORT_ENDIAN_STREAM_H_

#include <llvm/Support/Endian.h>
#include <llvm/Support/raw_ostream.h>

namespace llvm {
namespace support {

namespace endian {
/// Adapter to write values to a stream in a particular byte order.
template <endianness endian> struct Writer {
  raw_ostream &OS;
  Writer(raw_ostream &OS) : OS(OS) {}
  template <typename value_type> void write(value_type Val) {
    Val = byte_swap<value_type, endian>(Val);
    OS.write((const char *)&Val, sizeof(value_type));
  }
};
} // end namespace endian

} // end namespace support
} // end namespace llvm

#endif // _LLVM_SUPPORT_ENDIAN_STREAM_H_
