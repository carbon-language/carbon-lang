//===- HexDisassembler.h - Disassembler for hex strings -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the disassembler of strings of bytes written in
// hexadecimal, from standard input or from a file.
//
//===----------------------------------------------------------------------===//

#ifndef HEXDISASSEMBLER_H
#define HEXDISASSEMBLER_H

#include <string>

namespace llvm {

class Target;
class MemoryBuffer;

class HexDisassembler {
public:
  static int disassemble(const Target &target, 
                         const std::string &tripleString,
                         MemoryBuffer &buffer);
};
  
} // namespace llvm

#endif
