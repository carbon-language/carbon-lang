//===-- llvm/CodeGen/ByteStreamer.h - ByteStreamer class --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class that can take bytes that would normally be
// streamed via the AsmPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BYTESTREAMER_H
#define LLVM_CODEGEN_BYTESTREAMER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"
#include "DIEHash.h"

namespace llvm {
class ByteStreamer {
 public:
  // For now we're just handling the calls we need for dwarf emission/hashing.
  virtual void EmitInt8(uint8_t Byte, const Twine &Comment = "") = 0;
  virtual void EmitSLEB128(uint64_t DWord, const Twine &Comment = "") = 0;
  virtual void EmitULEB128(uint64_t DWord, const Twine &Comment = "") = 0;
};

class APByteStreamer : public ByteStreamer {
private:
  AsmPrinter &AP;

public:
  APByteStreamer(AsmPrinter &Asm) : AP(Asm) {}
  void EmitInt8(uint8_t Byte, const Twine &Comment) {
    AP.OutStreamer.AddComment(Comment);
    AP.EmitInt8(Byte);
  }
  void EmitSLEB128(uint64_t DWord, const Twine &Comment) {
    AP.OutStreamer.AddComment(Comment);
    AP.EmitSLEB128(DWord);
  }
  void EmitULEB128(uint64_t DWord, const Twine &Comment) {
    AP.OutStreamer.AddComment(Comment);
    AP.EmitULEB128(DWord);
  }
};

class HashingByteStreamer : public ByteStreamer {
 private:
  DIEHash &Hash;
 public:
 HashingByteStreamer(DIEHash &H) : Hash(H) {}
  void EmitInt8(uint8_t Byte, const Twine &Comment) {
    Hash.update(Byte);
  }
  void EmitSLEB128(uint64_t DWord, const Twine &Comment) {
    Hash.addSLEB128(DWord);
  }
  void EmitULEB128(uint64_t DWord, const Twine &Comment) {
    Hash.addULEB128(DWord);
  }
};
}

#endif
