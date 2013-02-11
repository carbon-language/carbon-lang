//===-- RelocVisitor.h - Visitor for object file relocations -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a wrapper around all the different types of relocations
// in different file formats, such that a client can handle them in a unified
// manner by only implementing a minimal number of functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_RELOCVISITOR_H
#define LLVM_OBJECT_RELOCVISITOR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace object {

struct RelocToApply {
  // The computed value after applying the relevant relocations.
  int64_t Value;

  // The width of the value; how many bytes to touch when applying the
  // relocation.
  char Width;
  RelocToApply(const RelocToApply &In) : Value(In.Value), Width(In.Width) {}
  RelocToApply(int64_t Value, char Width) : Value(Value), Width(Width) {}
  RelocToApply() : Value(0), Width(0) {}
};

/// @brief Base class for object file relocation visitors.
class RelocVisitor {
public:
  explicit RelocVisitor(StringRef FileFormat)
    : FileFormat(FileFormat), HasError(false) {}

  // TODO: Should handle multiple applied relocations via either passing in the
  // previously computed value or just count paired relocations as a single
  // visit.
  RelocToApply visit(uint32_t RelocType, RelocationRef R, uint64_t SecAddr = 0,
                     uint64_t Value = 0) {
    if (FileFormat == "ELF64-x86-64") {
      switch (RelocType) {
        case llvm::ELF::R_X86_64_NONE:
          return visitELF_X86_64_NONE(R);
        case llvm::ELF::R_X86_64_64:
          return visitELF_X86_64_64(R, Value);
        case llvm::ELF::R_X86_64_PC32:
          return visitELF_X86_64_PC32(R, Value, SecAddr);
        case llvm::ELF::R_X86_64_32:
          return visitELF_X86_64_32(R, Value);
        case llvm::ELF::R_X86_64_32S:
          return visitELF_X86_64_32S(R, Value);
        default:
          HasError = true;
          return RelocToApply();
      }
    } else if (FileFormat == "ELF32-i386") {
      switch (RelocType) {
      case llvm::ELF::R_386_NONE:
        return visitELF_386_NONE(R);
      case llvm::ELF::R_386_32:
        return visitELF_386_32(R, Value);
      case llvm::ELF::R_386_PC32:
        return visitELF_386_PC32(R, Value, SecAddr);
      default:
        HasError = true;
        return RelocToApply();
      }
    } else if (FileFormat == "ELF64-ppc64") {
      switch (RelocType) {
      case llvm::ELF::R_PPC64_ADDR32:
        return visitELF_PPC64_ADDR32(R, Value);
      default:
        HasError = true;
        return RelocToApply();
      }
    } else if (FileFormat == "ELF32-mips") {
      switch (RelocType) {
      case llvm::ELF::R_MIPS_32:
        return visitELF_MIPS_32(R, Value);
      default:
        HasError = true;
        return RelocToApply();
      }
    } else if (FileFormat == "ELF64-aarch64") {
      switch (RelocType) {
      case llvm::ELF::R_AARCH64_ABS32:
        return visitELF_AARCH64_ABS32(R, Value);
      case llvm::ELF::R_AARCH64_ABS64:
        return visitELF_AARCH64_ABS64(R, Value);
      default:
        HasError = true;
        return RelocToApply();
      }
    }
    HasError = true;
    return RelocToApply();
  }

  bool error() { return HasError; }

private:
  StringRef FileFormat;
  bool HasError;

  /// Operations

  /// 386-ELF
  RelocToApply visitELF_386_NONE(RelocationRef R) {
    return RelocToApply(0, 0);
  }

  // Ideally the Addend here will be the addend in the data for
  // the relocation. It's not actually the case for Rel relocations.
  RelocToApply visitELF_386_32(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    return RelocToApply(Value + Addend, 4);
  }

  RelocToApply visitELF_386_PC32(RelocationRef R, uint64_t Value,
                                 uint64_t SecAddr) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    uint64_t Address;
    R.getAddress(Address);
    return RelocToApply(Value + Addend - Address, 4);
  }

  /// X86-64 ELF
  RelocToApply visitELF_X86_64_NONE(RelocationRef R) {
    return RelocToApply(0, 0);
  }
  RelocToApply visitELF_X86_64_64(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    return RelocToApply(Value + Addend, 8);
  }
  RelocToApply visitELF_X86_64_PC32(RelocationRef R, uint64_t Value,
                                    uint64_t SecAddr) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    uint64_t Address;
    R.getAddress(Address);
    return RelocToApply(Value + Addend - Address, 4);
  }
  RelocToApply visitELF_X86_64_32(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    uint32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }
  RelocToApply visitELF_X86_64_32S(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    int32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }

  /// PPC64 ELF
  RelocToApply visitELF_PPC64_ADDR32(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    uint32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }

  /// MIPS ELF
  RelocToApply visitELF_MIPS_32(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    uint32_t Res = (Value + Addend) & 0xFFFFFFFF;
    return RelocToApply(Res, 4);
  }

  // AArch64 ELF
  RelocToApply visitELF_AARCH64_ABS32(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    int64_t Res =  Value + Addend;

    // Overflow check allows for both signed and unsigned interpretation.
    if (Res < INT32_MIN || Res > UINT32_MAX)
      HasError = true;

    return RelocToApply(static_cast<uint32_t>(Res), 4);
  }

  RelocToApply visitELF_AARCH64_ABS64(RelocationRef R, uint64_t Value) {
    int64_t Addend;
    R.getAdditionalInfo(Addend);
    return RelocToApply(Value + Addend, 8);
  }

};

}
}
#endif
