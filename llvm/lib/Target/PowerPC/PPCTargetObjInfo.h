//===-- PPCTargetObjInfo.h - Object File Info --------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target object file properties for PowerPC.
//
//===----------------------------------------------------------------------===//

#ifndef PPCTARGETOBJINFO_H
#define PPCTARGETOBJINFO_H

#include "llvm/Target/TargetObjInfo.h"

namespace llvm {

  class TargetMachine;

  struct MachOTargetObjInfo : public TargetObjInfo {
    MachOTargetObjInfo(const TargetMachine &PPC_TM);

    // align - Emit padding into the file until the current output position is
    // aligned to the specified power of two boundary.
    virtual void align(DataBuffer &Output, unsigned Boundary) const {
      assert(Boundary && (Boundary & (Boundary-1)) == 0 &&
             "Must align to 2^k boundary");
      size_t Size = Output.size();

      if (Size & (Boundary-1)) {
        // Add padding to get alignment to the correct place.
        size_t Pad = Boundary - (Size & (Boundary - 1));
        Output.resize(Size + Pad);
      }
    }

    //===------------------------------------------------------------------===//
    // Out Functions - Output the specified value to the data buffer.

    virtual void outbyte(DataBuffer &Output, unsigned char X) const {
      Output.push_back(X);
    }
    virtual void outhalf(DataBuffer &Output, unsigned short X) const {
      if (isLittleEndian) {
        Output.push_back(X & 255);
        Output.push_back(X >> 8);
      } else {
        Output.push_back(X >> 8);
        Output.push_back(X & 255);
      }
    }
    virtual void outword(DataBuffer &Output, unsigned X) const {
      if (isLittleEndian) {
        Output.push_back((X >>  0) & 255);
        Output.push_back((X >>  8) & 255);
        Output.push_back((X >> 16) & 255);
        Output.push_back((X >> 24) & 255);
      } else {
        Output.push_back((X >> 24) & 255);
        Output.push_back((X >> 16) & 255);
        Output.push_back((X >>  8) & 255);
        Output.push_back((X >>  0) & 255);
      }
    }
    virtual void outxword(DataBuffer &Output, uint64_t X) const {
      if (isLittleEndian) {
        Output.push_back(unsigned(X >>  0) & 255);
        Output.push_back(unsigned(X >>  8) & 255);
        Output.push_back(unsigned(X >> 16) & 255);
        Output.push_back(unsigned(X >> 24) & 255);
        Output.push_back(unsigned(X >> 32) & 255);
        Output.push_back(unsigned(X >> 40) & 255);
        Output.push_back(unsigned(X >> 48) & 255);
        Output.push_back(unsigned(X >> 56) & 255);
      } else {
        Output.push_back(unsigned(X >> 56) & 255);
        Output.push_back(unsigned(X >> 48) & 255);
        Output.push_back(unsigned(X >> 40) & 255);
        Output.push_back(unsigned(X >> 32) & 255);
        Output.push_back(unsigned(X >> 24) & 255);
        Output.push_back(unsigned(X >> 16) & 255);
        Output.push_back(unsigned(X >>  8) & 255);
        Output.push_back(unsigned(X >>  0) & 255);
      }
    }
    virtual void outaddr32(DataBuffer &Output, unsigned X) const {
      outword(Output, X);
    }
    virtual void outaddr64(DataBuffer &Output, uint64_t X) const {
      outxword(Output, X);
    }
    virtual void outaddr(DataBuffer &Output, uint64_t X) const {
      if (!is64Bit)
        outword(Output, (unsigned)X);
      else
        outxword(Output, X);
    }
    virtual void outstring(DataBuffer &Output, std::string &S,
                           unsigned Length) const {
      unsigned len_to_copy = S.length() < Length ? S.length() : Length;
      unsigned len_to_fill = S.length() < Length ? Length-S.length() : 0;

      for (unsigned i = 0; i < len_to_copy; ++i)
        outbyte(Output, S[i]);

      for (unsigned i = 0; i < len_to_fill; ++i)
        outbyte(Output, 0);
    }

    //===------------------------------------------------------------------===//
    // Fix Functions - Replace an existing entry at an offset.

    virtual void fixhalf(DataBuffer &Output, unsigned short X,
                         unsigned Offset) const {
      unsigned char *P = &Output[Offset];
      P[0] = (X >> (isLittleEndian ?  0 : 8)) & 255;
      P[1] = (X >> (isLittleEndian ?  8 : 0)) & 255;
    }
    virtual void fixword(DataBuffer &Output, unsigned X,
                         unsigned Offset) const {
      unsigned char *P = &Output[Offset];
      P[0] = (X >> (isLittleEndian ?  0 : 24)) & 255;
      P[1] = (X >> (isLittleEndian ?  8 : 16)) & 255;
      P[2] = (X >> (isLittleEndian ? 16 :  8)) & 255;
      P[3] = (X >> (isLittleEndian ? 24 :  0)) & 255;
    }
    virtual void fixaddr(DataBuffer &Output, uint64_t X,
                         unsigned Offset) const {
      // Not implemented
    }
  private:
    /// Target machine description.
    const TargetMachine &TM;

    /// is64Bit/isLittleEndian - This information is inferred from the target
    /// machine directly, indicating what header values and flags to set.
    bool is64Bit, isLittleEndian;
  };

} // end llvm namespace

#endif // PPCTARGETOBJINFO_H
