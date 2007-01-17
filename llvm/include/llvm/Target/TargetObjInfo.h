//===-- llvm/Target/TargetObjInfo.h - Object File Info ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to be used as the basis for target specific object
// writers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_OBJ_INFO_H
#define LLVM_TARGET_OBJ_INFO_H

#include <string>
#include <vector>

namespace llvm {

  struct TargetObjInfo {
    TargetObjInfo() {}
    virtual ~TargetObjInfo() {}

    typedef std::vector<unsigned char> DataBuffer;

    virtual void align(DataBuffer &Output, unsigned Boundary) const = 0;

    //===------------------------------------------------------------------===//
    // Out Functions - Output the specified value to the data buffer.

    virtual void outbyte(DataBuffer &Output, unsigned char X) const = 0;
    virtual void outhalf(DataBuffer &Output, unsigned short X) const = 0;
    virtual void outword(DataBuffer &Output, unsigned X) const = 0;
    virtual void outxword(DataBuffer &Output, uint64_t X) const = 0;
    virtual void outaddr32(DataBuffer &Output, unsigned X) const = 0;
    virtual void outaddr64(DataBuffer &Output, uint64_t X) const = 0;
    virtual void outaddr(DataBuffer &Output, uint64_t X) const = 0;
    virtual void outstring(DataBuffer &Output, std::string &S,
                           unsigned Length) const = 0;

    //===------------------------------------------------------------------===//
    // Fix Functions - Replace an existing entry at an offset.

    virtual void fixhalf(DataBuffer &Output, unsigned short X,
                         unsigned Offset) const = 0;
    virtual void fixword(DataBuffer &Output, unsigned X,
                         unsigned Offset) const = 0;
    virtual void fixaddr(DataBuffer &Output, uint64_t X,
                         unsigned Offset) const = 0;
  };

} // end llvm namespace

#endif // LLVM_TARGET_OBJ_INFO_H
