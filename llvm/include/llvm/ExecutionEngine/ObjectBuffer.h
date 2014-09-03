//===---- ObjectBuffer.h - Utility class to wrap object image memory -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a wrapper class to hold the memory into which an
// object will be generated.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_OBJECTBUFFER_H
#define LLVM_EXECUTIONENGINE_OBJECTBUFFER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// This class acts as a container for the memory buffer used during generation
/// and loading of executable objects using MCJIT and RuntimeDyld. The
/// underlying memory for the object will be owned by the ObjectBuffer instance
/// throughout its lifetime.
class ObjectBuffer {
  virtual void anchor();
public:
  ObjectBuffer() {}
  ObjectBuffer(std::unique_ptr<MemoryBuffer> Buf) : Buffer(std::move(Buf)) {}
  virtual ~ObjectBuffer() {}

  MemoryBufferRef getMemBuffer() const { return Buffer->getMemBufferRef(); }

  const char *getBufferStart() const { return Buffer->getBufferStart(); }
  size_t getBufferSize() const { return Buffer->getBufferSize(); }
  StringRef getBuffer() const { return Buffer->getBuffer(); }
  StringRef getBufferIdentifier() const {
    return Buffer->getBufferIdentifier();
  }

protected:
  // The memory contained in an ObjectBuffer
  std::unique_ptr<MemoryBuffer> Buffer;
};

/// This class encapsulates the SmallVector and raw_svector_ostream needed to
/// generate an object using MC code emission while providing a common
/// ObjectBuffer interface for access to the memory once the object has been
/// generated.
class ObjectBufferStream : public ObjectBuffer {
  void anchor() override;
public:
  ObjectBufferStream() : OS(SV) {}
  virtual ~ObjectBufferStream() {}

  raw_ostream &getOStream() { return OS; }
  void flush()
  {
    OS.flush();

    // Make the data accessible via the ObjectBuffer::Buffer
    Buffer =
        MemoryBuffer::getMemBuffer(StringRef(SV.data(), SV.size()), "", false);
  }

protected:
  SmallVector<char, 4096> SV; // Working buffer into which we JIT.
  raw_svector_ostream     OS; // streaming wrapper
};

} // namespace llvm

#endif
