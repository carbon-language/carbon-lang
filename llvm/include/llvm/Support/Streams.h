//===- llvm/Support/Streams.h - Wrappers for iostreams ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bill Wendling and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a wrapper for the std::cout and std::cerr I/O streams.
// It prevents the need to include <iostream> to each file just to get I/O.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_STREAMS_H
#define LLVM_SUPPORT_STREAMS_H

#include <ostream>              // Doesn't have static d'tors!!

namespace llvm {

  /// llvm_ostream - Acts like an ostream. It's a wrapper for the std::cerr and
  /// std::cout ostreams. However, it doesn't require #including <iostream> in
  /// every file, which increases static c'tors & d'tors in the object code.
  /// 
  class llvm_ostream {
    std::ostream* Stream;
  public:
    llvm_ostream() : Stream(0) {}
    llvm_ostream(std::ostream &OStream) : Stream(&OStream) {}

    template <typename Ty>
    llvm_ostream &operator << (const Ty &Thing) {
      if (Stream) *Stream << Thing;
      return *this;
    }

    bool operator == (const std::ostream &OS) { return &OS == Stream; }
  };

  extern llvm_ostream llvm_null;
  extern llvm_ostream llvm_cout;
  extern llvm_ostream llvm_cerr;

} // End llvm namespace

#endif
