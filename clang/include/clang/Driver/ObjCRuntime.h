//===--- ObjCRuntime.h - Objective C runtime features -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_DRIVER_OBJCRUNTIME_H_
#define CLANG_DRIVER_OBJCRUNTIME_H_

namespace clang {
namespace driver {

class ObjCRuntime {
public:
  enum Kind { GNU, NeXT };
private:
  unsigned RuntimeKind : 1;
public:
  void setKind(Kind k) { RuntimeKind = k; }
  Kind getKind() const { return static_cast<Kind>(RuntimeKind); }

  /// True if the runtime provides native ARC entrypoints.  ARC may
  /// still be usable without this if the tool-chain provides a
  /// statically-linked runtime support library.
  unsigned HasARC : 1;

  /// True if the runtime supports ARC zeroing __weak.
  unsigned HasWeak : 1;

  ObjCRuntime() : RuntimeKind(NeXT), HasARC(false), HasWeak(false) {}
};

}
}

#endif
