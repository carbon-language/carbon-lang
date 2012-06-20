//===--- ObjCRuntime.h - Objective-C Runtime Configuration ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines types useful for describing an Objective-C runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_OBJCRUNTIME_H
#define LLVM_CLANG_OBJCRUNTIME_H

#include "clang/Basic/VersionTuple.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {

/// The basic abstraction for the target ObjC runtime.
class ObjCRuntime {
public:
  /// The basic Objective-C runtimes that we know about.
  enum Kind {
    /// 'macosx' is the Apple-provided NeXT-derived runtime on Mac OS
    /// X platforms that use the non-fragile ABI; the version is a
    /// release of that OS.
    MacOSX,

    /// 'macosx-fragile' is the Apple-provided NeXT-derived runtime on
    /// Mac OS X platforms that use the fragile ABI; the version is a
    /// release of that OS.
    FragileMacOSX,

    /// 'ios' is the Apple-provided NeXT-derived runtime on iOS or the iOS
    /// simulator;  it is always non-fragile.  The version is a release
    /// version of iOS.
    iOS,

    /// 'gnu' is the non-fragile GNU runtime.
    GNU,

    /// 'gnu-fragile' is the fragile GNU runtime.
    FragileGNU
  };

private:
  Kind TheKind;
  VersionTuple Version;

public:
  /// A bogus initialization of the runtime.
  ObjCRuntime() : TheKind(MacOSX) {}

  ObjCRuntime(Kind kind, const VersionTuple &version)
    : TheKind(kind), Version(version) {}

  void set(Kind kind, VersionTuple version) {
    TheKind = kind;
    Version = version;
  }

  Kind getKind() const { return TheKind; }
  const VersionTuple &getVersion() const { return Version; }

  /// Does this runtime follow the set of implied behaviors for a
  /// "non-fragile" ABI?
  bool isNonFragile() const {
    switch (getKind()) {
    case FragileMacOSX: return false;
    case FragileGNU: return false;
    case MacOSX: return true;
    case GNU: return true;
    case iOS: return true;
    }
    llvm_unreachable("bad kind");
  }

  /// The inverse of isNonFragile():  does this runtiem follow the set of
  /// implied behaviors for a "fragile" ABI?
  bool isFragile() const { return !isNonFragile(); }

  /// Is this runtime basically of the GNU family of runtimes?
  bool isGNUFamily() const {
    switch (getKind()) {
    case FragileMacOSX:
    case MacOSX:
    case iOS:
      return false;
    case FragileGNU:
    case GNU:
      return true;
    }
    llvm_unreachable("bad kind");
  }

  /// Is this runtime basically of the NeXT family of runtimes?
  bool isNeXTFamily() const {
    // For now, this is just the inverse of isGNUFamily(), but that's
    // not inherently true.
    return !isGNUFamily();
  }

  /// Does this runtime natively provide the ARC entrypoints?  ARC
  /// cannot be directly supported on a platform that does not provide
  /// these entrypoints, although it may be supportable via a stub
  /// library.
  bool hasARC() const {
    switch (getKind()) {
    case FragileMacOSX: return false;
    case MacOSX: return getVersion() >= VersionTuple(10, 7);
    case iOS: return getVersion() >= VersionTuple(5);

    // This is really a lie, because some implementations and versions
    // of the runtime do not support ARC.  Probably -fgnu-runtime
    // should imply a "maximal" runtime or something?
    case FragileGNU: return true;
    case GNU: return true;
    }
    llvm_unreachable("bad kind");
  }

  /// Does this runtime natively provide ARC-compliant 'weak'
  /// entrypoints?
  bool hasWeak() const {
    // Right now, this is always equivalent to the ARC decision.
    return hasARC();
  }

  /// Does this runtime directly support the subscripting methods?
  /// This is really a property of the library, not the runtime.
  bool hasSubscripting() const {
    switch (getKind()) {
    case FragileMacOSX: return false;
    case MacOSX: return getVersion() >= VersionTuple(10, 8);
    case iOS: return false;

    // This is really a lie, because some implementations and versions
    // of the runtime do not support ARC.  Probably -fgnu-runtime
    // should imply a "maximal" runtime or something?
    case FragileGNU: return true;
    case GNU: return true;
    }
    llvm_unreachable("bad kind");
  }

  /// Does this runtime provide an objc_terminate function?  This is
  /// used in handlers for exceptions during the unwind process;
  /// without it, abort() must be used in pure ObjC files.
  bool hasTerminate() const {
    switch (getKind()) {
    case FragileMacOSX: return getVersion() >= VersionTuple(10, 8);
    case MacOSX: return getVersion() >= VersionTuple(10, 8);
    case iOS: return getVersion() >= VersionTuple(5);
    case FragileGNU: return false;
    case GNU: return false;
    }
    llvm_unreachable("bad kind");
  }

  /// Does this runtime support weakly importing classes?
  bool hasWeakClassImport() const {
    switch (getKind()) {
    case MacOSX: return true;
    case iOS: return true;
    case FragileMacOSX: return false;
    case FragileGNU: return false;
    case GNU: return false;
    }
    llvm_unreachable("bad kind");
  }

  /// Try to parse an Objective-C runtime specification from the given string.
  ///
  /// Return true on error.
  bool tryParse(StringRef input);

  std::string getAsString() const;

  friend bool operator==(const ObjCRuntime &left, const ObjCRuntime &right) {
    return left.getKind() == right.getKind() &&
           left.getVersion() == right.getVersion();
  }

  friend bool operator!=(const ObjCRuntime &left, const ObjCRuntime &right) {
    return !(left == right);
  }
};

raw_ostream &operator<<(raw_ostream &out, const ObjCRuntime &value);

}  // end namespace clang

#endif
