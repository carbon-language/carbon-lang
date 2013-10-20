//===-- ubsan_diag.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Diagnostics emission for Clang's undefined behavior sanitizer.
//
//===----------------------------------------------------------------------===//
#ifndef UBSAN_DIAG_H
#define UBSAN_DIAG_H

#include "ubsan_value.h"

namespace __ubsan {

/// \brief A location within a loaded module in the program. These are used when
/// the location can't be resolved to a SourceLocation.
class ModuleLocation {
  const char *ModuleName;
  uptr Offset;

public:
  ModuleLocation() : ModuleName(0), Offset(0) {}
  ModuleLocation(const char *ModuleName, uptr Offset)
    : ModuleName(ModuleName), Offset(Offset) {}
  const char *getModuleName() const { return ModuleName; }
  uptr getOffset() const { return Offset; }
};

/// A location of some data within the program's address space.
typedef uptr MemoryLocation;

/// \brief Location at which a diagnostic can be emitted. Either a
/// SourceLocation, a ModuleLocation, or a MemoryLocation.
class Location {
public:
  enum LocationKind { LK_Null, LK_Source, LK_Module, LK_Memory };

private:
  LocationKind Kind;
  // FIXME: In C++11, wrap these in an anonymous union.
  SourceLocation SourceLoc;
  ModuleLocation ModuleLoc;
  MemoryLocation MemoryLoc;

public:
  Location() : Kind(LK_Null) {}
  Location(SourceLocation Loc) :
    Kind(LK_Source), SourceLoc(Loc) {}
  Location(ModuleLocation Loc) :
    Kind(LK_Module), ModuleLoc(Loc) {}
  Location(MemoryLocation Loc) :
    Kind(LK_Memory), MemoryLoc(Loc) {}

  LocationKind getKind() const { return Kind; }

  bool isSourceLocation() const { return Kind == LK_Source; }
  bool isModuleLocation() const { return Kind == LK_Module; }
  bool isMemoryLocation() const { return Kind == LK_Memory; }

  SourceLocation getSourceLocation() const {
    CHECK(isSourceLocation());
    return SourceLoc;
  }
  ModuleLocation getModuleLocation() const {
    CHECK(isModuleLocation());
    return ModuleLoc;
  }
  MemoryLocation getMemoryLocation() const {
    CHECK(isMemoryLocation());
    return MemoryLoc;
  }
};

/// Try to obtain a location for the caller. This might fail, and produce either
/// an invalid location or a module location for the caller.
Location getCallerLocation(uptr CallerLoc = GET_CALLER_PC());

/// Try to obtain a location for the given function pointer. This might fail,
/// and produce either an invalid location or a module location for the caller.
/// If FName is non-null and the name of the function is known, set *FName to
/// the function name, otherwise *FName is unchanged.
Location getFunctionLocation(uptr Loc, const char **FName);

/// A diagnostic severity level.
enum DiagLevel {
  DL_Error, ///< An error.
  DL_Note   ///< A note, attached to a prior diagnostic.
};

/// \brief Annotation for a range of locations in a diagnostic.
class Range {
  Location Start, End;
  const char *Text;

public:
  Range() : Start(), End(), Text() {}
  Range(MemoryLocation Start, MemoryLocation End, const char *Text)
    : Start(Start), End(End), Text(Text) {}
  Location getStart() const { return Start; }
  Location getEnd() const { return End; }
  const char *getText() const { return Text; }
};

/// \brief A mangled C++ name. Really just a strong typedef for 'const char*'.
class MangledName {
  const char *Name;
public:
  MangledName(const char *Name) : Name(Name) {}
  const char *getName() const { return Name; }
};

/// \brief Representation of an in-flight diagnostic.
///
/// Temporary \c Diag instances are created by the handler routines to
/// accumulate arguments for a diagnostic. The destructor emits the diagnostic
/// message.
class Diag {
  /// The location at which the problem occurred.
  Location Loc;

  /// The diagnostic level.
  DiagLevel Level;

  /// The message which will be emitted, with %0, %1, ... placeholders for
  /// arguments.
  const char *Message;

public:
  /// Kinds of arguments, corresponding to members of \c Arg's union.
  enum ArgKind {
    AK_String, ///< A string argument, displayed as-is.
    AK_Mangled,///< A C++ mangled name, demangled before display.
    AK_UInt,   ///< An unsigned integer argument.
    AK_SInt,   ///< A signed integer argument.
    AK_Float,  ///< A floating-point argument.
    AK_Pointer ///< A pointer argument, displayed in hexadecimal.
  };

  /// An individual diagnostic message argument.
  struct Arg {
    Arg() {}
    Arg(const char *String) : Kind(AK_String), String(String) {}
    Arg(MangledName MN) : Kind(AK_Mangled), String(MN.getName()) {}
    Arg(UIntMax UInt) : Kind(AK_UInt), UInt(UInt) {}
    Arg(SIntMax SInt) : Kind(AK_SInt), SInt(SInt) {}
    Arg(FloatMax Float) : Kind(AK_Float), Float(Float) {}
    Arg(const void *Pointer) : Kind(AK_Pointer), Pointer(Pointer) {}

    ArgKind Kind;
    union {
      const char *String;
      UIntMax UInt;
      SIntMax SInt;
      FloatMax Float;
      const void *Pointer;
    };
  };

private:
  static const unsigned MaxArgs = 5;
  static const unsigned MaxRanges = 1;

  /// The arguments which have been added to this diagnostic so far.
  Arg Args[MaxArgs];
  unsigned NumArgs;

  /// The ranges which have been added to this diagnostic so far.
  Range Ranges[MaxRanges];
  unsigned NumRanges;

  Diag &AddArg(Arg A) {
    CHECK(NumArgs != MaxArgs);
    Args[NumArgs++] = A;
    return *this;
  }

  Diag &AddRange(Range A) {
    CHECK(NumRanges != MaxRanges);
    Ranges[NumRanges++] = A;
    return *this;
  }

  /// \c Diag objects are not copyable.
  Diag(const Diag &); // NOT IMPLEMENTED
  Diag &operator=(const Diag &);

public:
  Diag(Location Loc, DiagLevel Level, const char *Message)
    : Loc(Loc), Level(Level), Message(Message), NumArgs(0), NumRanges(0) {}
  ~Diag();

  Diag &operator<<(const char *Str) { return AddArg(Str); }
  Diag &operator<<(MangledName MN) { return AddArg(MN); }
  Diag &operator<<(unsigned long long V) { return AddArg(UIntMax(V)); }
  Diag &operator<<(const void *V) { return AddArg(V); }
  Diag &operator<<(const TypeDescriptor &V);
  Diag &operator<<(const Value &V);
  Diag &operator<<(const Range &R) { return AddRange(R); }
};

} // namespace __ubsan

#endif // UBSAN_DIAG_H
