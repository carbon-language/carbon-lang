//===- BuildLibCalls.h - Utility builder for libcalls -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to build some C language libcalls for
// optimization passes that need to call the various functions.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_UTILS_BUILDLIBCALLS_H
#define TRANSFORMS_UTILS_BUILDLIBCALLS_H

#include "llvm/Support/IRBuilder.h"

namespace llvm {
  class Value;
  class TargetData;
  
  /// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
  Value *CastToCStr(Value *V, IRBuilder<> &B);

  /// EmitStrLen - Emit a call to the strlen function to the builder, for the
  /// specified pointer.  Ptr is required to be some pointer type, and the
  /// return value has 'intptr_t' type.
  Value *EmitStrLen(Value *Ptr, IRBuilder<> &B, const TargetData *TD);

  /// EmitStrChr - Emit a call to the strchr function to the builder, for the
  /// specified pointer and character.  Ptr is required to be some pointer type,
  /// and the return value has 'i8*' type.
  Value *EmitStrChr(Value *Ptr, char C, IRBuilder<> &B, const TargetData *TD);

  /// EmitStrNCmp - Emit a call to the strncmp function to the builder.
  Value *EmitStrNCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                     const TargetData *TD);

  /// EmitStrCpy - Emit a call to the strcpy function to the builder, for the
  /// specified pointer arguments.
  Value *EmitStrCpy(Value *Dst, Value *Src, IRBuilder<> &B,
                    const TargetData *TD, StringRef Name = "strcpy");

  /// EmitStrNCpy - Emit a call to the strncpy function to the builder, for the
  /// specified pointer arguments and length.
  Value *EmitStrNCpy(Value *Dst, Value *Src, Value *Len, IRBuilder<> &B,
                    const TargetData *TD, StringRef Name = "strncpy");

  /// EmitMemCpyChk - Emit a call to the __memcpy_chk function to the builder.
  /// This expects that the Len and ObjSize have type 'intptr_t' and Dst/Src
  /// are pointers.
  Value *EmitMemCpyChk(Value *Dst, Value *Src, Value *Len, Value *ObjSize,
                       IRBuilder<> &B, const TargetData *TD);

  /// EmitMemChr - Emit a call to the memchr function.  This assumes that Ptr is
  /// a pointer, Val is an i32 value, and Len is an 'intptr_t' value.
  Value *EmitMemChr(Value *Ptr, Value *Val, Value *Len, IRBuilder<> &B,
                    const TargetData *TD);

  /// EmitMemCmp - Emit a call to the memcmp function.
  Value *EmitMemCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                    const TargetData *TD);

  /// EmitUnaryFloatFnCall - Emit a call to the unary function named 'Name'
  /// (e.g.  'floor').  This function is known to take a single of type matching
  /// 'Op' and returns one value with the same type.  If 'Op' is a long double,
  /// 'l' is added as the suffix of name, if 'Op' is a float, we add a 'f'
  /// suffix.
  Value *EmitUnaryFloatFnCall(Value *Op, StringRef Name, IRBuilder<> &B,
                              const AttrListPtr &Attrs);

  /// EmitPutChar - Emit a call to the putchar function.  This assumes that Char
  /// is an integer.
  Value *EmitPutChar(Value *Char, IRBuilder<> &B, const TargetData *TD);

  /// EmitPutS - Emit a call to the puts function.  This assumes that Str is
  /// some pointer.
  void EmitPutS(Value *Str, IRBuilder<> &B, const TargetData *TD);

  /// EmitFPutC - Emit a call to the fputc function.  This assumes that Char is
  /// an i32, and File is a pointer to FILE.
  void EmitFPutC(Value *Char, Value *File, IRBuilder<> &B,
                 const TargetData *TD);

  /// EmitFPutS - Emit a call to the puts function.  Str is required to be a
  /// pointer and File is a pointer to FILE.
  void EmitFPutS(Value *Str, Value *File, IRBuilder<> &B, const TargetData *TD);

  /// EmitFWrite - Emit a call to the fwrite function.  This assumes that Ptr is
  /// a pointer, Size is an 'intptr_t', and File is a pointer to FILE.
  void EmitFWrite(Value *Ptr, Value *Size, Value *File, IRBuilder<> &B,
                  const TargetData *TD);

  /// SimplifyFortifiedLibCalls - Helper class for folding checked library
  /// calls (e.g. __strcpy_chk) into their unchecked counterparts.
  class SimplifyFortifiedLibCalls {
  protected:
    CallInst *CI;
    virtual void replaceCall(Value *With) = 0;
    virtual bool isFoldable(unsigned SizeCIOp, unsigned SizeArgOp,
                            bool isString) const = 0;
  public:
    virtual ~SimplifyFortifiedLibCalls();
    bool fold(CallInst *CI, const TargetData *TD);
  };
}

#endif
