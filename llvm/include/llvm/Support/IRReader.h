//===---- llvm/Support/IRReader.h - Reader for LLVM IR files ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines functions for reading LLVM IR. They support both
// Bitcode and Assembly, automatically detecting the input format.
//
// These functions must be defined in a header file in order to avoid
// library dependencies, since they reference both Bitcode and Assembly
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_IRREADER_H
#define LLVM_SUPPORT_IRREADER_H

#include "llvm/Assembly/Parser.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ModuleProvider.h"

namespace llvm {

  /// If the given MemoryBuffer holds a bitcode image, return a ModuleProvider
  /// for it which does lazy deserialization of function bodies.  Otherwise,
  /// attempt to parse it as LLVM Assembly and return a fully populated
  /// ModuleProvider. This function *always* takes ownership of the given
  /// MemoryBuffer.
  inline ModuleProvider *getIRModuleProvider(MemoryBuffer *Buffer,
                                             const std::string &Filename,
                                             SMDiagnostic &Err,
                                             LLVMContext &Context) {
    if (isBitcode((const unsigned char *)Buffer->getBufferStart(),
                  (const unsigned char *)Buffer->getBufferEnd())) {
      std::string ErrMsg;
      ModuleProvider *MP = getBitcodeModuleProvider(Buffer, Context, &ErrMsg);
      if (MP == 0) {
        Err = SMDiagnostic(Filename, -1, -1, ErrMsg, "");
        // ParseBitcodeFile does not take ownership of the Buffer in the
        // case of an error.
        delete Buffer;
      }
      return MP;
    }

    Module *M = ParseAssembly(Buffer, Filename, 0, Err, Context);
    if (M == 0)
      return 0;
    return new ExistingModuleProvider(M);
  }

  /// If the given file holds a bitcode image, return a ModuleProvider
  /// for it which does lazy deserialization of function bodies.  Otherwise,
  /// attempt to parse it as LLVM Assembly and return a fully populated
  /// ModuleProvider.
  inline ModuleProvider *getIRFileModuleProvider(const std::string &Filename,
                                                 SMDiagnostic &Err,
                                                 LLVMContext &Context) {
    std::string ErrMsg;
    MemoryBuffer *F = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), &ErrMsg);
    if (F == 0) {
      Err = SMDiagnostic(Filename, -1, -1,
                         "Could not open input file '" + Filename + "'", "");
      return 0;
    }

    return getIRModuleProvider(F, Filename, Err, Context);
  }

  /// If the given MemoryBuffer holds a bitcode image, return a Module
  /// for it.  Otherwise, attempt to parse it as LLVM Assembly and return
  /// a Module for it. This function *always* takes ownership of the given
  /// MemoryBuffer.
  inline Module *ParseIR(MemoryBuffer *Buffer,
                         const std::string &Filename,
                         SMDiagnostic &Err,
                         LLVMContext &Context) {
    if (isBitcode((const unsigned char *)Buffer->getBufferStart(),
                  (const unsigned char *)Buffer->getBufferEnd())) {
      std::string ErrMsg;
      Module *M = ParseBitcodeFile(Buffer, Context, &ErrMsg);
      // ParseBitcodeFile does not take ownership of the Buffer.
      delete Buffer;
      if (M == 0)
        Err = SMDiagnostic(Filename, -1, -1, ErrMsg, "");
      return M;
    }

    return ParseAssembly(Buffer, Filename, 0, Err, Context);
  }

  /// If the given file holds a bitcode image, return a Module for it.
  /// Otherwise, attempt to parse it as LLVM Assembly and return a Module
  /// for it.
  inline Module *ParseIRFile(const std::string &Filename,
                             SMDiagnostic &Err,
                             LLVMContext &Context) {
    std::string ErrMsg;
    MemoryBuffer *F = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), &ErrMsg);
    if (F == 0) {
      Err = SMDiagnostic(Filename, -1, -1,
                         "Could not open input file '" + Filename + "'", "");
      return 0;
    }

    return ParseIR(F, Filename, Err, Context);
  }

}

#endif
