//===---- IRReader.cpp - Reader for LLVM IR files -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IRReader/IRReader.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/system_error.h"

using namespace llvm;

Module *llvm::getLazyIRModule(MemoryBuffer *Buffer, SMDiagnostic &Err,
                              LLVMContext &Context) {
  if (isBitcode((const unsigned char *)Buffer->getBufferStart(),
                (const unsigned char *)Buffer->getBufferEnd())) {
    std::string ErrMsg;
    Module *M = getLazyBitcodeModule(Buffer, Context, &ErrMsg);
    if (M == 0) {
      Err = SMDiagnostic(Buffer->getBufferIdentifier(), SourceMgr::DK_Error,
                         ErrMsg);
      // ParseBitcodeFile does not take ownership of the Buffer in the
      // case of an error.
      delete Buffer;
    }
    return M;
  }

  return ParseAssembly(Buffer, 0, Err, Context);
}

Module *llvm::getLazyIRFileModule(const std::string &Filename, SMDiagnostic &Err,
                                  LLVMContext &Context) {
  OwningPtr<MemoryBuffer> File;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), File)) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + ec.message());
    return 0;
  }

  return getLazyIRModule(File.take(), Err, Context);
}

Module *llvm::ParseIR(MemoryBuffer *Buffer, SMDiagnostic &Err,
                      LLVMContext &Context) {
  if (isBitcode((const unsigned char *)Buffer->getBufferStart(),
                (const unsigned char *)Buffer->getBufferEnd())) {
    std::string ErrMsg;
    Module *M = ParseBitcodeFile(Buffer, Context, &ErrMsg);
    if (M == 0)
      Err = SMDiagnostic(Buffer->getBufferIdentifier(), SourceMgr::DK_Error,
                         ErrMsg);
    // ParseBitcodeFile does not take ownership of the Buffer.
    delete Buffer;
    return M;
  }

  return ParseAssembly(Buffer, 0, Err, Context);
}

Module *llvm::ParseIRFile(const std::string &Filename, SMDiagnostic &Err,
                          LLVMContext &Context) {
  OwningPtr<MemoryBuffer> File;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename.c_str(), File)) {
    Err = SMDiagnostic(Filename, SourceMgr::DK_Error,
                       "Could not open input file: " + ec.message());
    return 0;
  }

  return ParseIR(File.take(), Err, Context);
}
