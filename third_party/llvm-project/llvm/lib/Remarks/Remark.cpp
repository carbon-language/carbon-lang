//===- Remark.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Remark type and the C API.
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/Remark.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::remarks;

std::string Remark::getArgsAsMsg() const {
  std::string Str;
  raw_string_ostream OS(Str);
  for (const Argument &Arg : Args)
    OS << Arg.Val;
  return OS.str();
}

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(StringRef, LLVMRemarkStringRef)

extern "C" const char *LLVMRemarkStringGetData(LLVMRemarkStringRef String) {
  return unwrap(String)->data();
}

extern "C" uint32_t LLVMRemarkStringGetLen(LLVMRemarkStringRef String) {
  return unwrap(String)->size();
}

extern "C" LLVMRemarkStringRef
LLVMRemarkDebugLocGetSourceFilePath(LLVMRemarkDebugLocRef DL) {
  return wrap(&unwrap(DL)->SourceFilePath);
}

extern "C" uint32_t LLVMRemarkDebugLocGetSourceLine(LLVMRemarkDebugLocRef DL) {
  return unwrap(DL)->SourceLine;
}

extern "C" uint32_t
LLVMRemarkDebugLocGetSourceColumn(LLVMRemarkDebugLocRef DL) {
  return unwrap(DL)->SourceColumn;
}

extern "C" LLVMRemarkStringRef LLVMRemarkArgGetKey(LLVMRemarkArgRef Arg) {
  return wrap(&unwrap(Arg)->Key);
}

extern "C" LLVMRemarkStringRef LLVMRemarkArgGetValue(LLVMRemarkArgRef Arg) {
  return wrap(&unwrap(Arg)->Val);
}

extern "C" LLVMRemarkDebugLocRef
LLVMRemarkArgGetDebugLoc(LLVMRemarkArgRef Arg) {
  if (const Optional<RemarkLocation> &Loc = unwrap(Arg)->Loc)
    return wrap(&*Loc);
  return nullptr;
}

extern "C" void LLVMRemarkEntryDispose(LLVMRemarkEntryRef Remark) {
  delete unwrap(Remark);
}

extern "C" LLVMRemarkType LLVMRemarkEntryGetType(LLVMRemarkEntryRef Remark) {
  // Assume here that the enums can be converted both ways.
  return static_cast<LLVMRemarkType>(unwrap(Remark)->RemarkType);
}

extern "C" LLVMRemarkStringRef
LLVMRemarkEntryGetPassName(LLVMRemarkEntryRef Remark) {
  return wrap(&unwrap(Remark)->PassName);
}

extern "C" LLVMRemarkStringRef
LLVMRemarkEntryGetRemarkName(LLVMRemarkEntryRef Remark) {
  return wrap(&unwrap(Remark)->RemarkName);
}

extern "C" LLVMRemarkStringRef
LLVMRemarkEntryGetFunctionName(LLVMRemarkEntryRef Remark) {
  return wrap(&unwrap(Remark)->FunctionName);
}

extern "C" LLVMRemarkDebugLocRef
LLVMRemarkEntryGetDebugLoc(LLVMRemarkEntryRef Remark) {
  if (const Optional<RemarkLocation> &Loc = unwrap(Remark)->Loc)
    return wrap(&*Loc);
  return nullptr;
}

extern "C" uint64_t LLVMRemarkEntryGetHotness(LLVMRemarkEntryRef Remark) {
  if (const Optional<uint64_t> &Hotness = unwrap(Remark)->Hotness)
    return *Hotness;
  return 0;
}

extern "C" uint32_t LLVMRemarkEntryGetNumArgs(LLVMRemarkEntryRef Remark) {
  return unwrap(Remark)->Args.size();
}

extern "C" LLVMRemarkArgRef
LLVMRemarkEntryGetFirstArg(LLVMRemarkEntryRef Remark) {
  ArrayRef<Argument> Args = unwrap(Remark)->Args;
  // No arguments to iterate on.
  if (Args.empty())
    return NULL;
  return reinterpret_cast<LLVMRemarkArgRef>(
      const_cast<Argument *>(Args.begin()));
}

extern "C" LLVMRemarkArgRef
LLVMRemarkEntryGetNextArg(LLVMRemarkArgRef ArgIt, LLVMRemarkEntryRef Remark) {
  // No more arguments to iterate on.
  if (ArgIt == NULL)
    return NULL;

  auto It = (ArrayRef<Argument>::const_iterator)ArgIt;
  auto Next = std::next(It);
  if (Next == unwrap(Remark)->Args.end())
    return NULL;

  return reinterpret_cast<LLVMRemarkArgRef>(const_cast<Argument *>(Next));
}
