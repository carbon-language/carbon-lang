//===-- BitWriter.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/BitWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;


/*===-- Operations on modules ---------------------------------------------===*/

int LLVMWriteBitcodeToFile(LLVMModuleRef M, const char *Path) {
  std::string ErrorInfo;
  raw_fd_ostream OS(Path, ErrorInfo,
                    raw_fd_ostream::F_Force|raw_fd_ostream::F_Binary);
  
  if (!ErrorInfo.empty())
    return -1;
  
  WriteBitcodeToFile(unwrap(M), OS);
  return 0;
}

#if defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR >= 4)
#include <ext/stdio_filebuf.h>

int LLVMWriteBitcodeToFileHandle(LLVMModuleRef M, int FileHandle) {
  raw_fd_ostream OS(FileHandle, false);
  
  WriteBitcodeToFile(unwrap(M), OS);
  return 0;
}

#else

int LLVMWriteBitcodeToFileHandle(LLVMModuleRef M, int FileHandle) {
  return -1; // Not supported.
}

#endif
