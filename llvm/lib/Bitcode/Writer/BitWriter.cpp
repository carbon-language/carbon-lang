//===-- BitWriter.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/BitWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include <fstream>

using namespace llvm;


/*===-- Operations on modules ---------------------------------------------===*/

int LLVMWriteBitcodeToFile(LLVMModuleRef M, const char *Path) {
  std::ofstream OS(Path);
  
  if (!OS.fail())
    WriteBitcodeToFile(unwrap(M), OS);
  
  if (OS.fail())
    return -1;
  
  return 0;
}

#ifdef __GNUC__
#include <ext/stdio_filebuf.h>

// FIXME: Control this with configure? Provide some portable abstraction in
// libSystem? As is, the user will just get a linker error if they use this on 
// non-GCC. Some C++ stdlibs even have ofstream::ofstream(int fd).
int LLVMWriteBitcodeToFileHandle(LLVMModuleRef M, int FileHandle) {
  __gnu_cxx::stdio_filebuf<char> Buffer(FileHandle, std::ios_base::out);
  std::ostream OS(&Buffer);
  
  if (!OS.fail())
    WriteBitcodeToFile(unwrap(M), OS);
  
  if (OS.fail())
    return -1;
  
  return 0;
}

#endif
