//===-- llvm/Bytecode/Reader.h - Reader for VM bytecode files ----*- C++ -*--=//
//
// This functionality is implemented by the lib/Bytecode/Reader library.
// This library is used to read VM bytecode files from an iostream.
//
// Note that performance of this library is _crucial_ for performance of the
// JIT type applications, so we have designed the bytecode format to support
// quick reading.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_READER_H
#define LLVM_BYTECODE_READER_H

#include "llvm/ModuleProvider.h"
#include <string>
#include <vector>

/// getBytecodeModuleProvider - lazy function-at-a-time loading from a file
///
AbstractModuleProvider*
getBytecodeModuleProvider(const std::string &Filename);

/// getBytecodeBufferModuleProvider - lazy function-at-a-time loading from a
/// buffer
///
AbstractModuleProvider* 
getBytecodeBufferModuleProvider(const unsigned char *Buffer,
                                unsigned BufferSize,
                                const std::string &ModuleID = "");

/// ParseBytecodeFile - Parse the given bytecode file
///
Module* ParseBytecodeFile(const std::string &Filename,
                          std::string *ErrorStr = 0);

/// ParseBytecodeBuffer - Parse a given bytecode buffer
///
Module* ParseBytecodeBuffer(const unsigned char *Buffer,
                            unsigned BufferSize,
                            const std::string &ModuleID = "",
                            std::string *ErrorStr = 0);

/// ReadArchiveFile - Read bytecode files from the specfied .a file, returning
/// true on error, or false on success.
///
bool ReadArchiveFile(const std::string &Filename,
                     std::vector<Module*> &Objects,
                     std::string *ErrorStr = 0);

#endif
