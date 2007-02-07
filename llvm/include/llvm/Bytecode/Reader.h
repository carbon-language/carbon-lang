//===-- llvm/Bytecode/Reader.h - Reader for VM bytecode files ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
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
#include "llvm/Module.h"
#include "llvm/Support/Compressor.h"
#include "llvm/System/Path.h"

namespace llvm {

// Forward declare the handler class
class BytecodeHandler;

typedef size_t BCDecompressor_t(const char *, size_t, char*&, std::string*);

/// This function returns a ModuleProvider that can be used to do lazy 
/// function-at-a-time loading from a bytecode file.
/// @returns NULL on error
/// @returns ModuleProvider* if successful
/// @brief Get a ModuleProvide for a bytecode file.
ModuleProvider *getBytecodeModuleProvider(
  const std::string &Filename,  ///< Name of file to be read
  BCDecompressor_t *BCDC = Compressor::decompressToNewBuffer,
  std::string* ErrMsg = 0,      ///< Optional error message holder 
  BytecodeHandler* H = 0        ///< Optional handler for reader events
);

/// This function returns a ModuleProvider that can be used to do lazy 
/// function function-at-a-time loading from a bytecode buffer.
/// @returns NULL on error
/// @returns ModuleProvider* if successful
/// @brief Get a ModuleProvider for a bytecode buffer.
ModuleProvider *getBytecodeBufferModuleProvider(
  const unsigned char *Buffer,    ///< Start of buffer to parse
  unsigned BufferSize,            ///< Size of the buffer
  const std::string &ModuleID,    ///< Name to give the module
  BCDecompressor_t *BCDC = Compressor::decompressToNewBuffer,
  std::string* ErrMsg = 0,        ///< Optional place to return an error message
  BytecodeHandler* H = 0          ///< Optional handler for reader events
);

/// This is the main interface to bytecode parsing. It opens the file specified
/// by \p Filename and parses the entire file, returing the corresponding Module
/// object if successful.
/// @returns NULL on error
/// @returns the module corresponding to the bytecode file, if successful
/// @brief Parse the given bytecode file
Module* ParseBytecodeFile(
  const std::string &Filename,    ///< Name of file to parse
  BCDecompressor_t *BCDC = Compressor::decompressToNewBuffer,
  std::string *ErrMsg = 0         ///< Optional place to return an error message
);

/// Parses a bytecode buffer specified by \p Buffer and \p BufferSize.
/// @returns NULL on error
/// @returns the module corresponding to the bytecode buffer, if successful
/// @brief Parse a given bytecode buffer
Module* ParseBytecodeBuffer(
  const unsigned char *Buffer,    ///< Start of buffer to parse
  unsigned BufferSize,            ///< Size of the buffer
  const std::string &ModuleID="", ///< Name to give the module
  BCDecompressor_t *BCDC = Compressor::decompressToNewBuffer,
  std::string *ErrMsg = 0         ///< Optional place to return an error message
);

} // End llvm namespace

#endif
