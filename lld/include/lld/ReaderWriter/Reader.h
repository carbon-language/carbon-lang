//===- ReaderWriter/Reader.h - Abstract File Format Reading Interface -----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_READER_H_
#define LLD_READERWRITER_READER_H_

#include "lld/Core/LLVM.h"
#include <memory>
#include <vector>

namespace lld {
class File;

///
/// The Reader is an abstract class for reading object files, 
/// library files, and executable files.  Each file format
/// (e.g. ELF, mach-o, PECOFF, native, etc) have a concrete subclass
/// of Reader.  
///
class Reader {
public:
  virtual ~Reader();
  
 
  /// Parse a file given its file system path and create a File object. 
  virtual error_code readFile(StringRef path,
                              std::vector<std::unique_ptr<File>> &result);

  /// Parse a supplied buffer (already filled with the contents of a file)
  /// and create a File object. 
  /// On success, the resulting File object takes ownership of 
  /// the MemoryBuffer.
  virtual error_code parseFile(std::unique_ptr<MemoryBuffer> mb,
                               std::vector<std::unique_ptr<File>> &result) = 0;
  
protected:
  // only concrete subclasses can be instantiated
  Reader();
};



///
/// The ReaderOptions encapsulates the options used by all Readers.  
/// Each file format defines a subclass of ReaderOptions
/// to hold file format specific options.  The option objects are the only
/// way to control the behaviour of Readers.
///
class ReaderOptions {
public:
  // Any options common to all file format Readers will go here.

protected:
  // only concrete subclasses can be instantiated
  ReaderOptions();
};





} // namespace lld

#endif // LLD_READERWRITER_READER_H_




