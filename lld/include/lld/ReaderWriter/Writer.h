//===- ReaderWriter/Writer.h - Abstract File Format Interface -------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_WRITER_H_
#define LLD_READERWRITER_WRITER_H_

#include "lld/Core/LLVM.h"
#include <memory>
#include <vector>

namespace lld {
class File;
class InputFiles;
class StubsPass;
class GOTPass;


///
/// The Writer is an abstract class for writing object files, 
/// shared library files, and executable files.  Each file format
/// (e.g. ELF, mach-o, PECOFF, native, etc) have a concrete subclass
/// of Writer.  
///
class Writer {
public:
  virtual ~Writer();
  
  /// Write a file from the supplied File object 
  virtual error_code writeFile(const lld::File &linkedFile, StringRef path) = 0;
  
  /// Return a Pass object for creating stubs/PLT entries
  virtual StubsPass *stubPass() {
    return nullptr;
  }
  
  /// Return a Pass object for creating GOT entries
  virtual GOTPass *gotPass() {
    return nullptr;
  }
  
  /// This method is called by Core Linking to give the Writer a chance to
  /// add file format specific "files" to set of files to be linked.
  /// This is how file format specific atoms can be added to the link.
  virtual void addFiles(InputFiles&) {
  }
  
  
protected:
  // only concrete subclasses can be instantiated
  Writer();
};



///
/// The WriterOptions encapsulates the options used by Writers.  
/// Each file format defines a subclass of WriterOptions
/// to hold file format specific options.  The option objects are the only
/// way to control the behaviour of Writers.
///
class WriterOptions {
public:
  // Any options common to all file formats will go here.

protected:
  // only concrete subclasses can be instantiated
  WriterOptions();
};





} // namespace lld

#endif // LLD_READERWRITER_WRITER_H_




