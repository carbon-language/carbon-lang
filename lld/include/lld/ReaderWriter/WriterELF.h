//===- ReaderWriter/WriterELF.h - ELF File Format Writing Interface -------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_WRITER_ELF_H_
#define LLD_READERWRITER_WRITER_ELF_H_

#include "lld/ReaderWriter/Writer.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"


namespace lld {

/// 
/// The WriterOptionsELF class encapsulates options needed 
/// to process mach-o files.  You can create an WriterOptionsELF 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class WriterOptionsELF : public WriterOptions {
public:
  virtual ~WriterOptionsELF();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  WriterOptionsELF(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  WriterOptionsELF();
  
protected:
};
 



///
/// The only way to instantiate a WriterELF object  
/// is via this createWriterELF function.  The is no public 
/// WriterELF class that you might be tempted to subclass.
/// Support for all variants must be represented in the WriterOptionsELF
/// object.
/// The Writer object created retains a reference to the 
/// WriterOptionsELF object supplied, so it must not be destroyed 
/// before the Writer object. 
///
Writer* createWriterELF(const WriterOptionsELF &options);



} // namespace lld

#endif // LLD_READERWRITER_WRITER_ELF_H_
