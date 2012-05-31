//===- ReaderWriter/WriterNative.h - Native File Format Reading Interface ---===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_WRITER_NATIVE_H_
#define LLD_READERWRITER_WRITER_NATIVE_H_

#include "lld/ReaderWriter/Writer.h"
#include "lld/Core/LLVM.h"


namespace lld {

/// 
/// The WriterOptionsNative class encapsulates options needed 
/// to process mach-o files.  You can create an WriterOptionsNative 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class WriterOptionsNative : public WriterOptions {
public:
  virtual ~WriterOptionsNative();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  WriterOptionsNative(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  WriterOptionsNative();
  
protected:
};
 



///
/// The only way to instantiate a WriterNative object  
/// is via this createWriterNative function.  The is no public 
/// WriterNative class that you might be tempted to subclass.
/// Support for all variants must be represented in the WriterOptionsNative
/// object.
/// The Writer object created retains a reference to the 
/// WriterOptionsNative object supplied, so it must not be destroyed 
/// before the Writer object. 
///
Writer* createWriterNative(const WriterOptionsNative &options);



} // namespace lld

#endif // LLD_READERWRITER_WRITER_NATIVE_H_
