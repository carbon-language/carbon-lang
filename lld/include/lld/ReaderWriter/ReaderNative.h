//===- ReaderWriter/ReaderNative.h - Native File Format Reading Interface ---===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_READER_NATIVE_H_
#define LLD_READERWRITER_READER_NATIVE_H_

#include "lld/ReaderWriter/Reader.h"
#include "lld/Core/LLVM.h"


namespace lld {

/// 
/// The ReaderOptionsNative class encapsulates options needed 
/// to process mach-o files.  You can create an ReaderOptionsNative 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class ReaderOptionsNative : public ReaderOptions {
public:
  virtual ~ReaderOptionsNative();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  ReaderOptionsNative(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  ReaderOptionsNative();
  
protected:
};
 



///
/// The only way to instantiate a ReaderNative object  
/// is via this createReaderNative function.  The is no public 
/// ReaderNative class that you might be tempted to subclass.
/// Support for all variants must be represented in the ReaderOptionsNative
/// object.
/// The Reader object created retains a reference to the 
/// ReaderOptionsNative object supplied, so the objects object must not be  
/// destroyed before the Reader object. 
///
Reader* createReaderNative(const ReaderOptionsNative &options);



} // namespace lld

#endif // LLD_READERWRITER_READER_NATIVE_H_
