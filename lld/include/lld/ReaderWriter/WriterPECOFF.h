//===- ReaderWriter/WriterPECOFF.h - PECOFF File Format Writing Interface -===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_WRITER_PECOFF_H_
#define LLD_READERWRITER_WRITER_PECOFF_H_

#include "lld/ReaderWriter/Writer.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"


namespace lld {

/// 
/// The WriterOptionsPECOFF class encapsulates options needed 
/// to process mach-o files.  You can create an WriterOptionsPECOFF 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class WriterOptionsPECOFF : public WriterOptions {
public:
  virtual ~WriterOptionsPECOFF();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  WriterOptionsPECOFF(int argc, const char* argv[]);

  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  WriterOptionsPECOFF();

protected:
};



///
/// The only way to instantiate a WriterPECOFF object  
/// is via this createWriterPECOFF function.  The is no public 
/// WriterPECOFF class that you might be tempted to subclass.
/// Support for all variants must be represented in the WriterOptionsPECOFF
/// object.
/// The Writer object created retains a reference to the 
/// WriterOptionsPECOFF object supplied, so it must not be destroyed 
/// before the Writer object. 
///
Writer* createWriterPECOFF(const WriterOptionsPECOFF &options);



} // namespace lld

#endif // LLD_READERWRITER_WRITER_PECOFF_H_
