//===- ReaderWriter/ReaderPECOFF.h - PECOFF File Format Reading Interface ---===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_READER_PECOFF_H_
#define LLD_READERWRITER_READER_PECOFF_H_

#include "lld/ReaderWriter/Reader.h"
#include "lld/Core/LLVM.h"


namespace lld {

/// 
/// The ReaderOptionsPECOFF class encapsulates options needed 
/// to process mach-o files.  You can create an ReaderOptionsPECOFF 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class ReaderOptionsPECOFF : public ReaderOptions {
public:
  virtual ~ReaderOptionsPECOFF();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  ReaderOptionsPECOFF(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  ReaderOptionsPECOFF();

  
protected:
};
 



///
/// The only way to instantiate a ReaderPECOFF object  
/// is via this createReaderPECOFF function.  The is no public 
/// ReaderPECOFF class that you might be tempted to subclass.
/// Support for all variants must be represented in the ReaderOptionsPECOFF
/// object.
/// The Reader object created retains a reference to the 
/// ReaderOptionsPECOFF object supplied, so the objects object must not be  
/// destroyed before the Reader object. 
///
Reader* createReaderPECOFF(const ReaderOptionsPECOFF &options);



} // namespace lld

#endif // LLD_READERWRITER_READER_PECOFF_H_
