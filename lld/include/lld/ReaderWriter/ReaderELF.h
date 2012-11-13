//===- ReaderWriter/ReaderELF.h - ELF File Format Reading Interface -------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_READER_ELF_H_
#define LLD_READERWRITER_READER_ELF_H_

#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/ReaderArchive.h"
#include "lld/Core/LLVM.h"


namespace lld {

/// 
/// The ReaderOptionsELF class encapsulates options needed 
/// to process mach-o files.  You can create an ReaderOptionsELF 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class ReaderOptionsELF : public ReaderOptions {
public:
  virtual ~ReaderOptionsELF();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  ReaderOptionsELF(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  ReaderOptionsELF();


  
protected:
};
 



///
/// The only way to instantiate a ReaderELF object  
/// is via this createReaderELF function.  The is no public 
/// ReaderELF class that you might be tempted to subclass.
/// Support for all variants must be represented in the ReaderOptionsELF
/// object.
/// The Reader object created retains a reference to the 
/// ReaderOptionsELF object supplied, so the objects object must not be  
/// destroyed before the Reader object. 
///
Reader* createReaderELF(const ReaderOptionsELF &options, 
                        ReaderOptionsArchive &optionsArchive);



} // namespace lld

#endif // LLD_READERWRITER_READER_ELF_H_
