//===- ReaderWriter/ReaderMachO.h - MachO File Format Reading Interface ---===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_READER_MACHO_H_
#define LLD_READER_WRITER_READER_MACHO_H_

#include "lld/ReaderWriter/Reader.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"


namespace lld {

/// 
/// The ReaderOptionsMachO class encapsulates options needed 
/// to process mach-o files.  You can create an ReaderOptionsMachO 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class ReaderOptionsMachO : public ReaderOptions {
public:
  virtual ~ReaderOptionsMachO()  { }

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  ReaderOptionsMachO(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  ReaderOptionsMachO();


  StringRef archName() const    { return _archName; }
  
protected:
  StringRef       _archName;
};
 



///
/// The only way to instantiate a ReaderMachO object  
/// is via this createReaderMachO function.  The is no public 
/// ReaderMachO class that you might be tempted to subclass.
/// Support for all variants must be represented in the ReaderOptionsMachO
/// object.
/// The Reader object created retains a reference to the 
/// ReaderOptionsMachO object supplied, so the objects object must not be  
/// destroyed before the Reader object. 
///
Reader* createReaderMachO(const ReaderOptionsMachO &options);



} // namespace lld

#endif // LLD_READER_WRITER_READER_MACHO_H_
