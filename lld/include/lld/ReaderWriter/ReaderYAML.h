//===- ReaderWriter/ReaderYAML.h - YAML File Format Reading Interface -----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_READER_YAML_H_
#define LLD_READERWRITER_READER_YAML_H_

#include "lld/ReaderWriter/Reader.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "llvm/ADT/StringRef.h"


namespace lld {

/// 
/// The ReaderOptionsYAML class encapsulates options needed 
/// to process mach-o files.  You can create an ReaderOptionsYAML 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class ReaderOptionsYAML : public ReaderOptions {
public:
  virtual ~ReaderOptionsYAML();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  ReaderOptionsYAML(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  ReaderOptionsYAML();


  /// Converts a reference kind string to a in-memory numeric value.
  /// Used when parsing YAML encoded object files.
  virtual Reference::Kind kindFromString(StringRef) const = 0;

  
protected:
};
 



///
/// The only way to instantiate a ReaderYAML object  
/// is via this createReaderYAML function.  The is no public 
/// ReaderYAML class that you might be tempted to subclass.
/// Support for all variants must be represented in the ReaderOptionsYAML
/// object.
/// The Reader object created retains a reference to the 
/// ReaderOptionsYAML object supplied, so the objects object must not be  
/// destroyed before the Reader object. 
///
Reader* createReaderYAML(const ReaderOptionsYAML &options);



} // namespace lld

#endif // LLD_READERWRITER_READER_YAML_H_


