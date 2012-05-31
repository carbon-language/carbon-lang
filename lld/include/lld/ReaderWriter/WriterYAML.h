//===- ReaderWriter/WriterYAML.h - YAML File Format Writing Interface -----===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_WRITER_YAML_H_
#define LLD_READERWRITER_WRITER_YAML_H_

#include "lld/ReaderWriter/Writer.h"

#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"
#include "lld/Core/Pass.h"

#include "llvm/ADT/StringRef.h"


namespace lld {

/// 
/// The WriterOptionsYAML class encapsulates options needed 
/// to process mach-o files.  You can create an WriterOptionsYAML 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class WriterOptionsYAML : public WriterOptions {
public:
  virtual ~WriterOptionsYAML();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  WriterOptionsYAML(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  WriterOptionsYAML();
  
  
  /// Converts an in-memory reference kind value to a string.
  /// Used when writing YAML encoded object files.
  virtual StringRef kindToString(Reference::Kind) const = 0;


  /// Enable Stubs pass to be run
  virtual StubsPass *stubPass() const {
    return nullptr;
  }
  
  /// Enable GOT pass to be run
  virtual GOTPass *gotPass() const {
    return nullptr;
  }
  
};
 



///
/// The only way to instantiate a WriterYAML object  
/// is via this createWriterYAML function.  The is no public 
/// WriterYAML class that you might be tempted to subclass.
/// Support for all variants must be represented in the WriterOptionsYAML
/// object.
/// The Writer object created retains a reference to the 
/// WriterOptionsYAML object supplied, so it must not be destroyed 
/// before the Writer object. 
///
Writer* createWriterYAML(const WriterOptionsYAML &options);


} // namespace lld

#endif // LLD_READERWRITER_WRITER_YAML_H_
