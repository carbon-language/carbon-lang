//===- ReaderWriter/WriterMachO.h - MachO File Format Reading Interface ---===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READERWRITER_WRITER_MACHO_H_
#define LLD_READERWRITER_WRITER_MACHO_H_

#include "lld/ReaderWriter/Writer.h"
#include "lld/Core/LLVM.h"
#include "llvm/ADT/StringRef.h"


namespace lld {

/// 
/// The WriterOptionsMachO class encapsulates options needed 
/// to process mach-o files.  You can create an WriterOptionsMachO 
/// instance from command line arguments or by subclassing and setting the 
/// instance variables in the subclass's constructor.
///
class WriterOptionsMachO : public WriterOptions {
public:
  virtual ~WriterOptionsMachO();

  ///
  /// Creates a Options object from darwin linker command line arguments.
  /// FIXME: to be replaced with new option processing mechanism.
  ///
  WriterOptionsMachO(int argc, const char* argv[]);
  
  ///
  /// Creates a Options object with default settings. For use when 
  /// programmatically constructing options.
  ///
  WriterOptionsMachO();



  enum OutputKind {
    outputDynamicExecutable,
    outputDylib,
    outputBundle,
    outputObjectFile
  };
  
  enum Architecture {
    arch_x86_64,
    arch_x86,
    arch_armv6,
    arch_armv7,
  };
  
  OutputKind   outputKind() const       { return _outputkind; }
  Architecture architecture() const     { return _architecture; }
  StringRef    archName() const;
  uint32_t     cpuType() const;
  uint32_t     cpuSubtype() const;
  uint64_t     pageZeroSize() const;
  bool         noTextRelocations() const { return _noTextRelocations; }
  bool         addEntryPointLoadCommand() const;
  bool         addUnixThreadLoadCommand() const;
  StringRef    entryPointName() const;

protected:
  OutputKind      _outputkind;
  Architecture    _architecture;
  uint64_t        _pageZeroSize;
  bool            _noTextRelocations;
  StringRef       _customEntryPointName;
};
 



///
/// The only way to instantiate a WriterMachO object  
/// is via this createWriterMachO function.  The is no public 
/// WriterMachO class that you might be tempted to subclass.
/// Support for all variants must be represented in the 
/// WriterOptionsMachO object.
/// The Writer object created retains a reference to the 
/// WriterOptionsMachO object supplied, so it must not be destroyed 
/// before the Writer object. 
///
Writer* createWriterMachO(const WriterOptionsMachO &options);


///
/// Returns an options object that can be used with the 
/// WriterYAML to write mach-o object files as YAML.
///
const class WriterOptionsYAML& writerOptionsMachOAsYAML(); 


///
/// Returns an options object that can be used with the 
/// ReaderYAML to reader YAML encoded mach-o files.
///
const class ReaderOptionsYAML& readerOptionsMachOAsYAML(); 
 



} // namespace lld

#endif // LLD_READERWRITER_WRITER_MACHO_H_
