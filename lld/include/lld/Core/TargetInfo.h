//===- lld/Core/TargetInfo.h - Linker Target Info Interface ---------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_TARGET_INFO_H
#define LLD_CORE_TARGET_INFO_H

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/range.h"
#include "lld/Core/Reference.h"

#include "lld/Driver/LinkerInput.h"
#include "lld/ReaderWriter/Reader.h"

#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace llvm {
  class Triple;
}

namespace lld {
class PassManager;
class File;
class Writer;
class InputFiles;

/// \brief The TargetInfo class encapsulates "what and how" to link.
///
/// The base class TargetInfo contains the options needed by core linking.
/// Subclasses of TargetInfo have additional options needed by specific Readers
/// and Writers. For example, ELFTargetInfo has methods that supplies options
/// to the ELF Reader and Writer.
///
/// \todo Consider renaming to something like "LinkingOptions".
class TargetInfo : public Reader {
public:
  virtual ~TargetInfo();

  /// \name Methods needed by core linking
  /// @{

  /// Name of symbol linker should use as "entry point" to program,
  /// usually "main" or "start".
  StringRef entrySymbolName() const {
    return _entrySymbolName;
  }

  /// Whether core linking should remove Atoms not reachable by following
  /// References from the entry point Atom or from all global scope Atoms
  /// if globalsAreDeadStripRoots() is true.
  bool deadStrip() const {
    return _deadStrip;
  }

  /// Only used if deadStrip() returns true.  Means all global scope Atoms
  /// should be marked live (along with all Atoms they reference).  Usually
  /// this method returns false for main executables, but true for dynamic
  /// shared libraries.
  bool globalsAreDeadStripRoots() const {
    assert(_deadStrip && "only applicable when deadstripping enabled");
    return _globalsAreDeadStripRoots;
  }

  /// Only used if deadStrip() returns true.  This method returns the names
  /// of DefinedAtoms that should be marked live (along with all Atoms they
  /// reference). Only Atoms with scope scopeLinkageUnit or scopeGlobal can
  /// be kept live using this method.
  const std::vector<StringRef> &deadStripRoots() const {
    return _deadStripRoots;
  }

  /// Archive files (aka static libraries) are normally lazily loaded.  That is,
  /// object files within an archive are only loaded and linked in, if the
  /// object file contains a DefinedAtom which will replace an existing
  /// UndefinedAtom.  If this method returns true, core linking will actively
  /// load every member object file from every archive.
  bool forceLoadAllArchives() const {
    return _forceLoadAllArchives;
  }

  /// Archive files (aka static libraries) are normally lazily loaded.  That is,
  /// object files within an archive are only loaded and linked in, if the
  /// object file contains a DefinedAtom which will replace an existing
  /// UndefinedAtom.  If this method returns true, core linking will also look
  /// for archive members to replace existing tentative definitions in addition
  /// to replacing undefines. Note: a "tentative definition" (also called a
  /// "common" symbols) is a C (but not C++) concept. They are modeled in lld
  /// as a DefinedAtom with merge() of mergeAsTentative.
  bool searchArchivesToOverrideTentativeDefinitions() const {
    return _searchArchivesToOverrideTentativeDefinitions;
  }

  /// Normally core linking will turn a tentative definition into a real
  /// definition if not replaced by a real DefinedAtom from some object file.
  /// If this method returns true, core linking will search all supplied
  /// dynamic shared libraries for symbol names that match remaining tentative
  /// definitions.  If any are found, the corresponding tentative definition
  /// atom is replaced with SharedLibraryAtom.
  bool searchSharedLibrariesToOverrideTentativeDefinitions() const {
    return _searchSharedLibrariesToOverrideTentativeDefinitions;
  }

  /// Normally, every UndefinedAtom must be replaced by a DefinedAtom or a
  /// SharedLibraryAtom for the link to be successful.  This method controls
  /// whether core linking prints out a list of remaining UndefinedAtoms.
  ///
  /// \todo This should be a method core linking calls with a list of the
  /// UndefinedAtoms so that different drivers can format the error message
  /// as needed.
  bool printRemainingUndefines() const {
    return _printRemainingUndefines;
  }

  /// Normally, every UndefinedAtom must be replaced by a DefinedAtom or a
  /// SharedLibraryAtom for the link to be successful.  This method controls
  /// whether core linking considers remaining undefines to be an error.
  bool allowRemainingUndefines() const {
    return _allowRemainingUndefines;
  }

  /// In the lld model, a SharedLibraryAtom is a proxy atom for something
  /// that will be found in a dynamic shared library when the program runs.
  /// A SharedLibraryAtom optionally contains the name of the shared library
  /// in which to find the symbol name at runtime.  Core linking may merge
  /// two SharedLibraryAtom with the same name.  If this method returns true,
  /// when merging core linking will also verify that they both have the same
  /// loadName() and if not print a warning.
  ///
  /// \todo This should be a method core linking calls so that drivers can
  /// format the warning as needed.
  bool warnIfCoalesableAtomsHaveDifferentLoadName() const {
    return _warnIfCoalesableAtomsHaveDifferentLoadName;
  }

  /// In C/C++ you can mark a function's prototype with
  /// __attribute__((weak_import)) or __attribute__((weak)) to say the function
  /// may not be available at runtime and/or build time and in which case its
  /// address will evaluate to NULL. In lld this is modeled using the
  /// UndefinedAtom::canBeNull() method.  During core linking, UndefinedAtom
  /// with the same name are automatically merged.  If this method returns
  /// true, core link also verfies that the canBeNull() value for merged
  /// UndefinedAtoms are the same and warns if not.
  ///
  /// \todo This should be a method core linking calls so that drivers can
  /// format the warning as needed.
  bool warnIfCoalesableAtomsHaveDifferentCanBeNull() const {
    return _warnIfCoalesableAtomsHaveDifferentCanBeNull;
  }

  /// Normally, every UndefinedAtom must be replaced by a DefinedAtom or a
  /// SharedLibraryAtom for the link to be successful.  This method controls
  /// whether core linking considers remaining undefines from the shared library
  /// to be an error.
  bool allowShlibUndefines() const {
    return _allowShlibUndefines;
  }

  /// If true, core linking will write the path to each input file to stdout
  /// (i.e. llvm::outs()) as it is used.  This is used to implement the -t
  /// linker option.
  ///
  /// \todo This should be a method core linking calls so that drivers can
  /// format the line as needed.
  bool logInputFiles() const {
    return _logInputFiles;
  }

  /// Parts of LLVM use global variables which are bound to command line
  /// options (see llvm::cl::Options). This method returns "command line"
  /// options which are used to configure LLVM's command line settings.
  /// For instance the -debug-only XXX option can be used to dynamically
  /// trace different parts of LLVM and lld.
  const std::vector<const char*> &llvmOptions() const {
    return _llvmOptions;
  }

  /// This method returns the sequence of input files for core linking to
  /// process.
  ///
  /// \todo Consider moving this out of TargetInfo so that the same TargetInfo
  /// object can be reused for different links.
  const std::vector<LinkerInput> &inputFiles() const {
    return _inputFiles;
  }
  /// @}


  /// \name Methods used by Drivers to configure TargetInfo
  /// @{
  void setOutputPath(StringRef str) { _outputPath = str; }
  void setEntrySymbolName(StringRef name) { _entrySymbolName = name; }
  void setDeadStripping(bool enable) { _deadStrip = enable; }
  void setGlobalsAreDeadStripRoots(bool v) { _globalsAreDeadStripRoots = v; }
  void setSearchArchivesToOverrideTentativeDefinitions(bool search) {
    _searchArchivesToOverrideTentativeDefinitions = search;
  }
  void setSearchSharedLibrariesToOverrideTentativeDefinitions(bool search) {
    _searchSharedLibrariesToOverrideTentativeDefinitions = search;
  }
  void setWarnIfCoalesableAtomsHaveDifferentCanBeNull(bool warn) {
    _warnIfCoalesableAtomsHaveDifferentCanBeNull = warn;
  }
  void setWarnIfCoalesableAtomsHaveDifferentLoadName(bool warn) {
    _warnIfCoalesableAtomsHaveDifferentLoadName = warn;
  }
  void setForceLoadAllArchives(bool force) {
    _forceLoadAllArchives = force;
  }
  void setPrintRemainingUndefines(bool print) {
    _printRemainingUndefines = print;
  }
  void setAllowRemainingUndefines(bool allow) {
    _allowRemainingUndefines = allow;
  }
  void setAllowShlibUndefines(bool allow) {
    _allowShlibUndefines = allow;
  }
  void setLogInputFiles(bool log) {
    _logInputFiles = log;
  }
  void appendInputFile(StringRef path) {
    _inputFiles.emplace_back(LinkerInput(path));
  }
  void appendInputFile(std::unique_ptr<llvm::MemoryBuffer> buffer) {
    _inputFiles.emplace_back(LinkerInput(std::move(buffer)));
  }
  void appendLLVMOption(const char *opt) {
    _llvmOptions.push_back(opt);
  }

  /// This method adds undefined symbols specified by the -u option to the
  /// to the list of undefined symbols known to the linker. This option
  /// essentially forces an undefined symbol to be create.
  void addUndefinedSymbol(StringRef symbolName) {
    _undefinedSymbols.push_back(symbolName);
  }

  /// Iterators for symbols that appear on the command line
  typedef std::vector<StringRef> StringRefVector;
  typedef StringRefVector::iterator StringRefVectorIter;
  typedef StringRefVector::const_iterator StringRefVectorConstIter;

  /// Return the list of undefined symbols that are specified in the
  /// linker command line, using the -u option.
  range<const StringRef *> undefinedSymbols() const {
    return _undefinedSymbols;
  }

  /// After all set* methods are called, the Driver calls this method
  /// to validate that there are no missing options or invalid combinations
  /// of options.  If there is a problem, a description of the problem
  /// is written to the supplied stream.
  ///
  /// \returns true if there is an error with the current settings.
  bool validate(raw_ostream &diagnostics);


  /// @}
  /// \name Methods used by Driver::link()
  /// @{

  /// Returns the file system path to which the linked output should be written.
  ///
  /// \todo To support in-memory linking, we need an abstraction that allows
  /// the linker to write to an in-memory buffer.
  StringRef outputPath() const {
    return _outputPath;
  }

  /// Abstract method to parse a supplied input file buffer into one or
  /// more lld::File objects. Subclasses of TargetInfo must implement this
  /// method.
  ///
  /// \param inputBuff This is an in-memory read-only copy of the input file.
  /// If the resulting lld::File object will contain pointers into
  /// this memory buffer, the lld::File object should take ownership
  /// of the buffer.  Otherwise core linking will maintain ownership of the
  /// buffer and delete it at some point.
  ///
  /// \param [out] result The instantiated lld::File object is returned here.
  /// The \p result is a vector because some input files parse into more than
  /// one lld::File (e.g. YAML).
  virtual error_code parseFile(std::unique_ptr<MemoryBuffer> &inputBuff,
                        std::vector<std::unique_ptr<File>> &result) const = 0;

  /// This is a wrapper around parseFile() where the input file is specified
  /// by file system path.  The default implementation reads the input file
  /// into a memory buffer and calls parseFile().
  ///
  /// \param path This is the file system path to the input file.
  /// \param [out] result The instantiated lld::File object is returned here.
  virtual error_code readFile(StringRef path,
                        std::vector<std::unique_ptr<File>> &result) const;

  /// This method is called by core linking to give the Writer a chance
  /// to add file format specific "files" to set of files to be linked. This is
  /// how file format specific atoms can be added to the link.
  virtual void addImplicitFiles(InputFiles&) const;

  /// This method is called by core linking to build the list of Passes to be
  /// run on the merged/linked graph of all input files.
  virtual void addPasses(PassManager &pm) const;

  /// Calls through to the writeFile() method on the specified Writer.
  ///
  /// \param linkedFile This is the merged/linked graph of all input file Atoms.
  virtual error_code writeFile(const File &linkedFile) const;

  /// @}


  /// \name Methods needed by YAML I/O and error messages to convert Kind values
  /// to and from strings.
  /// @{

  /// Abstract method to parse a kind name string into an integral
  /// Reference::Kind
  virtual ErrorOr<Reference::Kind> relocKindFromString(StringRef str) const = 0;

  /// Abstract method to return the name for a given integral
  /// Reference::Kind.
  virtual ErrorOr<std::string> stringFromRelocKind(Reference::Kind k) const = 0;

  /// @}



protected:
  TargetInfo(); // Must be subclassed

  /// Abstract method to lazily instantiate the Writer.
  virtual Writer &writer() const = 0;


  StringRef                _outputPath;
  StringRef                _entrySymbolName;
  bool                     _deadStrip;
  bool                     _globalsAreDeadStripRoots;
  bool                     _searchArchivesToOverrideTentativeDefinitions;
  bool                     _searchSharedLibrariesToOverrideTentativeDefinitions;
  bool                     _warnIfCoalesableAtomsHaveDifferentCanBeNull;
  bool                     _warnIfCoalesableAtomsHaveDifferentLoadName;
  bool                     _forceLoadAllArchives;
  bool                     _printRemainingUndefines;
  bool                     _allowRemainingUndefines;
  bool                     _logInputFiles;
  bool                     _allowShlibUndefines;
  std::vector<StringRef>   _deadStripRoots;
  std::vector<LinkerInput> _inputFiles;
  std::vector<const char*> _llvmOptions;
  std::unique_ptr<Reader>  _yamlReader;
  StringRefVector          _undefinedSymbols;

 private:
  /// Validate the subclass bits. Only called by validate.
  virtual bool validateImpl(raw_ostream &diagnostics) = 0;
};
} // end namespace lld

#endif
