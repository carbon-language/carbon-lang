//===- lld/ReaderWriter/ELFLinkingContext.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_LINKER_CONTEXT_H
#define LLD_READER_WRITER_ELF_LINKER_CONTEXT_H

#include "lld/Core/LinkingContext.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/Pass.h"
#include "lld/Core/range.h"
#include "lld/Core/STDExtras.h"

#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

#include <map>
#include <memory>

namespace lld {
class DefinedAtom;
class Reference;

namespace elf {
template <typename ELFT> class TargetHandler;
}

class TargetHandlerBase {
public:
  virtual ~TargetHandlerBase() {}
  virtual void registerRelocationNames(Registry &) = 0;

  virtual std::unique_ptr<Reader> getObjReader(bool) = 0;

  virtual std::unique_ptr<Reader> getDSOReader(bool) = 0;

  virtual std::unique_ptr<Writer> getWriter() = 0;
};

class ELFLinkingContext : public LinkingContext {
public:

  /// \brief The type of ELF executable that the linker
  /// creates.
  enum class OutputMagic : uint8_t {
    DEFAULT, // The default mode, no specific magic set
    NMAGIC,  // Disallow shared libraries and don't align sections
             // PageAlign Data, Mark Text Segment/Data segment RW
    OMAGIC   // Disallow shared libraries and don't align sections,
             // Mark Text Segment/Data segment RW
  };

  llvm::Triple getTriple() const { return _triple; }
  virtual bool is64Bits() const;
  virtual bool isLittleEndian() const;
  virtual uint64_t getPageSize() const { return 0x1000; }
  OutputMagic getOutputMagic() const { return _outputMagic; }
  uint16_t getOutputELFType() const { return _outputELFType; }
  uint16_t getOutputMachine() const;
  bool mergeCommonStrings() const { return _mergeCommonStrings; }
  virtual uint64_t getBaseAddress() const { return _baseAddress; }

  /// This controls if undefined atoms need to be created for undefines that are
  /// present in a SharedLibrary. If this option is set, undefined atoms are
  /// created for every undefined symbol that are present in the dynamic table
  /// in the shared library
  bool useShlibUndefines() const { return _useShlibUndefines; }
  /// @}

  /// \brief Does this relocation belong in the dynamic relocation table?
  ///
  /// This table is evaluated at loadtime by the dynamic loader and is
  /// referenced by the DT_RELA{,ENT,SZ} entries in the dynamic table.
  /// Relocations that return true will be added to the dynamic relocation
  /// table.
  virtual bool isDynamicRelocation(const DefinedAtom &,
                                   const Reference &) const {
    return false;
  }
  virtual bool validateImpl(raw_ostream &diagnostics);

  /// \brief Does the linker allow dynamic libraries to be linked with ?
  /// This is true when the output mode of the executable is set to be
  /// having NMAGIC/OMAGIC
  virtual bool allowLinkWithDynamicLibraries() const {
    if (_outputMagic == OutputMagic::NMAGIC ||
        _outputMagic == OutputMagic::OMAGIC || _noAllowDynamicLibraries)
      return false;
    return true;
  }

  static std::unique_ptr<ELFLinkingContext> create(llvm::Triple);

  /// \brief Use Elf_Rela format to output relocation tables.
  virtual bool isRelaOutputFormat() const { return true; }

  /// \brief Does this relocation belong in the dynamic plt relocation table?
  ///
  /// This table holds all of the relocations used for delayed symbol binding.
  /// It will be evaluated at load time if LD_BIND_NOW is set. It is referenced
  /// by the DT_{JMPREL,PLTRELSZ} entries in the dynamic table.
  /// Relocations that return true will be added to the dynamic plt relocation
  /// table.
  virtual bool isPLTRelocation(const DefinedAtom &, const Reference &) const {
    return false;
  }

  /// \brief The path to the dynamic interpreter
  virtual StringRef getDefaultInterpreter() const {
    return "/lib64/ld-linux-x86-64.so.2";
  }

  /// \brief The dynamic linker path set by the --dynamic-linker option
  virtual StringRef getInterpreter() const {
    if (_dynamicLinkerArg)
      return _dynamicLinkerPath;
    return getDefaultInterpreter();
  }

  /// \brief Does the output have dynamic sections.
  virtual bool isDynamic() const;

  /// \brief Are we creating a shared library ?
  virtual bool isDynamicLibrary() const {
    return _outputELFType == llvm::ELF::ET_DYN;
  }

  /// \brief Is the relocation a relative relocation
  virtual bool isRelativeReloc(const Reference &r) const;

  template <typename ELFT>
  lld::elf::TargetHandler<ELFT> &getTargetHandler() const {
    assert(_targetHandler && "Got null TargetHandler!");
    return static_cast<lld::elf::TargetHandler<ELFT> &>(*_targetHandler.get());
  }

  TargetHandlerBase *targetHandler() const { return _targetHandler.get(); }
  virtual void addPasses(PassManager &pm);

  void setTriple(llvm::Triple trip) { _triple = trip; }
  void setNoInhibitExec(bool v) { _noInhibitExec = v; }
  void setIsStaticExecutable(bool v) { _isStaticExecutable = v; }
  void setMergeCommonStrings(bool v) { _mergeCommonStrings = v; }
  void setUseShlibUndefines(bool use) { _useShlibUndefines = use; }

  void setOutputELFType(uint32_t type) { _outputELFType = type; }

  /// \brief Set the dynamic linker path
  void setInterpreter(StringRef dynamicLinker) {
    _dynamicLinkerArg = true;
    _dynamicLinkerPath = dynamicLinker;
  }

  /// \brief Set NMAGIC output kind when the linker specifies --nmagic
  /// or -n in the command line
  /// Set OMAGIC output kind when the linker specifies --omagic
  /// or -N in the command line
  virtual void setOutputMagic(OutputMagic magic) { _outputMagic = magic; }

  /// \brief Disallow dynamic libraries during linking
  virtual void setNoAllowDynamicLibraries() { _noAllowDynamicLibraries = true; }

  /// Searches directories for a match on the input File
  ErrorOr<StringRef> searchLibrary(StringRef libName) const;

  /// Get the entry symbol name
  virtual StringRef entrySymbolName() const;

  /// add to the list of initializer functions
  void addInitFunction(StringRef name) { _initFunctions.push_back(name); }

  /// add to the list of finalizer functions
  void addFiniFunction(StringRef name) { _finiFunctions.push_back(name); }

  /// Return the list of initializer symbols that are specified in the
  /// linker command line, using the -init option.
  range<const StringRef *> initFunctions() const {
    return _initFunctions;
  }

  /// Return the list of finalizer symbols that are specified in the
  /// linker command line, using the -fini option.
  range<const StringRef *> finiFunctions() const { return _finiFunctions; }

  void setSharedObjectName(StringRef soname) {
    _soname = soname;
  }

  StringRef sharedObjectName() const { return _soname; }

  /// \brief Set path to the system root
  void setSysroot(StringRef path) {
    _sysrootPath = path;
  }

  void addRpath(StringRef path) {
   _rpathList.push_back(path);
  }

  range<const StringRef *> getRpathList() const {
    return _rpathList;
  }

  void addRpathLink(StringRef path) {
   _rpathLinkList.push_back(path);
  }

  range<const StringRef *> getRpathLinkList() const {
    return _rpathLinkList;
  }

  virtual bool addUndefinedAtomsFromSharedLibrary(const SharedLibraryFile *s) {
    if (_undefinedAtomsFromFile.find(s) != _undefinedAtomsFromFile.end())
      return false;
    _undefinedAtomsFromFile[s] = true;
    return true;
  }

  /// \brief Helper function to allocate strings.
  StringRef allocateString(StringRef ref) const {
    char *x = _allocator.Allocate<char>(ref.size() + 1);
    memcpy(x, ref.data(), ref.size());
    x[ref.size()] = '\0';
    return x;
  }

  // add search path to list.
  virtual bool addSearchPath(StringRef ref) {
    _inputSearchPaths.push_back(ref);
    return true;
  }

private:
  ELFLinkingContext() LLVM_DELETED_FUNCTION;

protected:
  ELFLinkingContext(llvm::Triple, std::unique_ptr<TargetHandlerBase>);

  virtual Writer &writer() const;

  /// Method to create a internal file for an undefined symbol
  virtual std::unique_ptr<File> createUndefinedSymbolFile() const;

  uint16_t _outputELFType; // e.g ET_EXEC
  llvm::Triple _triple;
  std::unique_ptr<TargetHandlerBase> _targetHandler;
  uint64_t _baseAddress;
  bool _isStaticExecutable;
  bool _noInhibitExec;
  bool _mergeCommonStrings;
  bool _runLayoutPass;
  bool _useShlibUndefines;
  bool _dynamicLinkerArg;
  bool _noAllowDynamicLibraries;
  OutputMagic _outputMagic;
  StringRefVector _inputSearchPaths;
  std::unique_ptr<Writer> _writer;
  StringRef _dynamicLinkerPath;
  StringRefVector _initFunctions;
  StringRefVector _finiFunctions;
  StringRef _sysrootPath;
  StringRef _soname;
  StringRefVector _rpathList;
  StringRefVector _rpathLinkList;
  std::map<const SharedLibraryFile *, bool> _undefinedAtomsFromFile;
};
} // end namespace lld

#endif
