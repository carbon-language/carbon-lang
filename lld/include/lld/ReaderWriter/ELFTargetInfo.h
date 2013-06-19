//===- lld/ReaderWriter/ELFTargetInfo.h -----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_TARGET_INFO_H
#define LLD_READER_WRITER_ELF_TARGET_INFO_H

#include "lld/Core/PassManager.h"
#include "lld/Core/Pass.h"
#include "lld/Core/range.h"
#include "lld/Core/TargetInfo.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"

#include <memory>

namespace lld {
class DefinedAtom;
class Reference;

namespace elf { template <typename ELFT> class TargetHandler; }

class TargetHandlerBase {
public:
  virtual ~TargetHandlerBase() {}
};

class ELFTargetInfo : public TargetInfo {
public:
  enum class OutputMagic : uint8_t {
    DEFAULT, // The default mode, no specific magic set
    NMAGIC,  // Disallow shared libraries and dont align sections
             // PageAlign Data, Mark Text Segment/Data segment RW
    OMAGIC   // Disallow shared libraries and dont align sections,
             // Mark Text Segment/Data segment RW
  };
  llvm::Triple getTriple() const { return _triple; }
  virtual bool is64Bits() const;
  virtual bool isLittleEndian() const;
  virtual uint64_t getPageSize() const { return 0x1000; }
  OutputMagic getOutputMagic() const { return _outputMagic; }
  uint16_t getOutputType() const { return _outputFileType; }
  uint16_t getOutputMachine() const;
  bool outputYAML() const { return _outputYAML; }
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
        _outputMagic == OutputMagic::OMAGIC ||
        _noAllowDynamicLibraries)
      return false;
    return true;
  }

  virtual error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                        std::vector<std::unique_ptr<File>> &result) const;

  static std::unique_ptr<ELFTargetInfo> create(llvm::Triple);

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

  /// \brief Is the relocation a relative relocation
  virtual bool isRelativeReloc(const Reference &r) const;

  template <typename ELFT>
  lld::elf::TargetHandler<ELFT> &getTargetHandler() const {
    assert(_targetHandler && "Got null TargetHandler!");
    return static_cast<lld::elf::TargetHandler<ELFT> &>(*_targetHandler.get());
  }

  virtual void addPasses(PassManager &pm) const;

  void setTriple(llvm::Triple trip) { _triple = trip; }
  void setOutputFileType(uint32_t type) { _outputFileType = type; }
  void setOutputYAML(bool v) { _outputYAML = v; }
  void setNoInhibitExec(bool v) { _noInhibitExec = v; }
  void setIsStaticExecutable(bool v) { _isStaticExecutable = v; }
  void setMergeCommonStrings(bool v) { _mergeCommonStrings = v; }
  void setUseShlibUndefines(bool use) { _useShlibUndefines = use; }

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

  void appendSearchPath(StringRef dirPath) {
    _inputSearchPaths.push_back(dirPath);
  }
  /// Searches directories then calls appendInputFile()
  bool appendLibrary(StringRef libName);

  /// adds undefined symbols that are specified in the command line
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

private:
  ELFTargetInfo() LLVM_DELETED_FUNCTION;
protected:
  ELFTargetInfo(llvm::Triple, std::unique_ptr<TargetHandlerBase>);

  virtual Writer &writer() const;

  uint16_t                           _outputFileType; // e.g ET_EXEC
  llvm::Triple                       _triple;
  std::unique_ptr<TargetHandlerBase> _targetHandler;
  uint64_t                           _baseAddress;
  bool                               _isStaticExecutable;
  bool                               _outputYAML;
  bool                               _noInhibitExec;
  bool                               _mergeCommonStrings;
  bool                               _runLayoutPass;
  bool                               _useShlibUndefines;
  bool                               _dynamicLinkerArg;
  bool                               _noAllowDynamicLibraries;
  OutputMagic                        _outputMagic;
  StringRefVector                    _inputSearchPaths;
  llvm::BumpPtrAllocator             _extraStrings;
  std::unique_ptr<Reader>            _elfReader;
  std::unique_ptr<Writer>            _writer;
  std::unique_ptr<Reader>            _linkerScriptReader;
  StringRef                          _dynamicLinkerPath;
  StringRefVector                    _undefinedSymbols;
};
} // end namespace lld

#endif
