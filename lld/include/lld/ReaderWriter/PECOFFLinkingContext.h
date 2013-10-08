//===- lld/ReaderWriter/PECOFFLinkingContext.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PECOFF_LINKER_CONTEXT_H
#define LLD_READER_WRITER_PECOFF_LINKER_CONTEXT_H

#include <set>
#include <vector>

#include "lld/Core/LinkingContext.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/Allocator.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/ErrorHandling.h"

using llvm::COFF::MachineTypes;
using llvm::COFF::WindowsSubsystem;

namespace lld {

class PECOFFLinkingContext : public LinkingContext {
public:
  PECOFFLinkingContext()
      : _baseAddress(0x400000), _stackReserve(1024 * 1024), _stackCommit(4096),
        _heapReserve(1024 * 1024), _heapCommit(4096), _noDefaultLibAll(false),
        _sectionAlignment(4096),
        _subsystem(llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN),
        _machineType(llvm::COFF::IMAGE_FILE_MACHINE_I386), _imageVersion(0, 0),
        _minOSVersion(6, 0), _nxCompat(true), _largeAddressAware(false),
        _allowBind(true), _allowIsolation(true), _swapRunFromCD(false),
        _swapRunFromNet(false), _baseRelocationEnabled(true),
        _terminalServerAware(true), _dynamicBaseEnabled(true),
        _imageType(ImageType::IMAGE_EXE) {
    setDeadStripping(true);
  }

  struct Version {
    Version(int v1, int v2) : majorVersion(v1), minorVersion(v2) {}
    int majorVersion;
    int minorVersion;
  };

  /// \brief Casting support
  static inline bool classof(const LinkingContext *info) { return true; }

  enum ImageType {
    IMAGE_EXE,
    IMAGE_DLL
  };

  virtual Reader &getDefaultReader() const { return *_reader; }

  virtual Writer &writer() const;
  virtual bool validateImpl(raw_ostream &diagnostics);

  virtual void addPasses(PassManager &pm) const;

  virtual bool
  createImplicitFiles(std::vector<std::unique_ptr<File> > &result) const;

  void appendInputSearchPath(StringRef dirPath) {
    _inputSearchPaths.push_back(dirPath);
  }

  const std::vector<StringRef> getInputSearchPaths() {
    return _inputSearchPaths;
  }

  StringRef searchLibraryFile(StringRef path) const;

  /// Returns the decorated name of the given symbol name. On 32-bit x86, it
  /// adds "_" at the beginning of the string. On other architectures, the
  /// return value is the same as the argument.
  StringRef decorateSymbol(StringRef name) const {
    if (_machineType != llvm::COFF::IMAGE_FILE_MACHINE_I386)
      return name;
    std::string str = "_";
    str.append(name);
    return allocateString(str);
  }

  void setEntrySymbolName(StringRef name) {
    if (!name.empty())
      LinkingContext::setEntrySymbolName(decorateSymbol(name));
  }

  void setBaseAddress(uint64_t addr) { _baseAddress = addr; }
  uint64_t getBaseAddress() const { return _baseAddress; }

  void setStackReserve(uint64_t size) { _stackReserve = size; }
  void setStackCommit(uint64_t size) { _stackCommit = size; }
  uint64_t getStackReserve() const { return _stackReserve; }
  uint64_t getStackCommit() const { return _stackCommit; }

  void setHeapReserve(uint64_t size) { _heapReserve = size; }
  void setHeapCommit(uint64_t size) { _heapCommit = size; }
  uint64_t getHeapReserve() const { return _heapReserve; }
  uint64_t getHeapCommit() const { return _heapCommit; }

  void setSectionAlignment(uint32_t val) { _sectionAlignment = val; }
  uint32_t getSectionAlignment() const { return _sectionAlignment; }

  void setSubsystem(WindowsSubsystem ss) { _subsystem = ss; }
  WindowsSubsystem getSubsystem() const { return _subsystem; }

  void setMachineType(MachineTypes type) { _machineType = type; }
  MachineTypes getMachineType() const { return _machineType; }

  void setImageVersion(const Version &version) { _imageVersion = version; }
  Version getImageVersion() const { return _imageVersion; }

  void setMinOSVersion(const Version &version) { _minOSVersion = version; }
  Version getMinOSVersion() const { return _minOSVersion; }

  void setNxCompat(bool nxCompat) { _nxCompat = nxCompat; }
  bool isNxCompat() const { return _nxCompat; }

  void setLargeAddressAware(bool val) { _largeAddressAware = val; }
  bool getLargeAddressAware() const { return _largeAddressAware; }

  void setAllowBind(bool val) { _allowBind = val; }
  bool getAllowBind() const { return _allowBind; }

  void setAllowIsolation(bool val) { _allowIsolation = val; }
  bool getAllowIsolation() const { return _allowIsolation; }

  void setSwapRunFromCD(bool val) { _swapRunFromCD = val; }
  bool getSwapRunFromCD() const { return _swapRunFromCD; }

  void setSwapRunFromNet(bool val) { _swapRunFromNet = val; }
  bool getSwapRunFromNet() const { return _swapRunFromNet; }

  void setBaseRelocationEnabled(bool val) { _baseRelocationEnabled = val; }
  bool getBaseRelocationEnabled() const { return _baseRelocationEnabled; }

  void setTerminalServerAware(bool val) { _terminalServerAware = val; }
  bool isTerminalServerAware() const { return _terminalServerAware; }

  void setDynamicBaseEnabled(bool val) { _dynamicBaseEnabled = val; }
  bool getDynamicBaseEnabled() const { return _dynamicBaseEnabled; }

  void setImageType(ImageType type) { _imageType = type; }
  ImageType getImageType() const { return _imageType; }

  void addNoDefaultLib(StringRef libName) { _noDefaultLibs.insert(libName); }
  const std::set<std::string> &getNoDefaultLibs() const {
    return _noDefaultLibs;
  }

  void setNoDefaultLibAll(bool val) { _noDefaultLibAll = val; }
  bool getNoDefaultLibAll() const { return _noDefaultLibAll; }

  virtual ErrorOr<Reference::Kind> relocKindFromString(StringRef str) const;
  virtual ErrorOr<std::string> stringFromRelocKind(Reference::Kind kind) const;

  StringRef allocateString(StringRef ref) const {
    char *x = _allocator.Allocate<char>(ref.size() + 1);
    memcpy(x, ref.data(), ref.size());
    x[ref.size()] = '\0';
    return x;
  }

  virtual bool hasInputGraph() {
    if (_inputGraph)
      return true;
    return false;
  }

protected:
  /// Method to create a internal file for the entry symbol
  virtual std::unique_ptr<File> createEntrySymbolFile() const;

  /// Method to create a internal file for an undefined symbol
  virtual std::unique_ptr<File> createUndefinedSymbolFile() const;

private:
  // The start address for the program. The default value for the executable is
  // 0x400000, but can be altered using -base command line option.
  uint64_t _baseAddress;

  uint64_t _stackReserve;
  uint64_t _stackCommit;
  uint64_t _heapReserve;
  uint64_t _heapCommit;
  bool _noDefaultLibAll;
  uint32_t _sectionAlignment;
  WindowsSubsystem _subsystem;
  MachineTypes _machineType;
  Version _imageVersion;
  Version _minOSVersion;
  bool _nxCompat;
  bool _largeAddressAware;
  bool _allowBind;
  bool _allowIsolation;
  bool _swapRunFromCD;
  bool _swapRunFromNet;
  bool _baseRelocationEnabled;
  bool _terminalServerAware;
  bool _dynamicBaseEnabled;
  ImageType _imageType;

  // The set to store /nodefaultlib arguments.
  std::set<std::string> _noDefaultLibs;

  std::vector<StringRef> _inputSearchPaths;
  std::unique_ptr<Reader> _reader;
  std::unique_ptr<Writer> _writer;
};

} // end namespace lld

#endif
