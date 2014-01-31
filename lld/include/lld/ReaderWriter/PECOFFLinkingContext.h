//===- lld/ReaderWriter/PECOFFLinkingContext.h ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PECOFF_LINKING_CONTEXT_H
#define LLD_READER_WRITER_PECOFF_LINKING_CONTEXT_H

#include "lld/Core/LinkingContext.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileUtilities.h"

#include <map>
#include <set>
#include <vector>

using llvm::COFF::MachineTypes;
using llvm::COFF::WindowsSubsystem;

static const uint8_t DEFAULT_DOS_STUB[128] = {'M', 'Z'};

namespace lld {
class Group;

class PECOFFLinkingContext : public LinkingContext {
public:
  PECOFFLinkingContext()
      : _baseAddress(invalidBaseAddress), _stackReserve(1024 * 1024),
        _stackCommit(4096), _heapReserve(1024 * 1024), _heapCommit(4096),
        _noDefaultLibAll(false), _sectionDefaultAlignment(4096),
        _subsystem(llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN),
        _machineType(llvm::COFF::IMAGE_FILE_MACHINE_I386), _imageVersion(0, 0),
        _minOSVersion(6, 0), _nxCompat(true), _largeAddressAware(false),
        _allowBind(true), _allowIsolation(true), _swapRunFromCD(false),
        _swapRunFromNet(false), _baseRelocationEnabled(true),
        _terminalServerAware(true), _dynamicBaseEnabled(true),
        _createManifest(true), _embedManifest(false), _manifestId(1),
        _manifestLevel("'asInvoker'"), _manifestUiAccess("'false'"),
        _isDll(false), _dosStub(llvm::makeArrayRef(DEFAULT_DOS_STUB)) {
    setDeadStripping(true);
  }

  struct Version {
    Version(int v1, int v2) : majorVersion(v1), minorVersion(v2) {}
    int majorVersion;
    int minorVersion;
  };

  struct ExportDesc {
    ExportDesc() : ordinal(-1), noname(false), isData(false) {}
    bool operator<(const ExportDesc &other) const {
      return name.compare(other.name) < 0;
    }

    std::string name;
    int ordinal;
    bool noname;
    bool isData;
  };

  /// \brief Casting support
  static inline bool classof(const LinkingContext *info) { return true; }

  virtual Writer &writer() const;
  virtual bool validateImpl(raw_ostream &diagnostics);

  virtual void addPasses(PassManager &pm);

  virtual bool
  createImplicitFiles(std::vector<std::unique_ptr<File> > &result) const;

  bool is64Bit() const {
    return _machineType == llvm::COFF::IMAGE_FILE_MACHINE_AMD64;
  }

  void appendInputSearchPath(StringRef dirPath) {
    _inputSearchPaths.push_back(dirPath);
  }

  const std::vector<StringRef> getInputSearchPaths() {
    return _inputSearchPaths;
  }

  void registerTemporaryFile(StringRef path) {
    std::unique_ptr<llvm::FileRemover> fileRemover(
        new llvm::FileRemover(Twine(allocate(path))));
    _tempFiles.push_back(std::move(fileRemover));
  }

  StringRef searchLibraryFile(StringRef path) const;

  StringRef decorateSymbol(StringRef name) const;
  StringRef undecorateSymbol(StringRef name) const;

  void setEntrySymbolName(StringRef name) {
    if (!name.empty())
      LinkingContext::setEntrySymbolName(decorateSymbol(name));
  }

  void setBaseAddress(uint64_t addr) { _baseAddress = addr; }
  uint64_t getBaseAddress() const;

  void setStackReserve(uint64_t size) { _stackReserve = size; }
  void setStackCommit(uint64_t size) { _stackCommit = size; }
  uint64_t getStackReserve() const { return _stackReserve; }
  uint64_t getStackCommit() const { return _stackCommit; }

  void setHeapReserve(uint64_t size) { _heapReserve = size; }
  void setHeapCommit(uint64_t size) { _heapCommit = size; }
  uint64_t getHeapReserve() const { return _heapReserve; }
  uint64_t getHeapCommit() const { return _heapCommit; }

  void setSectionDefaultAlignment(uint32_t val) {
    _sectionDefaultAlignment = val;
  }
  uint32_t getSectionDefaultAlignment() const {
    return _sectionDefaultAlignment;
  }

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

  void setCreateManifest(bool val) { _createManifest = val; }
  bool getCreateManifest() const { return _createManifest; }

  void setManifestOutputPath(std::string val) { _manifestOutputPath = val; }
  const std::string &getManifestOutputPath() const {
    return _manifestOutputPath;
  }

  void setEmbedManifest(bool val) { _embedManifest = val; }
  bool getEmbedManifest() const { return _embedManifest; }

  void setManifestId(int val) { _manifestId = val; }
  int getManifestId() const { return _manifestId; }

  void setManifestLevel(std::string val) { _manifestLevel = std::move(val); }
  const std::string &getManifestLevel() const { return _manifestLevel; }

  void setManifestUiAccess(std::string val) { _manifestUiAccess = val; }
  const std::string &getManifestUiAccess() const { return _manifestUiAccess; }

  void setManifestDependency(std::string val) { _manifestDependency = val; }
  const std::string &getManifestDependency() const {
    return _manifestDependency;
  }

  void setIsDll(bool val) { _isDll = val; }
  bool isDll() const { return _isDll; }

  StringRef getOutputSectionName(StringRef sectionName) const;
  bool addSectionRenaming(raw_ostream &diagnostics,
                          StringRef from, StringRef to);

  StringRef getAlternateName(StringRef def) const;
  const std::map<std::string, std::string> &alternateNames() {
    return _alternateNames;
  }
  void setAlternateName(StringRef def, StringRef weak);

  void addNoDefaultLib(StringRef path) { _noDefaultLibs.insert(path); }
  bool hasNoDefaultLib(StringRef path) const {
    return _noDefaultLibs.count(path) == 1;
  }

  void setNoDefaultLibAll(bool val) { _noDefaultLibAll = val; }
  bool getNoDefaultLibAll() const { return _noDefaultLibAll; }

  void setSectionSetMask(StringRef sectionName, uint32_t flags);
  void setSectionClearMask(StringRef sectionName, uint32_t flags);
  uint32_t getSectionAttributes(StringRef sectionName, uint32_t flags) const;

  void setDosStub(ArrayRef<uint8_t> data) { _dosStub = data; }
  ArrayRef<uint8_t> getDosStub() const { return _dosStub; }

  void addDllExport(ExportDesc &desc);
  std::set<ExportDesc> &getDllExports() { return _dllExports; }
  const std::set<ExportDesc> &getDllExports() const { return _dllExports; }

  StringRef allocate(StringRef ref) const {
    char *x = _allocator.Allocate<char>(ref.size() + 1);
    memcpy(x, ref.data(), ref.size());
    x[ref.size()] = '\0';
    return x;
  }

  ArrayRef<uint8_t> allocate(ArrayRef<uint8_t> array) const {
    size_t size = array.size();
    uint8_t *p = _allocator.Allocate<uint8_t>(size);
    memcpy(p, array.data(), size);
    return ArrayRef<uint8_t>(p, p + array.size());
  }

  virtual bool hasInputGraph() { return !!_inputGraph; }

  void setLibraryGroup(Group *group) { _libraryGroup = group; }
  Group *getLibraryGroup() const { return _libraryGroup; }

protected:
  /// Method to create a internal file for the entry symbol
  virtual std::unique_ptr<File> createEntrySymbolFile() const;

  /// Method to create a internal file for an undefined symbol
  virtual std::unique_ptr<File> createUndefinedSymbolFile() const;

private:
  enum {
    invalidBaseAddress = UINT64_MAX,
    pe32DefaultBaseAddress = 0x400000UL,
    pe32PlusDefaultBaseAddress = 0x140000000UL
 };

  // The start address for the program. The default value for the executable is
  // 0x400000, but can be altered using /base command line option.
  uint64_t _baseAddress;

  uint64_t _stackReserve;
  uint64_t _stackCommit;
  uint64_t _heapReserve;
  uint64_t _heapCommit;
  bool _noDefaultLibAll;
  uint32_t _sectionDefaultAlignment;
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
  bool _createManifest;
  std::string _manifestOutputPath;
  bool _embedManifest;
  int _manifestId;
  std::string _manifestLevel;
  std::string _manifestUiAccess;
  std::string _manifestDependency;
  bool _isDll;

  // The set to store /nodefaultlib arguments.
  std::set<std::string> _noDefaultLibs;

  std::vector<StringRef> _inputSearchPaths;
  std::unique_ptr<Writer> _writer;

  // A map for weak aliases.
  std::map<std::string, std::string> _alternateNames;

  // A map for section renaming. For example, if there is an entry in the map
  // whose value is .rdata -> .text, the section contens of .rdata will be
  // merged to .text in the resulting executable.
  std::map<std::string, std::string> _renamedSections;

  // Section attributes specified by /section option.
  std::map<std::string, uint32_t> _sectionSetMask;
  std::map<std::string, uint32_t> _sectionClearMask;

  // DLLExport'ed symbols.
  std::set<ExportDesc> _dllExports;

  // List of files that will be removed on destruction.
  std::vector<std::unique_ptr<llvm::FileRemover> > _tempFiles;

  // DOS Stub. DOS stub is data located at the beginning of PE/COFF file.
  // Windows loader do not really care about DOS stub contents, but it's usually
  // a small DOS program that prints out a message "This program requires
  // Microsoft Windows." This feature was somewhat useful before Windows 95.
  ArrayRef<uint8_t> _dosStub;

  // The PECOFFGroup that contains all the .lib files.
  Group *_libraryGroup;
};

} // end namespace lld

#endif
