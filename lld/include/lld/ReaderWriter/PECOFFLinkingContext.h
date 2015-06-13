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
#include "lld/Core/Reader.h"
#include "lld/Core/Writer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileUtilities.h"
#include <map>
#include <mutex>
#include <set>
#include <vector>

using llvm::COFF::MachineTypes;
using llvm::COFF::WindowsSubsystem;

static const uint8_t DEFAULT_DOS_STUB[128] = {'M', 'Z'};

namespace lld {

class PECOFFLinkingContext : public LinkingContext {
public:
  PECOFFLinkingContext() { setDeadStripping(true); }

  struct Version {
    Version(int v1, int v2) : majorVersion(v1), minorVersion(v2) {}
    int majorVersion;
    int minorVersion;
  };

  struct ExportDesc {
    ExportDesc()
        : ordinal(-1), noname(false), isData(false), isPrivate(false) {}

    bool operator<(const ExportDesc &other) const {
      return getExternalName().compare(other.getExternalName()) < 0;
    }

    StringRef getRealName() const {
      return mangledName.empty() ? name : mangledName;
    }

    StringRef getExternalName() const {
      return externalName.empty() ? name : externalName;
    }

    std::string name;
    std::string externalName;
    std::string mangledName;
    int ordinal;
    bool noname;
    bool isData;
    bool isPrivate;
  };

  typedef bool (*ParseDirectives)(int, const char **, PECOFFLinkingContext &,
                                  raw_ostream &);

  /// \brief Casting support
  static bool classof(const LinkingContext *info) { return true; }

  Writer &writer() const override;
  bool validateImpl(raw_ostream &diagnostics) override;

  void addPasses(PassManager &pm) override;

  void createImplicitFiles(std::vector<std::unique_ptr<File>> &result) override;

  bool is64Bit() const {
    return _machineType == llvm::COFF::IMAGE_FILE_MACHINE_AMD64;
  }

  // Returns a set of all defined symbols in input files.
  const std::set<std::string> &definedSymbols();

  /// Page size of x86 processor. Some data needs to be aligned at page boundary
  /// when loaded into memory.
  uint64_t getPageSize() const {
    return 0x1000;
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

  void setEntrySymbolName(StringRef name) { _entry = name; }
  StringRef getEntrySymbolName() const { return _entry; }

  void setHasEntry(bool val) { _hasEntry = val; }
  bool hasEntry() const { return _hasEntry; }

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

  void setManifestUAC(bool val) { _manifestUAC = val; }
  bool getManifestUAC() const { return _manifestUAC; }

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

  void setSafeSEH(bool val) {
    if (val)
      _requireSEH = true;
    else
      _noSEH = true;
  }
  bool requireSEH() const { return _requireSEH; }
  bool noSEH() const { return _noSEH; }

  void setHighEntropyVA(bool val) { _highEntropyVA = val; }
  bool getHighEntropyVA() const { return _highEntropyVA; }

  void setOutputImportLibraryPath(const std::string &val) { _implib = val; }
  std::string getOutputImportLibraryPath() const;

  void setDebug(bool val) { _debug = val; }
  bool getDebug() { return _debug; }

  void setPDBFilePath(StringRef str) { _pdbFilePath = str; }
  std::string getPDBFilePath() const;

  void addDelayLoadDLL(StringRef dll) {
    _delayLoadDLLs.insert(dll.lower());
  }
  bool isDelayLoadDLL(StringRef dll) const {
    return _delayLoadDLLs.count(dll.lower()) == 1;
  }

  StringRef getOutputSectionName(StringRef sectionName) const;
  bool addSectionRenaming(raw_ostream &diagnostics,
                          StringRef from, StringRef to);

  const std::set<std::string> &getAlternateNames(StringRef name) {
    return _alternateNames[name];
  }

  void addAlternateName(StringRef weak, StringRef def) {
    _alternateNames[def].insert(weak);
  }

  void addNoDefaultLib(StringRef path) {
    if (path.endswith_lower(".lib"))
      _noDefaultLibs.insert(path.drop_back(4).lower());
    else
      _noDefaultLibs.insert(path.lower());
  }

  bool hasNoDefaultLib(StringRef path) const {
    if (path.endswith_lower(".lib"))
      return _noDefaultLibs.count(path.drop_back(4).lower()) > 0;
    return _noDefaultLibs.count(path.lower()) > 0;
  }

  void setNoDefaultLibAll(bool val) { _noDefaultLibAll = val; }
  bool getNoDefaultLibAll() const { return _noDefaultLibAll; }

  void setSectionSetMask(StringRef sectionName, uint32_t flags);
  void setSectionClearMask(StringRef sectionName, uint32_t flags);
  uint32_t getSectionAttributes(StringRef sectionName, uint32_t flags) const;

  void setDosStub(ArrayRef<uint8_t> data) { _dosStub = data; }
  ArrayRef<uint8_t> getDosStub() const { return _dosStub; }

  void addDllExport(ExportDesc &desc);
  std::vector<ExportDesc> &getDllExports() { return _dllExports; }
  const std::vector<ExportDesc> &getDllExports() const { return _dllExports; }

  StringRef getDelayLoadHelperName() const {
    return is64Bit() ? "__delayLoadHelper2" : "___delayLoadHelper2@8";
  }

  llvm::BumpPtrAllocator &getAllocator() { return _allocator; }

  StringRef allocate(StringRef ref) const {
    _allocMutex.lock();
    char *x = _allocator.Allocate<char>(ref.size() + 1);
    _allocMutex.unlock();
    memcpy(x, ref.data(), ref.size());
    x[ref.size()] = '\0';
    return x;
  }

  ArrayRef<uint8_t> allocate(ArrayRef<uint8_t> array) const {
    size_t size = array.size();
    _allocMutex.lock();
    uint8_t *p = _allocator.Allocate<uint8_t>(size);
    _allocMutex.unlock();
    memcpy(p, array.data(), size);
    return ArrayRef<uint8_t>(p, p + array.size());
  }

  template <typename T> T &allocateCopy(const T &x) const {
    _allocMutex.lock();
    T *r = new (_allocator) T(x);
    _allocMutex.unlock();
    return *r;
  }

  void addLibraryFile(std::unique_ptr<FileNode> file);

  void setModuleDefinitionFile(const std::string val) {
    _moduleDefinitionFile = val;
  }
  std::string getModuleDefinitionFile() const {
    return _moduleDefinitionFile;
  }

  std::recursive_mutex &getMutex() { return _mutex; }

  void setParseDirectives(ParseDirectives parseDirectives) {
    _parseDirectives = parseDirectives;
  }

  ParseDirectives getParseDirectives() {
    return _parseDirectives;
  }

protected:
  /// Method to create a internal file for the entry symbol
  std::unique_ptr<File> createEntrySymbolFile() const override;

  /// Method to create a internal file for an undefined symbol
  std::unique_ptr<File> createUndefinedSymbolFile() const override;

private:
  enum : uint64_t {
    invalidBaseAddress = UINT64_MAX,
    pe32DefaultBaseAddress = 0x400000U,
    pe32PlusDefaultBaseAddress = 0x140000000U
  };

  std::recursive_mutex _mutex;
  mutable std::mutex _allocMutex;

  std::string _entry;

  // False if /noentry option is given.
  bool _hasEntry = true;

  // The start address for the program. The default value for the executable is
  // 0x400000, but can be altered using /base command line option.
  uint64_t _baseAddress = invalidBaseAddress;
  uint64_t _stackReserve = 1024 * 1024;
  uint64_t _stackCommit = 4096;
  uint64_t _heapReserve = 1024 * 1024;
  uint64_t _heapCommit = 4096;
  bool _noDefaultLibAll = false;
  uint32_t _sectionDefaultAlignment = 4096;
  WindowsSubsystem _subsystem = llvm::COFF::IMAGE_SUBSYSTEM_UNKNOWN;
  MachineTypes _machineType = llvm::COFF::IMAGE_FILE_MACHINE_I386;
  Version _imageVersion = {0, 0};
  Version _minOSVersion = {6, 0};
  bool _nxCompat = true;
  bool _largeAddressAware = false;
  bool _allowBind = true;
  bool _allowIsolation = true;
  bool _swapRunFromCD = false;
  bool _swapRunFromNet = false;
  bool _baseRelocationEnabled = true;
  bool _terminalServerAware = true;
  bool _dynamicBaseEnabled = true;
  bool _createManifest = true;
  std::string _manifestOutputPath;
  bool _embedManifest = false;
  int _manifestId = 1;
  bool _manifestUAC = true;
  std::string _manifestLevel = "'asInvoker'";
  std::string _manifestUiAccess = "'false'";
  std::string _manifestDependency;
  bool _isDll = false;
  bool _highEntropyVA = true;

  // True if /SAFESEH option is specified. Valid only for x86. If true, LLD will
  // produce an image with SEH table. If any modules were not compatible with
  // SEH, LLD will exit with an error.
  bool _requireSEH = false;

  // True if /SAFESEH:no option is specified. Valid only for x86. If true, LLD
  // will not produce an image with SEH table even if all input object files are
  // compatible with SEH.
  bool _noSEH = false;

  // /IMPLIB command line option.
  std::string _implib = "";

  // True if /DEBUG is given.
  bool _debug = false;

  // PDB file output path. NB: this is dummy -- LLD just creates the empty file.
  std::string _pdbFilePath = "";

  // /DELAYLOAD option.
  std::set<std::string> _delayLoadDLLs;

  // The set to store /nodefaultlib arguments.
  std::set<std::string> _noDefaultLibs;

  std::vector<StringRef> _inputSearchPaths;
  std::unique_ptr<Writer> _writer;

  // A map for weak aliases.
  std::map<std::string, std::set<std::string>> _alternateNames;

  // A map for section renaming. For example, if there is an entry in the map
  // whose value is .rdata -> .text, the section contens of .rdata will be
  // merged to .text in the resulting executable.
  std::map<std::string, std::string> _renamedSections;

  // Section attributes specified by /section option.
  std::map<std::string, uint32_t> _sectionSetMask;
  std::map<std::string, uint32_t> _sectionClearMask;

  // DLLExport'ed symbols.
  std::vector<ExportDesc> _dllExports;

  // List of files that will be removed on destruction.
  std::vector<std::unique_ptr<llvm::FileRemover> > _tempFiles;

  // DOS Stub. DOS stub is data located at the beginning of PE/COFF file.
  // Windows loader do not really care about DOS stub contents, but it's usually
  // a small DOS program that prints out a message "This program requires
  // Microsoft Windows." This feature was somewhat useful before Windows 95.
  ArrayRef<uint8_t> _dosStub = llvm::makeArrayRef(DEFAULT_DOS_STUB);

  // Name of the temporary file for lib.exe subcommand. For debugging
  // only.
  std::string _moduleDefinitionFile;

  std::set<std::string> _definedSyms;
  std::set<Node *> _seen;

  ParseDirectives _parseDirectives = nullptr;
};

} // end namespace lld

#endif
