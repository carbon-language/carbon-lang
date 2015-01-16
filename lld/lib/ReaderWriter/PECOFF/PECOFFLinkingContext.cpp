//===- lib/ReaderWriter/PECOFF/PECOFFLinkingContext.cpp -------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Atoms.h"
#include "EdataPass.h"
#include "GroupedSectionsPass.h"
#include "IdataPass.h"
#include "InferSubsystemPass.h"
#include "LinkerGeneratedSymbolFile.h"
#include "LoadConfigPass.h"
#include "PDBPass.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/Simple.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/Passes/RoundTripNativePass.h"
#include "lld/Passes/RoundTripYAMLPass.h"
#include "lld/ReaderWriter/PECOFFLinkingContext.h"
#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Path.h"
#include <bitset>
#include <climits>
#include <set>

namespace lld {

bool PECOFFLinkingContext::validateImpl(raw_ostream &diagnostics) {
  if (_stackReserve < _stackCommit) {
    diagnostics << "Invalid stack size: reserve size must be equal to or "
                << "greater than commit size, but got " << _stackCommit
                << " and " << _stackReserve << ".\n";
    return false;
  }

  if (_heapReserve < _heapCommit) {
    diagnostics << "Invalid heap size: reserve size must be equal to or "
                << "greater than commit size, but got " << _heapCommit
                << " and " << _heapReserve << ".\n";
    return false;
  }

  // It's an error if the base address is not multiple of 64K.
  if (getBaseAddress() & 0xffff) {
    diagnostics << "Base address have to be multiple of 64K, but got "
                << getBaseAddress() << "\n";
    return false;
  }

  // Check for duplicate export ordinals.
  std::set<int> exports;
  for (const PECOFFLinkingContext::ExportDesc &desc : getDllExports()) {
    if (desc.ordinal == -1)
      continue;
    if (exports.count(desc.ordinal) == 1) {
      diagnostics << "Duplicate export ordinals: " << desc.ordinal << "\n";
      return false;
    }
    exports.insert(desc.ordinal);
  }

  // Check for /align.
  std::bitset<64> alignment(_sectionDefaultAlignment);
  if (alignment.count() != 1) {
    diagnostics << "Section alignment must be a power of 2, but got "
                << _sectionDefaultAlignment << "\n";
    return false;
  }

  _writer = createWriterPECOFF(*this);
  return true;
}

const std::set<std::string> &PECOFFLinkingContext::definedSymbols() {
  std::lock_guard<std::recursive_mutex> lock(_mutex);
  for (std::unique_ptr<Node> &node : getNodes()) {
    if (_seen.count(node.get()) > 0)
      continue;
    FileNode *fnode = dyn_cast<FileNode>(node.get());
    if (!fnode)
      continue;
    File *file = fnode->getFile();
    if (file->parse())
      continue;
    if (auto *archive = dyn_cast<ArchiveLibraryFile>(file)) {
      for (const std::string &sym : archive->getDefinedSymbols())
        _definedSyms.insert(sym);
      continue;
    }
    for (const DefinedAtom *atom : file->defined())
      if (!atom->name().empty())
        _definedSyms.insert(atom->name());
  }
  return _definedSyms;
}

std::unique_ptr<File> PECOFFLinkingContext::createEntrySymbolFile() const {
  return LinkingContext::createEntrySymbolFile("<command line option /entry>");
}

std::unique_ptr<File> PECOFFLinkingContext::createUndefinedSymbolFile() const {
  return LinkingContext::createUndefinedSymbolFile(
      "<command line option /include>");
}

static int getGroupStartPos(std::vector<std::unique_ptr<Node>> &nodes) {
  for (int i = 0, e = nodes.size(); i < e; ++i)
    if (GroupEnd *group = dyn_cast<GroupEnd>(nodes[i].get()))
      return i - group->getSize();
  llvm::report_fatal_error("internal error");
}

void PECOFFLinkingContext::addLibraryFile(std::unique_ptr<FileNode> file) {
  GroupEnd *currentGroupEnd;
  int pos = -1;
  std::vector<std::unique_ptr<Node>> &elements = getNodes();
  for (int i = 0, e = elements.size(); i < e; ++i) {
    if ((currentGroupEnd = dyn_cast<GroupEnd>(elements[i].get()))) {
      pos = i;
      break;
    }
  }
  assert(pos >= 0);
  elements.insert(elements.begin() + pos, std::move(file));
  elements[pos + 1] = llvm::make_unique<GroupEnd>(
      currentGroupEnd->getSize() + 1);
}

bool PECOFFLinkingContext::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &) {
  std::vector<std::unique_ptr<Node>> &members = getNodes();

  // Create a file for the entry point function.
  std::unique_ptr<FileNode> entry(new FileNode(
      llvm::make_unique<pecoff::EntryPointFile>(*this)));
  members.insert(members.begin() + getGroupStartPos(members), std::move(entry));

  // Create a file for __ImageBase.
  std::unique_ptr<FileNode> fileNode(new FileNode(
      llvm::make_unique<pecoff::LinkerGeneratedSymbolFile>(*this)));
  members.push_back(std::move(fileNode));

  // Create a file for _imp_ symbols.
  std::unique_ptr<FileNode> impFileNode(new FileNode(
      llvm::make_unique<pecoff::LocallyImportedSymbolFile>(*this)));
  members.push_back(std::move(impFileNode));

  // Create a file for dllexported symbols.
  std::unique_ptr<FileNode> exportNode(new FileNode(
      llvm::make_unique<pecoff::ExportedSymbolRenameFile>(*this)));
  addLibraryFile(std::move(exportNode));

  return true;
}

/// Returns the section name in the resulting executable.
///
/// Sections in object files are usually output to the executable with the same
/// name, but you can rename by command line option. /merge:from=to makes the
/// linker to combine "from" section contents to "to" section in the
/// executable. We have a mapping for the renaming. This method looks up the
/// table and returns a new section name if renamed.
StringRef
PECOFFLinkingContext::getOutputSectionName(StringRef sectionName) const {
  auto it = _renamedSections.find(sectionName);
  if (it == _renamedSections.end())
    return sectionName;
  return getOutputSectionName(it->second);
}

/// Adds a mapping to the section renaming table. This method will be used for
/// /merge command line option.
bool PECOFFLinkingContext::addSectionRenaming(raw_ostream &diagnostics,
                                              StringRef from, StringRef to) {
  auto it = _renamedSections.find(from);
  if (it != _renamedSections.end()) {
    if (it->second == to)
      // There's already the same mapping.
      return true;
    diagnostics << "Section \"" << from << "\" is already mapped to \""
                << it->second << ", so it cannot be mapped to \"" << to << "\".";
    return true;
  }

  // Add a mapping, and check if there's no cycle in the renaming mapping. The
  // cycle detection algorithm we use here is naive, but that's OK because the
  // number of mapping is usually less than 10.
  _renamedSections[from] = to;
  for (auto elem : _renamedSections) {
    StringRef sectionName = elem.first;
    std::set<StringRef> visited;
    visited.insert(sectionName);
    for (;;) {
      auto pos = _renamedSections.find(sectionName);
      if (pos == _renamedSections.end())
        break;
      if (visited.count(pos->second)) {
        diagnostics << "/merge:" << from << "=" << to << " makes a cycle";
        return false;
      }
      sectionName = pos->second;
      visited.insert(sectionName);
    }
  }
  return true;
}

StringRef PECOFFLinkingContext::getAlternateName(StringRef def) const {
  auto it = _alternateNames.find(def);
  if (it == _alternateNames.end())
    return "";
  return it->second;
}

void PECOFFLinkingContext::setAlternateName(StringRef weak, StringRef def) {
  _alternateNames[def] = weak;
}

/// Try to find the input library file from the search paths and append it to
/// the input file list. Returns true if the library file is found.
StringRef PECOFFLinkingContext::searchLibraryFile(StringRef filename) const {
  // Current directory always takes precedence over the search paths.
  if (llvm::sys::path::is_absolute(filename) || llvm::sys::fs::exists(filename))
    return filename;
  // Iterate over the search paths.
  for (StringRef dir : _inputSearchPaths) {
    SmallString<128> path = dir;
    llvm::sys::path::append(path, filename);
    if (llvm::sys::fs::exists(path.str()))
      return allocate(path.str());
  }
  return filename;
}

/// Returns the decorated name of the given symbol name. On 32-bit x86, it
/// adds "_" at the beginning of the string. On other architectures, the
/// return value is the same as the argument.
StringRef PECOFFLinkingContext::decorateSymbol(StringRef name) const {
  if (_machineType != llvm::COFF::IMAGE_FILE_MACHINE_I386)
    return name;
  std::string str = "_";
  str.append(name);
  return allocate(str);
}

StringRef PECOFFLinkingContext::undecorateSymbol(StringRef name) const {
  if (_machineType != llvm::COFF::IMAGE_FILE_MACHINE_I386)
    return name;
  if (!name.startswith("_"))
    return name;
  return name.substr(1);
}

uint64_t PECOFFLinkingContext::getBaseAddress() const {
  if (_baseAddress == invalidBaseAddress)
    return is64Bit() ? pe32PlusDefaultBaseAddress : pe32DefaultBaseAddress;
  return _baseAddress;
}

Writer &PECOFFLinkingContext::writer() const { return *_writer; }

void PECOFFLinkingContext::setSectionSetMask(StringRef sectionName,
                                             uint32_t newFlags) {
  _sectionSetMask[sectionName] |= newFlags;
  _sectionClearMask[sectionName] &= ~newFlags;
  const uint32_t rwx = (llvm::COFF::IMAGE_SCN_MEM_READ |
                        llvm::COFF::IMAGE_SCN_MEM_WRITE |
                        llvm::COFF::IMAGE_SCN_MEM_EXECUTE);
  if (newFlags & rwx)
    _sectionClearMask[sectionName] |= ~_sectionSetMask[sectionName] & rwx;
  assert((_sectionSetMask[sectionName] & _sectionClearMask[sectionName]) == 0);
}

void PECOFFLinkingContext::setSectionClearMask(StringRef sectionName,
                                               uint32_t newFlags) {
  _sectionClearMask[sectionName] |= newFlags;
  _sectionSetMask[sectionName] &= ~newFlags;
  assert((_sectionSetMask[sectionName] & _sectionClearMask[sectionName]) == 0);
}

uint32_t PECOFFLinkingContext::getSectionAttributes(StringRef sectionName,
                                                    uint32_t flags) const {
  auto si = _sectionSetMask.find(sectionName);
  uint32_t setMask = (si == _sectionSetMask.end()) ? 0 : si->second;
  auto ci = _sectionClearMask.find(sectionName);
  uint32_t clearMask = (ci == _sectionClearMask.end()) ? 0 : ci->second;
  return (flags | setMask) & ~clearMask;
}

// Returns true if two export descriptors are the same.
static bool sameExportDesc(const PECOFFLinkingContext::ExportDesc &a,
                           const PECOFFLinkingContext::ExportDesc &b) {
  return a.ordinal == b.ordinal && a.ordinal == b.ordinal &&
         a.noname == b.noname && a.isData == b.isData;
}

void PECOFFLinkingContext::addDllExport(ExportDesc &desc) {
  addInitialUndefinedSymbol(allocate(desc.name));

  // MSVC link.exe silently drops characters after the first atsign.
  // For example, /export:foo@4=bar is equivalent to /export:foo=bar.
  // We do the same thing for compatibility.
  if (!desc.externalName.empty()) {
    StringRef s(desc.externalName);
    size_t pos = s.find('@');
    if (pos != s.npos)
      desc.externalName = s.substr(0, pos);
  }

  // Scan the vector to look for existing entry. It's not very fast,
  // but because the number of exported symbol is usually not that
  // much, it should be okay.
  for (ExportDesc &e : _dllExports) {
    if (e.name != desc.name)
      continue;
    if (!sameExportDesc(e, desc))
      llvm::errs() << "Export symbol '" << desc.name
                   << "' specified more than once.\n";
    return;
  }
  _dllExports.push_back(desc);
}

static std::string replaceExtension(StringRef path, StringRef ext) {
  SmallString<128> ss = path;
  llvm::sys::path::replace_extension(ss, ext);
  return ss.str();
}

std::string PECOFFLinkingContext::getOutputImportLibraryPath() const {
  if (!_implib.empty())
    return _implib;
  return replaceExtension(outputPath(), ".lib");
}

std::string PECOFFLinkingContext::getPDBFilePath() const {
  assert(_debug);
  if (!_pdbFilePath.empty())
    return _pdbFilePath;
  return replaceExtension(outputPath(), ".pdb");
}

void PECOFFLinkingContext::addPasses(PassManager &pm) {
  pm.add(std::unique_ptr<Pass>(new pecoff::PDBPass(*this)));
  pm.add(std::unique_ptr<Pass>(new pecoff::EdataPass(*this)));
  pm.add(std::unique_ptr<Pass>(new pecoff::IdataPass(*this)));
  pm.add(std::unique_ptr<Pass>(new LayoutPass(registry())));
  pm.add(std::unique_ptr<Pass>(new pecoff::LoadConfigPass(*this)));
  pm.add(std::unique_ptr<Pass>(new pecoff::GroupedSectionsPass()));
  pm.add(std::unique_ptr<Pass>(new pecoff::InferSubsystemPass(*this)));
}

} // end namespace lld
