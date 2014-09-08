//===- lib/ReaderWriter/ELF/ELFLinkingContext.cpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/ELFLinkingContext.h"

#include "ArrayOrderPass.h"
#include "ELFFile.h"
#include "TargetHandler.h"
#include "Targets.h"

#include "lld/Core/Instrumentation.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/Passes/RoundTripYAMLPass.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace lld {

class CommandLineAbsoluteAtom : public AbsoluteAtom {
public:
  CommandLineAbsoluteAtom(const File &file, StringRef name, uint64_t value)
      : _file(file), _name(name), _value(value) {}

  const File &file() const override { return _file; }
  StringRef name() const override { return _name; }
  uint64_t value() const override { return _value; }
  Scope scope() const override { return scopeGlobal; }

private:
  const File &_file;
  StringRef _name;
  uint64_t _value;
};

class CommandLineUndefinedAtom : public SimpleUndefinedAtom {
public:
  CommandLineUndefinedAtom(const File &f, StringRef name)
      : SimpleUndefinedAtom(f, name) {}

  CanBeNull canBeNull() const override {
    return CanBeNull::canBeNullAtBuildtime;
  }
};

ELFLinkingContext::ELFLinkingContext(
    llvm::Triple triple, std::unique_ptr<TargetHandlerBase> targetHandler)
    : _outputELFType(elf::ET_EXEC), _triple(triple),
      _targetHandler(std::move(targetHandler)), _baseAddress(0),
      _isStaticExecutable(false), _noInhibitExec(false),
      _mergeCommonStrings(false), _runLayoutPass(true),
      _useShlibUndefines(true), _dynamicLinkerArg(false),
      _noAllowDynamicLibraries(false), _mergeRODataToTextSegment(true),
      _outputMagic(OutputMagic::DEFAULT), _sysrootPath("") {}

bool ELFLinkingContext::is64Bits() const { return getTriple().isArch64Bit(); }

bool ELFLinkingContext::isLittleEndian() const {
  // TODO: Do this properly. It is not defined purely by arch.
  return true;
}

void ELFLinkingContext::addPasses(PassManager &pm) {
  if (_runLayoutPass)
    pm.add(std::unique_ptr<Pass>(new LayoutPass(registry())));
  pm.add(std::unique_ptr<Pass>(new elf::ArrayOrderPass()));
}

uint16_t ELFLinkingContext::getOutputMachine() const {
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
    return llvm::ELF::EM_386;
  case llvm::Triple::x86_64:
    return llvm::ELF::EM_X86_64;
  case llvm::Triple::hexagon:
    return llvm::ELF::EM_HEXAGON;
  case llvm::Triple::mipsel:
    return llvm::ELF::EM_MIPS;
  case llvm::Triple::ppc:
    return llvm::ELF::EM_PPC;
  case llvm::Triple::aarch64:
    return llvm::ELF::EM_AARCH64;
  default:
    llvm_unreachable("Unhandled arch");
  }
}

StringRef ELFLinkingContext::entrySymbolName() const {
  if (_outputELFType == elf::ET_EXEC && _entrySymbolName.empty())
    return "_start";
  return _entrySymbolName;
}

bool ELFLinkingContext::validateImpl(raw_ostream &diagnostics) {
  switch (outputFileType()) {
  case LinkingContext::OutputFileType::YAML:
    _writer = createWriterYAML(*this);
    break;
  case LinkingContext::OutputFileType::Native:
    llvm_unreachable("Unimplemented");
    break;
  default:
    _writer = createWriterELF(this->targetHandler());
    break;
  }
  return true;
}

bool ELFLinkingContext::isDynamic() const {
  switch (_outputELFType) {
  case llvm::ELF::ET_EXEC:
    return !_isStaticExecutable;
  case llvm::ELF::ET_DYN:
    return true;
  }
  return false;
}

bool ELFLinkingContext::isRelativeReloc(const Reference &) const {
  return false;
}

Writer &ELFLinkingContext::writer() const { return *_writer; }

std::unique_ptr<ELFLinkingContext>
ELFLinkingContext::create(llvm::Triple triple) {
  switch (triple.getArch()) {
  case llvm::Triple::x86:
    return std::unique_ptr<ELFLinkingContext>(
        new lld::elf::X86LinkingContext(triple));
  case llvm::Triple::x86_64:
    return std::unique_ptr<ELFLinkingContext>(
        new lld::elf::X86_64LinkingContext(triple));
  case llvm::Triple::hexagon:
    return std::unique_ptr<ELFLinkingContext>(
        new lld::elf::HexagonLinkingContext(triple));
  case llvm::Triple::mipsel:
    return std::unique_ptr<ELFLinkingContext>(
        new lld::elf::MipsLinkingContext(triple));
  case llvm::Triple::ppc:
    return std::unique_ptr<ELFLinkingContext>(
        new lld::elf::PPCLinkingContext(triple));
  case llvm::Triple::aarch64:
    return std::unique_ptr<ELFLinkingContext>(
        new lld::elf::AArch64LinkingContext(triple));
  default:
    return nullptr;
  }
}

static void buildSearchPath(SmallString<128> &path, StringRef dir,
                            StringRef sysRoot) {
  if (!dir.startswith("=/"))
    path.assign(dir);
  else {
    path.assign(sysRoot);
    path.append(dir.substr(1));
  }
}

ErrorOr<StringRef> ELFLinkingContext::searchLibrary(StringRef libName) const {
  bool hasColonPrefix = libName[0] == ':';
  Twine soName =
      hasColonPrefix ? libName.drop_front() : Twine("lib", libName) + ".so";
  Twine archiveName =
      hasColonPrefix ? libName.drop_front() : Twine("lib", libName) + ".a";
  SmallString<128> path;
  for (StringRef dir : _inputSearchPaths) {
    // Search for dynamic library
    if (!_isStaticExecutable) {
      buildSearchPath(path, dir, _sysrootPath);
      llvm::sys::path::append(path, soName);
      if (llvm::sys::fs::exists(path.str()))
        return StringRef(*new (_allocator) std::string(path.str()));
    }
    // Search for static libraries too
    buildSearchPath(path, dir, _sysrootPath);
    llvm::sys::path::append(path, archiveName);
    if (llvm::sys::fs::exists(path.str()))
      return StringRef(*new (_allocator) std::string(path.str()));
  }
  if (!llvm::sys::fs::exists(libName))
    return make_error_code(llvm::errc::no_such_file_or_directory);

  return libName;
}

ErrorOr<StringRef> ELFLinkingContext::searchFile(StringRef fileName,
                                                 bool isSysRooted) const {
  SmallString<128> path;
  if (llvm::sys::path::is_absolute(fileName) && isSysRooted) {
    path.assign(_sysrootPath);
    path.append(fileName);
    if (llvm::sys::fs::exists(path.str()))
      return StringRef(*new (_allocator) std::string(path.str()));
  } else if (llvm::sys::fs::exists(fileName))
    return fileName;

  if (llvm::sys::path::is_absolute(fileName))
    return make_error_code(llvm::errc::no_such_file_or_directory);

  for (StringRef dir : _inputSearchPaths) {
    buildSearchPath(path, dir, _sysrootPath);
    llvm::sys::path::append(path, fileName);
    if (llvm::sys::fs::exists(path.str()))
      return StringRef(*new (_allocator) std::string(path.str()));
  }
  return make_error_code(llvm::errc::no_such_file_or_directory);
}

void ELFLinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File>> &files) const {
  std::unique_ptr<SimpleFile> file(
      new SimpleFile("<internal file for --defsym>"));
  for (auto &i : getAbsoluteSymbols()) {
    StringRef sym = i.first;
    uint64_t val = i.second;
    file->addAtom(*(new (_allocator) CommandLineAbsoluteAtom(*file, sym, val)));
  }
  files.push_back(std::move(file));
  LinkingContext::createInternalFiles(files);
}

std::unique_ptr<File> ELFLinkingContext::createUndefinedSymbolFile() const {
  if (_initialUndefinedSymbols.empty())
    return nullptr;
  std::unique_ptr<SimpleFile> undefinedSymFile(
      new SimpleFile("command line option -u"));
  for (auto undefSymStr : _initialUndefinedSymbols)
    undefinedSymFile->addAtom(*(new (_allocator) CommandLineUndefinedAtom(
        *undefinedSymFile, undefSymStr)));
  return std::move(undefinedSymFile);
}

static bool isSharedWeakAtom(const UndefinedAtom *ua) {
  return ua->canBeNull() != UndefinedAtom::canBeNullNever &&
         isa<SharedLibraryFile>(ua->file());
}

void ELFLinkingContext::notifySymbolTableCoalesce(const Atom *existingAtom,
                                                  const Atom *newAtom,
                                                  bool &useNew) {
  // First suppose that the `existingAtom` is defined
  // and the `newAtom` is undefined.
  auto *da = dyn_cast<DefinedAtom>(existingAtom);
  auto *ua = dyn_cast<UndefinedAtom>(newAtom);
  if (!da && !ua) {
    // Then try to reverse the assumption.
    da = dyn_cast<DefinedAtom>(newAtom);
    ua = dyn_cast<UndefinedAtom>(existingAtom);
  }

  if (da && ua && da->scope() == Atom::scopeGlobal && isSharedWeakAtom(ua))
    // If strong defined atom coalesces away weak atom declared
    // in the shared object the strong atom needs to be dynamicaly exported.
    // Save its name.
    _weakCoalescedSymbols.insert(ua->name());
}

} // end namespace lld
