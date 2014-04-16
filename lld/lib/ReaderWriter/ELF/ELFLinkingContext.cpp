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

// An AliasAtom is a zero-size atom representing an alias for other atom. It has
// a LayoutAfter reference to the target atom, so that this atom and the target
// atom will be layed out at the same location in the final result. Initially
// the target atom is an undefined atom. Resolver will replace it with a defined
// one.
//
// It does not have attributes itself. Most member function calls are forwarded
// to the target atom.
class AliasAtom : public SimpleDefinedAtom {
public:
  AliasAtom(const File &file, StringRef name)
      : SimpleDefinedAtom(file), _target(nullptr), _name(name) {}

  StringRef name() const override { return _name; }
  uint64_t size() const override { return 0; }
  ArrayRef<uint8_t> rawContent() const override { return ArrayRef<uint8_t>(); }

  Scope scope() const override {
    getTarget();
    return _target ? _target->scope() : scopeLinkageUnit;
  }

  Merge merge() const override {
    getTarget();
    return _target ? _target->merge() : mergeNo;
  }

  ContentType contentType() const override {
    getTarget();
    return _target ? _target->contentType() : typeUnknown;
  }

  Interposable interposable() const override {
    getTarget();
    return _target ? _target->interposable() : interposeNo;
  }

  SectionChoice sectionChoice() const override {
    getTarget();
    return _target ? _target->sectionChoice() : sectionBasedOnContent;
  }

  StringRef customSectionName() const override {
    getTarget();
    return _target ? _target->customSectionName() : StringRef("");
  }

private:
  void getTarget() const {
    if (_target)
      return;
    for (const Reference *r : *this) {
      if (r->kindNamespace() == lld::Reference::KindNamespace::all &&
          r->kindValue() == lld::Reference::kindLayoutAfter) {
        _target = dyn_cast<DefinedAtom>(r->target());
        return;
      }
    }
  }

  mutable const DefinedAtom *_target;
  StringRef _name;
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
      _noAllowDynamicLibraries(false), _outputMagic(OutputMagic::DEFAULT),
      _sysrootPath("") {}

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
  default:
    return nullptr;
  }
}

ErrorOr<StringRef> ELFLinkingContext::searchLibrary(StringRef libName) const {
  bool foundFile = false;
  StringRef pathref;
  SmallString<128> path;
  for (StringRef dir : _inputSearchPaths) {
    // Search for dynamic library
    if (!_isStaticExecutable) {
      path.clear();
      if (dir.startswith("=/")) {
        path.assign(_sysrootPath);
        path.append(dir.substr(1));
      } else {
        path.assign(dir);
      }
      llvm::sys::path::append(path, Twine("lib") + libName + ".so");
      pathref = path.str();
      if (llvm::sys::fs::exists(pathref)) {
        foundFile = true;
      }
    }
    // Search for static libraries too
    if (!foundFile) {
      path.clear();
      if (dir.startswith("=/")) {
        path.assign(_sysrootPath);
        path.append(dir.substr(1));
      } else {
        path.assign(dir);
      }
      llvm::sys::path::append(path, Twine("lib") + libName + ".a");
      pathref = path.str();
      if (llvm::sys::fs::exists(pathref)) {
        foundFile = true;
      }
    }
    if (foundFile)
      return StringRef(*new (_allocator) std::string(pathref));
  }
  if (!llvm::sys::fs::exists(libName))
    return llvm::make_error_code(llvm::errc::no_such_file_or_directory);

  return libName;
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
  for (auto &i : getAliases()) {
    StringRef from = i.first;
    StringRef to = i.second;
    SimpleDefinedAtom *fromAtom = new (_allocator) AliasAtom(*file, from);
    UndefinedAtom *toAtom = new (_allocator) SimpleUndefinedAtom(*file, to);
    fromAtom->addReference(Reference::KindNamespace::all,
                           Reference::KindArch::all, Reference::kindLayoutAfter,
                           0, toAtom, 0);
    file->addAtom(*fromAtom);
    file->addAtom(*toAtom);
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

} // end namespace lld
