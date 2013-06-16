//===- lib/ReaderWriter/ELF/ELFTargetInfo.cpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/ELFTargetInfo.h"

#include "TargetHandler.h"
#include "Targets.h"

#include "lld/Core/Instrumentation.h"
#include "lld/Passes/LayoutPass.h"
#include "lld/ReaderWriter/ReaderLinkerScript.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace lld {
ELFTargetInfo::ELFTargetInfo(llvm::Triple triple,
                             std::unique_ptr<TargetHandlerBase> targetHandler)
    : _outputFileType(elf::ET_EXEC),
      _triple(triple),
      _targetHandler(std::move(targetHandler)),
      _baseAddress(0),
      _isStaticExecutable(false),
      _outputYAML(false),
      _noInhibitExec(false),
      _mergeCommonStrings(false),
      _runLayoutPass(true),
      _useShlibUndefines(false),
      _dynamicLinkerArg(false),
      _outputMagic(OutputMagic::DEFAULT) {}

bool ELFTargetInfo::is64Bits() const { return getTriple().isArch64Bit(); }

bool ELFTargetInfo::isLittleEndian() const {
  // TODO: Do this properly. It is not defined purely by arch.
  return true;
}

void ELFTargetInfo::addPasses(PassManager &pm) const {
  if (_runLayoutPass)
    pm.add(std::unique_ptr<Pass>(new LayoutPass()));
}

uint16_t ELFTargetInfo::getOutputMachine() const {
  switch (getTriple().getArch()) {
  case llvm::Triple::x86:
    return llvm::ELF::EM_386;
  case llvm::Triple::x86_64:
    return llvm::ELF::EM_X86_64;
  case llvm::Triple::hexagon:
    return llvm::ELF::EM_HEXAGON;
  case llvm::Triple::ppc:
    return llvm::ELF::EM_PPC;
  default:
    llvm_unreachable("Unhandled arch");
  }
}

bool ELFTargetInfo::validateImpl(raw_ostream &diagnostics) {
  if (_outputFileType == elf::ET_EXEC &&
      _entrySymbolName.empty()) {
    _entrySymbolName = "_start";
  }

  if (_inputFiles.empty()) {
    diagnostics << "No input files\n";
    return true;
  }

  _elfReader = createReaderELF(*this);
  _linkerScriptReader.reset(new ReaderLinkerScript(*this));
  _writer = _outputYAML ? createWriterYAML(*this) : createWriterELF(*this);
  return false;
}

bool ELFTargetInfo::isDynamic() const {
  switch (_outputFileType) {
  case llvm::ELF::ET_EXEC:
    return !_isStaticExecutable;
  case llvm::ELF::ET_DYN:
    return true;
  }
  return false;
}

bool ELFTargetInfo::isRelativeReloc(const Reference &) const { return false; }

error_code
ELFTargetInfo::parseFile(std::unique_ptr<MemoryBuffer> &mb,
                         std::vector<std::unique_ptr<File> > &result) const {
  ScopedTask task(getDefaultDomain(), "parseFile");
  error_code ec = _elfReader->parseFile(mb, result);
  if (!ec)
    return ec;

  // Not an ELF file, check file extension to see if it might be yaml
  StringRef path = mb->getBufferIdentifier();
  if (path.endswith(".objtxt")) {
    ec = _yamlReader->parseFile(mb, result);
    if (!ec)
      return ec;
  }

  // Not a yaml file, assume it is a linkerscript
  return _linkerScriptReader->parseFile(mb, result);
}

Writer &ELFTargetInfo::writer() const {
  return *_writer;
}


std::unique_ptr<ELFTargetInfo> ELFTargetInfo::create(llvm::Triple triple) {
  switch (triple.getArch()) {
  case llvm::Triple::x86:
    return std::unique_ptr<ELFTargetInfo>(new lld::elf::X86TargetInfo(triple));
  case llvm::Triple::x86_64:
    return std::unique_ptr<
        ELFTargetInfo>(new lld::elf::X86_64TargetInfo(triple));
  case llvm::Triple::hexagon:
    return std::unique_ptr<
        ELFTargetInfo>(new lld::elf::HexagonTargetInfo(triple));
  case llvm::Triple::ppc:
    return std::unique_ptr<ELFTargetInfo>(new lld::elf::PPCTargetInfo(triple));
  default:
    return nullptr;
  }
}

bool ELFTargetInfo::appendLibrary(StringRef libName) {
  bool foundFile = false;
  StringRef pathref;
  for (StringRef dir : _inputSearchPaths) {
    // Search for dynamic library
    if (!_isStaticExecutable) {
      SmallString<128> dynlibPath;
      dynlibPath.assign(dir);
      llvm::sys::path::append(dynlibPath, Twine("lib") + libName + ".so");
      pathref = dynlibPath.str();
      if (llvm::sys::fs::exists(pathref)) {
        foundFile = true;
      }
    }
    // Search for static libraries too
    if (!foundFile) {
      SmallString<128> archivefullPath;
      archivefullPath.assign(dir);
      llvm::sys::path::append(archivefullPath, Twine("lib") + libName + ".a");
      pathref = archivefullPath.str();
      if (llvm::sys::fs::exists(pathref)) {
        foundFile = true;
      }
    }
    if (foundFile) {
      unsigned pathlen = pathref.size();
      char *x = _extraStrings.Allocate<char>(pathlen);
      memcpy(x, pathref.data(), pathlen);
      appendInputFile(StringRef(x, pathlen));
      return false;
    }
  }
  return true;
}

} // end namespace lld
