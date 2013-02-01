//===- lib/ReaderWriter/ELF/X86_64/X86_64TargetHandler.cpp ----------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86_64TargetHandler.h"

#include "X86_64TargetInfo.h"

using namespace lld;
using namespace elf;

using namespace llvm::ELF;

X86_64TargetHandler::X86_64TargetHandler(X86_64TargetInfo &targetInfo)
    : DefaultTargetHandler(targetInfo), _gotFile(targetInfo),
      _relocationHandler(targetInfo), _targetLayout(targetInfo) {}

namespace {
/// \brief R_X86_64_64 - word64: S + A
void reloc64(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint64_t result = S + A;
  *reinterpret_cast<llvm::support::ulittle64_t *>(location) = result |
            (uint64_t)*reinterpret_cast<llvm::support::ulittle64_t *>(location);
}

/// \brief R_X86_64_PC32 - word32: S + A - P
void relocPC32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  uint32_t result = (uint32_t)((S + A) - P);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result +
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
}

/// \brief R_X86_64_32 - word32:  S + A
void reloc32(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  int32_t result = (uint32_t)(S + A);
  *reinterpret_cast<llvm::support::ulittle32_t *>(location) = result |
            (uint32_t)*reinterpret_cast<llvm::support::ulittle32_t *>(location);
  // TODO: Make sure that the result zero extends to the 64bit value.
}

/// \brief R_X86_64_32S - word32:  S + A
void reloc32S(uint8_t *location, uint64_t P, uint64_t S, int64_t A) {
  int32_t result = (int32_t)(S + A);
  *reinterpret_cast<llvm::support::little32_t *>(location) = result |
            (int32_t)*reinterpret_cast<llvm::support::little32_t *>(location);
  // TODO: Make sure that the result sign extends to the 64bit value.
}
} // end anon namespace

ErrorOr<void> X86_64TargetRelocationHandler::applyRelocation(
    ELFWriter &writer, llvm::FileOutputBuffer &buf, const AtomLayout &atom,
    const Reference &ref) const {
  uint8_t *atomContent = buf.getBufferStart() + atom._fileOffset;
  uint8_t *location = atomContent + ref.offsetInAtom();
  uint64_t targetVAddress = writer.addressOfAtom(ref.target());
  uint64_t relocVAddress = atom._virtualAddr + ref.offsetInAtom();

  switch (ref.kind()) {
  case R_X86_64_64:
    reloc64(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_X86_64_PC32:
    relocPC32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_X86_64_32:
    reloc32(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_X86_64_32S:
    reloc32S(location, relocVAddress, targetVAddress, ref.addend());
    break;
  case R_X86_64_TPOFF64:
  case R_X86_64_TPOFF32: {
    // Get the start and end of the TLS segment.
    if (_tlsSize == 0) {
      auto tdata = _targetInfo.getTargetHandler<X86_64ELFType>().targetLayout()
          .findOutputSection(".tdata");
      auto tbss = _targetInfo.getTargetHandler<X86_64ELFType>().targetLayout()
          .findOutputSection(".tbss");
      // HACK: The tdata and tbss sections end up together to from the TLS
      // segment. This should actually use the TLS program header entry.
      if (tdata)
        _tlsSize = tdata->memSize();
      if (tbss)
        _tlsSize += tbss->memSize();
    }
    if (ref.kind() == R_X86_64_TPOFF32) {
      int32_t result = (int32_t)(targetVAddress - _tlsSize);
      *reinterpret_cast<llvm::support::little32_t *>(location) = result;
    } else {
      int64_t result = (int64_t)(targetVAddress - _tlsSize);
      *reinterpret_cast<llvm::support::little64_t *>(location) = result;
    }
    break;
  }
  // Runtime only relocations. Ignore here.
  case R_X86_64_IRELATIVE:
    break;
  default: {
    std::string str;
    llvm::raw_string_ostream s(str);
    auto name = _targetInfo.stringFromRelocKind(ref.kind());
    s << "Unhandled relocation: " << atom._atom->file().path() << ":"
      << atom._atom->name() << "@" << ref.offsetInAtom() << " "
      << (name ? *name : "<unknown>") << " (" << ref.kind() << ")";
    s.flush();
    llvm_unreachable(str.c_str());
  }
  }

  return error_code::success();
}

namespace {
class GLOBAL_OFFSET_TABLEAtom : public SimpleDefinedAtom {
public:
  GLOBAL_OFFSET_TABLEAtom(const File &f) : SimpleDefinedAtom(f) {}

  virtual StringRef name() const { return "_GLOBAL_OFFSET_TABLE_"; }

  virtual Scope scope() const { return scopeGlobal; }

  virtual SectionChoice sectionChoice() const { return sectionCustomRequired; }

  virtual StringRef customSectionName() const { return ".got"; }

  virtual ContentType contentType() const { return typeGOT; }

  virtual uint64_t size() const { return 0; }

  virtual ContentPermissions permissions() const { return permRW_; }

  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>();
  }
};
} // end anon namespace

void X86_64TargetHandler::addFiles(InputFiles &f) {
  _gotFile.addAtom(*new (_gotFile._alloc) GLOBAL_OFFSET_TABLEAtom(_gotFile));
  f.appendFile(_gotFile);
}
