//===- lib/ReaderWriter/YAML/ReaderWriterYAML.cpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lld/ReaderWriter/Reader.h"
#include "lld/ReaderWriter/Writer.h"

#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/DefinedAtom.h"
#include "lld/Core/Error.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Reference.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/YAMLTraits.h"

#include <string>

using llvm::yaml::MappingTraits;
using llvm::yaml::ScalarEnumerationTraits;
using llvm::yaml::ScalarTraits;
using llvm::yaml::IO;
using llvm::yaml::SequenceTraits;
using llvm::yaml::DocumentListTraits;

using namespace lld;

/// The conversion of Atoms to and from YAML uses LLVM's YAML I/O.  This
/// file just defines template specializations on the lld types which control
/// how the mapping is done to and from YAML.


namespace {
/// Most of the traits are context-free and always do the same transformation.
/// But, there are some traits that need some contextual information to properly
/// do their transform.  This struct is available via io.getContext() and
/// supplies contextual information.
class ContextInfo {
public:
  ContextInfo(const TargetInfo &ti) : _currentFile(nullptr), _targetInfo(ti) {}

  lld::File       *_currentFile;
  const TargetInfo &_targetInfo;
};

/// Used when writing yaml files.
/// In most cases, atoms names are unambiguous, so references can just
/// use the atom name as the target (e.g. target: foo).  But in a few
/// cases that does not work, so ref-names are added.  These are labels
/// used only in yaml.  The labels do not exist in the Atom model.
///
/// One need for ref-names are when atoms have no user supplied name
/// (e.g. c-string literal).  Another case is when two object files with
/// identically named static functions are merged (ld -r) into one object file.
/// In that case referencing the function by name is ambiguous, so a unique
/// ref-name is added.
class RefNameBuilder {
public:
  RefNameBuilder(const lld::File &file)
                                      : _collisionCount(0), _unnamedCounter(0) {
    if (&file == nullptr)
      return;
    // visit all atoms
    for (const lld::DefinedAtom *atom : file.defined()) {
      // Build map of atoms names to detect duplicates
      if (!atom->name().empty())
        buildDuplicateNameMap(*atom);

      // Find references to unnamed atoms and create ref-names for them.
      for (const lld::Reference *ref : *atom) {
        // create refname for any unnamed reference target
        const lld::Atom *target = ref->target();
        if ((target != nullptr) && target->name().empty()) {
          std::string storage;
          llvm::raw_string_ostream buffer(storage);
          buffer << llvm::format("L%03d", _unnamedCounter++);
          llvm::StringRef newName = copyString(buffer.str());
          _refNames[target] = newName;
          DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
                  << "unnamed atom: creating ref-name: '" << newName
                  << "' (" << (void*)newName.data() << ", "
                  << newName.size() << ")\n");
        }
      }
    }
    for (const lld::UndefinedAtom *undefAtom : file.undefined()) {
      buildDuplicateNameMap(*undefAtom);
    }
    for (const lld::SharedLibraryAtom *shlibAtom : file.sharedLibrary()) {
      buildDuplicateNameMap(*shlibAtom);
    }
    for (const lld::AbsoluteAtom *absAtom : file.absolute()) {
      buildDuplicateNameMap(*absAtom);
    }
  }

  void buildDuplicateNameMap(const lld::Atom &atom) {
    assert(!atom.name().empty());
    NameToAtom::iterator pos = _nameMap.find(atom.name());
    if ( pos != _nameMap.end() ) {
      // Found name collision, give each a unique ref-name.
      std::string Storage;
      llvm::raw_string_ostream buffer(Storage);
      buffer << atom.name() << llvm::format(".%03d", ++_collisionCount);
      llvm::StringRef newName = copyString(buffer.str());
      _refNames[&atom] = newName;
      DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
                  << "name collsion: creating ref-name: '"  << newName
                  << "' (" << (void*)newName.data() << ", "
                  << newName.size() << ")\n");
      const lld::Atom *prevAtom = pos->second;
      AtomToRefName::iterator pos2 = _refNames.find(prevAtom);
      if ( pos2 == _refNames.end() ) {
        // Only create ref-name for previous if none already created.
        std::string Storage2;
        llvm::raw_string_ostream buffer2(Storage2);
        buffer2 << prevAtom->name() << llvm::format(".%03d", ++_collisionCount);
        llvm::StringRef newName2 = copyString(buffer2.str());
        _refNames[prevAtom] = newName2;
        DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
                  << "name collsion: creating ref-name: '" << newName2
                  << "' (" << (void*)newName2.data() << ", "
                  << newName2.size() << ")\n");
      }
    }
    else {
      // First time we've seen this name, just add it to map.
      _nameMap[atom.name()] = &atom;
      DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
                  << "atom name seen for first time: '" << atom.name()
                  << "' (" << (void*)atom.name().data() << ", "
                  << atom.name().size() << ")\n");
    }
  }

  bool hasRefName(const lld::Atom *atom) {
     return _refNames.count(atom);
  }

  llvm::StringRef refName(const lld::Atom *atom) {
     return _refNames.find(atom)->second;
  }

private:
  typedef llvm::StringMap<const lld::Atom*> NameToAtom;
  typedef llvm::DenseMap<const lld::Atom*, std::string> AtomToRefName;

  // Allocate a new copy of this string and keep track of allocations
  // in _stringCopies, so they can be freed when RefNameBuilder is destroyed.
  llvm::StringRef copyString(llvm::StringRef str) {
     // We want _stringCopies to own the string memory so it is deallocated
    // when the File object is destroyed.  But we need a StringRef that
    // points into that memory.
    std::unique_ptr<char[]> s(new char[str.size()]);
    memcpy(s.get(), str.data(), str.size());
    llvm::StringRef r = llvm::StringRef(s.get(), str.size());
    _stringCopies.push_back(std::move(s));
    return r;
  }

  unsigned int                        _collisionCount;
  unsigned int                        _unnamedCounter;
  NameToAtom                          _nameMap;
  AtomToRefName                       _refNames;
  std::vector<std::unique_ptr<char[]>>  _stringCopies;
};


/// Used when reading yaml files to find the target of a reference
/// that could be a name or ref-name.
class RefNameResolver {
public:
  RefNameResolver(const lld::File *file, IO &io);

  const lld::Atom *lookup(llvm::StringRef name) const {
    NameToAtom::const_iterator pos = _nameMap.find(name);
    if (pos != _nameMap.end()) {
      return pos->second;
    }
    else {
      _io.setError(llvm::Twine("no such atom name: ") + name);
      return nullptr;
    }
  }

private:
  typedef llvm::StringMap<const lld::Atom*> NameToAtom;

  void add(llvm::StringRef name, const lld::Atom *atom) {
    if (_nameMap.count(name)) {
      _io.setError(llvm::Twine("duplicate atom name: ") + name);
    }
    else {
      _nameMap[name] = atom;
    }
  }

  IO               &_io;
  NameToAtom        _nameMap;
};


// Used in NormalizedFile to hold the atoms lists.
template <typename T>
class AtomList : public lld::File::atom_collection<T> {
public:
  virtual lld::File::atom_iterator<T> begin() const {
    return lld::File::atom_iterator<
        T>(*this,
           _atoms.empty() ? 0 : reinterpret_cast<const void *>(_atoms.data()));
  }
  virtual lld::File::atom_iterator<T> end() const{
    return lld::File::atom_iterator<
        T>(*this, _atoms.empty() ? 0 :
               reinterpret_cast<const void *>(_atoms.data() + _atoms.size()));
  }
  virtual const T *deref(const void *it) const {
    return *reinterpret_cast<const T *const*>(it);
  }
  virtual void next(const void *&it) const {
    const T *const *p = reinterpret_cast<const T *const *>(it);
    ++p;
    it = reinterpret_cast<const void*>(p);
  }
  virtual void push_back(const T *element) {
    _atoms.push_back(element);
  }
  std::vector<const T*>   _atoms;
};

/// Mapping of kind: field in yaml files.
enum FileKinds {
    fileKindObjectAtoms,  // atom based object file encoded in yaml
    fileKindArchive,      // static archive library encoded in yaml
    fileKindObjectELF,    // ELF object files encoded in yaml
    fileKindObjectMachO   // mach-o object files encoded in yaml
};

struct ArchMember {
  FileKinds           _kind;
  llvm::StringRef     _name;
  const lld::File    *_content;
};


// The content bytes in a DefinedAtom are just uint8_t but we want
// special formatting, so define a strong type.
LLVM_YAML_STRONG_TYPEDEF(uint8_t, ImplicitHex8)

// SharedLibraryAtoms have a bool canBeNull() method which we'd like to be
// more readable than just true/false.
LLVM_YAML_STRONG_TYPEDEF(bool, ShlibCanBeNull)

// lld::Reference::Kind is a typedef of int32.  We need a stronger
// type to make template matching work, so invent RefKind.
LLVM_YAML_STRONG_TYPEDEF(lld::Reference::Kind, RefKind)

} // namespace anon

LLVM_YAML_IS_SEQUENCE_VECTOR(ArchMember)
    LLVM_YAML_IS_SEQUENCE_VECTOR(const lld::Reference *)
    // Always write DefinedAtoms content bytes as a flow sequence.
    LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(ImplicitHex8)
    // for compatibility with gcc-4.7 in C++11 mode, add extra namespace
    namespace llvm {
  namespace yaml {

// This is a custom formatter for RefKind
template<>
struct ScalarTraits<RefKind> {
  static void output(const RefKind &value, void *ctxt,
                                                      llvm::raw_ostream &out) {
    assert(ctxt != nullptr);
    ContextInfo *info = reinterpret_cast<ContextInfo*>(ctxt);
    switch (value) {
    case lld::Reference::kindLayoutAfter:
      out << "layout-after";
      break;
    case lld::Reference::kindLayoutBefore:
      out << "layout-before";
      break;
    case lld::Reference::kindInGroup:
      out << "in-group";
      break;
    default:
      if (auto relocStr = info->_targetInfo.stringFromRelocKind(value))
        out << *relocStr;
      else 
        out << "<unknown>";
      break;
    }
  } 

  static StringRef input(StringRef scalar, void *ctxt, RefKind &value) {
    assert(ctxt != nullptr);
    ContextInfo *info = reinterpret_cast<ContextInfo*>(ctxt);
    auto relocKind = info->_targetInfo.relocKindFromString(scalar);
    if (!relocKind) {
      if (scalar.equals("layout-after")) {
        value = lld::Reference::kindLayoutAfter;
        return StringRef();
      }
      if (scalar.equals("layout-before")) {
        value = lld::Reference::kindLayoutBefore;
        return StringRef();
      }
      if (scalar.equals("in-group")) {
        value = lld::Reference::kindInGroup;
        return StringRef();
      }
      return "Invalid relocation kind";
    }
    value = *relocKind;
    return StringRef();
  }
};


template <>
struct ScalarEnumerationTraits<lld::File::Kind> {
  static void enumeration(IO &io, lld::File::Kind &value) {
    io.enumCase(value, "object", lld::File::kindObject);
    io.enumCase(value, "shared-library", lld::File::kindSharedLibrary);
    io.enumCase(value, "static-library", lld::File::kindArchiveLibrary);
  }
};

template <>
struct ScalarEnumerationTraits<lld::Atom::Scope> {
  static void enumeration(IO &io, lld::Atom::Scope &value) {
    io.enumCase(value, "global", lld::Atom::scopeGlobal);
    io.enumCase(value, "hidden", lld::Atom::scopeLinkageUnit);
    io.enumCase(value, "static", lld::Atom::scopeTranslationUnit);
  }
};

template <>
struct ScalarEnumerationTraits<lld::DefinedAtom::SectionChoice> {
  static void enumeration(IO &io, lld::DefinedAtom::SectionChoice &value) {
    io.enumCase(value, "content", lld::DefinedAtom::sectionBasedOnContent);
    io.enumCase(value, "custom",  lld::DefinedAtom::sectionCustomPreferred);
    io.enumCase(value, "custom-required",
                                  lld::DefinedAtom::sectionCustomRequired);
  }
};

template <>
struct ScalarEnumerationTraits<lld::DefinedAtom::SectionPosition> {
  static void enumeration(IO &io, lld::DefinedAtom::SectionPosition &value) {
    io.enumCase(value, "start",   lld::DefinedAtom::sectionPositionStart);
    io.enumCase(value, "early",   lld::DefinedAtom::sectionPositionEarly);
    io.enumCase(value, "any",     lld::DefinedAtom::sectionPositionAny);
    io.enumCase(value, "end",     lld::DefinedAtom::sectionPositionEnd);
  }
};

template <>
struct ScalarEnumerationTraits<lld::DefinedAtom::Interposable> {
  static void enumeration(IO &io, lld::DefinedAtom::Interposable &value) {
    io.enumCase(value, "no",  lld::DefinedAtom::interposeNo);
    io.enumCase(value, "yes", lld::DefinedAtom::interposeYes);
    io.enumCase(value, "yes-and-weak",
                              lld::DefinedAtom::interposeYesAndRuntimeWeak);
  }
};

template <>
struct ScalarEnumerationTraits<lld::DefinedAtom::Merge> {
  static void enumeration(IO &io, lld::DefinedAtom::Merge &value) {
    io.enumCase(value, "no",           lld::DefinedAtom::mergeNo);
    io.enumCase(value, "as-tentative", lld::DefinedAtom::mergeAsTentative);
    io.enumCase(value, "as-weak",      lld::DefinedAtom::mergeAsWeak);
    io.enumCase(value, "as-addressed-weak",
                                  lld::DefinedAtom::mergeAsWeakAndAddressUsed);
    io.enumCase(value, "by-content",   lld::DefinedAtom::mergeByContent);
  }
};

template <>
struct ScalarEnumerationTraits<lld::DefinedAtom::DeadStripKind> {
  static void enumeration(IO &io, lld::DefinedAtom::DeadStripKind &value) {
    io.enumCase(value, "normal", lld::DefinedAtom::deadStripNormal);
    io.enumCase(value, "never",  lld::DefinedAtom::deadStripNever);
    io.enumCase(value, "always", lld::DefinedAtom::deadStripAlways);
  }
};

template <>
struct ScalarEnumerationTraits<lld::DefinedAtom::ContentPermissions> {
  static void enumeration(IO &io, lld::DefinedAtom::ContentPermissions &value) {
    io.enumCase(value, "---",     lld::DefinedAtom::perm___);
    io.enumCase(value, "r--",     lld::DefinedAtom::permR__);
    io.enumCase(value, "r-x",     lld::DefinedAtom::permR_X);
    io.enumCase(value, "rw-",     lld::DefinedAtom::permRW_);
    io.enumCase(value, "rwx",     lld::DefinedAtom::permRWX);
    io.enumCase(value, "rw-l",    lld::DefinedAtom::permRW_L);
    io.enumCase(value, "unknown", lld::DefinedAtom::permUnknown);
  }
};

template <>
struct ScalarEnumerationTraits<lld::DefinedAtom::ContentType> {
  static void enumeration(IO &io, lld::DefinedAtom::ContentType &value) {
    io.enumCase(value, "unknown",
                          lld::DefinedAtom::typeUnknown);
    io.enumCase(value, "code",
                          lld::DefinedAtom::typeCode);
    io.enumCase(value, "stub",
                          lld::DefinedAtom::typeStub);
    io.enumCase(value, "constant", lld::DefinedAtom::typeConstant);
    io.enumCase(value, "data", lld::DefinedAtom::typeData);
    io.enumCase(value, "quick-data", lld::DefinedAtom::typeDataFast);
    io.enumCase(value, "zero-fill", lld::DefinedAtom::typeZeroFill);
    io.enumCase(value, "zero-fill-quick", lld::DefinedAtom::typeZeroFillFast);
    io.enumCase(value, "const-data", lld::DefinedAtom::typeConstData);
    io.enumCase(value, "got",
                          lld::DefinedAtom::typeGOT);
    io.enumCase(value, "resolver",
                          lld::DefinedAtom::typeResolver);
    io.enumCase(value, "branch-island",
                          lld::DefinedAtom::typeBranchIsland);
    io.enumCase(value, "branch-shim",
                          lld::DefinedAtom::typeBranchShim);
    io.enumCase(value, "stub-helper",
                          lld::DefinedAtom::typeStubHelper);
    io.enumCase(value, "c-string",
                          lld::DefinedAtom::typeCString);
    io.enumCase(value, "utf16-string",
                          lld::DefinedAtom::typeUTF16String);
    io.enumCase(value, "unwind-cfi",
                          lld::DefinedAtom::typeCFI);
    io.enumCase(value, "unwind-lsda",
                          lld::DefinedAtom::typeLSDA);
    io.enumCase(value, "const-4-byte",
                          lld::DefinedAtom::typeLiteral4);
    io.enumCase(value, "const-8-byte",
                          lld::DefinedAtom::typeLiteral8);
    io.enumCase(value, "const-16-byte",
                          lld::DefinedAtom::typeLiteral16);
    io.enumCase(value, "lazy-pointer",
                          lld::DefinedAtom::typeLazyPointer);
    io.enumCase(value, "lazy-dylib-pointer",
                          lld::DefinedAtom::typeLazyDylibPointer);
    io.enumCase(value, "cfstring",
                          lld::DefinedAtom::typeCFString);
    io.enumCase(value, "initializer-pointer",
                          lld::DefinedAtom::typeInitializerPtr);
    io.enumCase(value, "terminator-pointer",
                          lld::DefinedAtom::typeTerminatorPtr);
    io.enumCase(value, "c-string-pointer",
                          lld::DefinedAtom::typeCStringPtr);
    io.enumCase(value, "objc-class-pointer",
                          lld::DefinedAtom::typeObjCClassPtr);
    io.enumCase(value, "objc-category-list",
                          lld::DefinedAtom::typeObjC2CategoryList);
    io.enumCase(value, "objc-class1",
                          lld::DefinedAtom::typeObjC1Class);
    io.enumCase(value, "dtraceDOF",
                          lld::DefinedAtom::typeDTraceDOF);
    io.enumCase(value, "lto-temp",
                          lld::DefinedAtom::typeTempLTO);
    io.enumCase(value, "compact-unwind",
                          lld::DefinedAtom::typeCompactUnwindInfo);
    io.enumCase(value, "tlv-thunk",
                          lld::DefinedAtom::typeThunkTLV);
    io.enumCase(value, "tlv-data",
                          lld::DefinedAtom::typeTLVInitialData);
    io.enumCase(value, "tlv-zero-fill",
                          lld::DefinedAtom::typeTLVInitialZeroFill);
    io.enumCase(value, "tlv-initializer-ptr",
                          lld::DefinedAtom::typeTLVInitializerPtr);
  }
};

template <>
struct ScalarEnumerationTraits<lld::UndefinedAtom::CanBeNull> {
  static void enumeration(IO &io, lld::UndefinedAtom::CanBeNull &value) {
    io.enumCase(value, "never", lld::UndefinedAtom::canBeNullNever);
    io.enumCase(value, "at-runtime",  lld::UndefinedAtom::canBeNullAtRuntime);
    io.enumCase(value, "at-buildtime", lld::UndefinedAtom::canBeNullAtBuildtime);
  }
};


template <>
struct ScalarEnumerationTraits<ShlibCanBeNull> {
  static void enumeration(IO &io, ShlibCanBeNull &value) {
    io.enumCase(value, "never",       false);
    io.enumCase(value, "at-runtime",  true);
  }
};



/// This is a custom formatter for lld::DefinedAtom::Alignment.  Values look
/// like:
///     2^3          # 8-byte aligned
///     7 mod 2^4    # 16-byte aligned plus 7 bytes
template<>
struct ScalarTraits<lld::DefinedAtom::Alignment> {
  static void output(const lld::DefinedAtom::Alignment &value, void *ctxt,
                                                      llvm::raw_ostream &out) {
    if (value.modulus == 0) {
      out << llvm::format("2^%d", value.powerOf2);
    }
    else {
      out << llvm::format("%d mod 2^%d", value.modulus, value.powerOf2);
    }
  }

  static StringRef input(StringRef scalar, void *ctxt,
                                          lld::DefinedAtom::Alignment &value) {
    value.modulus = 0;
    size_t modStart = scalar.find("mod");
    if (modStart != StringRef::npos) {
      StringRef modStr = scalar.slice(0, modStart);
      modStr = modStr.rtrim();
      unsigned int modulus;
      if (modStr.getAsInteger(0, modulus)) {
        return "malformed alignment modulus";
      }
      value.modulus = modulus;
      scalar = scalar.drop_front(modStart+3);
      scalar = scalar.ltrim();
    }
    if (!scalar.startswith("2^")) {
      return "malformed alignment";
    }
    StringRef powerStr = scalar.drop_front(2);
    unsigned int power;
    if (powerStr.getAsInteger(0, power)) {
      return "malformed alignment power";
    }
    value.powerOf2 = power;
    if (value.modulus > (1<<value.powerOf2)) {
      return "malformed alignment, modulus too large for power";
    }
    return StringRef(); // returning empty string means success
  }
};




template <>
struct ScalarEnumerationTraits<FileKinds> {
  static void enumeration(IO &io, FileKinds &value) {
    io.enumCase(value, "object",        fileKindObjectAtoms);
    io.enumCase(value, "archive",       fileKindArchive);
    io.enumCase(value, "object-elf",    fileKindObjectELF);
    io.enumCase(value, "object-mach-o", fileKindObjectMachO);
  }
};

template <>
struct MappingTraits<ArchMember> {
  static void mapping(IO &io, ArchMember &member) {
    io.mapOptional("kind",    member._kind, fileKindObjectAtoms);
    io.mapOptional("name",    member._name);
    io.mapRequired("content", member._content);
  }
};



// Declare that an AtomList is a yaml sequence.
template<typename T>
struct SequenceTraits<AtomList<T>> {
  static size_t size(IO &io, AtomList<T> &seq) {
    return seq._atoms.size();
  }
  static const T *&element(IO &io, AtomList<T> &seq, size_t index) {
    if (index >= seq._atoms.size())
      seq._atoms.resize(index+1);
    return seq._atoms[index];
  }
};

// Used to allow DefinedAtom content bytes to be a flow sequence of
// two-digit hex numbers without the leading 0x (e.g. FF, 04, 0A)
template<>
struct ScalarTraits<ImplicitHex8> {
  static void output(const ImplicitHex8 &val, void*, llvm::raw_ostream &out) {
    uint8_t num = val;
    out << llvm::format("%02X", num);
  }

  static llvm::StringRef input(llvm::StringRef str, void*, ImplicitHex8 &val) {
    unsigned long long n;
    if (getAsUnsignedInteger(str, 16, n))
      return "invalid two-digit-hex number";
    if (n > 0xFF)
      return "out of range two-digit-hex number";
    val = n;
    return StringRef(); // returning empty string means success
  }
};


// YAML conversion for std::vector<const lld::File*>
template<>
struct DocumentListTraits< std::vector<const lld::File*> > {
  static size_t size(IO &io, std::vector<const lld::File*> &seq) {
    return seq.size();
  }
  static const lld::File *&element(IO &io, std::vector<const lld::File*> &seq,
                                                                 size_t index) {
    if (index >= seq.size())
      seq.resize(index+1);
    return seq[index];
  }
};


// YAML conversion for const lld::File*
template <>
struct MappingTraits<const lld::File*> {

    class NormArchiveFile : public lld::ArchiveLibraryFile {
    public:
      NormArchiveFile(IO &io)
          : ArchiveLibraryFile(((ContextInfo *)io.getContext())->_targetInfo,
                               ""),
            _path() {
      }
      NormArchiveFile(IO &io, const lld::File *file)
          : ArchiveLibraryFile(((ContextInfo *)io.getContext())->_targetInfo,
                               file->path()),
            _path(file->path()) {
        // If we want to support writing archives, this constructor would
        // need to populate _members.
      }

    const lld::File *denormalize(IO &io) {
      return this;
    }

    virtual void setOrdinalAndIncrement(uint64_t &ordinal) const {
      _ordinal = ordinal++;
      // Assign sequential ordinals to member files
      for (const ArchMember &member : _members) {
        member._content->setOrdinalAndIncrement(ordinal);
      }
    }

    virtual const atom_collection<lld::DefinedAtom> &defined() const {
      return _noDefinedAtoms;
    }
    virtual const atom_collection<lld::UndefinedAtom> &undefined() const {
      return _noUndefinedAtoms;
    }
    virtual const atom_collection<lld::SharedLibraryAtom> &sharedLibrary()const{
      return _noSharedLibaryAtoms;
    }
    virtual const atom_collection<lld::AbsoluteAtom> &absolute() const {
      return _noAbsoluteAtoms;
    }
    virtual const File *find(StringRef name, bool dataSymbolOnly) const {
      for (const ArchMember &member : _members) {
        for (const lld::DefinedAtom *atom : member._content->defined() ) {
          if (name == atom->name()) {
            if (!dataSymbolOnly)
              return member._content;
            switch (atom->contentType()) {
              case lld::DefinedAtom::typeData:
              case lld::DefinedAtom::typeZeroFill:
                return member._content;
              default:
                break;
            }
          }
        }
      }
      return nullptr;
    }

    StringRef                _path;
    std::vector<ArchMember>  _members;
  };

    class NormalizedFile : public lld::File {
    public:
      NormalizedFile(IO &io) : File("", kindObject), _IO(io), _rnb(nullptr) {}
      NormalizedFile(IO &io, const lld::File *file)
          : File(file->path(), kindObject), _IO(io), _rnb(new RefNameBuilder(*file)),
            _path(file->path()) {
        for (const lld::DefinedAtom *a : file->defined())
          _definedAtoms.push_back(a);
        for (const lld::UndefinedAtom *a : file->undefined())
        _undefinedAtoms.push_back(a);
      for (const lld::SharedLibraryAtom *a : file->sharedLibrary())
        _sharedLibraryAtoms.push_back(a);
      for (const lld::AbsoluteAtom *a : file->absolute())
        _absoluteAtoms.push_back(a);
    }
    const lld::File *denormalize(IO &io);

    virtual const atom_collection<lld::DefinedAtom> &defined() const {
      return _definedAtoms;
    }
    virtual const atom_collection<lld::UndefinedAtom> &undefined() const {
      return _undefinedAtoms;
    }
    virtual const atom_collection<lld::SharedLibraryAtom> &sharedLibrary()const{
      return _sharedLibraryAtoms;
    }
      virtual const atom_collection<lld::AbsoluteAtom> &absolute() const {
        return _absoluteAtoms;
      }

      virtual const TargetInfo &getTargetInfo() const {
        return ((ContextInfo *)_IO.getContext())->_targetInfo;
      }

      // Allocate a new copy of this string and keep track of allocations
      // in _stringCopies, so they can be freed when File is destroyed.
    StringRef copyString(StringRef str) {
      // We want _stringCopies to own the string memory so it is deallocated
      // when the File object is destroyed.  But we need a StringRef that
      // points into that memory.
      std::unique_ptr<char[]> s(new char[str.size()]);
      memcpy(s.get(), str.data(), str.size());
      llvm::StringRef r = llvm::StringRef(s.get(), str.size());
      _stringCopies.push_back(std::move(s));
        return r;
      }

      IO &_IO;
      RefNameBuilder *_rnb;
      StringRef _path;
      AtomList<lld::DefinedAtom> _definedAtoms;
    AtomList<lld::UndefinedAtom>       _undefinedAtoms;
    AtomList<lld::SharedLibraryAtom>   _sharedLibraryAtoms;
    AtomList<lld::AbsoluteAtom>        _absoluteAtoms;
    std::vector<std::unique_ptr<char[]>> _stringCopies;
  };


  static void mapping(IO &io, const lld::File *&file) {
    // We only support writing atom based YAML
    FileKinds kind = fileKindObjectAtoms;
    // If reading, peek ahead to see what kind of file this is.
    io.mapOptional("kind",   kind, fileKindObjectAtoms);
    //
    switch (kind) {
      case fileKindObjectAtoms:
        mappingAtoms(io, file);
        break;
      case fileKindArchive:
        mappingArchive(io, file);
        break;
      case fileKindObjectELF:
      case fileKindObjectMachO:
        // Eventually we will have an external function to call, similar
        // to mappingAtoms() and mappingArchive() but implememented
        // with coresponding file format code.
        llvm_unreachable("section based YAML not supported yet");
    }
  }

  static void mappingAtoms(IO &io, const lld::File *&file) {
    MappingNormalizationHeap<NormalizedFile, const lld::File*> keys(io, file);
    ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
    assert(info != nullptr);
    info->_currentFile = keys.operator->();

    io.mapOptional("path",                  keys->_path);
    io.mapOptional("defined-atoms",         keys->_definedAtoms);
    io.mapOptional("undefined-atoms",       keys->_undefinedAtoms);
    io.mapOptional("shared-library-atoms",  keys->_sharedLibraryAtoms);
    io.mapOptional("absolute-atoms",        keys->_absoluteAtoms);
  }

  static void mappingArchive(IO &io, const lld::File *&file) {
    MappingNormalizationHeap<NormArchiveFile, const lld::File*> keys(io, file);

    io.mapOptional("path",                keys->_path);
    io.mapOptional("members",             keys->_members);
  }

};



// YAML conversion for const lld::Reference*
template <>
struct MappingTraits<const lld::Reference*> {

  class NormalizedReference : public lld::Reference {
  public:
    NormalizedReference(IO &io)
      : _target(nullptr), _targetName(), _offset(0), _addend(0) {}

    NormalizedReference(IO &io, const lld::Reference *ref)
      : _target(nullptr),
        _targetName(targetName(io, ref)),
        _offset(ref->offsetInAtom()),
        _addend(ref->addend()),
        _mappedKind(ref->kind()) {
    }

    const lld::Reference *denormalize(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      typedef MappingTraits<const lld::File*>::NormalizedFile NormalizedFile;
      NormalizedFile *f = reinterpret_cast<NormalizedFile*>(info->_currentFile);
      if (!_targetName.empty())
        _targetName = f->copyString(_targetName);
        DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
                << "created Reference to name: '" << _targetName
                << "' (" << (void*)_targetName.data() << ", "
                << _targetName.size() << ")\n");
      setKind(_mappedKind);
      return this;
    }
    void bind(const RefNameResolver&);
    static StringRef targetName(IO &io, const lld::Reference *ref);

    virtual uint64_t         offsetInAtom() const { return _offset; }
    virtual const lld::Atom *target() const       { return _target; }
    virtual Addend           addend() const       { return _addend; }
    virtual void             setAddend(Addend a)  { _addend = a; }
    virtual void             setTarget(const lld::Atom *a) { _target = a; }

    const lld::Atom    *_target;
    StringRef           _targetName;
    uint32_t            _offset;
    Addend              _addend;
    RefKind             _mappedKind;
  };


  static void mapping(IO &io, const lld::Reference *&ref) {
    MappingNormalizationHeap<NormalizedReference,
                                          const lld::Reference*> keys(io, ref);

    io.mapRequired("kind",         keys->_mappedKind);
    io.mapOptional("offset",       keys->_offset);
    io.mapOptional("target",       keys->_targetName);
    io.mapOptional("addend",       keys->_addend,  (lld::Reference::Addend)0);
  }
};



// YAML conversion for const lld::DefinedAtom*
template <>
struct MappingTraits<const lld::DefinedAtom*> {

  class NormalizedAtom : public lld::DefinedAtom {
  public:
    NormalizedAtom(IO &io)
      : _file(fileFromContext(io)), _name(), _refName(),
        _alignment(0), _content(), _references() {
      static uint32_t ordinalCounter = 1;
      _ordinal = ordinalCounter++;
    }
    NormalizedAtom(IO &io, const lld::DefinedAtom *atom)
      : _file(fileFromContext(io)),
        _name(atom->name()),
        _refName(),
        _scope(atom->scope()),
        _interpose(atom->interposable()),
        _merge(atom->merge()),
        _contentType(atom->contentType()),
        _alignment(atom->alignment()),
        _sectionChoice(atom->sectionChoice()),
        _sectionPosition(atom->sectionPosition()),
        _deadStrip(atom->deadStrip()),
        _permissions(atom->permissions()),
        _size(atom->size()),
        _sectionName(atom->customSectionName()) {
          for ( const lld::Reference *r : *atom )
            _references.push_back(r);
          ArrayRef<uint8_t> cont = atom->rawContent();
          _content.reserve(cont.size());
          for (uint8_t x : cont)
              _content.push_back(x);
    }
    const lld::DefinedAtom *denormalize(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      typedef MappingTraits<const lld::File*>::NormalizedFile NormalizedFile;
      NormalizedFile *f = reinterpret_cast<NormalizedFile*>(info->_currentFile);
      if ( !_name.empty() )
        _name = f->copyString(_name);
      if ( !_refName.empty() )
        _refName = f->copyString(_refName);
      if ( !_sectionName.empty() )
        _sectionName = f->copyString(_sectionName);
      DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
              << "created DefinedAtom named: '" << _name
              << "' (" << (void*)_name.data() << ", "
              << _name.size() << ")\n");
      return this;
    }
    void bind(const RefNameResolver&);
    // Extract current File object from YAML I/O parsing context
    const lld::File &fileFromContext(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      assert(info->_currentFile != nullptr);
      return *info->_currentFile;
    }

    virtual const lld::File   &file() const          { return _file; }
    virtual StringRef          name() const          { return _name; }
    virtual uint64_t           size() const          { return _size; }
    virtual Scope              scope() const         { return _scope; }
    virtual Interposable       interposable() const  { return _interpose; }
    virtual Merge              merge() const         { return _merge; }
    virtual ContentType        contentType() const   { return _contentType; }
    virtual Alignment          alignment() const     { return _alignment; }
    virtual SectionChoice      sectionChoice() const { return _sectionChoice; }
    virtual StringRef          customSectionName() const { return _sectionName;}
    virtual SectionPosition    sectionPosition() const{return _sectionPosition;}
    virtual DeadStripKind      deadStrip() const     { return _deadStrip;  }
    virtual ContentPermissions permissions() const   { return _permissions; }
    virtual bool               isThumb() const       { return false; }
    virtual bool               isAlias() const       { return false; }
    ArrayRef<uint8_t>          rawContent() const    {
      return ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(_content.data()), _content.size());
    }

    virtual uint64_t           ordinal() const       { return _ordinal; }

    reference_iterator begin() const {
      uintptr_t index = 0;
      const void *it = reinterpret_cast<const void*>(index);
      return reference_iterator(*this, it);
    }
    reference_iterator end() const {
      uintptr_t index = _references.size();
      const void *it = reinterpret_cast<const void*>(index);
      return reference_iterator(*this, it);
    }
    const lld::Reference *derefIterator(const void *it) const {
      uintptr_t index = reinterpret_cast<uintptr_t>(it);
      assert(index < _references.size());
      return _references[index];
    }
    void incrementIterator(const void *&it) const {
      uintptr_t index = reinterpret_cast<uintptr_t>(it);
      ++index;
      it = reinterpret_cast<const void*>(index);
    }

    const lld::File          &_file;
    StringRef                 _name;
    StringRef                 _refName;
    Scope                     _scope;
    Interposable              _interpose;
    Merge                     _merge;
    ContentType               _contentType;
    Alignment                 _alignment;
    SectionChoice             _sectionChoice;
    SectionPosition           _sectionPosition;
    DeadStripKind             _deadStrip;
    ContentPermissions        _permissions;
    uint32_t                  _ordinal;
    std::vector<ImplicitHex8> _content;
    uint64_t                  _size;
    StringRef                 _sectionName;
    std::vector<const lld::Reference*> _references;
  };

  static void mapping(IO &io, const lld::DefinedAtom *&atom) {
    MappingNormalizationHeap<NormalizedAtom,
                                        const lld::DefinedAtom*> keys(io, atom);
    if ( io.outputting() ) {
      // If writing YAML, check if atom needs a ref-name.
      typedef MappingTraits<const lld::File*>::NormalizedFile NormalizedFile;
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      NormalizedFile *f = reinterpret_cast<NormalizedFile*>(info->_currentFile);
      assert(f);
      assert(f->_rnb);
      if ( f->_rnb->hasRefName(atom) ) {
        keys->_refName = f->_rnb->refName(atom);
      }
    }

    io.mapOptional("name",           keys->_name,
                                        StringRef());
    io.mapOptional("ref-name",       keys->_refName,
                                        StringRef());
    io.mapOptional("scope",          keys->_scope,
                                        lld::DefinedAtom::scopeTranslationUnit);
    io.mapOptional("type",           keys->_contentType,
                                        lld::DefinedAtom::typeCode);
    io.mapOptional("content",        keys->_content);
    io.mapOptional("size",           keys->_size,
                                        (uint64_t)keys->_content.size());
    io.mapOptional("interposable",   keys->_interpose,
                                        lld::DefinedAtom::interposeNo);
    io.mapOptional("merge",          keys->_merge,
                                        lld::DefinedAtom::mergeNo);
    io.mapOptional("alignment",      keys->_alignment,
                                        lld::DefinedAtom::Alignment(0));
    io.mapOptional("section-choice", keys->_sectionChoice,
                                        lld::DefinedAtom::sectionBasedOnContent);
    io.mapOptional("section-name",   keys->_sectionName,
                                        StringRef());
    io.mapOptional("section-position",keys->_sectionPosition,
                                        lld::DefinedAtom::sectionPositionAny);
    io.mapOptional("dead-strip",     keys->_deadStrip,
                                        lld::DefinedAtom::deadStripNormal);
    // default permissions based on content type
    io.mapOptional("permissions",    keys->_permissions,
                                                lld::DefinedAtom::permissions(
                                                           keys->_contentType));
    io.mapOptional("references",     keys->_references);
  }
};




// YAML conversion for const lld::UndefinedAtom*
template <>
struct MappingTraits<const lld::UndefinedAtom*> {

 class NormalizedAtom : public lld::UndefinedAtom {
  public:
    NormalizedAtom(IO &io)
      : _file(fileFromContext(io)), _name(), _canBeNull(canBeNullNever) {
    }
    NormalizedAtom(IO &io, const lld::UndefinedAtom *atom)
      : _file(fileFromContext(io)),
        _name(atom->name()),
        _canBeNull(atom->canBeNull()) {
    }
    const lld::UndefinedAtom *denormalize(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      typedef MappingTraits<const lld::File*>::NormalizedFile NormalizedFile;
      NormalizedFile *f = reinterpret_cast<NormalizedFile*>(info->_currentFile);
      if ( !_name.empty() )
        _name = f->copyString(_name);

      DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
              << "created UndefinedAtom named: '" << _name
              << "' (" << (void*)_name.data() << ", "
              << _name.size() << ")\n");
      return this;
    }
    // Extract current File object from YAML I/O parsing context
    const lld::File &fileFromContext(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      assert(info->_currentFile != nullptr);
      return *info->_currentFile;
    }

    virtual const lld::File  &file() const        { return _file; }
    virtual StringRef         name() const        { return _name; }
    virtual CanBeNull         canBeNull() const   { return _canBeNull; }

    const lld::File  &_file;
    StringRef         _name;
    CanBeNull         _canBeNull;
  };


  static void mapping(IO &io, const lld::UndefinedAtom* &atom) {
    MappingNormalizationHeap<NormalizedAtom,
                              const lld::UndefinedAtom*> keys(io, atom);

    io.mapRequired("name",             keys->_name);
    io.mapOptional("can-be-null",      keys->_canBeNull,
                                       lld::UndefinedAtom::canBeNullNever);
  }
};



// YAML conversion for const lld::SharedLibraryAtom*
template <>
struct MappingTraits<const lld::SharedLibraryAtom*> {

 class NormalizedAtom : public lld::SharedLibraryAtom {
  public:
    NormalizedAtom(IO &io)
      : _file(fileFromContext(io)), _name(), _loadName(), _canBeNull(false) {
    }
    NormalizedAtom(IO &io, const lld::SharedLibraryAtom *atom)
      : _file(fileFromContext(io)),
        _name(atom->name()),
        _loadName(atom->loadName()),
        _canBeNull(atom->canBeNullAtRuntime())  {
    }
    const lld::SharedLibraryAtom *denormalize(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      typedef MappingTraits<const lld::File*>::NormalizedFile NormalizedFile;
      NormalizedFile *f = reinterpret_cast<NormalizedFile*>(info->_currentFile);
      if ( !_name.empty() )
        _name = f->copyString(_name);
      if ( !_loadName.empty() )
        _loadName = f->copyString(_loadName);

      DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
              << "created SharedLibraryAtom named: '" << _name
              << "' (" << (void*)_name.data() << ", "
              << _name.size() << ")\n");
      return this;
    }
    // Extract current File object from YAML I/O parsing context
    const lld::File &fileFromContext(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      assert(info->_currentFile != nullptr);
      return *info->_currentFile;
    }

    virtual const lld::File  &file() const               { return _file; }
    virtual StringRef         name() const               { return _name; }
    virtual StringRef         loadName() const           { return _loadName;}
    virtual bool              canBeNullAtRuntime() const { return _canBeNull; }

    const lld::File          &_file;
    StringRef                 _name;
    StringRef                 _loadName;
    ShlibCanBeNull            _canBeNull;
  };


  static void mapping(IO &io, const lld::SharedLibraryAtom *&atom) {

    MappingNormalizationHeap<NormalizedAtom,
                              const lld::SharedLibraryAtom*> keys(io, atom);

    io.mapRequired("name",             keys->_name);
    io.mapOptional("load-name",        keys->_loadName);
    io.mapOptional("can-be-null",      keys->_canBeNull,
                                          (ShlibCanBeNull)false);
  }
};


// YAML conversion for const lld::AbsoluteAtom*
template <>
struct MappingTraits<const lld::AbsoluteAtom*> {

 class NormalizedAtom : public lld::AbsoluteAtom {
  public:
    NormalizedAtom(IO &io)
      : _file(fileFromContext(io)), _name(), _scope(), _value(0) {
    }
    NormalizedAtom(IO &io, const lld::AbsoluteAtom *atom)
      : _file(fileFromContext(io)),
        _name(atom->name()),
        _scope(atom->scope()),
        _value(atom->value()) {
    }
    const lld::AbsoluteAtom *denormalize(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      typedef MappingTraits<const lld::File*>::NormalizedFile NormalizedFile;
      NormalizedFile *f = reinterpret_cast<NormalizedFile*>(info->_currentFile);
      if ( !_name.empty() )
        _name = f->copyString(_name);

      DEBUG_WITH_TYPE("WriterYAML", llvm::dbgs()
              << "created AbsoluteAtom named: '" << _name
              << "' (" << (void*)_name.data() << ", "
              << _name.size() << ")\n");
      return this;
    }
    // Extract current File object from YAML I/O parsing context
    const lld::File &fileFromContext(IO &io) {
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      assert(info->_currentFile != nullptr);
      return *info->_currentFile;
    }

    virtual const lld::File  &file() const     { return _file; }
    virtual StringRef         name() const     { return _name; }
    virtual uint64_t          value() const    { return _value; }
    virtual Scope             scope() const    { return _scope; }

    const lld::File  &_file;
    StringRef         _name;
    StringRef         _refName;
    Scope             _scope;
    Hex64             _value;
  };


  static void mapping(IO &io, const lld::AbsoluteAtom *&atom) {
    MappingNormalizationHeap<NormalizedAtom,
                              const lld::AbsoluteAtom*> keys(io, atom);

    if ( io.outputting() ) {
      typedef MappingTraits<const lld::File*>::NormalizedFile NormalizedFile;
      ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
      assert(info != nullptr);
      NormalizedFile *f = reinterpret_cast<NormalizedFile*>(info->_currentFile);
      assert(f);
      assert(f->_rnb);
      if ( f->_rnb->hasRefName(atom) ) {
        keys->_refName = f->_rnb->refName(atom);
      }
    }

    io.mapRequired("name",      keys->_name);
    io.mapOptional("ref-name",  keys->_refName, StringRef());
    io.mapOptional("scope",     keys->_scope);
    io.mapRequired("value",     keys->_value);
  }
};

} // namespace llvm
} // namespace yaml


RefNameResolver::RefNameResolver(const lld::File *file, IO &io) : _io(io) {
  typedef MappingTraits<const lld::DefinedAtom*>::NormalizedAtom NormalizedAtom;
  for (const lld::DefinedAtom *a : file->defined() ) {
    NormalizedAtom *na = (NormalizedAtom*)a;
    if ( na->_refName.empty() )
      add(na->_name, a);
    else
      add(na->_refName, a);
  }

  for (const lld::UndefinedAtom *a : file->undefined() )
    add(a->name(), a);

  for (const lld::SharedLibraryAtom *a : file->sharedLibrary() )
    add(a->name(), a);

  typedef MappingTraits<const lld::AbsoluteAtom*>::NormalizedAtom NormAbsAtom;
  for (const lld::AbsoluteAtom *a : file->absolute() ) {
    NormAbsAtom *na = (NormAbsAtom*)a;
    if ( na->_refName.empty() )
      add(na->_name, a);
    else
      add(na->_refName, a);
   }
}



inline
const lld::File*
MappingTraits<const lld::File*>::NormalizedFile::denormalize(IO &io) {
  typedef MappingTraits<const lld::DefinedAtom*>::NormalizedAtom NormalizedAtom;

  RefNameResolver nameResolver(this, io);
  // Now that all atoms are parsed, references can be bound.
  for (const lld::DefinedAtom *a : this->defined() ) {
    NormalizedAtom *normAtom = (NormalizedAtom*)a;
    normAtom->bind(nameResolver);
  }
  return this;
}

inline
void MappingTraits<const lld::DefinedAtom*>::
              NormalizedAtom::bind(const RefNameResolver &resolver) {
  typedef MappingTraits<const lld::Reference*>::NormalizedReference
                                                            NormalizedReference;
  for (const lld::Reference *ref : _references) {
    NormalizedReference *normRef = (NormalizedReference*)ref;
    normRef->bind(resolver);
  }
}

inline
void MappingTraits<const lld::Reference*>::
         NormalizedReference::bind(const RefNameResolver &resolver) {
  _target = resolver.lookup(_targetName);
}


inline
llvm::StringRef MappingTraits<const lld::Reference*>::NormalizedReference::
                                targetName(IO &io, const lld::Reference *ref) {
  if ( ref->target() == nullptr )
    return llvm::StringRef();
  ContextInfo *info = reinterpret_cast<ContextInfo*>(io.getContext());
  assert(info != nullptr);
  typedef MappingTraits<const lld::File*>::NormalizedFile NormalizedFile;
  NormalizedFile *f = reinterpret_cast<NormalizedFile*>(info->_currentFile);
  RefNameBuilder *rnb = f->_rnb;
  if ( rnb->hasRefName(ref->target()) )
    return rnb->refName(ref->target());
  return ref->target()->name();
}



namespace lld {
namespace yaml {

class Writer : public lld::Writer {
public:
  Writer(const TargetInfo &ti) : _targetInfo(ti) {}

  virtual error_code writeFile(const lld::File &file, StringRef outPath) {
    // Create stream to path.
    std::string errorInfo;
    llvm::raw_fd_ostream out(outPath.data(), errorInfo);
    if (!errorInfo.empty())
      return llvm::make_error_code(llvm::errc::no_such_file_or_directory);

    // Create yaml Output writer, using yaml options for context.
    ContextInfo context(_targetInfo);
    llvm::yaml::Output yout(out, &context);

    // Write yaml output.
    const lld::File *fileRef = &file;
    yout << fileRef;

    return error_code::success();
  }

private:
  const TargetInfo &_targetInfo;
};

class ReaderYAML : public Reader {
public:
  ReaderYAML(const TargetInfo &ti) : Reader(ti) {}

  error_code parseFile(std::unique_ptr<MemoryBuffer> &mb,
                       std::vector<std::unique_ptr<File>> &result) const {
    // Note: we do not take ownership of the MemoryBuffer.  That is
    // because yaml may produce multiple File objects, so there is no
    // *one* File to take ownership.  Therefore, the yaml File objects
    // produced must make copies of all strings that come from YAML I/O.
    // Otherwise the strings will become invalid when this MemoryBuffer
    // is deallocated.

    // Create YAML Input parser.
    ContextInfo context(_targetInfo);
    llvm::yaml::Input yin(mb->getBuffer(), &context);

    // Fill vector with File objects created by parsing yaml.
    std::vector<const lld::File*> createdFiles;
    yin >> createdFiles;

    // Quit now if there were parsing errors.
    if ( yin.error() )
      return make_error_code(lld::yaml_reader_error::illegal_value);

    for (const File *file : createdFiles) {
      // Note: parseFile() should return vector of *const* File
      File *f = const_cast<File*>(file);
      result.emplace_back(f);
    }
    return make_error_code(lld::yaml_reader_error::success);
  }
};
} // end namespace yaml

std::unique_ptr<Writer> createWriterYAML(const TargetInfo &ti) {
  return std::unique_ptr<Writer>(new lld::yaml::Writer(ti));
}

std::unique_ptr<Reader> createReaderYAML(const TargetInfo &ti) {
  return std::unique_ptr<Reader>(new lld::yaml::ReaderYAML(ti));
}
} // end namespace lld
