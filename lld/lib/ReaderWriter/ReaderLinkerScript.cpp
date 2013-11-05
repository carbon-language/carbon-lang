//===- lib/ReaderWriter/ReaderLinkerScript.cpp ----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/ReaderLinkerScript.h"

#include "lld/Core/Error.h"
#include "lld/Core/File.h"
#include "lld/ReaderWriter/LinkerScript.h"

using namespace lld;
using namespace script;

namespace {
class LinkerScriptFile : public File {
public:
  static ErrorOr<std::unique_ptr<LinkerScriptFile> >
  create(const LinkingContext &context,
         std::unique_ptr<MemoryBuffer> mb) {
    std::unique_ptr<LinkerScriptFile> file(
        new LinkerScriptFile(context, std::move(mb)));
    file->_script = file->_parser.parse();
    if (!file->_script)
      return LinkerScriptReaderError::parse_error;
    return std::move(file);
  }

  static inline bool classof(const File *f) {
    return f->kind() == kindLinkerScript;
  }

  virtual const LinkingContext &getLinkingContext() const { return _context; }

  virtual const atom_collection<DefinedAtom> &defined() const {
    return _definedAtoms;
  }

  virtual const atom_collection<UndefinedAtom> &undefined() const {
    return _undefinedAtoms;
  }

  virtual const atom_collection<SharedLibraryAtom> &sharedLibrary() const {
    return _sharedLibraryAtoms;
  }

  virtual const atom_collection<AbsoluteAtom> &absolute() const {
    return _absoluteAtoms;
  }

  const LinkerScript *getScript() { return _script; }

private:
  LinkerScriptFile(const LinkingContext &context,
                   std::unique_ptr<MemoryBuffer> mb)
      : File(mb->getBufferIdentifier(), kindLinkerScript), _context(context),
        _lexer(std::move(mb)), _parser(_lexer), _script(nullptr) {}

  const LinkingContext &_context;
  atom_collection_vector<DefinedAtom> _definedAtoms;
  atom_collection_vector<UndefinedAtom> _undefinedAtoms;
  atom_collection_vector<SharedLibraryAtom> _sharedLibraryAtoms;
  atom_collection_vector<AbsoluteAtom> _absoluteAtoms;
  Lexer _lexer;
  Parser _parser;
  const LinkerScript *_script;
};
} // end anon namespace

namespace lld {
error_code ReaderLinkerScript::parseFile(
    std::unique_ptr<MemoryBuffer> &mb,
    std::vector<std::unique_ptr<File> > &result) const {
  auto lsf = LinkerScriptFile::create(_context, std::move(mb));
  if (!lsf)
    return lsf;
  const LinkerScript *ls = (*lsf)->getScript();
  result.push_back(std::move(*lsf));
  for (const auto &c : ls->_commands) {
    if (auto group = dyn_cast<lld::script::Group>(c))
      for (const auto &path : group->getPaths()) {
        OwningPtr<MemoryBuffer> opmb;
        if (error_code ec =
                MemoryBuffer::getFileOrSTDIN(path._path, opmb))
          return ec;
        std::unique_ptr<MemoryBuffer> eachMB(opmb.take());
        if (error_code ec =
                _context.getDefaultReader().parseFile(eachMB, result))
          return ec;
      }
  }
  return error_code::success();
}
} // end namespace lld
