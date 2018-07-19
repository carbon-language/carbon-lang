// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mod-file.h"
#include "scope.h"
#include "symbol.h"
#include "../parser/message.h"
#include <algorithm>
#include <cerrno>
#include <fstream>
#include <functional>
#include <ostream>
#include <set>
#include <sstream>
#include <vector>

namespace Fortran::semantics {

using namespace parser::literals;

class ModFileWriter {
public:
  // The initial characters of a file that identify it as a .mod file.
  static constexpr auto magic{"!mod$"};
  static constexpr auto extension{".mod"};

  // The .mod file format version number.
  void set_version(int version) { version_ = version; }
  // The directory to write .mod files in.
  void set_directory(const std::string &dir) { dir_ = dir; }

  // Errors encountered during writing. Non-empty if WriteAll returns false.
  const std::list<parser::MessageFormattedText> &errors() const {
    return errors_;
  }

  // Write out all .mod files; if error return false and report them on ostream.
  bool WriteAll(std::ostream &);
  // Write out all .mod files; if error return false.
  bool WriteAll();
  // Write out .mod file for one module; if error return false.
  bool WriteOne(const Symbol &);

private:
  using symbolSet = std::set<const Symbol *>;
  using symbolVector = std::vector<const Symbol *>;

  int version_{1};
  std::string dir_{"."};
  // The mod file consists of uses, declarations, and contained subprograms:
  std::stringstream uses_;
  std::stringstream useExtraAttrs_;  // attrs added to used entity
  std::stringstream decls_;
  std::stringstream contains_;
  // Any errors encountered during writing:
  std::list<parser::MessageFormattedText> errors_;

  std::string GetAsString(const std::string &);
  std::string GetHeader(const std::string &);
  void PutSymbols(const Scope &);
  symbolVector SortSymbols(const symbolSet);
  symbolSet CollectSymbols(const Scope &);
  void PutSymbol(const Symbol &);
  void PutDerivedType(const Symbol &);
  void PutSubprogram(const Symbol &);
  void PutGeneric(const Symbol &);
  void PutUse(const Symbol &);
  void PutUseExtraAttr(Attr, const Symbol &, const Symbol &);
  static void PutEntity(std::ostream &, const Symbol &);
  static void PutObjectEntity(std::ostream &, const Symbol &);
  static void PutProcEntity(std::ostream &, const Symbol &);
  static void PutEntity(std::ostream &, const Symbol &, std::function<void()>);
  static std::ostream &PutAttrs(std::ostream &, Attrs,
      std::string before = ","s, std::string after = ""s);
  static std::ostream &PutLower(std::ostream &, const Symbol &);
  static std::ostream &PutLower(std::ostream &, const DeclTypeSpec &);
  static std::ostream &PutLower(std::ostream &, const std::string &);
  static std::string CheckSum(const std::string &);
};

bool ModFileWriter::WriteAll(std::ostream &os) {
  if (!WriteAll()) {
    for (auto &message : errors()) {
      std::cerr << message.string() << '\n';
    }
    return false;
  }
  return true;
}

bool ModFileWriter::WriteAll() {
  for (const auto &scope : Scope::globalScope.children()) {
    if (scope.kind() == Scope::Kind::Module) {
      auto &symbol{*scope.symbol()};  // symbol must be present for module
      WriteOne(symbol);
    }
  }
  return errors_.empty();
}

bool ModFileWriter::WriteOne(const Symbol &modSymbol) {
  CHECK(modSymbol.has<ModuleDetails>());
  auto name{parser::ToLowerCaseLetters(modSymbol.name().ToString())};
  std::string path{dir_ + '/' + name + extension};
  std::ofstream os{path};
  PutSymbols(*modSymbol.scope());
  std::string all{GetAsString(name)};
  auto header{GetHeader(all)};
  os << header << all;
  os.close();
  if (!os) {
    errors_.emplace_back(
        "Error writing %s: %s"_err_en_US, path.c_str(), std::strerror(errno));
    return false;
  }
  return true;
}

// Return the entire body of the module file
// and clear saved uses, decls, and contains.
std::string ModFileWriter::GetAsString(const std::string &name) {
  std::stringstream all;
  all << "module " << name << '\n';
  all << uses_.str();
  uses_.str(""s);
  all << useExtraAttrs_.str();
  useExtraAttrs_.str(""s);
  all << decls_.str();
  decls_.str(""s);
  auto str{contains_.str()};
  contains_.str(""s);
  if (!str.empty()) {
    all << "contains\n" << str;
  }
  all << "end\n";
  return all.str();
}

// Return the header for this mod file.
std::string ModFileWriter::GetHeader(const std::string &all) {
  std::stringstream ss;
  ss << magic << " v" << version_ << " sum:" << CheckSum(all) << '\n';
  return ss.str();
}

// Put out the visible symbols from scope.
void ModFileWriter::PutSymbols(const Scope &scope) {
  for (const auto *symbol : SortSymbols(CollectSymbols(scope))) {
    PutSymbol(*symbol);
  }
}

// Sort symbols by their original order, not by name.
ModFileWriter::symbolVector ModFileWriter::SortSymbols(
    const ModFileWriter::symbolSet symbols) {
  ModFileWriter::symbolVector sorted;
  sorted.reserve(symbols.size());
  for (const auto *symbol : symbols) {
    sorted.push_back(symbol);
  }
  auto compare{[](const Symbol *x, const Symbol *y) {
    return x->name().begin() < y->name().begin();
  }};
  std::sort(sorted.begin(), sorted.end(), compare);
  return sorted;
}

// Return all symbols needed from this scope.
ModFileWriter::symbolSet ModFileWriter::CollectSymbols(const Scope &scope) {
  ModFileWriter::symbolSet symbols;
  for (const auto &pair : scope) {
    auto *symbol{pair.second};
    // include all components of derived types and other non-private symbols
    if (scope.kind() == Scope::Kind::DerivedType ||
        !symbol->attrs().test(Attr::PRIVATE)) {
      symbols.insert(symbol);
      // ensure the type symbol is included too, even if private
      if (const auto *type{symbol->GetType()}) {
        auto category{type->category()};
        if (category == DeclTypeSpec::TypeDerived ||
            category == DeclTypeSpec::ClassDerived) {
          auto *typeSymbol{type->derivedTypeSpec().scope()->symbol()};
          symbols.insert(typeSymbol);
        }
      }
      // TODO: other related symbols, e.g. in initial values
    }
  }
  return symbols;
}

void ModFileWriter::PutSymbol(const Symbol &symbol) {
  std::visit(
      common::visitors{
          [&](const ModuleDetails &) { /* should be current module */ },
          [&](const DerivedTypeDetails &) { PutDerivedType(symbol); },
          [&](const SubprogramDetails &) { PutSubprogram(symbol); },
          [&](const GenericDetails &) { PutGeneric(symbol); },
          [&](const UseDetails &) { PutUse(symbol); },
          [&](const UseErrorDetails &) {},
          [&](const auto &) { PutEntity(decls_, symbol); }},
      symbol.details());
}

void ModFileWriter::PutDerivedType(const Symbol &typeSymbol) {
  PutAttrs(decls_ << "type", typeSymbol.attrs(), ","s, ""s);
  PutLower(decls_ << "::", typeSymbol) << '\n';
  PutSymbols(*typeSymbol.scope());
  decls_ << "end type\n";
}

void ModFileWriter::PutSubprogram(const Symbol &symbol) {
  auto attrs{symbol.attrs()};
  Attrs bindAttrs{};
  if (attrs.test(Attr::BIND_C)) {
    // bind(c) is a suffix, not prefix
    bindAttrs.set(Attr::BIND_C, true);
    attrs.set(Attr::BIND_C, false);
  }
  bool isExternal{attrs.test(Attr::EXTERNAL)};
  std::ostream &os{isExternal ? decls_ : contains_};
  if (isExternal) {
    os << "interface\n";
  }
  PutAttrs(os, attrs, ""s, " "s);
  auto &details{symbol.get<SubprogramDetails>()};
  os << (details.isFunction() ? "function " : "subroutine ");
  PutLower(os, symbol) << '(';
  int n = 0;
  for (const auto &dummy : details.dummyArgs()) {
    if (n++ > 0) os << ',';
    PutLower(os, *dummy);
  }
  os << ')';
  PutAttrs(os, bindAttrs, " "s, ""s);
  if (details.isFunction()) {
    const Symbol &result{details.result()};
    if (result.name() != symbol.name()) {
      PutLower(os << " result(", result) << ')';
    }
    os << '\n';
    PutEntity(os, details.result());
  } else {
    os << '\n';
  }
  for (const auto &dummy : details.dummyArgs()) {
    PutEntity(os, *dummy);
  }
  os << "end\n";
  if (isExternal) {
    os << "end interface\n";
  }
}

void ModFileWriter::PutGeneric(const Symbol &symbol) {
  auto &details{symbol.get<GenericDetails>()};
  PutLower(decls_ << "interface ", symbol) << '\n';
  for (auto *specific : details.specificProcs()) {
    PutLower(decls_ << "procedure::", *specific) << '\n';
  }
  decls_ << "end interface\n";
}

void ModFileWriter::PutUse(const Symbol &symbol) {
  auto &details{symbol.get<UseDetails>()};
  auto &use{details.symbol()};
  PutLower(uses_ << "use ", details.module());
  PutLower(uses_ << ",only:", symbol);
  if (use.name() != symbol.name()) {
    PutLower(uses_ << "=>", use);
  }
  uses_ << '\n';
  PutUseExtraAttr(Attr::VOLATILE, symbol, use);
  PutUseExtraAttr(Attr::ASYNCHRONOUS, symbol, use);
}

// We have "USE local => use" in this module. If attr was added locally
// (i.e. on local but not on use), also write it out in the mod file.
void ModFileWriter::PutUseExtraAttr(
    Attr attr, const Symbol &local, const Symbol &use) {
  if (local.attrs().test(attr) && !use.attrs().test(attr)) {
    PutLower(useExtraAttrs_, AttrToString(attr)) << "::";
    PutLower(useExtraAttrs_, local) << '\n';
  }
}

void ModFileWriter::PutEntity(std::ostream &os, const Symbol &symbol) {
  std::visit(
      common::visitors{
          [&](const EntityDetails &) { PutObjectEntity(os, symbol); },
          [&](const ObjectEntityDetails &) { PutObjectEntity(os, symbol); },
          [&](const ProcEntityDetails &) { PutProcEntity(os, symbol); },
          [&](const auto &) {
            common::die("ModFileWriter::PutEntity: unexpected details: %s",
                DetailsToString(symbol.details()).c_str());
          },
      },
      symbol.details());
}

void ModFileWriter::PutObjectEntity(std::ostream &os, const Symbol &symbol) {
  PutEntity(os, symbol, [&]() {
    auto *type{symbol.GetType()};
    CHECK(type);
    PutLower(os, *type);
  });
}

void ModFileWriter::PutProcEntity(std::ostream &os, const Symbol &symbol) {
  const ProcInterface &interface{symbol.get<ProcEntityDetails>().interface()};
  PutEntity(os, symbol, [&]() {
    os << "procedure(";
    if (interface.symbol()) {
      PutLower(os, *interface.symbol());
    } else if (interface.type()) {
      PutLower(os, *interface.type());
    }
    os << ')';
  });
}

// Write an entity (object or procedure) declaration.
// writeType is called to write out the type.
void ModFileWriter::PutEntity(
    std::ostream &os, const Symbol &symbol, std::function<void()> writeType) {
  writeType();
  PutAttrs(os, symbol.attrs());
  PutLower(os << "::", symbol) << '\n';
}

// Put out each attribute to os, surrounded by `before` and `after` and
// mapped to lower case.
std::ostream &ModFileWriter::PutAttrs(
    std::ostream &os, Attrs attrs, std::string before, std::string after) {
  attrs.set(Attr::PUBLIC, false);  // no need to write PUBLIC
  attrs.set(Attr::EXTERNAL, false);  // no need to write EXTERNAL
  for (std::size_t i{0}; i < Attr_enumSize; ++i) {
    Attr attr{static_cast<Attr>(i)};
    if (attrs.test(attr)) {
      PutLower(os << before, AttrToString(attr)) << after;
    }
  }
  return os;
}

std::ostream &ModFileWriter::PutLower(std::ostream &os, const Symbol &symbol) {
  return PutLower(os, symbol.name().ToString());
}

std::ostream &ModFileWriter::PutLower(
    std::ostream &os, const DeclTypeSpec &type) {
  std::stringstream s;
  s << type;
  return PutLower(os, s.str());
}

std::ostream &ModFileWriter::PutLower(
    std::ostream &os, const std::string &str) {
  for (char c : str) {
    os << parser::ToLowerCaseLetter(c);
  }
  return os;
}

// Compute a simple hash of the contents of a module file and
// return it as a string of hex digits.
// This uses the Fowler-Noll-Vo hash function.
std::string ModFileWriter::CheckSum(const std::string &str) {
  std::uint64_t hash{0xcbf29ce484222325ull};
  for (char c : str) {
    hash ^= c & 0xff;
    hash *= 0x100000001b3;
  }
  static const char *digits = "0123456789abcdef";
  std::string result(16, '0');
  for (size_t i{16}; hash != 0; hash >>= 4) {
    result[--i] = digits[hash & 0xf];
  }
  return result;
}

void WriteModFiles() { ModFileWriter{}.WriteAll(std::cerr); }

}  // namespace Fortran::semantics
