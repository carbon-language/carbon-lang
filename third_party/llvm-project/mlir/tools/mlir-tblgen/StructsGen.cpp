//===- StructsGen.cpp - MLIR struct utility generator ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// StructsGen generates common utility functions for grouping attributes into a
// set of structured data.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::FmtContext;
using mlir::tblgen::StructAttr;

static void
emitStructClass(const Record &structDef, StringRef structName,
                llvm::ArrayRef<mlir::tblgen::StructFieldAttr> fields,
                StringRef description, raw_ostream &os) {
  const char *structInfo = R"(
// {0}
class {1} : public ::mlir::DictionaryAttr)";
  const char *structInfoEnd = R"( {
public:
  using ::mlir::DictionaryAttr::DictionaryAttr;
  static bool classof(::mlir::Attribute attr);
)";
  os << formatv(structInfo, description, structName) << structInfoEnd;

  // Declares a constructor function for the tablegen structure.
  //   TblgenStruct::get(MLIRContext context, Type1 Field1, Type2 Field2, ...);
  const char *getInfoDecl = "  static {0} get(\n";
  const char *getInfoDeclArg = "      {0} {1},\n";
  const char *getInfoDeclEnd = "      ::mlir::MLIRContext* context);\n\n";

  os << llvm::formatv(getInfoDecl, structName);

  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    os << llvm::formatv(getInfoDeclArg, storage, name);
  }
  os << getInfoDeclEnd;

  // Declares an accessor for the fields owned by the tablegen structure.
  //   namespace::storage TblgenStruct::field1() const;
  const char *fieldInfo = R"(  {0} {1}() const;
)";
  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    os << formatv(fieldInfo, storage, name);
  }

  os << "};\n\n";
}

static void emitStructDecl(const Record &structDef, raw_ostream &os) {
  StructAttr structAttr(&structDef);
  StringRef structName = structAttr.getStructClassName();
  StringRef cppNamespace = structAttr.getCppNamespace();
  StringRef description = structAttr.getSummary();
  auto fields = structAttr.getAllFields();

  // Wrap in the appropriate namespace.
  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  // Emit the struct class definition
  emitStructClass(structDef, structName, fields, description, os);

  // Close the declared namespace.
  for (auto ns : namespaces)
    os << "} // namespace " << ns << "\n";
}

static bool emitStructDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Struct Utility Declarations", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("StructAttr");
  for (const auto *def : defs) {
    emitStructDecl(*def, os);
  }

  return false;
}

static void emitFactoryDef(llvm::StringRef structName,
                           llvm::ArrayRef<mlir::tblgen::StructFieldAttr> fields,
                           raw_ostream &os) {
  const char *getInfoDecl = "{0} {0}::get(\n";
  const char *getInfoDeclArg = "    {0} {1},\n";
  const char *getInfoDeclEnd = "    ::mlir::MLIRContext* context) {";

  os << llvm::formatv(getInfoDecl, structName);

  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    os << llvm::formatv(getInfoDeclArg, storage, name);
  }
  os << getInfoDeclEnd;

  const char *fieldStart = R"(
  ::llvm::SmallVector<::mlir::NamedAttribute, {0}> fields;
)";
  os << llvm::formatv(fieldStart, fields.size());

  const char *getFieldInfo = R"(
  assert({0});
  auto {0}_id = ::mlir::Identifier::get("{0}", context);
  fields.emplace_back({0}_id, {0});
)";

  const char *getFieldInfoOptional = R"(
  if ({0}) {
    auto {0}_id = ::mlir::Identifier::get("{0}", context);
    fields.emplace_back({0}_id, {0});
  }
)";

  for (auto field : fields) {
    if (field.getType().isOptional() || field.getType().hasDefaultValue())
      os << llvm::formatv(getFieldInfoOptional, field.getName());
    else
      os << llvm::formatv(getFieldInfo, field.getName());
  }

  const char *getEndInfo = R"(
  ::mlir::Attribute dict = ::mlir::DictionaryAttr::get(context, fields);
  return dict.dyn_cast<{0}>();
}
)";
  os << llvm::formatv(getEndInfo, structName);
}

static void emitClassofDef(llvm::StringRef structName,
                           llvm::ArrayRef<mlir::tblgen::StructFieldAttr> fields,
                           raw_ostream &os) {
  const char *classofInfo = R"(
bool {0}::classof(::mlir::Attribute attr))";

  const char *classofInfoHeader = R"(
  if (!attr)
    return false;
  auto derived = attr.dyn_cast<::mlir::DictionaryAttr>();
  if (!derived)
    return false;
  int num_absent_attrs = 0;
)";

  os << llvm::formatv(classofInfo, structName) << " {";
  os << llvm::formatv(classofInfoHeader);

  FmtContext fctx;
  const char *classofArgInfo = R"(
  auto {0} = derived.get("{0}");
  if (!{0} || !({1}))
    return false;
)";
  const char *classofArgInfoOptional = R"(
  auto {0} = derived.get("{0}");
  if (!{0})
    ++num_absent_attrs;
  else if (!({1}))
    return false;
)";
  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    std::string condition =
        std::string(tgfmt(type.getConditionTemplate(), &fctx.withSelf(name)));
    if (type.isOptional() || type.hasDefaultValue())
      os << llvm::formatv(classofArgInfoOptional, name, condition);
    else
      os << llvm::formatv(classofArgInfo, name, condition);
  }

  const char *classofEndInfo = R"(
  return derived.size() + num_absent_attrs == {0};
}
)";
  os << llvm::formatv(classofEndInfo, fields.size());
}

static void
emitAccessorDef(llvm::StringRef structName,
                llvm::ArrayRef<mlir::tblgen::StructFieldAttr> fields,
                raw_ostream &os) {
  const char *fieldInfo = R"(
{0} {2}::{1}() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto {1} = derived.get("{1}");
  assert({1} && "attribute not found.");
  assert({1}.isa<{0}>() && "incorrect Attribute type found.");
  return {1}.cast<{0}>();
}
)";
  const char *fieldInfoOptional = R"(
{0} {2}::{1}() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto {1} = derived.get("{1}");
  if (!{1})
    return nullptr;
  assert({1}.isa<{0}>() && "incorrect Attribute type found.");
  return {1}.cast<{0}>();
}
)";
  const char *fieldInfoDefaultValued = R"(
{0} {2}::{1}() const {
  auto derived = this->cast<::mlir::DictionaryAttr>();
  auto {1} = derived.get("{1}");
  if (!{1}) {
    ::mlir::Builder builder(getContext());
    return {3};
  }
  assert({1}.isa<{0}>() && "incorrect Attribute type found.");
  return {1}.cast<{0}>();
}
)";
  FmtContext fmtCtx;
  fmtCtx.withBuilder("builder");

  for (auto field : fields) {
    auto name = field.getName();
    auto type = field.getType();
    auto storage = type.getStorageType();
    if (type.isOptional()) {
      os << llvm::formatv(fieldInfoOptional, storage, name, structName);
    } else if (type.hasDefaultValue()) {
      std::string defaultValue = tgfmt(type.getConstBuilderTemplate(), &fmtCtx,
                                       type.getDefaultValue());
      os << llvm::formatv(fieldInfoDefaultValued, storage, name, structName,
                          defaultValue);
    } else {
      os << llvm::formatv(fieldInfo, storage, name, structName);
    }
  }
}

static void emitStructDef(const Record &structDef, raw_ostream &os) {
  StructAttr structAttr(&structDef);
  StringRef cppNamespace = structAttr.getCppNamespace();
  StringRef structName = structAttr.getStructClassName();
  mlir::tblgen::FmtContext ctx;
  auto fields = structAttr.getAllFields();

  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  emitFactoryDef(structName, fields, os);
  emitClassofDef(structName, fields, os);
  emitAccessorDef(structName, fields, os);

  for (auto ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
}

static bool emitStructDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Struct Utility Definitions", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("StructAttr");
  for (const auto *def : defs)
    emitStructDef(*def, os);

  return false;
}

// Registers the struct utility generator to mlir-tblgen.
static mlir::GenRegistration
    genStructDecls("gen-struct-attr-decls",
                   "Generate struct utility declarations",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitStructDecls(records, os);
                   });

// Registers the struct utility generator to mlir-tblgen.
static mlir::GenRegistration
    genStructDefs("gen-struct-attr-defs", "Generate struct utility definitions",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    return emitStructDefs(records, os);
                  });
