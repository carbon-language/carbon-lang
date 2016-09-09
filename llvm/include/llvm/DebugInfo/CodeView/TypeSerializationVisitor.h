//===- TypeSerializationVisitor.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPESERIALIZATIONVISITOR_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPESERIALIZATIONVISITOR_H

#include "llvm/DebugInfo/CodeView/FieldListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/MemoryTypeTableBuilder.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace codeview {

class TypeSerializationVisitor : public TypeVisitorCallbacks {
public:
  TypeSerializationVisitor(FieldListRecordBuilder &FieldListBuilder,
                           MemoryTypeTableBuilder &TypeTableBuilder)
      : FieldListBuilder(FieldListBuilder), TypeTableBuilder(TypeTableBuilder) {
  }

  virtual Expected<TypeLeafKind> visitTypeBegin(const CVType &Record) override {
    if (Record.Type == TypeLeafKind::LF_FIELDLIST)
      FieldListBuilder.reset();
    return Record.Type;
  }

  virtual Error visitTypeEnd(const CVRecord<TypeLeafKind> &Record) override {
    if (Record.Type == TypeLeafKind::LF_FIELDLIST)
      TypeTableBuilder.writeFieldList(FieldListBuilder);
    return Error::success();
  }

#define TYPE_RECORD(EnumName, EnumVal, Name)                                   \
  virtual Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,            \
                                 Name##Record &Record) override {              \
    visitKnownRecordImpl(Record);                                              \
    return Error::success();                                                   \
  }
#define TYPE_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#define MEMBER_RECORD(EnumName, EnumVal, Name)                                 \
  virtual Error visitKnownRecord(const CVRecord<TypeLeafKind> &CVR,            \
                                 Name##Record &Record) override {              \
    visitMemberRecordImpl(Record);                                             \
    return Error::success();                                                   \
  }
#define MEMBER_RECORD_ALIAS(EnumName, EnumVal, Name, AliasName)
#include "llvm/DebugInfo/CodeView/TypeRecords.def"

private:
  template <typename RecordKind> void visitKnownRecordImpl(RecordKind &Record) {
    TypeTableBuilder.writeKnownType(Record);
  }
  template <typename RecordKind>
  void visitMemberRecordImpl(RecordKind &Record) {
    FieldListBuilder.writeMemberType(Record);
  }

  void visitKnownRecordImpl(FieldListRecord &FieldList) {}

  FieldListRecordBuilder &FieldListBuilder;
  MemoryTypeTableBuilder &TypeTableBuilder;
};
}
}

#endif
