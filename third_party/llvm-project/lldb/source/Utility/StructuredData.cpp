//===-- StructuredData.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StructuredData.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Status.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cerrno>
#include <cinttypes>
#include <cstdlib>

using namespace lldb_private;
using namespace llvm;

static StructuredData::ObjectSP ParseJSONValue(json::Value &value);
static StructuredData::ObjectSP ParseJSONObject(json::Object *object);
static StructuredData::ObjectSP ParseJSONArray(json::Array *array);

StructuredData::ObjectSP
StructuredData::ParseJSON(const std::string &json_text) {
  llvm::Expected<json::Value> value = json::parse(json_text);
  if (!value) {
    llvm::consumeError(value.takeError());
    return nullptr;
  }
  return ParseJSONValue(*value);
}

StructuredData::ObjectSP
StructuredData::ParseJSONFromFile(const FileSpec &input_spec, Status &error) {
  StructuredData::ObjectSP return_sp;

  auto buffer_or_error = llvm::MemoryBuffer::getFile(input_spec.GetPath());
  if (!buffer_or_error) {
    error.SetErrorStringWithFormatv("could not open input file: {0} - {1}.",
                                    input_spec.GetPath(),
                                    buffer_or_error.getError().message());
    return return_sp;
  }
  llvm::Expected<json::Value> value =
      json::parse(buffer_or_error.get()->getBuffer().str());
  if (value)
    return ParseJSONValue(*value);
  error.SetErrorString(toString(value.takeError()));
  return StructuredData::ObjectSP();
}

static StructuredData::ObjectSP ParseJSONValue(json::Value &value) {
  if (json::Object *O = value.getAsObject())
    return ParseJSONObject(O);

  if (json::Array *A = value.getAsArray())
    return ParseJSONArray(A);

  if (auto s = value.getAsString())
    return std::make_shared<StructuredData::String>(*s);

  if (auto b = value.getAsBoolean())
    return std::make_shared<StructuredData::Boolean>(*b);

  if (auto i = value.getAsInteger())
    return std::make_shared<StructuredData::Integer>(*i);

  if (auto d = value.getAsNumber())
    return std::make_shared<StructuredData::Float>(*d);

  return StructuredData::ObjectSP();
}

static StructuredData::ObjectSP ParseJSONObject(json::Object *object) {
  auto dict_up = std::make_unique<StructuredData::Dictionary>();
  for (auto &KV : *object) {
    StringRef key = KV.first;
    json::Value value = KV.second;
    if (StructuredData::ObjectSP value_sp = ParseJSONValue(value))
      dict_up->AddItem(key, value_sp);
  }
  return std::move(dict_up);
}

static StructuredData::ObjectSP ParseJSONArray(json::Array *array) {
  auto array_up = std::make_unique<StructuredData::Array>();
  for (json::Value &value : *array) {
    if (StructuredData::ObjectSP value_sp = ParseJSONValue(value))
      array_up->AddItem(value_sp);
  }
  return std::move(array_up);
}

StructuredData::ObjectSP
StructuredData::Object::GetObjectForDotSeparatedPath(llvm::StringRef path) {
  if (this->GetType() == lldb::eStructuredDataTypeDictionary) {
    std::pair<llvm::StringRef, llvm::StringRef> match = path.split('.');
    std::string key = match.first.str();
    ObjectSP value = this->GetAsDictionary()->GetValueForKey(key);
    if (value.get()) {
      // Do we have additional words to descend?  If not, return the value
      // we're at right now.
      if (match.second.empty()) {
        return value;
      } else {
        return value->GetObjectForDotSeparatedPath(match.second);
      }
    }
    return ObjectSP();
  }

  if (this->GetType() == lldb::eStructuredDataTypeArray) {
    std::pair<llvm::StringRef, llvm::StringRef> match = path.split('[');
    if (match.second.empty()) {
      return this->shared_from_this();
    }
    errno = 0;
    uint64_t val = strtoul(match.second.str().c_str(), nullptr, 10);
    if (errno == 0) {
      return this->GetAsArray()->GetItemAtIndex(val);
    }
    return ObjectSP();
  }

  return this->shared_from_this();
}

void StructuredData::Object::DumpToStdout(bool pretty_print) const {
  json::OStream stream(llvm::outs(), pretty_print ? 2 : 0);
  Serialize(stream);
}

void StructuredData::Array::Serialize(json::OStream &s) const {
  s.arrayBegin();
  for (const auto &item_sp : m_items) {
    item_sp->Serialize(s);
  }
  s.arrayEnd();
}

void StructuredData::Integer::Serialize(json::OStream &s) const {
  s.value(static_cast<int64_t>(m_value));
}

void StructuredData::Float::Serialize(json::OStream &s) const {
  s.value(m_value);
}

void StructuredData::Boolean::Serialize(json::OStream &s) const {
  s.value(m_value);
}

void StructuredData::String::Serialize(json::OStream &s) const {
  s.value(m_value);
}

void StructuredData::Dictionary::Serialize(json::OStream &s) const {
  s.objectBegin();
  for (const auto &pair : m_dict) {
    s.attributeBegin(pair.first.GetStringRef());
    pair.second->Serialize(s);
    s.attributeEnd();
  }
  s.objectEnd();
}

void StructuredData::Null::Serialize(json::OStream &s) const {
  s.value(nullptr);
}

void StructuredData::Generic::Serialize(json::OStream &s) const {
  s.value(llvm::formatv("{0:X}", m_object));
}
