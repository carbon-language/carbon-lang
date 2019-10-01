//===---------------------StructuredData.cpp ---------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/StructuredData.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/JSON.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cerrno>
#include <cstdlib>
#include <inttypes.h>
#include <limits>

using namespace lldb_private;
using namespace llvm;

// Functions that use a JSONParser to parse JSON into StructuredData
static StructuredData::ObjectSP ParseJSONValue(JSONParser &json_parser);
static StructuredData::ObjectSP ParseJSONObject(JSONParser &json_parser);
static StructuredData::ObjectSP ParseJSONArray(JSONParser &json_parser);

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

  JSONParser json_parser(buffer_or_error.get()->getBuffer());
  return_sp = ParseJSONValue(json_parser);
  return return_sp;
}

static StructuredData::ObjectSP ParseJSONObject(JSONParser &json_parser) {
  // The "JSONParser::Token::ObjectStart" token should have already been
  // consumed by the time this function is called
  auto dict_up = std::make_unique<StructuredData::Dictionary>();

  std::string value;
  std::string key;
  while (true) {
    JSONParser::Token token = json_parser.GetToken(value);

    if (token == JSONParser::Token::String) {
      key.swap(value);
      token = json_parser.GetToken(value);
      if (token == JSONParser::Token::Colon) {
        StructuredData::ObjectSP value_sp = ParseJSONValue(json_parser);
        if (value_sp)
          dict_up->AddItem(key, value_sp);
        else
          break;
      }
    } else if (token == JSONParser::Token::ObjectEnd) {
      return StructuredData::ObjectSP(dict_up.release());
    } else if (token == JSONParser::Token::Comma) {
      continue;
    } else {
      break;
    }
  }
  return StructuredData::ObjectSP();
}

static StructuredData::ObjectSP ParseJSONArray(JSONParser &json_parser) {
  // The "JSONParser::Token::ObjectStart" token should have already been
  // consumed by the time this function is called
  auto array_up = std::make_unique<StructuredData::Array>();

  std::string value;
  std::string key;
  while (true) {
    StructuredData::ObjectSP value_sp = ParseJSONValue(json_parser);
    if (value_sp)
      array_up->AddItem(value_sp);
    else
      break;

    JSONParser::Token token = json_parser.GetToken(value);
    if (token == JSONParser::Token::Comma) {
      continue;
    } else if (token == JSONParser::Token::ArrayEnd) {
      return StructuredData::ObjectSP(array_up.release());
    } else {
      break;
    }
  }
  return StructuredData::ObjectSP();
}

static StructuredData::ObjectSP ParseJSONValue(JSONParser &json_parser) {
  std::string value;
  const JSONParser::Token token = json_parser.GetToken(value);
  switch (token) {
  case JSONParser::Token::ObjectStart:
    return ParseJSONObject(json_parser);

  case JSONParser::Token::ArrayStart:
    return ParseJSONArray(json_parser);

  case JSONParser::Token::Integer: {
    uint64_t uval;
    if (llvm::to_integer(value, uval, 0))
      return std::make_shared<StructuredData::Integer>(uval);
  } break;

  case JSONParser::Token::Float: {
    double val;
    if (llvm::to_float(value, val))
      return std::make_shared<StructuredData::Float>(val);
  } break;

  case JSONParser::Token::String:
    return std::make_shared<StructuredData::String>(value);

  case JSONParser::Token::True:
  case JSONParser::Token::False:
    return std::make_shared<StructuredData::Boolean>(token ==
                                                     JSONParser::Token::True);

  case JSONParser::Token::Null:
    return std::make_shared<StructuredData::Null>();

  default:
    break;
  }
  return StructuredData::ObjectSP();
}

StructuredData::ObjectSP StructuredData::ParseJSON(std::string json_text) {
  JSONParser json_parser(json_text);
  StructuredData::ObjectSP object_sp = ParseJSONValue(json_parser);
  return object_sp;
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
    s.attributeBegin(pair.first.AsCString());
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
