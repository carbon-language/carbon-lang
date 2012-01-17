//===--- JSONParser.h - Simple JSON parser ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a JSON parser.
//
//  See http://www.json.org/ for an overview.
//  See http://www.ietf.org/rfc/rfc4627.txt for the full standard.
//
//  FIXME: Currently this supports a subset of JSON. Specifically, support
//  for numbers, booleans and null for values is missing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_JSON_PARSER_H
#define LLVM_SUPPORT_JSON_PARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"

namespace llvm {

class JSONContainer;
class JSONString;
class JSONValue;
class JSONKeyValuePair;

/// \brief Base class for a parsable JSON atom.
///
/// This class has no semantics other than being a unit of JSON data which can
/// be parsed out of a JSON document.
class JSONAtom {
public:
  /// \brief Possible types of JSON objects.
  enum Kind { JK_KeyValuePair, JK_Array, JK_Object, JK_String };

  /// \brief Returns the type of this value.
  Kind getKind() const { return MyKind; }

  static bool classof(const JSONAtom *Atom) { return true; }

protected:
  JSONAtom(Kind MyKind) : MyKind(MyKind) {}

private:
  Kind MyKind;
};

/// \brief A parser for JSON text.
///
/// Use an object of JSONParser to iterate over the values of a JSON text.
/// All objects are parsed during the iteration, so you can only iterate once
/// over the JSON text, but the cost of partial iteration is minimized.
/// Create a new JSONParser if you want to iterate multiple times.
class JSONParser {
public:
  /// \brief Create a JSONParser for the given input.
  ///
  /// Parsing is started via parseRoot(). Access to the object returned from
  /// parseRoot() will parse the input lazily.
  JSONParser(StringRef Input, SourceMgr *SM);

  /// \brief Returns the outermost JSON value (either an array or an object).
  ///
  /// Can return NULL if the input does not start with an array or an object.
  /// The object is not parsed yet - the caller must iterate over the
  /// returned object to trigger parsing.
  ///
  /// A JSONValue can be either a JSONString, JSONObject or JSONArray.
  JSONValue *parseRoot();

  /// \brief Parses the JSON text and returns whether it is valid JSON.
  ///
  /// In case validate() return false, failed() will return true and
  /// getErrorMessage() will return the parsing error.
  bool validate();

  /// \brief Returns true if an error occurs during parsing.
  ///
  /// If there was an error while parsing an object that was created by
  /// iterating over the result of 'parseRoot', 'failed' will return true.
  bool failed() const;

private:
  /// \brief These methods manage the implementation details of parsing new JSON
  /// atoms.
  /// @{
  JSONString *parseString();
  JSONValue *parseValue();
  JSONKeyValuePair *parseKeyValuePair();
  /// @}

  /// \brief Helpers to parse the elements out of both forms of containers.
  /// @{
  const JSONAtom *parseElement(JSONAtom::Kind ContainerKind);
  StringRef::iterator parseFirstElement(JSONAtom::Kind ContainerKind,
                                        char StartChar, char EndChar,
                                        const JSONAtom *&Element);
  StringRef::iterator parseNextElement(JSONAtom::Kind ContainerKind,
                                       char EndChar,
                                       const JSONAtom *&Element);
  /// @}

  /// \brief Whitespace parsing.
  /// @{
  void nextNonWhitespace();
  bool isWhitespace();
  /// @}

  /// \brief These methods are used for error handling.
  /// {
  void setExpectedError(StringRef Expected, StringRef Found);
  void setExpectedError(StringRef Expected, char Found);
  bool errorIfAtEndOfFile(StringRef Message);
  bool errorIfNotAt(char C, StringRef Message);
  /// }

  /// \brief Skips all elements in the given container.
  bool skipContainer(const JSONContainer &Container);

  /// \brief Skips to the next position behind the given JSON atom.
  bool skip(const JSONAtom &Atom);

  /// All nodes are allocated by the parser and will be deallocated when the
  /// parser is destroyed.
  BumpPtrAllocator ValueAllocator;

  /// \brief The original input to the parser.
  MemoryBuffer *InputBuffer;

  /// \brief The source manager used for diagnostics and buffer management.
  SourceMgr *SM;

  /// \brief The current position in the parse stream.
  StringRef::iterator Position;

  /// \brief The end position for fast EOF checks without introducing
  /// unnecessary dereferences.
  StringRef::iterator End;

  /// \brief If true, an error has occurred.
  bool Failed;

  friend class JSONContainer;
};


/// \brief Base class for JSON value objects.
///
/// This object represents an abstract JSON value. It is the root node behind
/// the group of JSON entities that can represent top-level values in a JSON
/// document. It has no API, and is just a placeholder in the type hierarchy of
/// nodes.
class JSONValue : public JSONAtom {
protected:
  JSONValue(Kind MyKind) : JSONAtom(MyKind) {}

public:
  /// \brief dyn_cast helpers
  ///@{
  static bool classof(const JSONAtom *Atom) {
    switch (Atom->getKind()) {
      case JK_Array:
      case JK_Object:
      case JK_String:
        return true;
      case JK_KeyValuePair:
        return false;
    };
    llvm_unreachable("Invalid JSONAtom kind");
  }
  static bool classof(const JSONValue *Value) { return true; }
  ///@}
};

/// \brief Gives access to the text of a JSON string.
///
/// FIXME: Implement a method to return the unescaped text.
class JSONString : public JSONValue {
public:
  /// \brief Returns the underlying parsed text of the string.
  ///
  /// This is the unescaped content of the JSON text.
  /// See http://www.ietf.org/rfc/rfc4627.txt for details.
  StringRef getRawText() const { return RawText; };

private:
  JSONString(StringRef RawText) : JSONValue(JK_String), RawText(RawText) {}

  StringRef RawText;

  friend class JSONParser;

public:
  /// \brief dyn_cast helpers
  ///@{
  static bool classof(const JSONAtom *Atom) {
    return Atom->getKind() == JK_String;
  }
  static bool classof(const JSONString *String) { return true; }
  ///@}
};

/// \brief A (key, value) tuple of type (JSONString *, JSONValue *).
///
/// Note that JSONKeyValuePair is not a JSONValue, it is a bare JSONAtom.
/// JSONKeyValuePairs can be elements of a JSONObject, but not of a JSONArray.
/// They are not viable as top-level values either.
class JSONKeyValuePair : public JSONAtom {
public:
  const JSONString * const Key;
  const JSONValue * const Value;

private:
  JSONKeyValuePair(const JSONString *Key, const JSONValue *Value)
      : JSONAtom(JK_KeyValuePair), Key(Key), Value(Value) {}

  friend class JSONParser;

public:
  /// \brief dyn_cast helpers
  ///@{
  static bool classof(const JSONAtom *Atom) {
    return Atom->getKind() == JK_KeyValuePair;
  }
  static bool classof(const JSONKeyValuePair *KeyValuePair) { return true; }
  ///@}
};

/// \brief Implementation of JSON containers (arrays and objects).
///
/// JSONContainers drive the lazy parsing of JSON arrays and objects via
/// forward iterators.
class JSONContainer : public JSONValue {
private:
  /// \brief An iterator that parses the underlying container during iteration.
  ///
  /// Iterators on the same collection use shared state, so when multiple copies
  /// of an iterator exist, only one is allowed to be used for iteration;
  /// iterating multiple copies of an iterator of the same collection will lead
  /// to undefined behavior.
  class AtomIterator {
  public:
    AtomIterator(const AtomIterator &I) : Container(I.Container) {}

    /// \brief Iterator interface.
    ///@{
    bool operator==(const AtomIterator &I) const {
      if (isEnd() || I.isEnd())
        return isEnd() == I.isEnd();
      return Container->Position == I.Container->Position;
    }
    bool operator!=(const AtomIterator &I) const {
      return !(*this == I);
    }
    AtomIterator &operator++() {
      Container->parseNextElement();
      return *this;
    }
    const JSONAtom *operator*() {
      return Container->Current;
    }
    ///@}

  private:
    /// \brief Create an iterator for which 'isEnd' returns true.
    AtomIterator() : Container(0) {}

    /// \brief Create an iterator for the given container.
    AtomIterator(const JSONContainer *Container) : Container(Container) {}

    bool isEnd() const {
      return Container == 0 || Container->Position == StringRef::iterator();
    }

    const JSONContainer * const Container;

    friend class JSONContainer;
  };

protected:
  /// \brief An iterator for the specified AtomT.
  ///
  /// Used for the implementation of iterators for JSONArray and JSONObject.
  template <typename AtomT>
  class IteratorTemplate : public std::iterator<std::forward_iterator_tag,
                                                const AtomT*> {
  public:
    explicit IteratorTemplate(const AtomIterator& AtomI)
      : AtomI(AtomI) {}

    bool operator==(const IteratorTemplate &I) const {
      return AtomI == I.AtomI;
    }
    bool operator!=(const IteratorTemplate &I) const { return !(*this == I); }

    IteratorTemplate &operator++() {
      ++AtomI;
      return *this;
    }

    const AtomT *operator*() { return dyn_cast<AtomT>(*AtomI); }

  private:
    AtomIterator AtomI;
  };

  JSONContainer(JSONParser *Parser, char StartChar, char EndChar,
                JSONAtom::Kind ContainerKind)
    : JSONValue(ContainerKind), Parser(Parser),
      Position(), Current(0), Started(false),
      StartChar(StartChar), EndChar(EndChar) {}

  /// \brief Returns a lazy parsing iterator over the container.
  ///
  /// As the iterator drives the parse stream, begin() must only be called
  /// once per container.
  AtomIterator atom_begin() const {
    if (Started)
      report_fatal_error("Cannot parse container twice.");
    Started = true;
    // Set up the position and current element when we begin iterating over the
    // container.
    Position = Parser->parseFirstElement(getKind(), StartChar, EndChar, Current);
    return AtomIterator(this);
  }
  AtomIterator atom_end() const {
    return AtomIterator();
  }

private:
  AtomIterator atom_current() const {
    if (!Started)
      return atom_begin();

    return AtomIterator(this);
  }

  /// \brief Parse the next element in the container into the Current element.
  ///
  /// This routine is called as an iterator into this container walks through
  /// its elements. It mutates the container's internal current node to point to
  /// the next atom of the container.
  void parseNextElement() const {
    Parser->skip(*Current);
    Position = Parser->parseNextElement(getKind(), EndChar, Current);
  }

  // For parsing, JSONContainers call back into the JSONParser.
  JSONParser * const Parser;

  // 'Position', 'Current' and 'Started' store the state of the parse stream
  // for iterators on the container, they don't change the container's elements
  // and are thus marked as mutable.
  mutable StringRef::iterator Position;
  mutable const JSONAtom *Current;
  mutable bool Started;

  const char StartChar;
  const char EndChar;

  friend class JSONParser;

public:
  /// \brief dyn_cast helpers
  ///@{
  static bool classof(const JSONAtom *Atom) {
    switch (Atom->getKind()) {
      case JK_Array:
      case JK_Object:
        return true;
      case JK_KeyValuePair:
      case JK_String:
        return false;
    };
    llvm_unreachable("Invalid JSONAtom kind");
  }
  static bool classof(const JSONContainer *Container) { return true; }
  ///@}
};

/// \brief A simple JSON array.
class JSONArray : public JSONContainer {
public:
  typedef IteratorTemplate<JSONValue> const_iterator;

  /// \brief Returns a lazy parsing iterator over the container.
  ///
  /// As the iterator drives the parse stream, begin() must only be called
  /// once per container.
  const_iterator begin() const { return const_iterator(atom_begin()); }
  const_iterator end() const { return const_iterator(atom_end()); }

private:
  JSONArray(JSONParser *Parser)
    : JSONContainer(Parser, '[', ']', JSONAtom::JK_Array) {}

public:
  /// \brief dyn_cast helpers
  ///@{
  static bool classof(const JSONAtom *Atom) {
    return Atom->getKind() == JSONAtom::JK_Array;
  }
  static bool classof(const JSONArray *Array) { return true; }
  ///@}

  friend class JSONParser;
};

/// \brief A JSON object: an iterable list of JSON key-value pairs.
class JSONObject : public JSONContainer {
public:
  typedef IteratorTemplate<JSONKeyValuePair> const_iterator;

  /// \brief Returns a lazy parsing iterator over the container.
  ///
  /// As the iterator drives the parse stream, begin() must only be called
  /// once per container.
  const_iterator begin() const { return const_iterator(atom_begin()); }
  const_iterator end() const { return const_iterator(atom_end()); }

private:
  JSONObject(JSONParser *Parser)
    : JSONContainer(Parser, '{', '}', JSONAtom::JK_Object) {}

public:
  /// \brief dyn_cast helpers
  ///@{
  static bool classof(const JSONAtom *Atom) {
    return Atom->getKind() == JSONAtom::JK_Object;
  }
  static bool classof(const JSONObject *Object) { return true; }
  ///@}

  friend class JSONParser;
};

} // end namespace llvm

#endif // LLVM_SUPPORT_JSON_PARSER_H
