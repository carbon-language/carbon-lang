//===- lld/Core/InputGraph.h - Files to be linked -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Inputs to the linker in the form of a Graph.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_INPUTGRAPH_H
#define LLD_INPUTGRAPH_H

#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Driver/LinkerInput.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <stack>
#include <vector>

namespace lld {

class InputElement;
class LinkingContext;

/// \brief The inputs to the linker are represented by an InputGraph. The
/// nodes in the input graph contains Input elements. The InputElements are
/// either Input Files or Control Options. The Input Files represent each Input
/// File to the linker and the control option specify what the linker needs
/// to do when it processes the option. Each InputElement that is part of the
/// Graph has also an Ordinal value associated with it. The ordinal value is
/// needed for components to figure out the relative position of the arguments
/// that appeared in the Command Line. One such example is adding the list of
/// dynamic dynamic libraries to the DT_NEEDED list with the ELF Flavor. The
/// InputElements also have a weight function that can be used to determine the
/// weight of the file, for statistical purposes. The InputGraph also would
/// contain a set of General options that are processed by the linker, which
/// control the output
class InputGraph {
public:
  typedef std::vector<std::unique_ptr<InputElement> > InputElementVectorT;
  typedef InputElementVectorT::iterator InputElementIterT;

  /// \brief Initialize the inputgraph
  InputGraph() : _ordinal(0), _numElements(0), _numFiles(0) {}

  /// \brief Adds a node into the InputGraph
  virtual bool addInputElement(std::unique_ptr<InputElement>);

  /// \brief Set Ordinals for all the InputElements that form the InputGraph
  virtual bool assignOrdinals();

  /// Destructor
  virtual ~InputGraph() {}

  /// Total number of InputFiles
  virtual int64_t numFiles() const { return _numFiles; }

  /// Total number of InputElements
  virtual int64_t numElements() const { return _numElements; }

  /// \brief Do postprocessing of the InputGraph if there is a need for the
  /// to provide additional information to the user, also rearranges
  /// InputElements by their ordinals. If an user wants to place an input file
  /// at the desired position, the user can do that
  virtual void doPostProcess();

  /// \brief Iterators
  InputElementIterT begin() { return _inputArgs.begin(); }

  InputElementIterT end() { return _inputArgs.end(); }

  /// \brief Validate the input graph
  virtual bool validate();

  /// \brief Dump the input Graph
  virtual bool dump(raw_ostream &diagnostics = llvm::errs());

  InputElement &operator[](uint32_t index) const {
    return (*_inputArgs[index]);
  }

private:
  // Input arguments
  InputElementVectorT _inputArgs;
  // Ordinals
  int64_t _ordinal;
  // Total number of InputElements
  int64_t _numElements;
  // Total number of FileNodes
  int64_t _numFiles;
};

/// \brief This describes each element in the InputGraph. The Kind
/// determines what the current node contains.
class InputElement {
public:
  /// Each input element in the graph can be a File or a control
  enum class Kind : uint8_t {
    Control,  // Represents a type associated with ControlNodes
        File, // Represents a type associated with File Nodes
  };

  /// \brief Initialize the Input Element, The ordinal value of an input Element
  /// is initially set to -1, if the user wants to override its ordinal,
  /// let the user do it
  InputElement(Kind type, int64_t ordinal = -1)
      : _kind(type), _ordinal(-1), _weight(0) {}

  virtual ~InputElement() {}

  /// Return the Element Type for an Input Element
  virtual Kind kind() const { return _kind; }

  virtual void setOrdinal(int64_t ordinal) {
    if (_ordinal != -1)
      _ordinal = ordinal;
  }

  virtual int64_t getOrdinal() const { return _ordinal; }

  virtual int64_t weight() const { return _weight; }

  virtual void setWeight(int64_t weight) { _weight = weight; }

  /// \brief validates the Input Element
  virtual bool validate() = 0;

  /// \brief Dump the Input Element
  virtual bool dump(raw_ostream &diagnostics) = 0;

private:
  Kind _kind;
  int64_t _ordinal;
  int64_t _weight;
};

/// \brief The Control class represents a control node in the InputGraph
class ControlNode : public InputElement {
public:
  /// A control node could be of several types supported by InputGraph
  /// Future kinds of Control node could be added
  enum class ControlKind : uint8_t {
    Simple,   // Represents a simple control node
        Group // Represents a type associated with ControlNodes
  };

  ControlNode(ControlNode::ControlKind controlKind =
                  ControlNode::ControlKind::Simple,
              int64_t _ordinal = -1)
      : InputElement(InputElement::Kind::Control, _ordinal),
        _controlKind(controlKind) {}

  virtual ~ControlNode() {}

  /// \brief Return the kind of control node
  virtual ControlNode::ControlKind controlKind() { return _controlKind; }

  /// \brief Process control start/exit
  virtual bool processControlEnter() { return true; }

  /// \brief Process control start/exit
  virtual bool processControlExit() { return true; }

  /// Process the input Elemenet
  virtual bool processInputElement(std::unique_ptr<InputElement> element) = 0;

  /// \brief Casting support
  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::Control;
  }

  /// Does the control node have any more elements
  bool hasMoreElements() const { return (_elements.size() != 0); }

  /// \brief Iterators to iterate the
  InputGraph::InputElementIterT begin() { return _elements.begin(); }

  InputGraph::InputElementIterT end() { return _elements.end(); }

  /// \brief Create a lld::File node from the FileNode
  virtual std::unique_ptr<LinkerInput>
  createLinkerInput(const LinkingContext &targetInfo) = 0;

private:
  ControlKind _controlKind;

protected:
  InputGraph::InputElementVectorT _elements;
};

/// \brief Represents an Input file in the graph
class FileNode : public InputElement {
public:
  FileNode(StringRef path, int64_t ordinal = -1)
      : InputElement(InputElement::Kind::File, ordinal), _path(path) {}

  virtual StringRef path(const LinkingContext &) const { return _path; }

  virtual ~FileNode() {}

  /// \brief Casting support
  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::File;
  }

  /// \brief Create a lld::File node from the FileNode
  virtual std::unique_ptr<LinkerInput>
  createLinkerInput(const LinkingContext &targetInfo) = 0;

protected:
  StringRef _path;
};

/// \brief A Control node which contains a group of InputElements
/// This affects the resolver so that it resolves undefined symbols
/// in the group completely before looking at other input files that
/// follow the group
class Group : public ControlNode {
public:
  Group() : ControlNode(ControlNode::ControlKind::Group) {}

  static inline bool classof(const InputElement *a) {
    return a->kind() == InputElement::Kind::Control;
  }

  virtual bool processInputElement(std::unique_ptr<InputElement> element) {
    _elements.push_back(std::move(element));
    return true;
  }

  virtual std::unique_ptr<lld::LinkerInput>
  createLinkerInput(const lld::LinkingContext &) = 0;
};

} // namespace lld

#endif // LLD_INPUTGRAPH_H
