//===- lib/Driver/InputGraph.cpp ------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "lld/Core/Resolver.h"
#include "lld/Driver/InputGraph.h"

using namespace lld;

namespace {
bool sortInputElements(const std::unique_ptr<InputElement> &a,
                       const std::unique_ptr<InputElement> &b) {
  return a->getOrdinal() < b->getOrdinal();
}
}

bool InputGraph::addInputElement(std::unique_ptr<InputElement> ie) {
  _inputArgs.push_back(std::move(ie));
  return true;
}

bool InputGraph::assignOrdinals() {
  for (auto &ie : _inputArgs)
    ie->setOrdinal(++_ordinal);
  return true;
}

bool InputGraph::assignFileOrdinals(uint64_t &startOrdinal) {
  for (auto &ie : _inputArgs)
    ie->assignFileOrdinals(startOrdinal);
  return true;
}

void InputGraph::doPostProcess() {
  std::stable_sort(_inputArgs.begin(), _inputArgs.end(), sortInputElements);
}

bool InputGraph::validate() {
  for (auto &ie : _inputArgs)
    if (!ie->validate())
      return false;
  return true;
}

bool InputGraph::dump(raw_ostream &diagnostics) {
  for (auto &ie : _inputArgs)
    if (!ie->dump(diagnostics))
      return false;
  return true;
}

void InputGraph::insertElementsAt(
    std::vector<std::unique_ptr<InputElement> > inputElements,
    Position position, int32_t pos) {
  if (position == InputGraph::Position::BEGIN)
    pos = 0;
  else if (position == InputGraph::Position::END)
    pos = _inputArgs.size();
  _inputArgs.insert(_inputArgs.begin() + pos,
                    std::make_move_iterator(inputElements.begin()),
                    std::make_move_iterator(inputElements.end()));
}

void InputGraph::insertOneElementAt(std::unique_ptr<InputElement> element,
                                    Position position, int32_t pos) {
  if (position == InputGraph::Position::BEGIN)
    pos = 0;
  else if (position == InputGraph::Position::END)
    pos = _inputArgs.size();
  _inputArgs.insert(_inputArgs.begin() + pos, std::move(element));
}

/// \brief Helper functions for the resolver
ErrorOr<InputElement *> InputGraph::getNextInputElement() {
  if (_nextElementIndex >= _inputArgs.size())
    return make_error_code(input_graph_error::no_more_elements);
  return _inputArgs[_nextElementIndex++].get();
}

/// \brief Set the index on what inputElement has to be returned
ErrorOr<void> InputGraph::setNextElementIndex(uint32_t index) {
  if (index > _inputArgs.size())
    return make_error_code(llvm::errc::invalid_argument);
  _nextElementIndex = index;
  return error_code::success();
}

/// InputElement

/// \brief Initialize the Input Element, The ordinal value of an input Element
/// is initially set to -1, if the user wants to override its ordinal,
/// let the user do it
InputElement::InputElement(Kind type, int64_t ordinal)
    : _kind(type), _ordinal(ordinal), _weight(0),
      _resolveState(Resolver::StateNoChange), _nextFileIndex(0) {}

/// \brief Assign File ordinals for files contained
/// in the InputElement
void FileNode::assignFileOrdinals(uint64_t &startOrdinal) {
  for (auto &file : _files)
    file->setOrdinalAndIncrement(startOrdinal);
}

/// \brief Assign File ordinals for files contained
/// in the InputElement
void ControlNode::assignFileOrdinals(uint64_t &startOrdinal) {
  for (auto &elem : _elements)
    elem->assignFileOrdinals(startOrdinal);
}

/// \brief Assign File ordinals for files contained
/// in the InputElement
void SimpleFileNode::assignFileOrdinals(uint64_t &startOrdinal) {
  for (auto &file : _files)
    file->setOrdinalAndIncrement(startOrdinal);
}
