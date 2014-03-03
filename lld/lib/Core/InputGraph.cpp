//===- lib/Core/InputGraph.cpp --------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/InputGraph.h"

#include "lld/Core/Resolver.h"
#include "llvm/ADT/OwningPtr.h"

using namespace lld;

static bool sortInputElements(const std::unique_ptr<InputElement> &a,
                              const std::unique_ptr<InputElement> &b) {
  return a->getOrdinal() < b->getOrdinal();
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

/// \brief Insert element at position
void InputGraph::insertElementsAt(
    std::vector<std::unique_ptr<InputElement> > inputElements,
    Position position, size_t pos) {
  if (position == InputGraph::Position::BEGIN)
    pos = 0;
  else if (position == InputGraph::Position::END)
    pos = _inputArgs.size();
  _inputArgs.insert(_inputArgs.begin() + pos,
                    std::make_move_iterator(inputElements.begin()),
                    std::make_move_iterator(inputElements.end()));
}

void InputGraph::insertOneElementAt(std::unique_ptr<InputElement> element,
                                    Position position, size_t pos) {
  if (position == InputGraph::Position::BEGIN)
    pos = 0;
  else if (position == InputGraph::Position::END)
    pos = _inputArgs.size();
  _inputArgs.insert(_inputArgs.begin() + pos, std::move(element));
}

/// \brief Helper functions for the resolver
ErrorOr<InputElement *> InputGraph::getNextInputElement() {
  if (_nextElementIndex >= _inputArgs.size())
    return make_error_code(InputGraphError::no_more_elements);
  auto elem = _inputArgs[_nextElementIndex++].get();
  // Do not return Hidden elements.
  if (!elem->isHidden())
    return elem;
  return getNextInputElement();
}

/// \brief Set the index on what inputElement has to be returned
error_code InputGraph::setNextElementIndex(uint32_t index) {
  if (index > _inputArgs.size())
    return make_error_code(llvm::errc::invalid_argument);
  _nextElementIndex = index;
  return error_code::success();
}

// Normalize the InputGraph.
void InputGraph::normalize() {
  auto iterb = _inputArgs.begin();
  auto itere = _inputArgs.end();
  auto currentIter = _inputArgs.begin();
  bool replaceCurrentNode = false;
  bool expand = false;

  std::vector<std::unique_ptr<InputElement> > _workInputArgs;
  while (iterb != itere) {
    replaceCurrentNode = false;
    expand = false;
    InputElement::ExpandType expandType = (*iterb)->expandType();
    if (expandType == InputElement::ExpandType::ReplaceAndExpand) {
      replaceCurrentNode = true;
      expand = true;
    } else if (expandType == InputElement::ExpandType::ExpandOnly) {
      replaceCurrentNode = false;
      expand = true;
    }
    currentIter = iterb++;
    if (expand)
      _workInputArgs.insert(
          _workInputArgs.end(),
          std::make_move_iterator((*currentIter)->expandElements().begin()),
          std::make_move_iterator((*currentIter)->expandElements().end()));
    if (!replaceCurrentNode)
      _workInputArgs.push_back(std::move(*currentIter));
  }
  _inputArgs = std::move(_workInputArgs);
}

/// InputElement

/// \brief Initialize the Input Element, The ordinal value of an input Element
/// is initially set to -1, if the user wants to override its ordinal,
/// let the user do it
InputElement::InputElement(Kind type, int64_t ordinal)
    : _kind(type), _ordinal(ordinal), _weight(0) {}

/// FileNode
FileNode::FileNode(StringRef path, int64_t ordinal)
    : InputElement(InputElement::Kind::File, ordinal), _path(path),
      _resolveState(Resolver::StateNoChange), _nextFileIndex(0) {}

/// \brief Read the file into _buffer.
error_code FileNode::getBuffer(StringRef filePath) {
  // Create a memory buffer
  OwningPtr<MemoryBuffer> opmb;

  if (error_code ec = MemoryBuffer::getFileOrSTDIN(filePath, opmb))
    return ec;

  std::unique_ptr<MemoryBuffer> mb(opmb.take());
  _buffer = std::move(mb);

  return error_code::success();
}

// Reset the next file that would be be processed by the resolver.
// Reset the resolve state too.
void FileNode::resetNextIndex() {
  _nextFileIndex = 0;
  setResolveState(Resolver::StateNoChange);
}

/// ControlNode

/// \brief Get the resolver State. The return value of the resolve
/// state for a control node is the or'ed value of the resolve states
/// contained in it.
uint32_t ControlNode::getResolveState() const {
  uint32_t resolveState = Resolver::StateNoChange;
  for (auto &elem : _elements)
    resolveState |= elem->getResolveState();
  return resolveState;
}

/// \brief Set the resolve state for the current element
/// thats processed by the resolver.
void ControlNode::setResolveState(uint32_t resolveState) {
  if (_elements.empty())
    return;
  _elements[_currentElementIndex]->setResolveState(resolveState);
}

/// SimpleFileNode

SimpleFileNode::SimpleFileNode(StringRef path, int64_t ordinal)
    : FileNode(path, ordinal) {}

/// Group

/// \brief Return the next file that need to be processed by the resolver.
/// This also processes input elements depending on the resolve status
/// of the input elements contained in the group.
ErrorOr<File &> Group::getNextFile() {
  // If there are no elements, move on to the next input element
  if (_elements.empty())
    return make_error_code(InputGraphError::no_more_files);

  for (;;) {
    // If we have processed all the elements as part of this node
    // check the resolver status for each input element and if the status
    // has not changed, move onto the next file.
    if (_nextElementIndex == _elements.size()) {
      if (getResolveState() == Resolver::StateNoChange)
        return make_error_code(InputGraphError::no_more_files);
      resetNextIndex();
    }
    _currentElementIndex = _nextElementIndex;
    auto file = _elements[_nextElementIndex]->getNextFile();
    // Move on to the next element if we have finished processing all
    // the files in the input element
    if (file.getError() == InputGraphError::no_more_files) {
      _nextElementIndex++;
      continue;
    }
    return *file;
  }
}
