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

#include <memory>

using namespace lld;

static bool sortInputElements(const std::unique_ptr<InputElement> &a,
                              const std::unique_ptr<InputElement> &b) {
  return a->getOrdinal() < b->getOrdinal();
}

ErrorOr<File &> InputGraph::nextFile() {
  // When nextFile() is called for the first time, _currentInputElement is not
  // initialized. Initialize it with the first element of the input graph.
  if (_currentInputElement == nullptr) {
    ErrorOr<InputElement *> elem = getNextInputElement();
    if (elem.getError() == InputGraphError::no_more_elements)
      return make_error_code(InputGraphError::no_more_files);
    _currentInputElement = *elem;
  }

  // Otherwise, try to get the next file of _currentInputElement. If the current
  // input element points to an archive file, and there's a file left in the
  // archive, it will succeed. If not, try to get the next file in the input
  // graph.
  for (;;) {
    ErrorOr<File &> nextFile = _currentInputElement->getNextFile();
    if (nextFile.getError() != InputGraphError::no_more_files)
      return std::move(nextFile);

    ErrorOr<InputElement *> elem = getNextInputElement();
    if (elem.getError() == InputGraphError::no_more_elements ||
        *elem == nullptr)
      return make_error_code(InputGraphError::no_more_files);
    _currentInputElement = *elem;
  }
}

void InputGraph::setResolverState(uint32_t state) {
  _currentInputElement->setResolveState(state);
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

bool InputGraph::dump(raw_ostream &diagnostics) {
  for (auto &ie : _inputArgs)
    if (!ie->dump(diagnostics))
      return false;
  return true;
}

/// \brief Insert element at position
void InputGraph::insertElementAt(std::unique_ptr<InputElement> element,
                                 Position position) {
  if (position == InputGraph::Position::BEGIN) {
    _inputArgs.insert(_inputArgs.begin(), std::move(element));
    return;
  }
  assert(position == InputGraph::Position::END);
  _inputArgs.push_back(std::move(element));
}

/// \brief Helper functions for the resolver
ErrorOr<InputElement *> InputGraph::getNextInputElement() {
  if (_nextElementIndex >= _inputArgs.size())
    return make_error_code(InputGraphError::no_more_elements);
  return _inputArgs[_nextElementIndex++].get();
}

void InputGraph::normalize() {
  auto iterb = _inputArgs.begin();
  auto itere = _inputArgs.end();
  auto currentIter = _inputArgs.begin();

  std::vector<std::unique_ptr<InputElement> > _workInputArgs;
  while (iterb != itere) {
    bool expand = (*iterb)->shouldExpand();
    currentIter = iterb++;
    if (expand) {
      _workInputArgs.insert(
          _workInputArgs.end(),
          std::make_move_iterator((*currentIter)->expandElements().begin()),
          std::make_move_iterator((*currentIter)->expandElements().end()));
    } else {
      _workInputArgs.push_back(std::move(*currentIter));
    }
  }
  _inputArgs = std::move(_workInputArgs);
}

/// InputElement

/// \brief Initialize the Input Element, The ordinal value of an input Element
/// is initially set to -1, if the user wants to override its ordinal,
/// let the user do it
InputElement::InputElement(Kind type, int64_t ordinal)
    : _kind(type), _ordinal(ordinal) {}

/// FileNode
FileNode::FileNode(StringRef path, int64_t ordinal)
    : InputElement(InputElement::Kind::File, ordinal), _path(path),
      _resolveState(Resolver::StateNoChange), _nextFileIndex(0) {}

/// \brief Read the file into _buffer.
error_code FileNode::getBuffer(StringRef filePath) {
  // Create a memory buffer
  std::unique_ptr<MemoryBuffer> mb;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(filePath, mb))
    return ec;
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
uint32_t Group::getResolveState() const {
  uint32_t resolveState = Resolver::StateNoChange;
  for (auto &elem : _elements)
    resolveState |= elem->getResolveState();
  return resolveState;
}

/// \brief Set the resolve state for the current element
/// thats processed by the resolver.
void Group::setResolveState(uint32_t resolveState) {
  if (_elements.empty())
    return;
  _elements[_currentElementIndex]->setResolveState(resolveState);
}

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
