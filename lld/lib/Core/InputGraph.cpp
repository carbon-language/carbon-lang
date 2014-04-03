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

void InputGraph::notifyProgress() { _currentInputElement->notifyProgress(); }

bool InputGraph::addInputElement(std::unique_ptr<InputElement> ie) {
  _inputArgs.push_back(std::move(ie));
  return true;
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

/// \brief Read the file into _buffer.
error_code FileNode::getBuffer(StringRef filePath) {
  // Create a memory buffer
  std::unique_ptr<MemoryBuffer> mb;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(filePath, mb))
    return ec;
  _buffer = std::move(mb);
  return error_code::success();
}

/// \brief Return the next file that need to be processed by the resolver.
/// This also processes input elements depending on the resolve status
/// of the input elements contained in the group.
ErrorOr<File &> Group::getNextFile() {
  // If there are no elements, move on to the next input element
  if (_elements.empty())
    return make_error_code(InputGraphError::no_more_files);

  for (;;) {
    // If we have processed all the elements, and have made no progress on
    // linking, we cannot resolve any symbol from this group. Continue to the
    // next one by returning no_more_files.
    if (_nextElementIndex == _elements.size()) {
      if (!_madeProgress)
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
