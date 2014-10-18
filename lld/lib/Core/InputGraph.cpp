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

ErrorOr<File &> InputGraph::getNextFile() {
  // Try to get the next file of _currentInputElement. If the current input
  // element points to an archive file, and there's a file left in the archive,
  // it will succeed. If not, try to get the next file in the input graph.
  for (;;) {
    if (_currentInputElement) {
      ErrorOr<File &> next = _currentInputElement->getNextFile();
      if (next.getError() != InputGraphError::no_more_files) {
        for (const std::function<void(File *)> &observer : _observers)
          observer(&next.get());
        return std::move(next);
      }
    }

    ErrorOr<InputElement *> elt = getNextInputElement();
    if (elt.getError() == InputGraphError::no_more_elements || *elt == nullptr)
      return make_error_code(InputGraphError::no_more_files);
    _currentInputElement = *elt;
  }
}

void InputGraph::notifyProgress() { _currentInputElement->notifyProgress(); }

void InputGraph::registerObserver(std::function<void(File *)> fn) {
  _observers.push_back(fn);
}

void InputGraph::addInputElement(std::unique_ptr<InputElement> ie) {
  _inputArgs.push_back(std::move(ie));
}

void InputGraph::addInputElementFront(std::unique_ptr<InputElement> ie) {
  _inputArgs.insert(_inputArgs.begin(), std::move(ie));
}

bool InputGraph::dump(raw_ostream &diagnostics) {
  for (std::unique_ptr<InputElement> &ie : _inputArgs)
    if (!ie->dump(diagnostics))
      return false;
  return true;
}

/// \brief Helper functions for the resolver
ErrorOr<InputElement *> InputGraph::getNextInputElement() {
  if (_nextElementIndex >= _inputArgs.size())
    return make_error_code(InputGraphError::no_more_elements);
  return _inputArgs[_nextElementIndex++].get();
}

void InputGraph::normalize() {
  for (std::unique_ptr<InputElement> &elt : _inputArgs)
    elt->expand();
  std::vector<std::unique_ptr<InputElement>> vec;
  for (std::unique_ptr<InputElement> &elt : _inputArgs) {
    if (elt->getReplacements(vec))
      continue;
    vec.push_back(std::move(elt));
  }
  _inputArgs = std::move(vec);
}

/// \brief Read the file into _buffer.
std::error_code FileNode::getBuffer(StringRef filePath) {
  // Create a memory buffer
  ErrorOr<std::unique_ptr<MemoryBuffer>> mb =
      MemoryBuffer::getFileOrSTDIN(filePath);
  if (std::error_code ec = mb.getError())
    return ec;
  _buffer = std::move(mb.get());
  return std::error_code();
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
