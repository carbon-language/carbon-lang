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

InputGraph::~InputGraph() { }

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
  InputElement *elem = _inputArgs[_nextElementIndex++].get();
  if (isa<GroupEnd>(elem))
    return getNextInputElement();
  return elem;
}

void InputGraph::normalize() {
  std::vector<std::unique_ptr<InputElement>> vec;
  for (std::unique_ptr<InputElement> &elt : _inputArgs) {
    if (elt->getReplacements(vec))
      continue;
    vec.push_back(std::move(elt));
  }
  _inputArgs = std::move(vec);
}

// If we are at the end of a group, return its size (which indicates
// how many files we need to go back in the command line).
// Returns 0 if we are not at the end of a group.
int InputGraph::getGroupSize() {
  if (_nextElementIndex >= _inputArgs.size())
    return 0;
  InputElement *elem = _inputArgs[_nextElementIndex].get();
  if (const GroupEnd *group = dyn_cast<GroupEnd>(elem))
    return group->getSize();
  return 0;
}

void InputGraph::skipGroup() {
  if (_nextElementIndex >= _inputArgs.size())
    return;
  if (isa<GroupEnd>(_inputArgs[_nextElementIndex].get()))
    _nextElementIndex++;
}

bool FileNode::getReplacements(InputGraph::InputElementVectorT &result) {
  if (_files.size() < 2)
    return false;
  for (std::unique_ptr<File> &file : _files)
    result.push_back(llvm::make_unique<SimpleFileNode>(_path, std::move(file)));
  return true;
}
